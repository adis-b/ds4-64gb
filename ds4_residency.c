#include "ds4_residency.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define DS4_RESIDENCY_CACHE_MAGIC   "DS4_RES1"
#define DS4_RESIDENCY_CACHE_VERSION 1u

struct ds4_residency {
    ds4_residency_options opt;
    char     *cache_path;        /* owned copy of opt.cache_path; NULL if disabled */
    bool      has_loaded_state;  /* true iff cache_path was read in create() */
    uint64_t *hits;
    uint64_t  observed_tokens;
    uint64_t  total_tokens;
    uint64_t  apply_count;
    /* mlock bookkeeping */
    uint8_t  *locked_mask;       /* n_layers * n_experts bytes, 0/1 */
    uint64_t  locked_essentials; /* bytes locked from warm_essentials */
    uint64_t  locked_experts;    /* bytes currently locked from routed experts */
    /* One-shot warning so we don't spam the log when the budget is tight */
    bool      lock_overflow_warned;
    bool      lock_failed_warned;
};

/* Routing-cache header. Written/read verbatim in host byte order. We don't
 * try to make the cache portable across machines; it's a perf cache local
 * to one install. The magic + version + (n_layers, n_experts) check is just
 * to reject obviously-wrong files. */
typedef struct {
    char     magic[8];      /* "DS4_RES1" exactly */
    uint32_t version;       /* DS4_RESIDENCY_CACHE_VERSION */
    uint32_t n_layers;      /* must equal opt.n_layers on load */
    uint32_t n_experts;     /* must equal opt.n_experts on load */
    uint32_t reserved0;
    uint64_t total_tokens;
    uint64_t apply_count;
} ds4_residency_cache_header;

static uint64_t res_page(void) {
    long ps = sysconf(_SC_PAGESIZE);
    return ps > 0 ? (uint64_t)ps : 4096u;
}

/* Try to read a routing-cache file written by save_state. On success, *hits
 * is populated and the file's total_tokens / apply_count are returned via
 * the out params. Returns 0 on success, -1 on missing file, malformed
 * header, or shape mismatch. */
static int res_read_cache_file(const char *path,
                               uint32_t    n_layers,
                               uint32_t    n_experts,
                               uint64_t   *hits,
                               uint64_t   *total_tokens_out,
                               uint64_t   *apply_count_out) {
    if (!path || !path[0]) return -1;
    FILE *fp = fopen(path, "rb");
    if (!fp) return -1;
    ds4_residency_cache_header h;
    if (fread(&h, sizeof(h), 1, fp) != 1) { fclose(fp); return -1; }
    if (memcmp(h.magic, DS4_RESIDENCY_CACHE_MAGIC, 8) != 0 ||
        h.version   != DS4_RESIDENCY_CACHE_VERSION ||
        h.n_layers  != n_layers ||
        h.n_experts != n_experts) {
        fclose(fp);
        return -1;
    }
    const size_t n_cells = (size_t)n_layers * n_experts;
    if (fread(hits, sizeof(hits[0]), n_cells, fp) != n_cells) {
        fclose(fp);
        return -1;
    }
    if (total_tokens_out) *total_tokens_out = h.total_tokens;
    if (apply_count_out)  *apply_count_out  = h.apply_count;
    fclose(fp);
    return 0;
}

ds4_residency *ds4_residency_create(const ds4_residency_options *opt) {
    if (!opt || opt->n_layers == 0 || opt->n_experts == 0) return NULL;
    ds4_residency *r = calloc(1, sizeof(*r));
    if (!r) return NULL;
    r->opt = *opt;
    r->hits = calloc((size_t)opt->n_layers * opt->n_experts, sizeof(r->hits[0]));
    if (!r->hits) { free(r); return NULL; }
    if (opt->lock_budget_bytes > 0) {
        r->locked_mask = calloc((size_t)opt->n_layers * opt->n_experts,
                                sizeof(r->locked_mask[0]));
        if (!r->locked_mask) { free(r->hits); free(r); return NULL; }
    }
    /* Routing-cache: store a private copy of the path so the caller can
     * reuse / free its string, and attempt to seed counters from prior runs.
     * A missing or malformed cache is not an error -- we just start empty. */
    if (opt->cache_path && opt->cache_path[0]) {
        size_t n = strlen(opt->cache_path);
        r->cache_path = malloc(n + 1);
        if (r->cache_path) memcpy(r->cache_path, opt->cache_path, n + 1);
        uint64_t loaded_total = 0, loaded_applies = 0;
        if (res_read_cache_file(r->cache_path,
                                opt->n_layers, opt->n_experts,
                                r->hits,
                                &loaded_total, &loaded_applies) == 0) {
            r->has_loaded_state = true;
            r->total_tokens = loaded_total;
            r->apply_count  = loaded_applies;
            fprintf(stderr,
                    "ds4: residency: loaded routing cache from %s "
                    "(lifetime=%llu applies=%llu)\n",
                    r->cache_path,
                    (unsigned long long)loaded_total,
                    (unsigned long long)loaded_applies);
        } else {
            fprintf(stderr,
                    "ds4: residency: no usable routing cache at %s; "
                    "will populate on first apply\n",
                    r->cache_path);
        }
    }
    return r;
}

void ds4_residency_free(ds4_residency *r) {
    if (!r) return;
    /* Persist whatever partial-window observations have accumulated since
     * the last apply. apply() saves on every fire, but a short run that
     * doesn't reach learn_tokens would otherwise discard up to learn_tokens
     * worth of routing data. Best-effort: a failed save is logged inside
     * save_state's caller-path elsewhere; here it just gets ignored. */
    if (r->cache_path) {
        (void)ds4_residency_save_state(r, r->cache_path);
    }
    /* Note: we intentionally do NOT munlock here. The mmap is owned by the
     * caller; when it goes away the mlocks go away with it. Unlocking from
     * here would require knowing every region we ever locked, which is more
     * book-keeping than the engine shutdown path actually needs. */
    free(r->locked_mask);
    free(r->cache_path);
    free(r->hits);
    free(r);
}

bool ds4_residency_has_loaded_state(const ds4_residency *r) {
    return r && r->has_loaded_state;
}

int ds4_residency_save_state(const ds4_residency *r, const char *path) {
    if (!r || !path || !path[0]) return -1;
    /* Atomic write: serialize to <path>.tmp, then rename. A crash mid-write
     * therefore can't leave behind a truncated cache that we'd later try to
     * load. */
    size_t plen = strlen(path);
    char *tmp = malloc(plen + 5);
    if (!tmp) return -1;
    memcpy(tmp, path, plen);
    memcpy(tmp + plen, ".tmp", 5);
    FILE *fp = fopen(tmp, "wb");
    if (!fp) { free(tmp); return -1; }

    ds4_residency_cache_header h;
    memset(&h, 0, sizeof(h));
    memcpy(h.magic, DS4_RESIDENCY_CACHE_MAGIC, 8);
    h.version      = DS4_RESIDENCY_CACHE_VERSION;
    h.n_layers     = r->opt.n_layers;
    h.n_experts    = r->opt.n_experts;
    h.total_tokens = r->total_tokens;
    h.apply_count  = r->apply_count;
    const size_t n_cells = (size_t)r->opt.n_layers * r->opt.n_experts;
    int ok = 1;
    if (fwrite(&h, sizeof(h), 1, fp) != 1) ok = 0;
    if (ok && fwrite(r->hits, sizeof(r->hits[0]), n_cells, fp) != n_cells) ok = 0;
    if (fclose(fp) != 0) ok = 0;
    if (!ok) { remove(tmp); free(tmp); return -1; }
    if (rename(tmp, path) != 0) { remove(tmp); free(tmp); return -1; }
    free(tmp);
    return 0;
}

uint64_t ds4_residency_locked_bytes(const ds4_residency *r) {
    if (!r) return 0;
    return r->locked_essentials + r->locked_experts;
}

void ds4_residency_set_locked_essentials(ds4_residency *r, uint64_t bytes) {
    if (!r) return;
    r->locked_essentials = bytes;
}

static void res_madvise(ds4_residency_mmap m, uint64_t off, uint64_t len, int advice) {
    if (!m.map || len == 0 || off >= m.size) return;
    if (len > m.size - off) len = m.size - off;
    uint64_t page = res_page();
    uint64_t a_off = off & ~(page - 1u);
    uint64_t a_end = ((off + len) + page - 1u) & ~(page - 1u);
    if (a_end > m.size) a_end = m.size;
    if (a_off >= a_end) return;
    void *base = (void *)((const uint8_t *)m.map + a_off);
    size_t span = (size_t)(a_end - a_off);
#if defined(POSIX_MADV_WILLNEED) && defined(POSIX_MADV_DONTNEED)
    (void)posix_madvise(base, span, advice);
#else
    (void)base; (void)span; (void)advice;
#endif
}

/* Page-aligned mlock of a file region.
 *
 * Returns the number of bytes actually pinned (>=0) on success, or 0 on
 * failure. The returned size is the aligned span, which is what callers
 * need to feed to munlock() later. */
static size_t res_lock(ds4_residency_mmap m, uint64_t off, uint64_t len) {
    if (!m.map || len == 0 || off >= m.size) return 0;
    if (len > m.size - off) len = m.size - off;
    uint64_t page = res_page();
    uint64_t a_off = off & ~(page - 1u);
    uint64_t a_end = ((off + len) + page - 1u) & ~(page - 1u);
    if (a_end > m.size) a_end = m.size;
    if (a_off >= a_end) return 0;
    void *base = (void *)((const uint8_t *)m.map + a_off);
    size_t span = (size_t)(a_end - a_off);
    if (mlock(base, span) != 0) {
        return 0;
    }
    return span;
}

static void res_unlock(ds4_residency_mmap m, uint64_t off, uint64_t len) {
    if (!m.map || len == 0 || off >= m.size) return;
    if (len > m.size - off) len = m.size - off;
    uint64_t page = res_page();
    uint64_t a_off = off & ~(page - 1u);
    uint64_t a_end = ((off + len) + page - 1u) & ~(page - 1u);
    if (a_end > m.size) a_end = m.size;
    if (a_off >= a_end) return;
    void *base = (void *)((const uint8_t *)m.map + a_off);
    size_t span = (size_t)(a_end - a_off);
    (void)munlock(base, span);
}

void ds4_residency_warm_region(ds4_residency_mmap m, uint64_t off, uint64_t len) {
#if defined(POSIX_MADV_WILLNEED)
    res_madvise(m, off, len, POSIX_MADV_WILLNEED);
#else
    (void)m; (void)off; (void)len;
#endif
}

void ds4_residency_cool_region(ds4_residency_mmap m, uint64_t off, uint64_t len) {
#if defined(POSIX_MADV_DONTNEED)
    res_madvise(m, off, len, POSIX_MADV_DONTNEED);
#else
    (void)m; (void)off; (void)len;
#endif
}

uint64_t ds4_residency_lock_region(ds4_residency_mmap m, uint64_t off, uint64_t len) {
    return (uint64_t)res_lock(m, off, len);
}

void ds4_residency_unlock_region(ds4_residency_mmap m, uint64_t off, uint64_t len) {
    res_unlock(m, off, len);
}

uint64_t ds4_residency_warm_essentials(ds4_residency *r,
                                       ds4_residency_mmap m,
                                       const uint64_t *offsets,
                                       const uint64_t *lengths,
                                       uint32_t        n) {
    if (!offsets || !lengths) return 0;
    uint64_t total = 0;
    uint64_t locked = 0;
    uint64_t lock_failed = 0;
    const uint64_t budget = r ? r->opt.lock_budget_bytes : 0;
    for (uint32_t i = 0; i < n; i++) {
        ds4_residency_warm_region(m, offsets[i], lengths[i]);
        total += lengths[i];
        if (budget > 0 && locked < budget) {
            size_t pinned = res_lock(m, offsets[i], lengths[i]);
            if (pinned > 0) {
                locked += pinned;
            } else {
                lock_failed += lengths[i];
            }
        }
    }
    if (r) {
        r->locked_essentials = locked;
        if (lock_failed > 0) {
            r->lock_failed_warned = true;
        }
    }
    if (budget > 0) {
        fprintf(stderr,
                "ds4: residency: warm-essentials WILLNEED %.2f GiB across %u tensors"
                " (mlock %.2f GiB ok, %.2f GiB unlocked)\n",
                (double)total / (1024.0 * 1024.0 * 1024.0), n,
                (double)locked / (1024.0 * 1024.0 * 1024.0),
                (double)lock_failed / (1024.0 * 1024.0 * 1024.0));
    } else {
        fprintf(stderr,
                "ds4: residency: warm-essentials WILLNEED %.2f GiB across %u tensors\n",
                (double)total / (1024.0 * 1024.0 * 1024.0), n);
    }
    return total;
}

void ds4_residency_record_token(ds4_residency *r,
                                const int32_t *selected,
                                uint32_t       n_expert_used) {
    if (!r || !selected || n_expert_used == 0) return;
    for (uint32_t L = 0; L < r->opt.n_layers; L++) {
        const int32_t *row = selected + (size_t)L * n_expert_used;
        for (uint32_t k = 0; k < n_expert_used; k++) {
            int32_t e = row[k];
            if (e < 0 || (uint32_t)e >= r->opt.n_experts) continue;
            r->hits[(size_t)L * r->opt.n_experts + (uint32_t)e]++;
        }
    }
    r->observed_tokens++;
    r->total_tokens++;
}

bool ds4_residency_should_apply(const ds4_residency *r) {
    if (!r) return false;
    if (r->opt.learn_tokens == 0) return false;
    if (r->opt.resident_per_layer == 0) return false;
    return r->observed_tokens >= r->opt.learn_tokens;
}

/* Min-heap based partial sort: keeps the K largest values, in any order. */
static void res_top_k(const uint64_t *hits, uint32_t n, uint32_t k, uint32_t *out) {
    if (k == 0) return;
    if (k > n) k = n;
    for (uint32_t i = 0; i < k; i++) out[i] = i;
    for (int32_t i = (int32_t)(k / 2) - 1; i >= 0; i--) {
        uint32_t root = (uint32_t)i;
        for (;;) {
            uint32_t l = 2 * root + 1, rr = 2 * root + 2, sm = root;
            if (l  < k && hits[out[l]]  < hits[out[sm]]) sm = l;
            if (rr < k && hits[out[rr]] < hits[out[sm]]) sm = rr;
            if (sm == root) break;
            uint32_t t = out[root]; out[root] = out[sm]; out[sm] = t;
            root = sm;
        }
    }
    for (uint32_t i = k; i < n; i++) {
        if (hits[i] <= hits[out[0]]) continue;
        out[0] = i;
        uint32_t root = 0;
        for (;;) {
            uint32_t l = 2 * root + 1, rr = 2 * root + 2, sm = root;
            if (l  < k && hits[out[l]]  < hits[out[sm]]) sm = l;
            if (rr < k && hits[out[rr]] < hits[out[sm]]) sm = rr;
            if (sm == root) break;
            uint32_t t = out[root]; out[root] = out[sm]; out[sm] = t;
            root = sm;
        }
    }
}

uint32_t ds4_residency_layer_top_k(const ds4_residency *r,
                                   uint32_t layer,
                                   uint32_t *out,
                                   uint64_t *total_hits) {
    if (!r || !out || layer >= r->opt.n_layers) return 0;
    uint32_t k = r->opt.resident_per_layer;
    if (k > r->opt.n_experts) k = r->opt.n_experts;
    if (k == 0) {
        if (total_hits) *total_hits = 0;
        return 0;
    }
    const uint64_t *row = r->hits + (size_t)layer * r->opt.n_experts;
    if (total_hits) {
        uint64_t sum = 0;
        for (uint32_t e = 0; e < r->opt.n_experts; e++) sum += row[e];
        *total_hits = sum;
    }
    res_top_k(row, r->opt.n_experts, k, out);
    return k;
}

void ds4_residency_apply(ds4_residency                  *r,
                         ds4_residency_mmap              m,
                         ds4_residency_expert_regions_fn fn,
                         void                           *ud,
                         bool                            reset) {
    if (!r || !fn) return;
    if (r->opt.resident_per_layer == 0) return;

    uint32_t k = r->opt.resident_per_layer;
    if (k > r->opt.n_experts) k = r->opt.n_experts;

    uint32_t *top  = malloc((size_t)k * sizeof(uint32_t));
    uint8_t  *mark = malloc((size_t)r->opt.n_experts * sizeof(uint8_t));
    if (!top || !mark) { free(top); free(mark); return; }

    uint64_t warmed = 0, cooled = 0;

    /* Mlock accounting.
     *
     * Locking is layered: essentials are locked in warm_essentials and stay
     * locked. The lock budget that's still available to routed experts is
     * (budget - essentials). On each apply we first walk every layer to
     * (1) unlock previously-locked experts that are no longer in the top-K
     * and clear the mask, then (2) walk again to lock the new top-K within
     * the remaining budget. We unlock first so the second pass has the most
     * possible headroom; without that, an apply that touches a similar set
     * to the previous one could fail to relock anything because the budget
     * still appears full.
     */
    const uint64_t budget = r->opt.lock_budget_bytes;
    uint64_t budget_left =
        budget > r->locked_essentials ? budget - r->locked_essentials : 0;
    uint64_t lock_attempted = 0;
    uint64_t lock_overflow  = 0;
    bool lock_active = (budget > 0) && (r->locked_mask != NULL);

    if (lock_active) {
        /* Pass 1: unlock experts that were previously locked. We blow away
         * the entire mask here and rebuild it in pass 2; that avoids needing
         * a second buffer for the new top-K set. */
        for (uint32_t L = 0; L < r->opt.n_layers; L++) {
            const uint64_t *row = r->hits + (size_t)L * r->opt.n_experts;
            res_top_k(row, r->opt.n_experts, k, top);
            memset(mark, 0, r->opt.n_experts);
            for (uint32_t i = 0; i < k; i++) mark[top[i]] = 1;

            for (uint32_t e = 0; e < r->opt.n_experts; e++) {
                uint8_t *cell = &r->locked_mask[(size_t)L * r->opt.n_experts + e];
                if (!*cell) continue;
                if (mark[e]) continue; /* still in top-K, keep locked */
                ds4_residency_region g = {0,0}, u = {0,0}, d = {0,0};
                fn(ud, L, e, &g, &u, &d);
                res_unlock(m, g.off, g.len);
                res_unlock(m, u.off, u.len);
                res_unlock(m, d.off, d.len);
                uint64_t freed = g.len + u.len + d.len;
                if (r->locked_experts > freed) r->locked_experts -= freed;
                else r->locked_experts = 0;
                *cell = 0;
            }
        }
        budget_left =
            budget > r->locked_essentials + r->locked_experts
                ? budget - r->locked_essentials - r->locked_experts
                : 0;
    }

    /* Pass 2: WILLNEED + (optional) mlock for top-K, DONTNEED for the rest. */
    for (uint32_t L = 0; L < r->opt.n_layers; L++) {
        const uint64_t *row = r->hits + (size_t)L * r->opt.n_experts;
        res_top_k(row, r->opt.n_experts, k, top);
        memset(mark, 0, r->opt.n_experts);
        for (uint32_t i = 0; i < k; i++) mark[top[i]] = 1;

        for (uint32_t e = 0; e < r->opt.n_experts; e++) {
            ds4_residency_region g = {0,0}, u = {0,0}, d = {0,0};
            fn(ud, L, e, &g, &u, &d);
            if (mark[e]) {
                ds4_residency_warm_region(m, g.off, g.len); warmed += g.len;
                ds4_residency_warm_region(m, u.off, u.len); warmed += u.len;
                ds4_residency_warm_region(m, d.off, d.len); warmed += d.len;
                if (lock_active) {
                    uint8_t *cell = &r->locked_mask[(size_t)L * r->opt.n_experts + e];
                    if (!*cell) {
                        uint64_t needed = g.len + u.len + d.len;
                        lock_attempted += needed;
                        if (budget_left >= needed) {
                            size_t pg = res_lock(m, g.off, g.len);
                            size_t pu = res_lock(m, u.off, u.len);
                            size_t pd = res_lock(m, d.off, d.len);
                            uint64_t pinned = (uint64_t)pg + pu + pd;
                            if (pg && pu && pd) {
                                r->locked_experts += pinned;
                                budget_left = budget_left >= pinned ? budget_left - pinned : 0;
                                *cell = 1;
                            } else {
                                /* Partial failure: unwind so the page-cache
                                 * accounting matches the locked_mask state. */
                                if (pg) res_unlock(m, g.off, g.len);
                                if (pu) res_unlock(m, u.off, u.len);
                                if (pd) res_unlock(m, d.off, d.len);
                                if (!r->lock_failed_warned) {
                                    fprintf(stderr,
                                            "ds4: residency: mlock failed for layer %u expert %u"
                                            " (errno=%d), falling back to WILLNEED\n",
                                            L, e, errno);
                                    r->lock_failed_warned = true;
                                }
                            }
                        } else {
                            lock_overflow += needed;
                        }
                    }
                }
            } else if (r->opt.evict_cold) {
                ds4_residency_cool_region(m, g.off, g.len); cooled += g.len;
                ds4_residency_cool_region(m, u.off, u.len); cooled += u.len;
                ds4_residency_cool_region(m, d.off, d.len); cooled += d.len;
            }
        }
    }

    if (lock_active && lock_overflow > 0 && !r->lock_overflow_warned) {
        fprintf(stderr,
                "ds4: residency: %.2f GiB of hot experts exceeded the %.2f GiB lock budget"
                " and remain pageable\n",
                (double)lock_overflow / (1024.0 * 1024.0 * 1024.0),
                (double)budget / (1024.0 * 1024.0 * 1024.0));
        r->lock_overflow_warned = true;
    }

    free(top);
    free(mark);

    r->apply_count++;
    if (lock_active) {
        fprintf(stderr,
                "ds4: residency: apply #%llu top-%u/layer "
                "warm=%.2f GiB cool=%.2f GiB locked=%.2f GiB (ess %.2f + exp %.2f) "
                "observed=%llu lifetime=%llu\n",
                (unsigned long long)r->apply_count,
                r->opt.resident_per_layer,
                (double)warmed / (1024.0 * 1024.0 * 1024.0),
                (double)cooled / (1024.0 * 1024.0 * 1024.0),
                (double)(r->locked_essentials + r->locked_experts) /
                    (1024.0 * 1024.0 * 1024.0),
                (double)r->locked_essentials / (1024.0 * 1024.0 * 1024.0),
                (double)r->locked_experts / (1024.0 * 1024.0 * 1024.0),
                (unsigned long long)r->observed_tokens,
                (unsigned long long)r->total_tokens);
        (void)lock_attempted;
    } else {
        fprintf(stderr,
                "ds4: residency: apply #%llu top-%u/layer "
                "warm=%.2f GiB cool=%.2f GiB observed=%llu lifetime=%llu\n",
                (unsigned long long)r->apply_count,
                r->opt.resident_per_layer,
                (double)warmed / (1024.0 * 1024.0 * 1024.0),
                (double)cooled / (1024.0 * 1024.0 * 1024.0),
                (unsigned long long)r->observed_tokens,
                (unsigned long long)r->total_tokens);
    }

    if (reset) {
        if (r->opt.decay <= 0.0f) {
            memset(r->hits, 0,
                   (size_t)r->opt.n_layers * r->opt.n_experts * sizeof(r->hits[0]));
        } else if (r->opt.decay < 1.0f) {
            const size_t n = (size_t)r->opt.n_layers * r->opt.n_experts;
            for (size_t i = 0; i < n; i++) {
                r->hits[i] = (uint64_t)((double)r->hits[i] * (double)r->opt.decay);
            }
        }
        r->observed_tokens = 0;
    }

    /* Persist the post-apply (post-decay) state so the next run starts from
     * exactly the same counters. We do this every apply rather than only on
     * shutdown so a kill -9 still leaves a usable cache behind. The file is
     * ~n_layers * n_experts * 8 bytes -- on DS4 Flash that's ~88 KiB,
     * which is well below the cost of mistakes here. */
    if (r->cache_path) {
        if (ds4_residency_save_state(r, r->cache_path) != 0) {
            /* Saving is best-effort; a broken cache disk shouldn't kill the
             * inference run. Warn once and move on. */
            static bool save_failed_warned = false;
            if (!save_failed_warned) {
                fprintf(stderr,
                        "ds4: residency: failed to write routing cache to %s "
                        "(errno=%d); continuing without persistence\n",
                        r->cache_path, errno);
                save_failed_warned = true;
            }
        }
    }
}

void ds4_residency_dump_stats(const ds4_residency *r, FILE *fp) {
    if (!r || !fp) return;
    fprintf(fp,
            "ds4: residency stats: layers=%u experts=%u observed=%llu lifetime=%llu applies=%llu"
            " locked=%.2f GiB (ess %.2f + exp %.2f, budget %.2f)\n",
            r->opt.n_layers, r->opt.n_experts,
            (unsigned long long)r->observed_tokens,
            (unsigned long long)r->total_tokens,
            (unsigned long long)r->apply_count,
            (double)(r->locked_essentials + r->locked_experts) /
                (1024.0 * 1024.0 * 1024.0),
            (double)r->locked_essentials / (1024.0 * 1024.0 * 1024.0),
            (double)r->locked_experts / (1024.0 * 1024.0 * 1024.0),
            (double)r->opt.lock_budget_bytes / (1024.0 * 1024.0 * 1024.0));
    if (r->total_tokens == 0) return;

    const uint32_t TOP = 8;
    uint32_t *top = malloc((size_t)TOP * sizeof(uint32_t));
    if (!top) return;
    for (uint32_t L = 0; L < r->opt.n_layers; L++) {
        const uint64_t *row = r->hits + (size_t)L * r->opt.n_experts;
        uint64_t sum = 0;
        for (uint32_t e = 0; e < r->opt.n_experts; e++) sum += row[e];
        if (sum == 0) continue;
        res_top_k(row, r->opt.n_experts, TOP, top);
        uint64_t covered = 0;
        for (uint32_t i = 0; i < TOP; i++) covered += row[top[i]];
        for (uint32_t i = 0; i < TOP; i++) {
            for (uint32_t j = i + 1; j < TOP; j++) {
                if (row[top[j]] > row[top[i]]) {
                    uint32_t t = top[i]; top[i] = top[j]; top[j] = t;
                }
            }
        }
        fprintf(fp, "  layer %2u: top%u covers %.1f%% (",
                L, TOP, 100.0 * (double)covered / (double)sum);
        for (uint32_t i = 0; i < TOP; i++) {
            fprintf(fp, "%s%u:%llu",
                    i ? " " : "", top[i], (unsigned long long)row[top[i]]);
        }
        fprintf(fp, ")\n");
    }
    free(top);
}
