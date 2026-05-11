#include "ds4_residency.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

struct ds4_residency {
    ds4_residency_options opt;
    uint64_t *hits;
    uint64_t  observed_tokens;
    uint64_t  total_tokens;
    uint64_t  apply_count;
};

static uint64_t res_page(void) {
    long ps = sysconf(_SC_PAGESIZE);
    return ps > 0 ? (uint64_t)ps : 4096u;
}

ds4_residency *ds4_residency_create(const ds4_residency_options *opt) {
    if (!opt || opt->n_layers == 0 || opt->n_experts == 0) return NULL;
    ds4_residency *r = calloc(1, sizeof(*r));
    if (!r) return NULL;
    r->opt = *opt;
    r->hits = calloc((size_t)opt->n_layers * opt->n_experts, sizeof(r->hits[0]));
    if (!r->hits) { free(r); return NULL; }
    return r;
}

void ds4_residency_free(ds4_residency *r) {
    if (!r) return;
    free(r->hits);
    free(r);
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

uint64_t ds4_residency_warm_essentials(ds4_residency *r,
                                       ds4_residency_mmap m,
                                       const uint64_t *offsets,
                                       const uint64_t *lengths,
                                       uint32_t        n) {
    (void)r;
    if (!offsets || !lengths) return 0;
    uint64_t total = 0;
    for (uint32_t i = 0; i < n; i++) {
        ds4_residency_warm_region(m, offsets[i], lengths[i]);
        total += lengths[i];
    }
    fprintf(stderr,
            "ds4: residency: warm-essentials WILLNEED %.2f GiB across %u tensors\n",
            (double)total / (1024.0 * 1024.0 * 1024.0), n);
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
            } else if (r->opt.evict_cold) {
                ds4_residency_cool_region(m, g.off, g.len); cooled += g.len;
                ds4_residency_cool_region(m, u.off, u.len); cooled += u.len;
                ds4_residency_cool_region(m, d.off, d.len); cooled += d.len;
            }
        }
    }

    free(top);
    free(mark);

    r->apply_count++;
    fprintf(stderr,
            "ds4: residency: apply #%llu top-%u/layer "
            "warm=%.2f GiB cool=%.2f GiB observed=%llu lifetime=%llu\n",
            (unsigned long long)r->apply_count,
            r->opt.resident_per_layer,
            (double)warmed / (1024.0 * 1024.0 * 1024.0),
            (double)cooled / (1024.0 * 1024.0 * 1024.0),
            (unsigned long long)r->observed_tokens,
            (unsigned long long)r->total_tokens);

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
}

void ds4_residency_dump_stats(const ds4_residency *r, FILE *fp) {
    if (!r || !fp) return;
    fprintf(fp,
            "ds4: residency stats: layers=%u experts=%u observed=%llu lifetime=%llu applies=%llu\n",
            r->opt.n_layers, r->opt.n_experts,
            (unsigned long long)r->observed_tokens,
            (unsigned long long)r->total_tokens,
            (unsigned long long)r->apply_count);
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
