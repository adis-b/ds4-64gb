/*
 * ds4_residency: page-cache residency policy for the mmap'd model.
 *
 * On Macs with less RAM than the model file (notably the 64 GB target),
 * we don't want every weight page warm. Instead we:
 *   - keep "always-hot" weights (norms, attention, embeddings, output) WILLNEED'd
 *     and (optionally) mlock'd so the OS cannot evict them
 *   - track which routed experts the router actually selected per layer
 *   - WILLNEED+mlock the top-K most-used experts (subject to a byte budget),
 *     DONTNEED the rest
 *
 * Without mlock, hints are advisory and the OS can evict warm pages under
 * memory pressure. On hosts where the mapped model exceeds physical RAM that
 * causes continuous SSD page-ins during inference. With a sensible
 * lock_budget_bytes the hot working set becomes genuinely pinned and only the
 * cold tail thrashes.
 *
 * This module deliberately knows nothing about ds4_tensor / ds4_model. The
 * caller passes the mmap base + size and supplies file regions as
 * (offset, length) pairs. That keeps the module compilable in isolation
 * and avoids ABI coupling with ds4.c internals.
 */

#ifndef DS4_RESIDENCY_H
#define DS4_RESIDENCY_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

typedef struct ds4_residency ds4_residency;

typedef struct {
    /* Total layers and experts in this model */
    uint32_t n_layers;
    uint32_t n_experts;
    /* If > 0, after this many tokens worth of routing observations the policy
     * is reapplied (top-K warm, rest cool) and learning re-windows. 0 leaves
     * apply timing entirely up to the caller. */
    uint32_t learn_tokens;
    /* Number of experts per layer to keep warm. 0 disables top-K policy. */
    uint32_t resident_per_layer;
    /* Decay multiplier applied at each apply window (0 = zero counters,
     * 1 = sticky, 0.5 = half-life of one window). */
    float    decay;
    /* If true, the apply call also issues DONTNEED on the cold experts.
     * Set false to only WILLNEED hot experts and leave the rest to LRU. */
    bool     evict_cold;
    /* If > 0, the residency module will mlock warm regions (essentials and
     * top-K routed experts) up to this many bytes in total. Beyond the
     * budget it falls back to WILLNEED so we never starve the OS.
     *
     * Essentials are locked first and remain locked for the engine lifetime.
     * The remainder of the budget is used by each apply() call: regions that
     * fall out of the new top-K are munlocked, regions that newly enter the
     * top-K are mlocked as long as the running total stays under budget.
     *
     * 0 disables mlock entirely (legacy madvise-only behavior). */
    uint64_t lock_budget_bytes;
} ds4_residency_options;

ds4_residency *ds4_residency_create(const ds4_residency_options *opt);
void ds4_residency_free(ds4_residency *r);

/* The mmap of the model file. */
typedef struct {
    const void *map;
    uint64_t    size;
} ds4_residency_mmap;

/* WILLNEED / DONTNEED a single (offset, length) region. Page-aligned and
 * clamped to the file. No-ops if the platform lacks posix_madvise. */
void ds4_residency_warm_region(ds4_residency_mmap m, uint64_t off, uint64_t len);
void ds4_residency_cool_region(ds4_residency_mmap m, uint64_t off, uint64_t len);

/* mlock / munlock a single region, page-aligned and clamped to the file.
 * lock_region returns the number of bytes actually pinned (the aligned span
 * on success, 0 on failure). These are stateless and have no budget; the
 * caller is responsible for tracking the running total. Useful for pinning
 * the always-hot essentials before the residency policy object exists. */
uint64_t ds4_residency_lock_region(ds4_residency_mmap m, uint64_t off, uint64_t len);
void     ds4_residency_unlock_region(ds4_residency_mmap m, uint64_t off, uint64_t len);

/* Returns the residency module's current mlock footprint (essentials plus
 * the routed experts currently locked by apply()). 0 if no module or no
 * budget. */
uint64_t ds4_residency_locked_bytes(const ds4_residency *r);

/* Tells the residency module that the caller has already mlocked this many
 * bytes for the always-hot essentials, so the running total reported by
 * apply() and dump_stats matches reality. Idempotent: each call replaces
 * the prior value. */
void ds4_residency_set_locked_essentials(ds4_residency *r, uint64_t bytes);

/* Walk a list of "always-hot" file regions and WILLNEED them. Intended to be
 * called once at engine startup for everything that isn't a routed expert.
 * Returns total bytes hinted. */
uint64_t ds4_residency_warm_essentials(ds4_residency *r,
                                       ds4_residency_mmap m,
                                       const uint64_t *offsets,
                                       const uint64_t *lengths,
                                       uint32_t        n);

/* Record one token worth of routing decisions. selected is laid out as
 * [n_layers][n_expert_used] of int32. NOT thread-safe per object. */
void ds4_residency_record_token(ds4_residency *r,
                                const int32_t *selected,
                                uint32_t       n_expert_used);

/* True when the policy has enough samples to be (re)applied. */
bool ds4_residency_should_apply(const ds4_residency *r);

/* For a given layer, returns (in *out) the top-K expert ids by historical
 * hit count, plus the total observed hits in *total_hits. K is at most
 * resident_per_layer (clamped). out must hold resident_per_layer entries. */
uint32_t ds4_residency_layer_top_k(const ds4_residency *r,
                                   uint32_t layer,
                                   uint32_t *out,
                                   uint64_t *total_hits);

/* Convenience: walk every layer, ask the caller for its routed-expert file
 * regions, and apply WILLNEED on the top-K + (optionally) DONTNEED on the
 * rest. The fn callback receives layer + expert and must fill (gate, up,
 * down) regions; setting len=0 marks "absent".
 *
 * If reset is true, counters are decayed/zeroed per the options. */
typedef struct {
    uint64_t off;
    uint64_t len;
} ds4_residency_region;

typedef void (*ds4_residency_expert_regions_fn)(
        void                    *ud,
        uint32_t                 layer,
        uint32_t                 expert,
        ds4_residency_region    *gate,
        ds4_residency_region    *up,
        ds4_residency_region    *down);

void ds4_residency_apply(ds4_residency                  *r,
                         ds4_residency_mmap              m,
                         ds4_residency_expert_regions_fn fn,
                         void                           *ud,
                         bool                            reset);

void ds4_residency_dump_stats(const ds4_residency *r, FILE *fp);

#endif /* DS4_RESIDENCY_H */
