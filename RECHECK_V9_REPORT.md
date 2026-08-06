# Recheck v9: Corrected SJPA and DCSPF-Guard

## Executive finding

No implementation or theorem can guarantee that a learned detector is “100% better” on every seed, corruption, dataset, and metric. The recheck therefore used a stricter target:

1. eliminate known statistical and comparison defects;
2. preserve a fair backbone and initialization protocol;
3. require reproducible numerical tests;
4. add a mathematically explicit fallback candidate;
5. separate proxy evidence from real VEDAI evidence.

The corrected SJPA remains strongly better than direct concatenation in the controlled proxy. DCSPF-Guard slightly raises mean clean proxy performance and improves the simultaneous mixed-corruption mean relative to SJPA, but it does **not** significantly dominate SJPA across five seeds. DCSPF is therefore an experimental stronger candidate, not a proven replacement.

## Newly discovered experimental defect

The earlier proxy constructed methods sequentially from one random-number stream. Methods with different constructors consumed different random draws, so their downstream heads did not receive component-wise identical initialization. The v9 fair benchmark resets deterministic seeds independently for:

- data generation;
- fusion construction;
- downstream head construction;
- data-loader shuffling;
- each corruption realization.

This removes a hidden initialization confound.

## Fair controlled proxy, five seeds

| Method | Clean mean ± SD | Robust mean ± SD | Mixed corruption |
|---|---:|---:|---:|
| Direct concatenation | 0.37262 ± 0.06248 | 0.04584 ± 0.01099 | 0.01519 |
| Corrected SJPA | 0.49638 ± 0.09389 | **0.26510 ± 0.05462** | 0.00347 |
| DCSPF-Guard | **0.50449 ± 0.09028** | 0.26094 ± 0.05694 | 0.00839 |

Paired DCSPF comparisons:

| Comparison | Mean difference | Two-sided paired p | Seed wins |
|---|---:|---:|---:|
| Clean vs concatenation | +0.13186 | 0.02358 | 5/5 |
| Robust vs concatenation | +0.21511 | 0.00171 | 5/5 |
| Clean vs corrected SJPA | +0.00811 | 0.27645 | 3/5 |
| Robust vs corrected SJPA | −0.00416 | 0.46576 | 2/5 |
| Mixed vs corrected SJPA | +0.00492 | 0.22417 | 4/5 |

Interpretation:

- both corrected methods beat concatenation strongly in this proxy;
- DCSPF has the highest clean mean;
- corrected SJPA has the highest aggregate robustness mean;
- DCSPF's advantage over SJPA is not statistically established;
- neither method is proven superior on real VEDAI by these proxy values.

## Threshold routing audit

For the selected thresholds

\[
(\tau_c,\tau_t,\tau_d)=(0.35,0.60,0.80),
\]

mean canonical routing rates were:

| Condition | Canonical route rate |
|---|---:|
| Clean | 0.9949 |
| RGB noise | 1.0000 |
| IR noise | 1.0000 |
| Missing RGB | 1.0000 |
| Missing IR | 1.0000 |
| IR shift | 0.9349 |
| Simultaneous mixed corruption | **0.0000** |

The normalized coherence plus reliability-dominance guard detects the synthetic both-bad region that raw nuclear cross-correlation failed to identify. Noise can increase an unnormalized cross-covariance norm; normalizing by within-modality covariance energy is essential.

## Production model audit

| Model | Trainable parameters | Fusion parameters |
|---|---:|---:|
| Direct concatenation | 11,205,554 | 0 |
| Corrected SJPA | 11,205,554 | 0 |
| DCSPF-Guard | 11,271,602 | 66,048 |

DCSPF overhead is approximately 0.589% of the direct-concatenation model.

## Verification status

- Full repository test suite: 105 tests after adding certificate tests.
- Mathematical/numerical tests cover bounded coherence, identity initialization, guard logic, finite forward/backward values, corrected reliability statistics, and candidate certification.
- All model YAML configurations construct and execute in the repository tests.
- The corrected main comparison keeps the C3k2 backbone unchanged.

## Novelty position

Broad claims such as “the first alignment-and-reliability RGB–IR detector” are unsafe. Recent work already includes learned offset alignment with dynamic gating, uncertainty-aware fusion, distribution alignment, and reliability-guided expert routing.

The narrower, defensible DCSPF contribution is:

> A dual-coordinate RGB–IR fusion operator combining groupwise closed-form covariance/Procrustes canonicalization with bounded RV-style coherence, same-domain energy reliability, margin-certified shift acceptance, and deterministic evidence routing between canonical and raw coordinate experts.

The strongest mathematical distinctions are:

1. the channel transform is the closed-form optimizer of an orthogonal Procrustes problem rather than a learned attention alignment;
2. the coherence statistic is explicitly normalized and proven bounded;
3. spatial correction requires a quantified score margin over zero shift;
4. the guard has an exact logical fallback region based on joint coherence/typicality and single-stream dominance;
5. modality gain has a closed-form bound;
6. the adapters preserve selected representations exactly at initialization;
7. candidate deployment can be subjected to a family-wise Hoeffding lower-confidence certificate.

## Required full experiment before a superiority claim

Use the corrected SJPA and DCSPF YAMLs with:

- identical C3k2 backbone;
- same pretrained or random initialization policy;
- component-wise saved initial weights;
- at least five seeds;
- 100 or more epochs with identical schedules;
- no threshold selection on the test set;
- mean ± SD for mAP50 and mAP50–95;
- paired image bootstrap confidence intervals;
- clean, natural day/night, missing-modality, noise, and spatial-shift tests;
- latency, FLOPs, parameters, and peak memory;
- ablations for Procrustes, coherence, typicality, dominance, spatial margin, and expert adapters.

A valid strongest-result statement requires the proposal's lower confidence bound to exceed every baseline on the primary metric. Until then, write “achieved the highest mean in the controlled proxy,” not “is 100% better.”

## Recommended manuscript positioning

- **Main conservative model:** corrected parameter-free SJPA, because it has the highest robust proxy mean and no parameter increase.
- **Experimental enhanced model:** DCSPF-Guard, because it has the highest clean proxy mean and a meaningful mixed-failure fallback, but has not significantly beaten corrected SJPA.
- Do not merge their results or choose the better model after viewing final test outcomes. Predeclare the primary model and metric before the final test.
