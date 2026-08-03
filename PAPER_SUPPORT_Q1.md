# Q1-Level Paper Positioning and Evidence Plan

## 1. Source paper being extended

The base paper is **ETFNet: An efficient transformer-based RGB--IR fusion network for UAV object detection**, published in *Information Fusion*. ETFNet reports a TGF transformer fusion module and CAFEM context enhancement. The released extension preserves the paired RGB--IR UAV detection setting and replaces TGF with SJPA-TR.

ETFNet's reported public-benchmark reference points are:

| Benchmark | ETFNet result |
|---|---:|
| DroneVehicle | 82.8 mAP50 |
| VTUAV-det | 78.0 mAP50 |
| RGBTDronePerson | 45.50 mAP50 |
| DroneVehicle model cost | 18.9M parameters, 32.3 GFLOPs |

These are paper-reported values, not results reproduced by this repository.

---

## 2. Why a generic alignment/reliability claim is not novel enough

Current literature already covers the broad ingredients:

| Direction | Representative paper | Why it narrows the novelty claim |
|---|---|---|
| Weak-alignment offset learning | OAFA, CVPR 2024 | Learns a common subspace and deformable offsets for UAV RGB--IR alignment. |
| Unified offset alignment and gated fusion | CoDAF, 2026 | Combines offset-guided alignment, deformable convolution, shared semantics, and adaptive gating. |
| Reliability-aware alignment/fusion | LER-YOLO, 2026 preprint | Estimates spatial reliability and routes sparse fusion experts. |
| Uncertainty-aware implicit alignment | UMFNet, CVPR 2026 | Uses Gaussian latent uncertainty and confidence-guided fusion for unaligned RGB-T SOD. |
| Distribution-aligned robust fusion | CVPR 2026 | Aligns fused features to a pretrained detector distribution for unseen degradations. |
| Orthogonal Procrustes in learned features | Deep Orthogonal Procrustes; PEARL, CVPR 2026 | Establishes that Procrustes itself and parameter-free feature rotation are not new in vision. |
| Sparse foreground-only fusion | Efficient RGB-T Object Detection via Sparse Cross-Modality Fusion, 2026 | Shows that efficiency via selective fusion is also occupied. |

Consequently, a paper cannot safely claim novelty from any one of these phrases: *alignment*, *reliability*, *uncertainty*, *gating*, *parameter-free*, or *Procrustes*.

---

## 3. Defensible novelty statement

A cautious, testable statement is:

> We introduce a parameter-free RGB--IR UAV fusion operator that combines running groupwise whitening and closed-form orthogonal channel alignment with a gauge-invariant Procrustes spatial score over a finite zero-padded translation set, followed by a simplex-valued multiplicative reliability correction with a formal norm bound.

The likely contribution is the **specific mathematical construction and its coupling**, not the individual components.

A stronger “first” claim should only be used after a documented systematic review. Suggested wording:

> To the best of our knowledge, prior RGB--IR UAV detectors have not combined these four properties in one parameter-free early-fusion operator.

Do not write “the first Procrustes RGB--IR method” unless the final literature search supports it.

---

## 4. Contributions suitable for the manuscript

### Contribution 1 — Latent-coordinate formulation

Formulate RGB--IR channel fusion as a latent coordinate compatibility problem rather than assuming corresponding learned channels are directly commensurate.

### Contribution 2 — Closed-form, groupwise channel alignment

Use running groupwise statistics and the exact orthogonal Procrustes solution. Prove constrained optimality and preserve feature energy under orthogonal reparameterization.

### Contribution 3 — Procrustes-optimal selective spatial score

Score each finite zero-padded shift by the nuclear norm of the aligned cross-covariance. Prove that this equals the maximum orthogonal correlation available for that candidate. Use reliability and zero-score gates to avoid unconditional registration.

### Contribution 4 — Bounded competitive reliability

Use a two-simplex reliability vector and prove that each modality correction is bounded by its trigger-scaled feature norm.

### Contribution 5 — Reproducible paired RGB--IR system

Provide strict pair validation, six-channel cache correctness, exact epoch-boundary resume, deterministic manifests, model-config validation, paired video inference, and export tests.

---

## 5. Evidence already available

The release currently establishes software and mathematical correctness, not public-benchmark superiority:

- 49 automated unit/integration tests pass.
- All 12 shipped model configurations build and run forward.
- Six-channel forward/backward and BF16 autocast pass.
- Orthogonality, whitening, simplex, correction-bound, gauge-equivariance, and no-wrap shift tests pass.
- Exact interrupted/resumed training is tensor-identical at an epoch boundary.
- Train, validation, paired image/video inference, TorchScript export, and TorchScript prediction pass on a synthetic OBB fixture.
- A built wheel installs into an isolated target, imports from the installed artifact, constructs the full SJPA-TR graph, completes a six-channel forward pass, and exposes the unified CLI.
- A five-seed synthetic proxy shows strong results under individual noise, missing-modality, and translation conditions.
- The same proxy exposes a severe compound noise-plus-shift failure; this limitation is retained rather than hidden.

These tests support implementation claims. They cannot replace DroneVehicle, VTUAV-det, or RGBTDronePerson experiments.

---

## 6. Minimum experimental package for a Q1-level submission

### Public benchmark comparison

Train ETFNet-TGF and SJPA-TR from the same corrected repository, under the same data loader, backbone, augmentation, optimizer, image size, epoch count, and seeds.

Use at least three independent seeds; five are preferable for the main dataset. Report mean, sample standard deviation, 95% confidence interval, and per-seed values.

### Mandatory ablation matrix

| Variant | Whitening | Channel Procrustes | Spatial score | Selective gate | Reliability bound |
|---|---:|---:|---:|---:|---:|
| Corrected ETFNet-TGF | — | — | — | — | — |
| Direct concatenation | — | — | — | — | — |
| GOCI | Yes | Yes | No | — | Yes |
| SJPA without reliability | Yes | Yes | Yes | Yes | No |
| SJPA without selectivity | Yes | Yes | Yes | No | Yes |
| SJPA-TR | Yes | Yes | Yes | Yes | Yes |

Also compare nuclear versus Frobenius candidate scoring, group widths, anchor sizes, maximum shift, and fusion stage.

### Capacity-matched controls

Because SJPA-TR removes trainable TGF parameters, include:

1. Original ETFNet-TGF.
2. TGF reduced to match SJPA-TR total parameters where feasible.
3. SJPA-TR plus an unrelated parameter-matched convolutional control.
4. Direct concatenation with the same downstream channel width.

This separates mathematical design from model capacity.

### Robustness suite

Evaluate clean and controlled corruptions separately and jointly:

- RGB and IR Gaussian/Poisson/impulse noise;
- blur and low contrast;
- RGB darkness and IR saturation/crossover;
- local occlusion;
- modality dropout;
- translations of 1--8 pixels and small rotations;
- compound noise plus misalignment.

Report mAP retention, not only absolute mAP:

\[
R_c=\frac{\operatorname{mAP}_{c}}{\operatorname{mAP}_{clean}}.
\]

### Geometry diagnostics

Report:

- selected-shift accuracy on synthetically shifted pairs;
- false shift rate on aligned pairs;
- shift-selection entropy;
- native/export shift disagreement;
- zero-padding boundary effects;
- results stratified by small/medium/large objects.

### Efficiency and deployment

Report parameters, GFLOPs, peak GPU memory, end-to-end latency, throughput, and energy on at least one workstation GPU and one representative edge platform. Measure data loading and postprocessing separately from the network kernel.

---

## 7. Decision thresholds

A Q1 submission should not be based on a tiny average gain. A defensible target is:

- statistically supported clean mAP improvement over corrected ETFNet-TGF on the primary benchmark, or clean parity with a substantial and statistically supported robustness/efficiency improvement;
- no material regression on the other two datasets;
- clear small-object or weak-alignment gains matching the stated mechanism;
- lower parameters without hidden latency or memory inflation;
- compound-degradation results that either improve or are explicitly bounded as a limitation.

If SJPA-TR fails to exceed TGF on real data, the paper should be reframed as a robustness/geometry study rather than claiming state of the art.

---

## 8. Suggested paper structure

1. Introduction: latent coordinate ambiguity, weak spatial alignment, reliability conflict.
2. Related work: UAV RGB--IR detection; weak alignment; reliability-aware fusion; closed-form feature alignment.
3. Method: corrected ETFNet base; whitening; Procrustes channel map; finite spatial score; reliability bound.
4. Theory: Propositions 1--3 and groupwise gauge property.
5. Experiments: clean benchmarks, ablations, robustness, geometry diagnostics, deployment.
6. Limitations: finite translations, compound corruption, export surrogate, groupwise assumption.
7. Reproducibility statement and artifact checklist.

---

## 9. Paper-support references

- ETFNet, *Information Fusion*, DOI: 10.1016/j.inffus.2026.104658.
- Chen et al., “Weakly Misalignment-free Adaptive Feature Alignment for UAVs-based Multimodal Object Detection,” CVPR 2024.
- Liu et al., “Cross-modal Offset-guided Dynamic Alignment and Fusion for Weakly Aligned UAV Object Detection,” *Applied Soft Computing*, 2026, DOI: 10.1016/j.asoc.2026.116028.
- Hou et al., “LER-YOLO: Reliability-Aware Expert Routing for Misaligned RGB-Infrared UAV Detection,” arXiv:2605.20667, 2026.
- Wang et al., “Uncertainty-Aware Modality Fusion for Unaligned RGB-T Salient Object Detection,” CVPR 2026.
- Hao et al., “Distribution-Aligned Multimodal Fusion for Robust Object Detection,” CVPR 2026.
- Pei et al., “PEARL: Geometry Aligns Semantics for Training-Free Open-Vocabulary Semantic Segmentation,” CVPR 2026.
- Thopalli et al., “The Surprising Effectiveness of Deep Orthogonal Procrustes Alignment in Unsupervised Domain Adaptation,” *IEEE Access*, 2023.
- Tian et al., “Efficient RGB-T Object Detection via Sparse Cross-Modality Fusion,” arXiv:2606.30215, 2026.
