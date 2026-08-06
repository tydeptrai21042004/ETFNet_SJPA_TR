# Corrected SJPA-TR Proposal: Test Results, Mathematics, and Novelty Positioning

## 1. Decision

The recommended **main proposal** is:

> **Statistically Calibrated SJPA-TR on the C3k2 dual-stream backbone**

Configuration:

```text
ultralytics/cfg/models/etfnet/etfnet_P2_C3k2_SJPA_TR_Corrected.yaml
```

This is preferred over the original `etfnet_P2_CAFEM_SJPA.yaml` because the original experiment changed both the fusion rule and the backbone. The corrected configuration is identical to `dual_concat` except at fusion layer 9. It therefore provides a fair comparison of the proposed fusion rule.

The optional `RTPF` model is retained as a safety ablation, not as the main performance model:

```text
ultralytics/cfg/models/etfnet/etfnet_P2_C3k2_RTPF.yaml
```

RTPF is mathematically safer, but its controlled proxy training was too conservative and did not outperform direct SJPA.

---

## 2. Confirmed root defect

The original code estimated whitening statistics on an `8 x 8` pooled feature grid but measured reliability energy on the unpooled P2 feature map. For a `512 x 512` image, P2 is approximately `128 x 128`.

The old computation was

\[
E_m^{\mathrm{old}}
=\frac{1}{CHW}\|A_m\|_F^2,
\]

where the whitening matrix had been estimated from pooled averages. Average pooling reduces covariance, so applying its inverse square root to the full-resolution map inflated feature energy.

A direct diagnostic produced:

| Quantity | Measured value |
|---|---:|
| Full-resolution RGB energy | 209.67 |
| Full-resolution IR energy | 185.36 |
| Old reliability trigger | 1.0000 |
| Corrected pooled RGB energy | 0.7843 |
| Corrected pooled IR energy | 0.6981 |
| Corrected trigger | 0.0528 |

Thus, the old reliability correction was effectively always active. It also prevented the spatial search from operating because the same anomaly score was used as an eligibility condition.

The corrected energy is computed in the same anchor domain as the covariance:

\[
E_m
=\frac{1}{C a^2}
\left\|P_a(A_m)\right\|_F^2,
\]

where \(P_a\) is adaptive pooling to an \(a\times a\) anchor grid.

---

## 3. Corrected mathematical method

Let RGB and IR features be

\[
R,I\in\mathbb{R}^{B\times C\times H\times W}.
\]

Channels are divided into \(G\) groups of width \(d=C/G\).

### 3.1 Groupwise statistical calibration

For each group \(g\), pooled tokens are centered and covariance matrices are estimated:

\[
\Sigma_{r,g}=\frac{1}{N-1}X_{r,g}^{\top}X_{r,g},\qquad
\Sigma_{i,g}=\frac{1}{N-1}X_{i,g}^{\top}X_{i,g}.
\]

Whitening operators are

\[
W_{r,g}=(\Sigma_{r,g}+\varepsilon I)^{-1/2},\qquad
W_{i,g}=(\Sigma_{i,g}+\varepsilon I)^{-1/2}.
\]

### 3.2 Closed-form orthogonal Procrustes alignment

The cross-covariance in the whitened domain is

\[
M_g=W_{r,g}\Sigma_{ri,g}W_{i,g}.
\]

If

\[
M_g=U_gS_gV_g^{\top},
\]

then

\[
Q_g^{\star}=U_gV_g^{\top}
\]

solves

\[
Q_g^{\star}
=\arg\min_{Q^{\top}Q=I}
\|X_{r,g}W_{r,g}Q-X_{i,g}W_{i,g}\|_F^2.
\]

This gives a closed-form channel-coordinate alignment without trainable fusion parameters.

### 3.3 Selective spatial correction

For candidate translation \(\delta\in\{-s,\ldots,s\}^2\), define the normalized score

\[
S(\delta)
=\frac{1}{Gd}\sum_{g=1}^{G}
\left\|C_g(\delta)\right\|_*
-\lambda\|\delta\|_2^2,
\]

where \(\|\cdot\|_*\) is the nuclear norm of the groupwise cross-covariance.

The corrected implementation accepts a nonzero shift only if

\[
S(\delta^\star)-S(0)>m,
\]

where \(m\ge 0\) is a required score-gain margin. This avoids choosing a shift from negligible numerical differences.

### 3.4 Reliability simplex

Using pooled-domain energies,

\[
d_m=|\log(E_m+\epsilon)|,
\]

and modality probabilities are

\[
p_m=\frac{\exp(-\gamma d_m)}
{\exp(-\gamma d_r)+\exp(-\gamma d_i)}.
\]

Therefore,

\[
p_r+p_i=1,
\qquad p_r,p_i\in[0,1].
\]

The reliability trigger is

\[
t=\sigma\left(k\left[\max(d_r,d_i)-\tau\right]\right).
\]

The bounded rescaling factors are

\[
s_r=\sqrt{2p_r},\qquad s_i=\sqrt{2p_i},
\]

and the output branches are

\[
\widetilde A=(1-t)A+t s_rA,
\qquad
\widetilde B=(1-t)B+t s_iB.
\]

Because \(s_m\in[0,\sqrt{2}]\), the correction cannot create an unbounded reliability gain.

---

## 4. Controlled test results

### 4.1 Repository and numerical validation

- Full repository test suite: **98 passed**.
- Mathematical SJPA/RTPF tests: all passed.
- All shipped model configurations: build successfully.
- Corrected proposal parameter count: **11,205,570**.
- `dual_concat` parameter count: **11,205,570**.
- The main comparison is therefore parameter matched.

### 4.2 Five-seed controlled detection proxy

The proxy is a synthetic controlled detection problem. It is useful for mechanism testing but is **not a substitute for VEDAI mAP**.

| Method | Clean proxy mAP50 | Robust mean |
|---|---:|---:|
| `dual_concat` | 0.35584 ± 0.04707 | 0.04664 ± 0.01218 |
| TGF | 0.26307 ± 0.03343 | 0.05020 ± 0.01001 |
| PBTR | 0.25751 ± 0.09699 | 0.04998 ± 0.01834 |
| GOCI | 0.35822 ± 0.04912 | 0.04177 ± 0.00973 |
| **SJPA** | **0.46150 ± 0.08595** | **0.26748 ± 0.04807** |

Against the strongest clean baseline, SJPA improved by

\[
\frac{0.46150-0.35822}{0.35822}\times100
=28.83\%.
\]

Against the strongest robust-mean baseline, it improved by

\[
\frac{0.26748-0.05020}{0.05020}\times100
\approx432.9\%.
\]

Paired one-sided tests across five seeds gave:

- Clean: \(p=0.00217\).
- RGB noise: \(p=0.00190\).
- IR noise: \(p=0.00036\).
- Missing RGB: \(p=0.00099\).
- Missing IR: \(p=0.00054\).
- IR shift: \(p=0.00083\).

### 4.3 Remaining failure

Under simultaneous mixed corruption, SJPA obtained only

\[
0.00025
\]

versus

\[
0.01826
\]

for direct concatenation. The current method is therefore **not universally best**. This condition should be reported, not hidden.

---

## 5. Novelty assessment

A literature search found neighboring ideas in:

- scene-specific RGB-X fusion;
- cosine-similarity channel resampling;
- uncertainty-aware alignment;
- condition-aware dynamic modality weighting;
- distribution-aligned multimodal detection;
- feature purification and selection.

The search did **not** identify the exact combination of:

1. parameter-free groupwise whitening;
2. closed-form orthogonal Procrustes channel alignment;
3. finite zero-padded spatial search scored by normalized groupwise nuclear norm;
4. a strict score-gain acceptance margin;
5. reliability estimated in the same pooled statistical domain;
6. simplex-constrained bounded modality rescaling;
7. insertion into a parameter-matched RGB-IR OBB detector.

This supports a **plausible novelty claim**, but it is not proof of worldwide novelty. A paper should state the contribution precisely rather than claim that Procrustes alignment, reliability weighting, or spatial alignment are individually new.

Recommended claim:

> We introduce a statistically calibrated, parameter-free RGB-IR fusion operator that couples groupwise closed-form Procrustes alignment with margin-gated spatial correction and pooled-domain simplex reliability. Unlike learned attention fusion, the operator has an explicit optimization solution, a normalized shift-selection criterion, and no increase in detector parameter count.

Avoid the claim:

> We are the first method to align or reliability-weight RGB and infrared features.

---

## 6. Required real-data experiment

The corrected real VEDAI result was not generated in the CPU-only test environment. The supplied Kaggle script defaults to:

```text
ETFNET_EPOCHS=100
ETFNET_SEEDS=0,1,2
```

Main proposal label:

```text
sjpa_tr_proposal
```

The experiment matrix also includes:

- `dual_concat`;
- `sjpa_cafem_legacy`;
- `sjpa_no_reliability`;
- `rtpf_safety_ablation`;
- the remaining original baselines.

Do not claim the proposal has the highest VEDAI mAP until that GPU experiment finishes and the three-seed mean exceeds `dual_concat` with uncertainty reported.

---

## 7. Final recommendation

Use the corrected C3k2-SJPA configuration as the main proposal. Its design is mathematically interpretable, parameter matched, and strongly supported by the controlled proxy. Keep RTPF as a safety-oriented ablation. The real-data paper claim remains conditional on the new 100-epoch, three-seed VEDAI benchmark.
