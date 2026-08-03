# Fast SJPA-TR Validation Report

**Date:** 2026-08-02  
**Base repository:** ETFNet  
**Selected proposal:** Fast SJPA-TR — Selective Joint Spatial–Channel Procrustes Alignment with Trust-Region Reliability

## 1. Scope and claim boundary

This package validates candidate fusion ideas at three levels:

1. Mathematical and numerical properties.
2. A controlled multimodal detection proxy with identical training conditions.
3. Forward/backward and latency integration in the complete ETFNet graph.

The public DroneVehicle, VTUAV-det, and RGBTDronePerson datasets were not included in the supplied archive. Therefore, this report does **not** claim that Fast SJPA-TR already exceeds ETFNet's published real-dataset mAP. That claim requires retraining after repairing the repository's RGB/IR loading and prediction defects.

## 2. Ideas tested before selection

| Candidate | Main purpose | Test outcome | Decision |
|---|---|---|---|
| Per-sample one-sided Procrustes | Align RGB and IR channel bases independently per image | Poor clean detection proxy and unstable samplewise orientation | Rejected |
| Per-sample symmetric consensus–innovation SVD | Produce shared and differential features | Sign/permutation/gauge instability; substantially weaker clean result | Rejected |
| Fisher innovation loss | Improve fine-grained class separation | Small clean gain in one trainable proxy but degraded several corruption cases | Rejected as core contribution |
| PBTR | Reliability-weighted barycenter with bounded innovation | Useful under the hardest compound corruption, but weaker clean and overall robustness than the final method | Rejected as final method |
| Global whitening + Procrustes | Stable latent-coordinate alignment | Stronger clean result than samplewise alignment, but insufficient noise/missing-modality robustness | Retained as alignment core |
| GOPA-TR | Global Procrustes plus trust reliability | Strong clean/noise/missing-modality performance, but weak spatial-shift handling | Retained as intermediate ablation |
| Fast SJPA-TR | Global alignment, selective spatial correction, trust reliability | Best overall clean/single-corruption balance; full graph passed | **Selected** |

## 3. Final mathematical formulation

Let pooled, grouped, centered RGB and IR features be `R` and `I`. Running covariance statistics define whitening matrices `W_r` and `W_i`. The whitened cross-covariance is

\[
C=W_r\,\Sigma_{ri}\,W_i.
\]

For each permitted spatial shift \(\delta\in\mathcal D\), solve

\[
(\delta^*,Q^*)=
\arg\min_{\delta\in\mathcal D,\;Q^TQ=I}
\|T_\delta(RW_r)Q-IW_i\|_F^2.
\]

For a fixed shift, if

\[
(T_\delta RW_r)^T(IW_i)=U_\delta\Sigma_\delta V_\delta^T,
\]

then the global orthogonal Procrustes solution is

\[
Q_\delta^*=U_\delta V_\delta^T.
\]

After substitution, minimizing the residual over the finite shift set is equivalent to

\[
\delta^*=\arg\max_{\delta\in\mathcal D}
\left\|(T_\delta RW_r)^T(IW_i)\right\|_*-eta\|\delta\|_2^2,
\]

where \(\|\cdot\|_*\) is the nuclear norm and \(\beta\) discourages unnecessary motion.

### Reliability simplex

For aligned full-resolution features \(Z_r,Z_i\), define energy deviations

\[
d_m=\left|\log(\operatorname{mean}(Z_m^2)+\epsilon)\right|.
\]

The modality probabilities are the solution of an entropy-regularized simplex weighting rule:

\[
[p_r,p_i]=\operatorname{softmax}(-\gamma[d_r,d_i]),
\quad p_r+p_i=1.
\]

A smooth degradation trigger is

\[
\alpha=\sigma\left(k(\max(d_r,d_i)-\tau)\right).
\]

The final pair is

\[
F_m=(1-\alpha)Z_m+
\alpha\sqrt{2p_m}\,Z_m.
\]

Because \(0\leq p_m\leq1\),

\[
\|F_m-Z_m\|_F
\leq \alpha\|Z_m\|_F.
\]

Thus, reliability correction is bounded and is nearly inactive for statistically typical modality pairs.

## 4. Controlled detection-proxy comparison

All methods used the same synthetic multimodal detection generator, three random seeds, identical downstream head, and the same corruption-augmented training schedule. Values below are **proxy mAP50**, not public-dataset mAP.

| Method | Clean | RGB noise | IR noise | Missing RGB | Missing IR | IR shift | Severe mixed | Robust mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Concatenation | 0.3069 | 0.0110 | 0.0404 | 0.0492 | 0.0408 | 0.0670 | 0.0120 | 0.0367 |
| TGF proxy | 0.2016 | 0.0206 | 0.0506 | 0.0353 | 0.0461 | 0.0596 | 0.0097 | 0.0370 |
| PBTR | 0.2212 | 0.0492 | 0.0371 | 0.0567 | 0.0666 | 0.0552 | **0.0201** | 0.0475 |
| Global Procrustes | 0.2585 | 0.0123 | 0.0291 | 0.0366 | 0.0586 | 0.0976 | 0.0082 | 0.0404 |
| GOPA-TR | **0.3989** | 0.2172 | 0.2474 | 0.1709 | **0.2350** | 0.0916 | 0.0012 | 0.1605 |
| Full SJPA-TR | 0.3800 | 0.2320 | **0.2482** | 0.1724 | 0.2246 | **0.3656** | 0.0011 | **0.2073** |
| Fast SJPA-TR, threshold 0.30 | 0.3810 | **0.2363** | 0.2410 | **0.1737** | 0.2251 | 0.3333 | 0.0011 | 0.2017 |

### Interpretation

- Full SJPA-TR improved clean proxy mAP from the strongest non-proposed baseline, concatenation, from 0.3069 to 0.3800: **+0.0731 absolute**.
- Its robust mean was 0.2073 versus 0.0475 for the strongest alternative baseline, PBTR: approximately **4.36×**.
- Its shifted-IR result was 0.3656 versus 0.0976 for Global Procrustes: approximately **3.74×**.
- Fast SJPA-TR preserved most of these gains while avoiding shift search for typical/noisy/missing pairs.
- Fast SJPA-TR is **not universally superior**: PBTR remained better under the deliberately severe simultaneous noise-plus-shift condition. This is a remaining research limitation, not a hidden result.

## 5. Fast search behavior

For the selected precheck threshold 0.30:

- Clean search rate: 18.7%.
- Shifted-IR search rate: 95.2%.
- RGB noise, IR noise, and missing-modality search rate: 0%.

This is the desired behavior: geometric search is mainly activated for likely spatial mismatch, not for photometric corruption or modality loss.

## 6. Complete ETFNet graph tests

| Check | ETFNet TGF | Fast SJPA-TR |
|---|---:|---:|
| Parameters | 10,540,023 | **9,981,746** |
| Reduction | — | **558,277 (5.30%)** |
| Random six-channel forward | Passed | Passed |
| Backward | Passed | Passed |
| Finite gradients | Passed | Passed |
| Median CPU forward, 320×320, batch 1 | 68.694 ms | 68.664 ms |

The measured CPU medians are effectively indistinguishable; the test does not support a claim that Fast SJPA-TR is definitively faster. It does support equal observed latency with fewer trainable parameters in this environment.

## 7. Numerical checks

The inherited global Procrustes/trust core produced:

- Maximum orthogonality error of \(Q\): `1.43e-6`.
- Probability-simplex error: `0.0`.
- No observed trust-bound violation.
- Finite input gradients.

The current complete Fast SJPA-TR graph was also re-tested after the final precheck implementation: forward and backward passed, all gradients were finite, and the parameter count was 9,981,746.

## 8. What is supported and unsupported

### Supported by the tests

- Per-sample Procrustes variants should not be used.
- Global/running-statistic alignment is substantially more stable.
- Reliability correction must be conditionally activated rather than always applied.
- Spatial correction materially improves shifted-modality behavior.
- Fast SJPA-TR offers the strongest tested overall clean and single-degradation trade-off.
- The full module is parameter-free and reduces complete ETFNet parameters by 5.30% relative to TGF.

### Not yet supported

- Real DroneVehicle mAP above ETFNet's published 82.8%.
- Real VTUAV-det or RGBTDronePerson superiority.
- Better Jetson/GPU energy or latency.
- Robustness to severe simultaneous noise and misalignment.

## 9. Required real benchmark gate

Do not present Fast SJPA-TR as a successful final paper method unless corrected end-to-end runs meet at least:

- DroneVehicle mAP50 ≥ 82.8%, preferably ≥ 83.3%.
- VTUAV-det mAP50 ≥ 78.0%.
- RGBTDronePerson mAP50 > 45.57%.
- No meaningful increase in measured GPU latency.
- Statistically significant retention improvements under controlled noise, shift, and modality-drop tests.

The original archive's modality-order, paired-path, caching, prediction, dependency, and checkpoint issues must be fixed before these benchmarks are scientifically trustworthy.
