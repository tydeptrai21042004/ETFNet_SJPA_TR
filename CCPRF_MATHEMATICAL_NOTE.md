# CCPRF v10.3 final mathematical recheck

## Scope and honest claim

The target is deliberately modest: obtain an exact improvement over the same
trained `dual_concat` checkpoint using only a small fusion residual. Detection
mAP is data- and optimization-dependent, so no algebraic construction can
honestly guarantee that it will beat the baseline on every real-data run.
CCPRF v10.3 is designed to maximize the probability of a positive result while
preserving the baseline exactly at initialization and limiting downside.

## Defects found in CCPRF v10.2

### 1. Weak correlation generated a large residual input

Version 10.2 used the cross-prediction error

\[
\widehat R_w-R_w,\qquad \widehat I_w-I_w.
\]

When the cross-covariance is weak, \(\widehat R_w\approx0\) and
\(\widehat I_w\approx0\), so this input approaches \(-R_w\) and \(-I_w\).
Thus the least reliable directions generated the largest raw residual basis,
which contradicted the intended reliability shrinkage.

### 2. Production grouped convolution did not pair RGB and IR groups

The input was stored as `[all RGB channels, all IR channels]` and passed
straight to a 16-group convolution. Consequently, groups 0--7 received only
RGB channels and groups 8--15 received only IR channels. No group received the
corresponding RGB and IR canonical channels together, although the derivation
assumed joint groupwise processing. The one-group proxy could not expose this
production-layout bug.

### 3. Absolute covariance loading was not scale equivariant

A fixed \(\varepsilon I\) was applied directly to feature covariance. Its
relative effect therefore depended on feature magnitude. Version 10.3 first
normalizes each sample/group by its RMS scale, making the statistical
construction equivariant to positive modality rescaling.

## v10.3 construction

For one sample and one corresponding channel group, let centered point-sampled
features be \(R_c,I_c\in\mathbb R^{N\times d}\). Define scalar group scales

\[
s_r=\sqrt{\operatorname{mean}(R_c^2)},\qquad
s_i=\sqrt{\operatorname{mean}(I_c^2)}.
\]

After normalization, form ridge covariances

\[
\Sigma_r=\frac{\bar R_c^\top\bar R_c}{N-1}+\varepsilon I,
\qquad
\Sigma_i=\frac{\bar I_c^\top\bar I_c}{N-1}+\varepsilon I,
\]

with Cholesky factors \(L_rL_r^\top=\Sigma_r\) and
\(L_iL_i^\top=\Sigma_i\). The whitened coordinates are

\[
R_w=\bar R_cL_r^{-\top},\qquad I_w=\bar I_cL_i^{-\top},
\]

and the whitened cross-covariance is

\[
C=\frac{R_w^\top I_w}{N-1}.
\]

### Bounded cross operator

Because ridge whitening gives

\[
\frac{R_w^\top R_w}{N-1}\preceq I,
\qquad
\frac{I_w^\top I_w}{N-1}\preceq I,
\]

Cauchy--Schwarz implies

\[
\|C\|_2\le1.
\]

Therefore the singular values \(\sigma_j\) act as bounded direction-wise
reliability coefficients.

### Cross support

The two linear cross-supported components are

\[
S_r=I_wC^\top,\qquad S_i=R_wC.
\]

They vanish when the modalities are uncorrelated.

### Reliability-weighted disagreement

Define

\[
A_r=CC^\top,\qquad A_i=C^\top C,
\]

and

\[
D_r=(S_r-R_w)A_r,\qquad D_i=(S_i-I_w)A_i.
\]

The final cross-confirmed basis is

\[
H_r=S_r+D_r,\qquad H_i=S_i+D_i.
\]

If \(C=U\operatorname{diag}(\sigma_j)V^\top\), then in a matched canonical
direction

\[
H_{r,j}=\sigma_j(1+\sigma_j^2)I_{w,j}-\sigma_j^2R_{w,j},
\]

with the symmetric expression for \(H_{i,j}\). Hence:

- \(\sigma_j=0\Rightarrow H_{r,j}=H_{i,j}=0\);
- weak directions enter only at order \(O(\sigma_j)\);
- disagreement correction is weighted by \(\sigma_j^2\);
- strongly correlated directions receive the strongest cross-confirmation.

This directly fixes the v10.2 weak-correlation contradiction.

## Learned residual and safety bound

After recoloring, corresponding RGB and IR groups are explicitly interleaved
as

```text
[group 0 RGB, group 0 IR, group 1 RGB, group 1 IR, ...]
```

before the grouped `1x1` map. The output is converted back to modality-major
order afterward. The fusion rule is

\[
Z=Z_0+\Pi_\rho(W[H_r;H_i]),\qquad Z_0=[R;I],\qquad \rho=0.05.
\]

`W` is created exactly as zero without consuming random numbers. Therefore

\[
Z=Z_0
\]

at initialization, all common detector weights and initial predictions match
`dual_concat` exactly, and

\[
\|Z-Z_0\|_F\le0.05\|Z_0\|_F
\]

for every sample after training.

## Numerical safeguards

- statistics are per-sample and independent of batch companions;
- deterministic point sampling avoids average-pooling variance collapse;
- group RMS normalization prevents scale-induced ill-conditioning;
- covariance is explicitly symmetrized;
- `torch.linalg.cholesky_ex` is retried with increasing ridge loading;
- a finite diagonal factor is used as a last-resort per-sample fallback;
- all statistical linear algebra runs in float32 with autocast disabled;
- no forward-path eigendecomposition or SVD is used;
- zero, constant, rank-deficient, very small, and very large inputs were tested.

## Parameter cost

At the P2 fusion point, \(C=128\), \(G=16\), and \(d=8\). The joint groupwise
map contains

\[
(2C)(2d)=256\times16=4096
\]

parameters, about `0.037%` of the 11.2M-parameter detector.

## Ten-seed screening using the exact v10.3 implementation

The actual repository `CCPRF` class—not a simplified surrogate—was inserted
into the controlled detector proxy. The trained concatenation model was kept
fixed and only the 4096 innovation parameters were trained for two epochs.

| Setting | Clean mean | Clean gain | Clean wins | Robust mean | Robust gain | Robust wins |
|---|---:|---:|---:|---:|---:|---:|
| dual concat | 0.361544 | -- | -- | 0.048532 | -- | -- |
| CCPRF, LR 0.003 | 0.385550 | +0.024006 (+6.64%) | 10/10 | 0.049365 | +0.000833 (+1.72%) | 9/10 |
| CCPRF, LR 0.010 | 0.385117 | +0.023574 (+6.52%) | 10/10 | 0.049597 | +0.001066 (+2.20%) | 8/10 |

For LR `0.003`, the one-sided paired clean-score tests were

- paired t-test: `p = 0.00268`;
- Wilcoxon signed-rank: `p = 0.00098`.

For robust averages, the one-sided Wilcoxon result was `p = 0.03223`.
These are proxy results, not real VEDAI claims, but they are substantially
stronger and more consistent than v10.2.

## Execution validation

- 13/13 CCPRF-specific mathematical tests passed;
- 126/126 full repository tests passed;
- full model random and zero forward paths passed;
- CPU autocast forward/backward passed;
- one real optimizer step changed only `model.9.innovation_weight`;
- all common tensors remained bitwise unchanged;
- checkpoint `deepcopy` and serialization passed;
- post-step zero and random evaluation remained finite.

## Minimal real VEDAI protocol

Use the existing `dual_concat` best checkpoint. Run the primary candidate for
two residual-only epochs with AdamW LR `0.003`. Evaluate exact metrics. Only if
it does not beat the baseline, run the secondary LR `0.010` candidate. This
limits the experiment to two or four cheap residual-only epochs.

A real-data win cannot be guaranteed before execution. The proposal should be
retained only when exact VEDAI `mAP50-95` is greater than the same checkpoint's
baseline result and `mAP50` does not decrease materially.
