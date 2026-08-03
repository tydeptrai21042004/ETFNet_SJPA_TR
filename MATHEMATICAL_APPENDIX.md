# Mathematical Appendix: SJPA-TR

## 1. Scope and notation

SJPA-TR denotes **Selective Spatial Procrustes Alignment with Trust-Region Reliability**. The implementation is sequential and does **not** claim to solve one unrestricted joint spatial--channel optimization problem.

At an early fusion stage, let the RGB and IR feature maps be

\[
X_r,X_i\in\mathbb{R}^{B\times C\times H\times W}.
\]

Channels are divided into \(G\) groups of width \(d=C/G\). Statistics are estimated on an \(a\times a\) pooled anchor grid, so each group has \(N=B a^2\) tokens. The spatial candidate set is

\[
\mathcal D=\{-s,\ldots,s\}^2.
\]

The reference implementation uses zero-padded translations; it never uses circular wraparound.

---

## 2. Running groupwise whitening

For modality \(m\in\{r,i\}\) and group \(g\), let

\[
Z_m^{(g)}\in\mathbb{R}^{N\times d}
\]

be the pooled tokens. Define

\[
\mu_m^{(g)}=\frac1N\mathbf 1^\top Z_m^{(g)},
\qquad
\Sigma_m^{(g)}=\frac{(Z_m^{(g)}-\mathbf1\mu_m^{(g)})^\top
(Z_m^{(g)}-\mathbf1\mu_m^{(g)})}{N-1}.
\]

The cross-covariance is

\[
\Sigma_{ri}^{(g)}=
\frac{(Z_r^{(g)}-\mathbf1\mu_r^{(g)})^\top
(Z_i^{(g)}-\mathbf1\mu_i^{(g)})}{N-1}.
\]

Exponential running statistics are stored as buffers. With \(\varepsilon>0\), symmetric whitening maps are

\[
W_m^{(g)}=(\Sigma_m^{(g)}+\varepsilon I_d)^{-1/2}.
\]

For the full-resolution features, whitening is applied groupwise:

\[
R_w^{(g)}=(R^{(g)}-\mu_r^{(g)})W_r^{(g)},
\qquad
I_w^{(g)}=(I^{(g)}-\mu_i^{(g)})W_i^{(g)}.
\]

Running statistics make validation and deployment independent of test-batch composition. All eigendecompositions and SVDs are solved in FP32, including under AMP/BF16 execution.

---

## 3. Closed-form channel alignment

For each group, solve the orthogonal Procrustes problem

\[
Q_g^\star=
\arg\min_{Q^\top Q=I_d}
\|R_w^{(g)}Q-I_w^{(g)}\|_F^2.
\]

Let

\[
M_g=(W_r^{(g)}\Sigma_{ri}^{(g)}W_i^{(g)})=U_g S_g V_g^\top.
\]

Then the implemented solution is

\[
\boxed{Q_g^\star=U_gV_g^\top.}
\]

The aligned RGB feature is

\[
A^{(g)}=R_w^{(g)}Q_g^\star,
\]

while the whitened IR feature \(I_w^{(g)}\) is the reference coordinate system.

### Proposition 1: Procrustes optimality

For fixed whitened features, \(Q_g^\star\) is a global minimizer of the constrained problem above.

**Proof.** Expanding the objective gives a constant minus
\(2\operatorname{tr}(Q^\top M_g)\). Von Neumann's trace inequality yields

\[
\operatorname{tr}(Q^\top M_g)\leq\sum_j\sigma_j(M_g),
\]

with equality for \(Q=U_gV_g^\top\). \(\square\)

---

## 4. Selective spatial score

After channel alignment, each candidate shift \(\delta=(\delta_y,\delta_x)\in\mathcal D\) produces

\[
B_\delta=T_\delta(I_w),
\]

where \(T_\delta\) is a zero-padded translation. For a candidate and a group, define

\[
C_{g,\delta}=\frac{(A^{(g)})^\top B_\delta^{(g)}}{N-1}.
\]

The native PyTorch score is

\[
S(\delta)=
\sum_{g=1}^G\|C_{g,\delta}\|_*
-\beta\|\delta\|_2^2,
\]

and

\[
\delta^\star=\arg\max_{\delta\in\mathcal D}S(\delta).
\]

### Proposition 2: Procrustes interpretation of the spatial score

For every candidate \(\delta\),

\[
\|C_{g,\delta}\|_*
=
\max_{P^\top P=I_d}\operatorname{tr}(P^\top C_{g,\delta}).
\]

Thus, the nuclear-norm score is the maximum groupwise orthogonal correlation available at that translation, not an arbitrary similarity heuristic.

**Proof.** Apply von Neumann's trace inequality to the SVD of \(C_{g,\delta}\). \(\square\)

### Selectivity rule

A nonzero shift is permitted only when both conditions hold:

1. The pre-shift pair is not classified as severely anomalous by the energy reliability statistic.
2. The zero-shift Procrustes score is below a threshold, indicating that spatial correction may be useful.

Otherwise, \(\delta=(0,0)\) is forced. The finite argmax is exact over \(\mathcal D\), but it is not a continuous geometric registration claim.

---

## 5. Simplex reliability and bounded correction

After alignment, define modality energies

\[
e_r=\operatorname{mean}(A^2),
\qquad
e_i=\operatorname{mean}(B_{\delta^\star}^2).
\]

Because whitening targets unit energy, deviations are

\[
d_r=|\log(e_r+\eta)|,
\qquad
d_i=|\log(e_i+\eta)|.
\]

Competitive reliability probabilities are

\[
[p_r,p_i]=\operatorname{softmax}(-\gamma[d_r,d_i]),
\]

so

\[
p_r,p_i\geq0,
\qquad p_r+p_i=1.
\]

The correction trigger is

\[
\alpha=\sigma(k(\max(d_r,d_i)-\tau)).
\]

Each modality is corrected by

\[
F_m=\left[(1-\alpha)+\alpha\sqrt{2p_m}\right]Z_m,
\]

where \(Z_r=A\) and \(Z_i=B_{\delta^\star}\). The module output is

\[
F=\operatorname{Concat}(F_r,F_i).
\]

### Proposition 3: Bounded trust correction

For \(m\in\{r,i\}\),

\[
\boxed{\|F_m-Z_m\|_F\leq\alpha\|Z_m\|_F.}
\]

**Proof.** Since \(p_m\in[0,1]\),
\(\sqrt{2p_m}\in[0,\sqrt2]\), hence
\(|\sqrt{2p_m}-1|\leq1\). Therefore

\[
\|F_m-Z_m\|_F
=\alpha|\sqrt{2p_m}-1|\|Z_m\|_F
\leq\alpha\|Z_m\|_F.
\]

\(\square\)

This is a multiplicative trust-region bound; it does not claim Bayesian calibration.

---

## 6. Groupwise channel-gauge property

Let \(O_r\) and \(O_i\) be block-diagonal orthogonal matrices whose blocks follow the same channel grouping as SJPA-TR. Reparameterize the modality features as

\[
R'=RO_r,
\qquad I'=IO_i.
\]

The whitening maps transform equivariantly, and the whitened cross-covariance becomes

\[
M'_g=O_{r,g}^\top M_gO_{i,g}.
\]

An associated Procrustes solution is

\[
Q_g'=O_{r,g}^\top Q_g^\star O_{i,g}.
\]

Therefore,

\[
R'_wQ'=R_wQ^\star O_i,
\qquad I'_w=I_wO_i.
\]

For each spatial candidate,

\[
C'_{g,\delta}=O_{i,g}^\top C_{g,\delta}O_{i,g}.
\]

The nuclear norm and Frobenius norm are unitarily invariant, and feature energies are preserved. Consequently:

- candidate shift scores are invariant;
- the selected shift is invariant, up to deterministic tie handling;
- reliability probabilities and triggers are invariant;
- the output is equivariant under the common IR-side block rotation \(O_i\).

This is the defensible invariance claim. It is restricted to group-compatible orthogonal reparameterizations, not arbitrary nonlinear feature changes.

---

## 7. Parameter count and complexity

SJPA-TR has **zero trainable fusion parameters**. It stores running means, covariances, cross-covariances, and export transforms as buffers.

Let \(S=(2s+1)^2\). Dominant costs are approximately

\[
O(GNd^2+Gd^3)
\]

for running whitening and global channel alignment, and

\[
O\!\left(SBGNd^2+SBGd^3+SBCHW\right)
\]

for finite candidate scoring and zero-padded feature selection. With the released configuration \(C=128\), \(G=32\), \(d=4\), and \(s=1\), all matrix factorizations are only \(4\times4\).

The \(SBCHW\) shifted-feature stack can still become a memory bottleneck for larger \(s\). This is a limitation and should be reported.

---

## 8. Native versus exported scoring

Native PyTorch uses \(\|C\|_*\). Export mode uses \(\|C\|_F\) because SVD support is inconsistent across ONNX/TensorRT runtimes. Both scores are orthogonally invariant, but they can rank candidates differently.

Therefore:

- native and exported predictions must be compared on a held-out set;
- exact export equivalence must not be claimed for \(s>0\);
- `max_shift=0` gives exact native/export module equivalence up to floating-point tolerance;
- deployment papers should report the disagreement rate of selected shifts and the resulting mAP delta.

---

## 9. Claims that the paper must avoid

The implementation does not justify claims that SJPA-TR:

- is the first use of orthogonal Procrustes in deep learning;
- is the first reliability-aware RGB--IR fusion method;
- solves unrestricted continuous image registration;
- is universally superior under compound degradation;
- already exceeds ETFNet on public datasets.

The supported contribution is the specific parameter-free construction and its proven properties, contingent on real benchmark validation.
