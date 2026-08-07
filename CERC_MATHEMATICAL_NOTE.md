# CERC v10.6 — mathematical recheck

## 1. One operator for one or many evidence fields

At one backbone stage let the available aligned evidence be

\[
\mathcal X=\{X^{(m)}\}_{m=1}^{M},\qquad M\ge1,
\]

with every embedded field in \(\mathbb R^{B\times C\times H\times W}\).
No single-view/multi-view case appears in the relation equations. Each field is
split into \(G=C/d\) channel atoms, so the atom index is \(a=(m,g)\) and there
are \(K=MG\) atoms. Thus even \(M=1\) still has relational structure whenever
\(G>1\).

## 2. Scale-normalized whitening

For sampled tokens \(X_a\in\mathbb R^{N\times d}\), define

\[
\bar X_a=X_a-\mu_a,
\qquad
s_a=\frac{\|\bar X_a\|_F}{\sqrt{Nd}}+\varepsilon,
\qquad
Y_a=\bar X_a/s_a.
\]

The covariance is regularized by a scale-adaptive ridge and Cholesky factored,

\[
\Sigma_a=\frac{Y_a^\top Y_a}{N-1},\qquad
L_aL_a^\top=\Sigma_a+\lambda_a I,
\]

and

\[
Z_a=Y_aL_a^{-\top}.
\]

The RMS normalization makes the canonical supports insensitive to positive
rescaling of an evidence field up to the finite ridge term.

## 3. Canonical evidence relation

For every atom pair,

\[
C_{ab}=\frac{Z_a^\top Z_b}{N-1},
\qquad
A_{ab}=C_{ab}C_{ab}^\top.
\]

A smooth bounded support is

\[
e_{ab}=\frac{1}{d}\operatorname{tr}(A_{ab}),
\qquad
r_{ab}=\frac{e_{ab}}{1+e_{ab}}.
\]

Self-relations are removed by one algebraic off-diagonal matrix
\(J=\mathbf 1-I_K\): \(\tilde r_{ab}=J_{ab}r_{ab}\). This is not a runtime
single/multi-view branch and gives exactly zero self support.

## 4. Relational innovation

The least-squares canonical prediction of atom \(a\) from atom \(b\) is

\[
T_{a\leftarrow b}=Z_bC_{ab}^\top.
\]

CERC suppresses weak directions using \(A_{ab}\) and aggregates

\[
E_a=
\frac{
\sum_b\tilde r_{ab}(T_{a\leftarrow b}-Z_a)A_{ab}
}{
\varepsilon+\sum_b\tilde r_{ab}
}.
\]

The implementation uses the algebraically identical, memory-efficient form

\[
E_a=
\frac{
\sum_b\tilde r_{ab}Z_b(C_{ab}^\top A_{ab})
-Z_a\sum_b\tilde r_{ab}A_{ab}
}{
\varepsilon+\sum_b\tilde r_{ab}
}.
\]

A dedicated unit test evaluates both formulas numerically and requires them to
match.

If \(C_{ab}=0\), then \(A_{ab}=0\), \(r_{ab}=0\), and that pair contributes
exactly zero. Hence unrelated evidence cannot create a large residual merely
because it is unrelated.

## 5. Convolution is inside the proposal

Recolor the innovation to the native atom scale,

\[
D_a=s_aE_aL_a^\top.
\]

A single shared spatial convolution \(\mathcal K_\theta\) is applied to every
atom from every field,

\[
R_a=\mathcal K_\theta*D_a.
\]

The same kernel is shared for all \(M\) and \(G\), so its parameter count is
independent of evidence count. For group width \(d\) and a \(k\times k\)
kernel it has only \(d^2k^2\) parameters.

The kernel is zero initialized. Therefore for any \(M\ge1\),

\[
R_a=0,\qquad X'_a=X_a
\]

at initialization, while the convolution receives a nonzero gradient on the
first optimization step whenever the relational innovation is nonzero.

## 6. Trust bound

Each atom correction is projected onto

\[
\|\Delta_a\|_F\le\rho\|X_a\|_F,
\qquad \rho=0.05.
\]

This bound is tested after replacing the zero kernel by random nonzero weights.

## 7. Fixed-width consensus without a view-count branch

Relational centrality is

\[
q_{m,g}=\frac{1}{K}\sum_b\tilde r_{(m,g),b}.
\]

For each group,

\[
\pi_{m,g}=\operatorname{softmax}_m(q_{m,g}/\tau),
\qquad
F_g=\sum_m\pi_{m,g}X'_{m,g}.
\]

When \(M=1\), the same softmax equation gives \(\pi_{1,g}=1\) exactly. With
missing modalities, the softmax automatically renormalizes over the fields that
are present. No CERC equation changes.

## 8. Scope

CERC v10.6 is a **2-D** feature-relation operator. It is suitable for ordinary
single images, aligned multi-sensor images, fabric/surface inspection, 2-D
medical images, and task heads for classification, segmentation, detection, or
anomaly features. MedMNIST3D or volumetric CT/MRI require a future Conv3D CERC
extension and are not claimed as supported by this version.
