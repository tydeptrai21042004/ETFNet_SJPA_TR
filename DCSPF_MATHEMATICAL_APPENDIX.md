# Mathematical Appendix: DCSPF-Guard

## 1. Scope and notation

Let the paired P2 feature tensors be

\[
R,I\in\mathbb{R}^{B\times C\times H\times W}.
\]

Channels are partitioned into \(G\) groups of width \(d=C/G\). A fixed pooling operator \(P_a\) maps each feature map to \(a\times a\) anchors. For one sample and one group, the centered pooled token matrices are

\[
X_g,Y_g\in\mathbb{R}^{n\times d},\qquad n=a^2.
\]

DCSPF has two coordinate experts:

1. the raw expert \(Z_{\rm raw}=[R;I]\);
2. the canonical expert \(Z_{\rm can}\), produced by covariance calibration, orthogonal Procrustes alignment, margin-gated spatial correction, and bounded reliability rescaling.

The final hard guard is

\[
Z=(1-g)A_{\rm raw}(Z_{\rm raw})+gA_{\rm can}(Z_{\rm can}),\qquad g\in\{0,1\},
\]

where both grouped \(1\times1\) adapters are initialized as exact identities.

---

## 2. Statistical-domain consistency

For modality \(m\in\{r,i\}\), DCSPF estimates mean and covariance in the pooled anchor domain:

\[
\mu_{m,g}=\frac1n\sum_{j=1}^{n}X_{m,g,j},
\]

\[
\Sigma_{m,g}=\frac{1}{n-1}(X_{m,g}-\mathbf 1\mu_{m,g}^{\top})^{\top}
(X_{m,g}-\mathbf 1\mu_{m,g}^{\top})+\varepsilon I_d.
\]

The whitening matrix is

\[
W_{m,g}=\Sigma_{m,g}^{-1/2}.
\]

### Proposition 0 — bounded whitening gain

Because the regularized covariance eigenvalues satisfy

\[
\lambda_{\min}(\Sigma_{m,g})\geq\varepsilon,
\]

we have

\[
\|W_{m,g}\|_2\leq\varepsilon^{-1/2}.
\]

Consequently,

\[
\|XW_{m,g}\|_F\leq\varepsilon^{-1/2}\|X\|_F.
\]

The eigenvalue floor therefore prevents an arbitrarily large whitening operator. Moreover, a convex running-statistics update preserves positive semidefiniteness of the covariance buffers when initialized from positive semidefinite estimates.

Reliability is measured in the same pooled domain:

\[
E_m=\frac{1}{Ca^2}\left\|P_a(\widetilde F_m)\right\|_F^2,
\qquad
D_m=\left|\log(E_m+\epsilon_0)\right|.
\]

This removes the former mismatch in which covariance was estimated after aggressive pooling but reliability energy was measured on the unpooled P2 tensor.

---

## 3. Closed-form groupwise Procrustes alignment

Define the whitened cross-covariance

\[
M_g=W_{r,g}\Sigma_{ri,g}W_{i,g}.
\]

Let

\[
M_g=U_gS_gV_g^{\top}
\]

be its singular-value decomposition. The canonical channel rotation is

\[
Q_g^{\star}=U_gV_g^{\top}.
\]

### Proposition 1 — optimality

\(Q_g^{\star}\) solves

\[
\min_{Q^{\top}Q=I_d}\|X_{r,g}W_{r,g}Q-X_{i,g}W_{i,g}\|_F^2.
\]

### Proof

Expanding the squared norm yields constants minus

\[
2\operatorname{tr}(Q^{\top}M_g).
\]

Thus the minimization is equivalent to maximizing \(\operatorname{tr}(Q^{\top}M_g)\) over orthogonal \(Q\). The orthogonal Procrustes theorem gives the maximizer \(Q_g^{\star}=U_gV_g^{\top}\). ∎

### Corollary 1 — energy preservation

Because \(Q_g^{\star}\) is orthogonal,

\[
\|XQ_g^{\star}\|_F=\|X\|_F.
\]

Therefore, channel alignment itself cannot amplify the Frobenius energy of the whitened RGB group.

---

## 4. Bounded normalized cross-modal coherence

For each group define

\[
C_{xy,g}=\frac{X_g^{\top}Y_g}{n-1},\quad
C_{xx,g}=\frac{X_g^{\top}X_g}{n-1},\quad
C_{yy,g}=\frac{Y_g^{\top}Y_g}{n-1}.
\]

DCSPF uses

\[
\kappa=
\frac{\sum_{g=1}^{G}\|C_{xy,g}\|_F}
{\sqrt{\left(\sum_{g=1}^{G}\|C_{xx,g}\|_F\right)
       \left(\sum_{g=1}^{G}\|C_{yy,g}\|_F\right)}+\epsilon}.
\]

### Proposition 2 — coherence bound

Ignoring the positive numerical stabilizer in the denominator,

\[
0\leq\kappa\leq1.
\]

### Proof

For every group,

\[
\|X_g^{\top}Y_g\|_F^2
=\operatorname{tr}(X_gX_g^{\top}Y_gY_g^{\top})
\leq\|X_gX_g^{\top}\|_F\|Y_gY_g^{\top}\|_F.
\]

The nonzero singular values of \(X_gX_g^{\top}\) and \(X_g^{\top}X_g\) coincide, hence

\[
\|C_{xy,g}\|_F
\leq\sqrt{\|C_{xx,g}\|_F\|C_{yy,g}\|_F}.
\]

Summing over groups and applying Cauchy--Schwarz gives

\[
\sum_g\|C_{xy,g}\|_F
\leq
\sqrt{\left(\sum_g\|C_{xx,g}\|_F\right)
      \left(\sum_g\|C_{yy,g}\|_F\right)}.
\]

Nonnegativity is immediate from the norm definition. ∎

The stabilizer and implementation clamp preserve the numerical interval \([0,1]\).

### Proposition 2b — coordinate and scale invariance

For orthogonal channel transforms \(Q_x,Q_y\) and nonzero scalars \(a,b\),

\[
\kappa(aXQ_x,bYQ_y)=\kappa(X,Y).
\]

Indeed, the Frobenius norm is unitarily invariant, while the factors \(|a|\), \(|b|\) cancel between numerator and denominator. Thus the routing evidence does not depend on an arbitrary orthogonal channel basis or global feature scale.

---

## 5. Bounded reliability rescaling

Define reliability probabilities

\[
p_m=\frac{e^{-\gamma D_m}}{e^{-\gamma D_r}+e^{-\gamma D_i}},
\qquad p_r+p_i=1.
\]

The modality scales are

\[
s_m=\sqrt{2p_m}.
\]

### Proposition 3 — gain and joint-energy bounds

\[
0\leq s_m\leq\sqrt 2,
\]

and, for canonical modality tensors \(A,B\),

\[
\|[s_rA;s_iB]\|_F^2
=2p_r\|A\|_F^2+2p_i\|B\|_F^2
\leq2\max\{\|A\|_F^2,\|B\|_F^2\}.
\]

Thus the reliability rule cannot create unbounded feature gain.

---

## 6. Margin-certified spatial selection

For displacement \(\delta\) in a finite candidate set \(\mathcal D\), the normalized score is

\[
S(\delta)=\frac1{Gd}\sum_{g=1}^{G}\|C_g(\delta)\|_*
-\lambda\|\delta\|_2^2.
\]

A nonzero displacement is accepted only if

\[
S(\delta^{\star})\geq S(0)+m.
\]

### Proposition 4 — no weak-evidence shift

The returned displacement is either zero or has a score improvement of at least \(m\) over zero displacement. This follows directly from the acceptance predicate. Consequently, a nonzero translation cannot be selected solely because of an arbitrarily small numerical argmax difference.

---

## 7. Dual-evidence guard

Let

\[
T=\min(D_r,D_i),\qquad
\Delta_p=|p_r-p_i|.
\]

The guard is

\[
g=
\mathbf 1\left[
(\kappa\geq\tau_c\land T\leq\tau_t)
\lor
(\Delta_p\geq\tau_d)
\right].
\]

It selects the canonical expert under either of two interpretable conditions:

1. **joint evidence:** the pair is coherent and at least one stream is statistically typical;
2. **single-modality evidence:** one stream has a decisive reliability advantage.

### Proposition 5 — exact fallback condition

The raw expert is selected exactly when

\[
(\kappa<\tau_c\lor T>\tau_t)\land\Delta_p<\tau_d.
\]

This is De Morgan's law applied to the guard. It characterizes the intended mixed-failure region: insufficient pair coherence/typicality and no trustworthy modality winner.

---

## 8. Initialization invariance

Each grouped adapter has block-diagonal identity weights and zero bias at initialization:

\[
A_{\rm raw}(z)=z,\qquad A_{\rm can}(z)=z.
\]

### Proposition 6 — branch preservation at initialization

At initialization,

\[
Z=
\begin{cases}
Z_{\rm raw},&g=0,\\
Z_{\rm can},&g=1.
\end{cases}
\]

Therefore, introducing DCSPF does not perturb either selected coordinate representation before learning begins. This is a representation-level statement, not a guarantee of detector-level accuracy after training.

---

## 9. Parameter overhead

For feature width \(C\), each grouped adapter maps \(2C\) channels to \(2C\) channels with two groups. It has

\[
2C^2+2C
\]

parameters. Two adapters have

\[
N_{\rm DCSPF}=4C^2+4C.
\]

At the repository's P2 fusion width \(C=128\),

\[
N_{\rm DCSPF}=66{,}048.
\]

The measured complete model counts are:

- direct concatenation: \(11{,}205{,}570\);
- corrected parameter-free SJPA: \(11{,}205{,}570\);
- DCSPF-Guard: \(11{,}271{,}618\).

Thus DCSPF adds approximately \(0.589\%\) parameters relative to direct concatenation.

---

## 10. Finite-family validation certificate

Architecture alone cannot prove that a detector is more accurate. For a bounded, predeclared, per-image utility \(u\in[0,1]\), define paired differences

\[
D_i=u_i^{\rm candidate}-u_i^{\rm baseline}\in[-1,1].
\]

Suppose \(K\) candidate configurations are evaluated on \(n\) i.i.d. or exchangeable validation examples. For family-wise error probability \(\delta\), define

\[
\operatorname{LCB}_k
=\widehat\Delta_k-
\sqrt{\frac{2\log(K/\delta)}{n}},
\qquad
\widehat\Delta_k=\frac1n\sum_{i=1}^{n}D_{i,k}.
\]

### Proposition 7 — simultaneous lower confidence bound

With probability at least \(1-\delta\),

\[
\mathbb E[D_k]\geq\operatorname{LCB}_k
\]

simultaneously for every candidate \(k\in\{1,\ldots,K\}\).

### Proof

Since \(D_i\in[-1,1]\), Hoeffding's inequality gives

\[
\Pr\left(\widehat\Delta_k-\mathbb E[D_k]\geq t\right)
\leq e^{-nt^2/2}.
\]

Set \(t=\sqrt{2\log(K/\delta)/n}\), making the failure probability at most \(\delta/K\) for each candidate. A union bound over \(K\) candidates gives total failure probability at most \(\delta\). ∎

The repository utility `tools/certify_candidate.py` implements this rule. It must not be applied directly to a single aggregate mAP number; use a bounded per-image utility fixed before test inspection. Final mAP superiority should additionally be evaluated by paired image bootstrap on the untouched test set.

---

## 11. What the mathematics does and does not establish

The derivations establish:

- closed-form optimality of each orthogonal channel alignment;
- bounded coherence and bounded reliability gain;
- an explicit minimum-evidence condition for spatial shifts;
- exact logical conditions for canonical routing and fallback;
- identity preservation at adapter initialization;
- a finite-sample validation certificate under stated assumptions.

They do **not** prove that DCSPF has higher mAP than every competing network. That claim requires matched full-dataset experiments, multiple seeds, confidence intervals, and an untouched test set.
