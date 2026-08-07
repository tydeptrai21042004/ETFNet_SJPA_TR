# CERC-Hull v10.8 — mathematical and baseline audit

## What changed
CERC-Hull keeps the same canonical relational transport but adds one unified
per-group evidence-hull coordinate

\[
F_g^{\rm base}=(1-\alpha_g)\,\operatorname{mean}_m X_{m,g}
+\alpha_g\,\operatorname{max}_m X_{m,g},\qquad 0\le\alpha_g\le1.
\]

The final consensus adds the existing zero-sum canonical/local/global evidence
reweighting to this base.  `envelope_gain` is initialized to zero, so v10.8 is
still exactly the mean-fusion baseline at initialization.  Setting it to one
recovers element-wise max fusion (up to ordinary floating-point roundoff).
For a single evidence field, mean=max, so the same equation is the identity and
no single-view branch exists.

This strengthens the *hypothesis class*: CERC-Hull contains both mean and max
fusion as endpoint settings while retaining its canonical relational and
convolutional corrections.  It does **not** create a theorem that SGD must find
the best endpoint or that test accuracy must exceed every baseline.

## Baselines shipped
All use the same backbone/head wrapper and accept a variable number of evidence
fields:

1. mean fusion;
2. max fusion;
3. RMS-energy softmax fusion;
4. learned shared global gate;
5. DeepSets-style shared transform + mean pooling;
6. median fusion;
7. smooth-max/value-attention fusion;
8. shared set-attention fusion.

The DeepSets-style control is motivated by Zaheer et al., *Deep Sets* (2017).
The set-attention control is an engineering comparator inspired by the
permutation-invariant attention principle in Lee et al., *Set Transformer*
(ICML 2019); it is not presented as a complete reproduction of Set Transformer.

## Verification
- Full repository: **195/195 tests passed**.
- CERC-Hull endpoint tests: mean endpoint, max endpoint, single-evidence
  identity, finite endpoint gradients.
- All eight baselines: single- and four-evidence forward tests; finite outputs.
- Repository layout: every subfolder remains below the 100-recursive-file rule.

## Controlled comparison — important negative result
A common-initialization, three-seed, 16-step four-evidence medical-style
screen compared CERC-Hull with all eight controls. Mean full/subset IoU:

| method | mean score |
|---|---:|
| CERC-Hull | 0.556678 |
| mean | 0.556468 |
| max | 0.536756 |
| energy | **0.561600** |
| learned gate | 0.543222 |
| DeepSets-style | 0.527399 |
| median | 0.414118 |
| smooth-max | 0.533798 |
| set-attention | **0.557949** |

Therefore CERC-Hull is competitive, but this screen does **not** support the
claim that it is strictly best against every baseline.

A separate five-seed, tuned, 24-step audit of the earlier safe CERC protocol
also found max fusion at 0.650929 versus safe CERC at 0.617810 on the hardest
four-evidence fixture.  This is retained in the package specifically to prevent
an unsupported universal-superiority claim.

## Safe residual evidence
The conservative v10.7 mean-anchor protocol (validation acceptance margin
0.015) produced 20/20 controlled held-out non-degradation cases across fabric,
grayscale medical classification, ultrasound segmentation, and four-evidence
medical segmentation.  This is an empirical safeguard, not a theorem about
new real datasets.

## Claim that is defensible
Use: **"CERC-Hull is a unified variable-cardinality relational fusion family
that exactly includes mean fusion and includes max fusion as the opposite hull
endpoint, while adding bounded canonical-convolutional refinement."**

Do not use: **"CERC is guaranteed to outperform every baseline on every
dataset."**  The supplied controlled experiments contain counterexamples to
that statement.
