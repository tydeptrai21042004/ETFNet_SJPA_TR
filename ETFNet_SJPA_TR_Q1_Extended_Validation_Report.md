# ETFNet–SJPA-TR v3: Extended Runability, Mathematics, and Q1 Evidence Report

## 1. Decision

The v3 release is **software-runnable under the tested CPU environment** and is materially more reproducible than the supplied ETFNet archive. The proposed SJPA-TR operator has a defensible mathematical contribution, but this report does **not** claim Q1 acceptance or real-benchmark superiority. Those claims require matched multi-seed training on DroneVehicle, VTUAV-det, and RGBTDronePerson.

## 2. Tested environment

- Version: `8.0.238+etfnetsjpa.3`
- Python: `3.13.5`
- Platform: `Linux-6.12.13-x86_64-with-glibc2.41`
- Tested PyTorch runtime: `2.10.0+cpu`
- Validation source hashes are stored in every `extended_validation_*_v3.json` file.

## 3. Expanded runability tests

| Test layer | Result | Evidence |
|---|---:|---|
| Source compilation | Passed | `VALIDATION/logs/compileall.log` |
| Automated unit/integration suite | **49 passed** | `VALIDATION/logs/pytest.log` |
| Mathematical/invariant tests | Passed | whitening, orthogonality, simplex, trust bound, no-wrap shift, gauge behavior |
| BF16/autocast and finite gradients | Passed | automated tests |
| RGB–IR pair/channel/cache edge cases | Passed | corrupt image, stale IR cache, invalid OBB, pair mismatch tests |
| Shipped model configurations | **12/12 passed** | `VALIDATION/model_configs_v3.json` |
| Exact epoch-boundary resume | **Tensor-identical** | no raw-model or EMA differences |
| One-epoch paired training | Passed | full pipeline |
| Validation | Passed | full pipeline |
| Paired image prediction | Passed | full pipeline |
| Synchronized paired video prediction | Passed | full pipeline |
| TorchScript export | Passed | full pipeline |
| TorchScript paired prediction | Passed | full pipeline |
| Built wheel install/import | Passed | `VALIDATION/wheel_install_v3.json` |
| Installed full SJPA graph forward | Passed | 6-channel input, 9,981,746 parameters |
| Unified installed CLI | Passed | `etfnet_cli --help` |

The complete train → validate → paired image/video predict → TorchScript export → TorchScript predict workflow completed in **34.06 s** on the synthetic fixture. Exact resume completed in **13.11 s**.

### Exact resume result

```json
{
  "continuous_best_fitness": 0.18905,
  "continuous_epoch": 1,
  "continuous_updates": 3,
  "ema_differences": [],
  "exact": true,
  "model_differences": [],
  "resumed_best_fitness": 0.18905,
  "resumed_epoch": 1,
  "resumed_updates": 3
}
```

## 4. What was added beyond ordinary smoke testing

1. Eight signed translation-recovery cases using zero padding, preventing false success caused by circular wraparound.
2. State-dict round-trip and evaluation-output equality.
3. Native-versus-export-mode checks and a traced TorchScript new-input equivalence test.
4. Degenerate-anchor validation and covariance edge cases.
5. Corrupt-image detection, empty/missing pairs, invalid class indices, invalid OBB polygons, and stale IR cache rebuilding.
6. Full model-YAML construction for every shipped configuration.
7. Exact interrupted-versus-continuous training comparison, including raw FP32 model, EMA, optimizer update count, epoch, and best fitness.
8. Isolated built-wheel import and full six-channel model execution, avoiding accidental imports from the source checkout.
9. Four isolated validation stages to avoid hidden OpenMP/runtime interference between independent training workloads.
10. CI definitions for Python 3.9, 3.11, and 3.13. Only Python 3.13.5 was executed in this environment; the other matrix entries remain CI targets rather than locally certified runtimes.

## 5. Mathematical contribution that can be defended

SJPA-TR is a **sequential parameter-free early-fusion operator**:

1. Running groupwise whitening creates stable train/evaluation coordinates.
2. Closed-form orthogonal Procrustes channel alignment solves
   \[
   Q_g^\star=\arg\min_{Q^\top Q=I}\|R_w^{(g)}Q-I_w^{(g)}\|_F^2,
   \qquad Q_g^\star=U_gV_g^\top.
   \]
3. Each finite zero-padded translation is scored by
   \[
   S(\delta)=\sum_g\|C_{g,\delta}\|_\ast-\beta\|\delta\|_2^2,
   \]
   where the nuclear norm equals the maximum orthogonal correlation achievable at that candidate shift.
4. Competitive reliability lies on a two-simplex:
   \[
   [p_r,p_i]=\operatorname{softmax}(-\gamma[d_r,d_i]),
   \qquad p_r+p_i=1.
   \]
5. The multiplicative correction satisfies
   \[
   \|F_m-Z_m\|_F\leq\alpha\|Z_m\|_F.
   \]

The mathematical contribution is **the coupling of these properties**, not whitening, Procrustes, alignment, reliability, or gating individually. The method must not be described as solving an unrestricted joint continuous spatial–channel optimization.

## 6. Controlled five-seed proxy

This is a synthetic detection proxy, not public-dataset mAP. Mean ± sample standard deviation:

| Method | Clean | Robust mean | IR shift | Severe mixed |
|---|---:|---:|---:|---:|
| Concatenation | 0.25314 ± 0.03855 | 0.03645 ± 0.00781 | 0.08400 | 0.01139 |
| TGF proxy | 0.21173 ± 0.04355 | 0.04188 ± 0.00429 | 0.09150 | 0.00991 |
| PBTR | 0.22831 ± 0.07293 | 0.05359 ± 0.02191 | 0.10784 | **0.01536** |
| GOCI | 0.28904 ± 0.08269 | 0.03331 ± 0.00764 | 0.06359 | 0.01019 |
| **SJPA-TR** | **0.40402 ± 0.08802** | **0.23021 ± 0.04051** | **0.38193** | 0.00062 |

One-sided paired tests versus the strongest alternative in each condition:

| Condition | Mean difference | p-value |
|---|---:|---:|
| Clean | 0.11498 | 0.01465 |
| RGB noise | 0.21844 | 0.00079 |
| IR noise | 0.25720 | 0.00175 |
| Missing RGB | 0.14804 | 0.00236 |
| Missing IR | 0.17668 | 0.00394 |
| IR shift | 0.27409 | 0.00115 |
| Severe mixed noise+shift | **-0.01474** | 0.97904 |

### Required disclosure

SJPA-TR failed the severe compound noise-plus-shift proxy. PBTR scored 0.01536, while SJPA-TR scored 0.00062. This cannot be hidden or converted into a universal-superiority claim. The public benchmark must include compound corruptions and report this behavior.

## 7. Novelty boundary for a Q1 manuscript

### Claims supported by the current package

- The operator is parameter-free at the fusion stage.
- The channel map is the exact groupwise orthogonal Procrustes minimizer for fixed running whitened statistics.
- The finite spatial candidate is selected exactly under the stated native nuclear-norm score.
- Reliability probabilities sum to one.
- The correction has a formal norm bound.
- The software provides strict paired RGB–IR data handling and exact-resume evidence.

### Claims not yet supported

- State-of-the-art mAP on any public dataset.
- Superiority over the published ETFNet results.
- Real-time onboard UAV performance.
- Universal robustness.
- Bayesian uncertainty calibration.
- Exact native/export equivalence for nonzero-shift selection; export uses a documented surrogate.
- “First Procrustes method” or “first alignment-reliability method.”

## 8. Public-benchmark acceptance gates

Before submission, run the corrected TGF baseline and SJPA-TR under identical data, augmentations, optimizer, image size, epochs, hardware, and at least three seeds. The paper should report per-seed values, mean, standard deviation, 95% confidence intervals, paired significance/effect sizes, parameters, GFLOPs, peak memory, end-to-end GPU latency, and edge-device latency.

Minimum evidence for a strong paper:

- clean mAP improvement over corrected ETFNet-TGF, **or** clean parity with substantial statistically supported robustness and efficiency gains;
- no material regression on VTUAV-det or RGBTDronePerson;
- gains stratified by small/medium/large objects and alignment severity;
- capacity-matched controls;
- native-versus-export shift agreement;
- explicit compound-corruption results;
- real deployment measurement rather than inference from GFLOPs.

## 9. Reproduction commands

```bash
python tools/run_extended_validation.py --stage quick --timeout 900
python tools/run_extended_validation.py --stage exact-resume --timeout 900
python tools/run_extended_validation.py --stage proxy --timeout 1200
python tools/run_extended_validation.py --stage pipeline --timeout 1200
```

See `MATHEMATICAL_APPENDIX.md`, `PAPER_SUPPORT_Q1.md`, and `EXPERIMENT_PROTOCOL_Q1.md` for the proof package, literature positioning, and full experiment plan.
