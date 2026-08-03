# Reproducible Experiment Protocol for the SJPA-TR Paper

## 1. Freeze the experimental contract

Before the first full run, record:

- Git commit and release archive SHA-256;
- Python, PyTorch, CUDA, cuDNN, driver, GPU and OS;
- dataset YAML and full-data SHA-256 fingerprint;
- exact train/validation/test split manifests;
- model YAML;
- all command-line overrides;
- seed list;
- whether AMP and deterministic algorithms are enabled.

Use `data_fingerprint=sha256` for final paper runs.

## 2. Primary commands

Dataset preflight:

```bash
python etfnet_cli.py check-data \
  --data /path/to/data.yaml \
  --task obb \
  --fingerprint sha256 \
  --output runs/preflight.json
```

Five-seed SJPA-TR experiment:

```bash
python tools/run_reproducible_experiment.py \
  --data /path/to/data.yaml \
  --model ultralytics/cfg/models/etfnet/etfnet_P2_CAFEM_SJPA.yaml \
  --seeds 0 1 2 3 4 \
  --epochs 100 \
  --device 0 \
  --full-data-hash
```

Run the same command with `etfnet_P2_CAFEM_TGF.yaml`, GOCI, direct concatenation, and all ablations.

## 3. Reporting table

For each method and dataset, retain one row per seed:

| Method | Seed | mAP50 | mAP50-95 | AP-small | AP-medium | AP-large | Params | GFLOPs | Latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|

Aggregate only after all runs are complete. Never select the best seed as the headline result.

## 4. Statistical analysis

Use paired seeds and paired data splits. Report:

\[
\bar d=\frac1K\sum_{k=1}^K(m_k^{SJPA}-m_k^{base}),
\]

sample standard deviation of \(d_k\), a 95% confidence interval, and a paired two-sided test. Use a nonparametric paired test as a sensitivity analysis when \(K\) is small. Correct for multiple comparisons when testing many corruptions.

## 5. Robustness protocol

Apply corruptions only to the held-out test pairs and retain the exact transformed-pair manifests. For spatial shifts, use zero padding and report the valid overlap. Avoid circular shifts because they introduce unrealistic content at the opposite boundary.

Use independent severity levels and a compound suite. The current synthetic proxy shows that compound noise plus shift is a known risk and must be tested explicitly.

## 6. Export equivalence

For every exported runtime:

1. Compare raw logits against native PyTorch on at least 1,000 pairs.
2. Report maximum/mean absolute logit error.
3. Report mAP delta.
4. Log native/export selected-shift disagreement where diagnostics are available.
5. Report runtime and peak memory.

Because export uses a Frobenius score surrogate while native PyTorch uses the nuclear norm, exact equivalence is not assumed for `max_shift>0`.

## 7. Artifact release checklist

- source archive and wheel;
- exact environment files;
- dataset preparation scripts, not redistributed restricted data;
- split manifests and fingerprints;
- all model YAMLs;
- raw per-seed CSV/JSON;
- training logs and checkpoints;
- corruption manifests;
- analysis scripts;
- proof appendix;
- known limitations and failed experiments.
