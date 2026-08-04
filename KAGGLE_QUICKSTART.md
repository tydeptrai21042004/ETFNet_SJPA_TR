# Kaggle quick start

This corrected repository is version `8.0.238+etfnetsjpa.7`.

## Recommended path: attached corrected ZIP

1. Upload `ETFNet_SJPA_TR_Corrected_v7.zip` as a private Kaggle Dataset.
2. Create a Kaggle notebook and attach that Dataset.
3. Enable **Internet** and a **GPU accelerator**.
4. Open `kaggle/ETFNet_SJPA_TR_Corrected_Kaggle_VEDAI10_All_Baselines.ipynb`, or paste the matching `.py` file into one cell.
5. Run the cell from the beginning.

The cell prefers the attached corrected ZIP. It will reject an older GitHub checkout whose package version is not `8.0.238+etfnetsjpa.7`.

## Benchmark settings

- Dataset: complete VEDAI-512
- Task: oriented object detection
- Models: ten baseline/proposal configurations
- Epochs: 10
- Seed: 0
- Image size: 512
- Initial batch: 4, with automatic smaller-batch retry on CUDA OOM
- Precision: CUDA AMP when the corrected SJPA probe passes; otherwise FP32 for every model

Results are written under `/kaggle/working/etfnet_corrected_vedai512_10epoch_all_baselines` and packaged as `/kaggle/working/ETFNet_SJPA_TR_Corrected_VEDAI10_results.zip`.

Ten epochs with one seed are a development comparison, not a paper-grade final benchmark.
