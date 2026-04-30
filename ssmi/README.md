# MOSI SSMI

This folder contains a MOSI-specific version of the MOSEI SSM model.

The replacement `data/aligned_50.pkl` has this structure:

- `train/valid/test`
- each split contains `text`, `vision`, `audio`, `regression_labels`
- feature shapes are `text=[N, 50, 768]`, `vision=[N, 50, 20]`, `audio=[N, 50, 5]`

Model input flow:

1. Each modality is projected with its own linear layer into a shared 128-dimensional input space.
2. Padding masks are preserved so zero-padded timesteps do not become valid tokens after projection bias.
3. The projected streams use the same MOSEI-style structure: temporal encoder, shared/private decoupling, shared selective state mixer, private TMoEs, 4-token fusion, and direct regression.

Run locally:

```bash
python ssmi/train.py
```

Useful environment variables:

```bash
MOSI_PKL=/path/to/aligned_50.pkl
EPOCHS=50
BATCH_SIZE=32
INPUT_HIDDEN=128
REQUIRE_OFFICIAL_MAMBA=auto
```

`REQUIRE_OFFICIAL_MAMBA=auto` uses official `mamba_ssm` when it is available and falls back to a lightweight state mixer when Kaggle cannot build `causal-conv1d` or `mamba-ssm`. Set it to `1` only when the official wheels install cleanly in your environment.
