# MOSI SSMI

This folder contains a MOSI-specific version of the MOSEI SSM model.

The replacement `data/aligned_50.pkl` has this structure:

- `train/valid/test`
- each split contains `text`, `vision`, `audio`, `regression_labels`
- feature shapes are `text=[N, 50, 768]`, `vision=[N, 50, 20]`, `audio=[N, 50, 5]`

Model input flow:

1. Each modality is projected with its own linear layer into a shared 128-dimensional input space.
2. After that projection, the model uses the same MOSEI SSM structure: temporal encoder, shared/private decoupling, official-Mamba shared selective state mixer, private TMoEs, 4-token fusion, and direct regression.
3. Official `mamba_ssm` is required. There is no lightweight fallback in this strict version.

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
```
