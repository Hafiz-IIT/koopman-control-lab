# Results

This directory contains outputs from the scalar EDMD prediction benchmark.

- `manifest.json`: fixed experiment configuration and scope limitations.
- `koopman_scalar_summary.csv`: per-system/per-method mean, sample standard deviation, and 95% t-interval across 10 seeds.
- `summary.md`: human-readable summary and preliminary interpretation.
- `koopman_scalar_per_seed.csv`: generated per-seed output. It is produced by `python experiment.py` and uploaded by CI as an artifact so the full raw run data can be inspected.

Regenerate all outputs with:

```bash
python experiment.py
```

All data are synthetic. These results are a prediction pilot, not evidence of Koopman invariance, control utility, or physical-system validity.
