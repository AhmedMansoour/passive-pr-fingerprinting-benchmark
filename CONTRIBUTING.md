# Contributing

Contributions are welcome if they improve reproducibility, documentation, or benchmark coverage without changing the meaning of the released results.

## Suggested contribution types

- Bug fixes in data loading or evaluation scripts.
- Additional documentation or examples.
- New baselines that follow the same train/test protocol.
- Reproducibility reports from different platforms.

## Development checklist

Before submitting a pull request, run:

```bash
python scripts/check_repository.py
python experiments/run_kwnn_benchmark.py
```

Please document any new dependency in `requirements-full.txt` unless it is needed by the core smoke benchmark.
