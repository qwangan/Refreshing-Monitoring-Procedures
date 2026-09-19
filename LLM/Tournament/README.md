# Tournament human/watermarked paragraph experiment

This package reproduces one prespecified H-W-H-W-H realization from the paper.
It contains no multi-trial experiment, 1,400-path simulation study, or saved
results.

The experiment tokenizes the four locked Efron paragraphs into 443 OPT tokens,
keeps three 81-token human blocks, and replaces two 100-token spans by
Tournament-watermarked OPT-1.3B continuations. It then applies the refreshing
weight-adaptive e-process with threshold 49 and last-global-minimum
localization.

## Locked design

- Model: `facebook/opt-1.3b`
- Revision: `3f5c25d0bc631cb57ac65913f76e22c2dfb61d62`
- Precision: float32
- Temperature: 1
- Watermark: 30 exact two-competitor Tournament layers with fresh
  full-vocabulary Bernoulli(1/2) tables
- Pivot: randomized Binomial(30, 1/2) transform
- Master seed: `20260805090442`
- Blocks: H(81)-W(100)-H(81)-W(100)-H(81)
- No EOS stopping, top-k, top-p, repetition penalty, seed search, or
  outcome-based retry

The normalized text and source PDF are hash-locked in `efron_contract.py`.

## Install

Use Python 3.11 or newer. Install the official PyTorch build appropriate for
your machine, then install the remaining packages:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch
python -m pip install -r requirements.txt
```

The first run downloads the pinned OPT-1.3B checkpoint unless it is already in
the Hugging Face cache.

## Test

The unit tests do not load the language model:

```bash
python -m pytest -q
```

Run a 13-token real-model smoke test:

```bash
python run_efron_case.py --smoke --device auto
```

## Run the prespecified case once

```bash
python run_efron_case.py --device auto
```

Use `--device cuda` to require CUDA or `--device cpu` to require CPU. The
runner refuses to overwrite an existing scientific result. Outputs are written
to `results/efron_case/` and include the pivot arrays, JSON metrics, the mixed
passage as TeX, and PDF/PNG figures.

To rebuild figures from saved pivots without regenerating text:

```bash
python run_efron_case.py --rebuild-derived --local-files-only
```
