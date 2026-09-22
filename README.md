# Sequential resetting procedures

Code and frozen input data for the numerical studies in "Sequential resetting procedures and false discovery rate" by Q. Wang, R. Wang, and Z. Zhang (2026). Generated tables, figures, reports, checkpoints, and model caches are intentionally excluded.

## Independent simulations

`IID.R` generates Figures 6-8 and the 1,000-run Monte Carlo results in Table 1.

```sh
Rscript IID.R
```

## LLM watermark experiments

The `LLM` directory is organized by watermarking method:

* `LLM/Gumbel-max` contains the complete Gumbel-max OPT-1.3B study for Section 5.2 and Appendix B.
* `LLM/Tournament` contains the complete Tournament OPT-1.3B study for Section 5.2 and Appendix B.

### Gumbel-max watermark

The `LLM/Gumbel-max` directory contains:

* `generate_fresh_opt13b.py` fixes the prompts, schedules, seeds, model revision, and generates the 1,400 paths used in the paper: 1,000 primary paths and 400 four-interval stress paths.
* `resetting_swz.py` implements the resetting and localization procedures.
* `analyze_opt13b_paths.py` calculates Tables 2 and B.6.
* `analyze_eprocess_comparison.py` and `analyze_fixed_lambda_benchmarks.py` calculate Table B.8.
* `make_paper_outputs.py` generates Figures 9, 10, 12, B.17, B.18, and B.20 and the corresponding table files.
* `run_gpu_study.sh` runs the full resumable study on a CUDA GPU.

Install the packages in `LLM/Gumbel-max/requirements.txt`, then run:

```sh
bash LLM/Gumbel-max/run_gpu_study.sh
```

The mixed Efron/OPT-1.3B text example in Table 4 and Figure 12 uses four fixed paragraphs from printed page x of Efron's *Large-Scale Inference*. The normalized excerpt is supplied as `efron_excerpt.txt`. Before generation, `run_efron_case.py` uses the shared `LLM/sentence_aligned_case.py` helper to choose complete-sentence cuts closest to the target token geometry `80/100/80/100/80`. The two approximately 100-token source passages are then replaced by complete, context-conditioned watermarked sentences. The accepted realization uses master seed `20260922000100`; it was the first completed run after the geometry was locked, with no outcome-based retry.

```sh
cd LLM/Gumbel-max
python run_efron_case.py --device cuda
python -c "import make_paper_outputs as p; p.build_human_watermark_figure()"
```

### Tournament watermark

`LLM/Tournament` mirrors the Gumbel-max layout while retaining the Tournament-specific sampling and randomized pivot:

* `generate_tournament_opt13b.py` generates the 1,400 primary and stress-study paths used in Tables 3, B.7, and B.9.
* `analyze_tournament_paths.py` performs the resetting-process replay and calculates those tables.
* `make_paper_outputs.py` generates Figures 11, 13, B.19, and B.21.
* `run_efron_case.py` produces the balanced sentence-aligned mixed-document experiment in Table 5 and Figure 13 from the hash-checked `efron_excerpt.txt`, using the same source cuts and master seed as the Gumbel-max case.
* `tournament_watermark.py` implements the 30-layer Tournament sampler and randomized null pivot, while `resetting_swz.py` implements the resetting detector.
* `run_gpu_study.sh` runs the complete resumable Tournament study on a CUDA GPU.

```sh
bash LLM/Tournament/run_gpu_study.sh
```

Generated outputs are written under `results/llm/` and are intentionally excluded from the repository. The Gumbel-max mixed-document run writes to `results/llm/efron_case`; the Tournament run writes to `results/llm/tournament_efron_case`. Either set of figures can be rebuilt from saved pivots without regenerating text by adding `--rebuild-derived --local-files-only` to the corresponding `run_efron_case.py` command.

## Financial backtesting

The `Financial Backtesting` directory contains the complete data and code for Section 5.3. `NASDAQ.csv` contains the daily closing values of the NASDAQ Composite index (`^IXIC`) from January 16, 1996 through December 31, 2025. `NASDAQ-forecasts.R` contains all functions needed to construct the negated percentage log-returns and the rolling 97.5% VaR and ES forecasts under normal, t, skewed-t, and empirical specifications. No code from the earlier E-backtesting repository is required.

The forecast calculation uses a 500-day rolling AR(1)-GARCH(1,1) estimation window. It is computationally intensive, so it saves resumable checkpoints after every 100 dates by default.

```sh
cd "Financial Backtesting"
Rscript NASDAQ-forecasts.R --cores=6
Rscript E-backtesting.R --input=NASDAQ-regenerated.rds
```

The required R packages are `rugarch` and `sgt`. A single-date check can be run before the complete calculation:

```sh
cd "Financial Backtesting"
Rscript NASDAQ-forecasts.R --validate-date=2021-12-31
```

`NASDAQ.rds` is the frozen forecast bundle used for the reported results. It is retained so that Figures 14 and 15 can be reproduced immediately:

```sh
cd "Financial Backtesting"
Rscript E-backtesting.R
```

Figure 14 contains 6,539 forecast dates from January 3, 2000 through December 31, 2025. The resetting GREM analysis in Figure 15 contains 5,282 monitored dates from January 4, 2005 through December 31, 2025; the additional 500 preceding observations are used to initialize the betting rule.

All generated tables and figures are written to `results/`.
