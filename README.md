# Refreshing monitoring procedures

Code and data for the numerical studies in "False discovery rates for refreshing monitoring procedures" by Q. Wang, R. Wang, and Z. Zhang (2026).

## Independent simulations

`IID.R` generates Figures 5-7 and the 1,000-run Monte Carlo results in Table 1.

```sh
Rscript IID.R
```

## LLM watermark experiment

The `LLM` directory contains the complete OPT-1.3B generation and analysis code for Section 5.2 and Appendix B:

* `generate_fresh_opt13b.py` fixes the prompts, schedules, seeds, model revision, and generates all model paths.
* `refreshing_swz.py` implements the refreshing and localization procedures.
* `analyze_opt13b_paths.py`, `analyze_eprocess_comparison.py`, and `analyze_fixed_lambda_benchmarks.py` calculate Tables 2, 4, and 5.
* `make_paper_outputs.py` generates the reported tables and representative figures.
* `run_gpu_study.sh` runs the full resumable study on a CUDA GPU.

Install the packages in `LLM/requirements.txt`, then run:

```sh
bash LLM/run_gpu_study.sh
```

The mixed Efron/OPT-1.3B text example uses four fixed paragraphs from printed page x of Efron's *Large-Scale Inference*. `LLM/prepare_efron_source.py` reproduces the excerpt from the source PDF, and `LLM/run_efron_case.py` regenerates the prespecified GPU realization. The supplied `LLM/human_watermark.npz` is the coauthor's exact saved realization used for the paper figure.

```sh
cd LLM
python prepare_efron_source.py --pdf BradleyEfron_2010_Prologue_Large-ScaleInferenceE.pdf
python run_efron_case.py --device cuda
HUMAN_WATERMARK_NPZ=../results/llm/efron_case/case_arrays.npz \
  python -c "import make_paper_outputs as p; p.build_human_watermark_figure()"
```

### Tournament watermark

`LLM/Tournament` contains the coauthor's locked, code-only reproduction of the Tournament-watermark mixed-text experiment. It uses the same Efron source passage and a prespecified H-W-H-W-H realization, with two 100-token Tournament-watermarked OPT-1.3B continuations between three 81-token human blocks. The directory includes the design lock, source materials, detector implementation, tests, and code for regenerating the reported metrics and figures; generated results are intentionally excluded.

```sh
cd LLM/Tournament
python -m pytest -q
python run_efron_case.py --smoke --device auto
python run_efron_case.py --device cuda
```

See `LLM/Tournament/README.md` for the pinned model revision, watermark construction, seed, and commands for rebuilding figures from saved pivots.

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

`NASDAQ.rds` is the frozen forecast bundle used for the reported results. It is retained so that Table 3 and Figures 10 and 15 can be reproduced immediately:

```sh
cd "Financial Backtesting"
Rscript E-backtesting.R
```

Figure 10 contains 6,539 forecast dates from January 3, 2000 through December 31, 2025. The refreshing GREM analysis reported in Table 3 and Figure 15 contains 5,282 monitored dates from January 4, 2005 through December 31, 2025; the additional 500 preceding observations are used to initialize the betting rule.

All generated tables and figures are written to `results/`.
