# Refreshing monitoring procedures
Code and data for the numerical studies in "False discovery rates for refreshing monitoring procedures" by Q. Wang, R. Wang, and Z. Zhang (2026)

R code for the simulation study under the independence setting in Section 5.1

* IID.R: generate Figures 5-7 and the 1,000-run Monte Carlo results in Table 1

Python code for the LLM watermark experiment in Section 5.2 and Appendix B

* LLM/generate_fresh_opt13b.py: generate the fixed OPT-1.3B paths
* LLM/refreshing_swz.py: refreshing and localization procedures
* LLM/analyze_opt13b_paths.py: one-shot, whole-block, and localized comparisons in Tables 2 and 4
* LLM/analyze_eprocess_comparison.py: WA, OG, and 50/50-average comparisons in Table 5
* LLM/analyze_fixed_lambda_benchmarks.py: fixed-lambda comparisons in Table 5
* LLM/make_paper_outputs.py: generate the LLM tables, representative figures, and the mixed human/watermarked refreshing-process figure
* LLM/human_watermark.npz: exact saved OPT-1.3B pivotal statistics for the mixed Efron/watermarked text example
* LLM/run_gpu_study.sh: run the full resumable study on a CUDA GPU
* LLM/requirements.txt: required Python packages

R code and data for the financial analysis in Section 5.3

* E-backtesting.R: generate Table 3 and Figures 10 and 15
* NASDAQ.rds: NASDAQ Composite returns and VaR/ES forecasts used in the paper

The R studies can be run using `Rscript IID.R` and `Rscript E-backtesting.R`. The full LLM study requires a CUDA GPU and can be run using `bash LLM/run_gpu_study.sh`. The mixed human/watermarked figure can be rebuilt without a GPU using `cd LLM && python -c "import make_paper_outputs as p; p.build_human_watermark_figure()"`. Generated tables and figures are written to `results/`.
