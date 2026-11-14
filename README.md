# Simulation-Based Two-Stage Staffing

## 1. Introduction

This repository contains a discrete-event simulation framework for two-stage
staffing, together with the code for the numerical experiments, tables, and
figures. It covers single-shift and multi-shift staffing optimization and
offline lookup-table construction.

## 2. Installation

Use Python 3.11 and install the required packages:

```bash
python -m pip install -r requirements.txt
```

## 3. Details

Run the experiment scripts from the repository root, for example:

```bash
python experiment1.py
```

| Experiment | Script | Results |
| --- | --- | --- |
| 1: Single shift starting empty | `experiment1.py` | `output/experiment1/` |
| 2: Different initial patient counts | `experiment2.py` | `output/experiment2/` |
| 3: Cost and demand-variance sensitivity | `experiment3.py` | `output/experiment3/` |
| 4: Non-stylized model with general distributions and balking | `experiment4.py` | `output/experiment4/` |
| 5: Two-shift comparison of OPT, SOA-D, and SOA | `experiment5.py` | `output/experiment5/` |
| 6: Nurse-staffing case study and sensitivity analysis | `experiment6.py` | `output/experiment6/` |
| 7: Continuous 52-week nurse-staffing evaluation | `experiment7.py` | `output/experiment7_initialization75/` |
| Appendix A: Three-round outer coordinate-search test | `appendix_search.py` | `output/appendix_search/` |

The result folders contain comparison tables (`benchmark_table.csv`), detailed
results, and plots where applicable. Lookup tables are saved as CSV files.
Experiment 3 reuses the baseline results from Experiment 2; the stylized-model
rows in Experiment 4 also come from Experiment 2.

For Experiment 6, baseline results and the full staffing lookup table are under
`output/experiment6/rho0.3_sigma20.118/`; the combined sensitivity results are
in `output/experiment6/sensitivity_table.csv`.

`experiment7.py` implements the continuous 52-week rolling-horizon evaluation in
the nurse-staffing case study: it first fits the one-week policies starting with
75 unfinished patients (`output/experiment6_initialization75/rho0.3_sigma20.118/`)
and then evaluates them over 1,000 independent 52-week trajectories
(`output/experiment7_initialization75/`).

The OPT and SOA-D procedures for Experiment 5 are briefly explained in
[Experiment 5 notes](docs/experiment5_benchmarks.md).

## 4. Version Note

- **November 2026 (`main`)**: Updated experimental configurations and added the continuous 52-week evaluation (`experiment7.py`) and outer search-round test (`appendix_search.py`).
- **October 2026 (`v2026.10`)**: Earlier version containing Experiments 1–6 only, archived under release tag `v2026.10`.