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

The result folders contain comparison tables (`benchmark_table.csv`), detailed
results, and plots where applicable. Lookup tables are saved as CSV files.
Experiment 3 reuses the baseline results from Experiment 2; the stylized-model
rows in Experiment 4 also come from Experiment 2.

For Experiment 6, baseline results and the appendix staffing table are under
`output/experiment6/rho0.3_sigma20.118/`; the combined sensitivity results are
in `output/experiment6/sensitivity_table.csv`.

The OPT and SOA-D procedures for Experiment 5 are briefly explained in
[Experiment 5 notes](docs/experiment5_benchmarks.md).
