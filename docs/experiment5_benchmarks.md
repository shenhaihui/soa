# Experiment 5: OPT, SOA-D, and SOA

This experiment compares three methods for two consecutive shifts, starting
empty and carrying unfinished patients into the second shift.

## OPT: sequential benchmark (`SEQ_REF`)

1. For each candidate base staffing vector and first-shift arrival rate,
   consider a candidate first-shift surge level. Simulate 40 first-shift
   outcomes and pair them with conditional samples of the second-shift rate.
2. For each pair, search second-shift surge staffing given that state and rate.
   Average the resulting two-shift costs to evaluate the first-shift surge
   level. Search over first-shift surge levels, then average across first-rate
   scenarios and add base staffing costs to evaluate the base vector.
3. For final evaluation, keep the selected base vector fixed. Choose first-shift
   surge staffing using only the first rate and the initial state. After the
   first shift, search second-shift surge staffing using the observed
   carry-over state and second rate.

The sequential search is implemented in [`two_shift.py`](../two_shift.py).
OPT is a simulation-based reference using finite samples and local searches,
not a certified global optimum.

## SOA-D: direct conditional averaging (`COND_MEAN`)

1. Keep the base staffing vector obtained from the auxiliary-model search.
2. For each shift's observed rate, hold that rate fixed and draw 10 samples of
   the other shift's rate from its conditional distribution. Solve the
   auxiliary surge problem for each completed two-shift rate vector.
3. Average the surge decisions for the target shift and round to the nearest
   integer, with halves rounded up.

This averages **surge decisions**, not system states. The decision depends on
the current rate but not the realized carry-over state. The implementation is
in [`revision_two_shift_cond_mean.py`](../revision_two_shift_cond_mean.py).

## SOA: offline lookup (`BIN_LOOKUP`)

Using the same auxiliary-model base vector, collect surge solutions for 1,000
joint rate scenarios. For each shift, divide the rates into 20 equal-count
groups, average and round their surge decisions, and merge adjacent intervals
with the same decision. Online decisions are obtained by lookup. See
[`revision_two_shift_lookup.py`](../revision_two_shift_lookup.py).

Run `python experiment5.py` for the complete comparison. Final results are
saved in `output/experiment5/`, particularly `benchmark_table.csv` and
`lookup.csv`.
