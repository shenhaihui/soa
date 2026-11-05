# Experiment 5: OPT, SOA-D, and SOA

This experiment compares three methods for two consecutive shifts ($I = 2$), starting from an empty system ($\pi_0$) and carrying unfinished patients at the end of shift 1 over to shift 2.

For the SOA and OPT policies, the base staffing search uses $M = 100$ arrival-rate scenarios with $R = 20$ simulation replications per scenario, and sets the outer coordinate-search round to $K_{\text{out}} = 1$. For SOA, the inner joint surge search uses $K_{\text{in}} = 2$ coordinate-search rounds.

---

## 1. OPT: sequential benchmark (`SEQ_REF`)

We optimize the base staffing vector $\boldsymbol{b} = (b_1, b_2)$ via coordinate search, using simulation-based backward induction as the evaluation function:

1. **Base staffing search (offline):**
   - For each candidate $\boldsymbol{b}$ and first-shift arrival-rate scenario $\lambda_{m1}$, each candidate first-shift surge level $s_{m1}$ is evaluated by its downstream two-shift cost: we simulate the first shift for 40 replications to obtain the first-shift performance costs and resulting terminal states.
   - For each of the 40 terminal states, we draw one second-shift arrival rate conditional on $\lambda_{m1}$ and search for $s_2$ given that state and rate using single-shift hill-climbing.
   - The total costs across both shifts (including surge staffing and performance costs) are averaged over the 40 replications to evaluate $s_{m1}$.
   - After searching for $s_{m1}$ under each scenario $\lambda_{m1}$ via hill-climbing, we average the resulting minimum costs across all $M = 100$ scenarios and add the base staffing cost to evaluate $\boldsymbol{b}$.
2. **Out-of-sample evaluation (online execution):**
   - We fix the selected base staffing vector $\boldsymbol{b}$ and use 100 independently sampled evaluation arrival-rate vectors with 100 simulation replications per vector.
   - Before the first shift, $s_1$ is determined using only the realized $\lambda_1$ and initial state $\pi_0$, following the backward-induction search procedure above.
   - We then simulate the first shift to obtain its realized cost and terminal state $\pi_1$.
   - Before the second shift, we search for $s_2$ given $\pi_1$ and the observed $\lambda_2$, and then simulate the second shift.
   - The simulation sample paths used to execute both shifts are independent of those used to search for the surge decisions. Final performance is estimated by averaging the total costs over all evaluation scenarios and replications.

The sequential search is implemented in [`two_shift.py`](../two_shift.py). 

---

## 2. SOA-D: direct conditional averaging (`COND_MEAN`)

1. **Fix base staffing:** We fix the base staffing vector $\boldsymbol{b}$ obtained by SOA.
2. **Conditional sampling ($2{,}000$ vectors):** For each of the 100 evaluation arrival-rate vectors $(\lambda_{m1}, \lambda_{m2})$, we generate 10 additional vectors by fixing the first component at $\lambda_{m1}$ and sampling the second from its conditional distribution, and another 10 by fixing the second component at $\lambda_{m2}$ and conditionally sampling the first component.
3. **Joint surge search:** Across these $2{,}000$ vectors, we apply the joint surge staffing search with fixed $\boldsymbol{b}$ to obtain a surge decision vector $\boldsymbol{s} = (s_1, s_2)$ for each vector.
4. **Conditional averaging and evaluation:**
   - The surge decision $s_{m1}$ for the first shift is obtained by averaging the first components $s_1$ across the 10 vectors conditional on $\lambda_{m1}$ and rounding to the nearest integer (with halves rounded up).
   - Similarly, $s_{m2}$ is obtained by averaging the second components $s_2$ across the 10 vectors conditional on $\lambda_{m2}$ and rounding to the nearest integer.
   - The resulting surge vector $\boldsymbol{s}_m = (s_{m1}, s_{m2})$ is then combined with $\boldsymbol{b}$ to evaluate performance under the original evaluation vector $(\lambda_{m1}, \lambda_{m2})$.

Note that this procedure averages **surge decisions**.

---

## 3. SOA: offline lookup (`BIN_LOOKUP`)

1. **Scenario augmentation ($1{,}000$ vectors):** We retain the selected SOA base staffing vector $\boldsymbol{b}$, supplement the 100 search scenarios with 900 additional arrival-rate vectors sampled from the joint distribution, and apply the joint surge search to each new vector to obtain $1{,}000$ surge decision vectors $\boldsymbol{s}_m = (s_{m1}, s_{m2})$ in total.
2. **Lookup table construction:** For each shift, we divide the arrival rates into $N = 20$ equal-count bins, average and round their surge decisions, and merge adjacent bins with identical decisions.
3. **Online evaluation:** During evaluation, surge decisions $\boldsymbol{s}_m = (s_{m1}, s_{m2})$ are obtained directly by looking up $\lambda_{m1}$ and $\lambda_{m2}$ in the pre-computed table.

---

## Evaluation Protocol and Execution

In the final evaluation, all three policies use the exact same 100 evaluation arrival-rate scenarios (sampled independently of the base staffing search), with 100 DES replications per scenario. The evaluation DES sample paths are independent of those used in the surge searches, with common random numbers (CRNs) shared across policies.

Run `python experiment5.py` for the complete comparison. Final results are saved in `output/experiment5/`, particularly `benchmark_table.csv` and `lookup.csv`.
