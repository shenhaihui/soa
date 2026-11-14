"""Experiment 5: retain the fitted bases, then build exact Direct and a 1000-path lookup."""
import hashlib
from pathlib import Path

from experiment_common import OUTPUT_ROOT, ROOT, run_point
from two_shift import two_params

METHODS = ("SEQ_REF", "COND_MEAN", "BIN_LOOKUP")


def run(output: Path = OUTPUT_ROOT / "experiment5", use_parallel: bool = True,
        pilot: bool = False) -> dict:
    p = two_params(pilot)
    p["pipeline_source_sha256"] = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in ("experiment5.py", "two_shift_policies.py")}
    return run_point(output, p, "two", "test" if pilot else "formal", use_parallel)


if __name__ == "__main__":
    run()
