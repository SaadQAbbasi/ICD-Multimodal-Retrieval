"""Compare re-run experiments (outputs/<run>/) with the shipped results (results/<run>/).

Prints, per run folder found in both places: best epoch, test fused micro AP, P@5 and retrieval Jaccard@10,
new vs shipped, and flags |delta micro AP| > 0.003 (GPU non-determinism gives differences in the third decimal;
the seed-to-seed std of Round 4 is 0.0003-0.0004).

    python -m experiments.compare_with_shipped            # every run folder present in outputs/ and results/
    python -m experiments.compare_with_shipped experiment03_supcon_w0.001_e120_cos_s42
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def summary(run_dir: Path) -> dict | None:
    if not (run_dir / "results.json").is_file():
        return None
    res = json.loads((run_dir / "results.json").read_text())
    fused = res["test"]["modalities"]["fused"]
    ret_path = run_dir / "retrieval.json"
    ret = json.loads(ret_path.read_text()) if ret_path.is_file() else {}
    return {"best": res["best_epoch"], "ap": fused["micro_ap"], "p5": fused["precision_at_5"],
            "jac": ret.get("fused", {}).get("mean_jaccard")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", help="run folder names (default: all present in outputs/ and results/)")
    parser.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    parser.add_argument("--shipped", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    runs = args.runs or sorted(p.name for p in args.outputs.glob("*")
                               if (p / "results.json").is_file() and (args.shipped / p.name / "results.json").is_file())
    if not runs:
        print(f"No run folder of {args.shipped} found in {args.outputs}.")
        return
    fmt = lambda v: "-" if v is None else f"{v:.4f}"
    print(f"{'run':56s} {'best ep':>9s} {'micro AP new / shipped':>24s} {'delta':>8s} {'P@5 new / shipped':>19s} {'Jacc@10 new / shipped':>22s}")
    for run in runs:
        new, old = summary(args.outputs / run), summary(args.shipped / run)
        if new is None or old is None:
            print(f"{run:56s} missing results.json in {'outputs' if new is None else 'results'}")
            continue
        delta = new["ap"] - old["ap"]
        flag = "ok" if abs(delta) <= 0.003 else "CHECK"
        print(f"{run:56s} {new['best']:>4d}/{old['best']:<4d} {fmt(new['ap']):>11s} / {fmt(old['ap']):<10s} {delta:+.4f} {flag:5s}"
              f" {fmt(new['p5']):>8s} / {fmt(old['p5']):<8s} {fmt(new['jac']):>10s} / {fmt(old['jac']):<8s}")


if __name__ == "__main__":
    main()
