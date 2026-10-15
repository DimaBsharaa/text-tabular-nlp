"""Summary of scripts/dev_eval.py results: mean dev score per method and the difference vs its baseline
(frozen0 for label-free tf_* and cte; cte for cte_* variants), per dataset.
    python scripts/dev_summary.py reports/dev/p7_dev_results.csv
"""
import sys, pandas as pd
d = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "reports/dev/p7_dev_results.csv")
d["ds"] = d.dataset.str.replace("BIN_TEXT_|MUL_TEXT_|REG_TEXT_", "", regex=True).str[:12]
w = d.pivot_table(index=["ds", "seed"], columns="method", values="score")
m = w.groupby("ds").mean()
order = [c for c in ["frozen0", "tf_num", "tf_lsa", "tf_row", "tf_num_lsa", "tf_all", "cte", "cte_probe", "cte_rowlex",
                     "cte_lsa_row", "cte_probe_rowlex", "cte_all"] if c in m]
print("mean dev score (2 seeds):"); print(m[order].round(4).to_string())
rows = {}
for c in order:
    base = "frozen0" if c.startswith("tf_") or c == "cte" else ("cte" if c.startswith("cte_") else None)
    if base and base in m and c != base:
        diff = (w[c] - w[base]).groupby("ds").mean()
        rows[f"{c} - {base}"] = diff
t = pd.DataFrame(rows).T
t["mean"] = t.mean(axis=1); t["wins"] = (t.drop(columns="mean") > 0).sum(axis=1)
print("\ndifference vs baseline (mean of seeds), per dataset:"); print(t.round(4).to_string())
print("\njobs:", len(d), "| total seconds:", int(d.seconds.sum()))
