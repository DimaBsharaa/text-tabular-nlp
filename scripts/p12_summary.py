"""P12 dev-round summary with the decision rule fixed in PROGRESS.md P12 (before any P12 run):
advance a candidate iff mean over dataset-splits > +0.002 vs p9_base AND better on >= 6 of 9 datasets
(each dataset = mean over its dev splits). Only complete (dataset, fold, seed) pairs are compared.
    python scripts/p12_summary.py .explore/dev/results_p12.csv
"""
import sys

import pandas as pd

d = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else ".explore/dev/results_p12.csv")
d["ds"] = d.dataset.str.replace("BIN_TEXT_|MUL_TEXT_|REG_TEXT_", "", regex=True).str[:14]
w = d.pivot_table(index=["ds", "fold", "seed"], columns="method", values="score")
print(f"jobs {len(d)} | job hours {d.seconds.sum() / 3600:.2f} | datasets {w.index.get_level_values(0).nunique()}")
for m in [c for c in w.columns if c != "p9_base"]:
    x = (w[m] - w["p9_base"]).dropna()
    per = x.groupby(level=0).mean()
    advance = x.mean() > 0.002 and (per > 0).sum() >= 6 and len(per) >= 9
    print(f"{m:<11} splits {len(x):>3} | mean {x.mean():+.4f} | datasets better {(per > 0).sum()}/{len(per)} "
          f"-> {'ADVANCE' if advance else 'no'}")
    print("            ", {k: round(v, 4) for k, v in per.items()})
