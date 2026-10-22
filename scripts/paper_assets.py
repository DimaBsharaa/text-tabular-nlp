"""Everything the paper shows that comes from data: LaTeX number macros, tables and figures, generated from the
committed results/ and reports/ files, so no number in the paper is typed by hand.
    python scripts/paper_assets.py
Writes paper/generated/numbers.tex (\\newcommand macros), paper/generated/table_*.tex and paper/figures/fig_*.pdf.
"""
from __future__ import annotations

import json
import os
import sys
from os.path import abspath, dirname, join

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

REPO_ROOT = dirname(dirname(abspath(__file__)))
sys.path.insert(0, join(REPO_ROOT, "scripts"))
GEN = join(REPO_ROOT, "paper", "generated")
FIGS = join(REPO_ROOT, "paper", "figures")
R = lambda *p: join(REPO_ROOT, *p)

LEARNERS = ["CatBoost", "LightGBM", "TabM", "TabPFN-v2", "TabPFN-2.5"]
MACRO = {"CatBoost": "CatBoost", "LightGBM": "LightGBM", "TabM": "TabM", "TabPFN-v2": "PFNtwo",
         "TabPFN-2.5": "PFNtwofive"}
HOLM_KEY = {"CatBoost": "C9", "LightGBM": "C10_lgbm", "TabM": "C10_tabm", "TabPFN-v2": "C11_tabpfnv2",
            "TabPFN-2.5": "C11_tabpfnv2p5"}
# E8 (archive/PROGRESS.md): Joint TAR's head quality, its own training rows minus test rows, 6 pilot datasets x 3 folds
LEAK = {"gap": "+0.031", "pos": "16", "n": "18", "datasets": "6", "p": "0.0002"}


def name(d: str) -> str:
    n = d.split("_TEXT_", 1)[1].replace("_", " ").title()
    return n.replace("Us ", "US ").replace("Scimagojr", "SCImago")


def f3(x: float, sign: bool = False) -> str:
    if sign and abs(x) < 0.0005:
        return "0.000"
    s = f"{x:+.3f}" if sign else f"{x:.3f}"
    return s.replace("-", "$-$") if sign else s


def fp(p: float) -> str:
    if p < 0.0001:
        return "$<$0.0001"
    return f"{p:.4f}" if p < 0.001 else f"{p:.3f}"


def main() -> None:
    os.makedirs(GEN, exist_ok=True)
    os.makedirs(FIGS, exist_ok=True)
    summ = pd.read_csv(R("results", "summary.csv")).set_index("learner").loc[LEARNERS]
    per_ds = pd.read_csv(R("results", "scores_per_dataset.csv"))
    folds = pd.read_csv(R("results", "scores_per_fold.csv"))
    holm = json.load(open(R("reports", "p16", "p16_report.json")))["holm_all_preregistered"]
    p16 = json.load(open(R("reports", "p16", "p16_report.json")))
    m: dict[str, str] = {}

    # ---- headline per learner
    for l in LEARNERS:
        s, k = summ.loc[l], MACRO[l]
        m[f"n{k}"] = str(int(s.datasets))
        m[f"tar{k}"], m[f"ours{k}"], m[f"frozen{k}"] = f3(s.joint_tar), f3(s.ours), f3(s.joint_frozen)
        m[f"gain{k}"] = f3(s.mean_gain, True)
        m[f"wins{k}"] = str(int(s.wins))
        m[f"p{k}"] = fp(s.wilcoxon_p)
        m[f"holm{k}"] = fp(holm[HOLM_KEY[l]])
    m["gainMin"], m["gainMax"] = f3(summ.mean_gain.min(), True), f3(summ.mean_gain.max(), True)
    m["pMax"] = fp(summ.wilcoxon_p.max())
    m["nHolmPass"] = str(int(sum(holm[HOLM_KEY[l]] < 0.05 for l in LEARNERS)))
    mult = (summ.ours - summ.joint_frozen) / (summ.joint_tar - summ.joint_frozen)
    m["multMin"], m["multMax"] = f"{mult.min():.1f}", f"{mult.max():.1f}"
    m["nTests"] = str(len(holm))

    # ---- consistency
    m["runsWonPct"] = f"{100 * (folds.gain_vs_tar > 0).mean():.0f}"
    m["nRuns"] = str(len(folds))
    piv = per_ds.pivot(index="dataset", columns="learner", values="gain_vs_tar")[LEARNERS]
    npos, nobs = (piv > 0).sum(axis=1), piv.notna().sum(axis=1)
    m["dsWonAll"], m["dsWonFour"] = str(int((npos == nobs).sum())), str(int((npos >= 4).sum()))
    by_task = per_ds.assign(t=per_ds.dataset.str[:3]).groupby("t").gain_vs_tar
    lo_hi = per_ds.assign(t=per_ds.dataset.str[:3]).groupby(["t", "learner"]).gain_vs_tar.mean()
    for t, key in (("REG", "Reg"), ("BIN", "Bin"), ("MUL", "Mul")):
        m[f"gain{key}Min"], m[f"gain{key}Max"] = f3(lo_hi[t].min(), True), f3(lo_hi[t].max(), True)
    noise = pd.read_csv(R("reports", "phase0", "baseline_noise.csv")).set_index("dataset").paired_diff_std
    mg = per_ds.groupby("dataset").gain_vs_tar.mean()
    m["dsAboveNoise"] = str(int((mg / noise.loc[mg.index] > 1).sum()))
    ph0 = pd.read_csv(R("reports", "phase0", "summary.csv")).set_index("dataset")
    m["jigsawTarGain"] = f3(ph0.loc["BIN_TEXT_JIGSAW_TOXICITY", "gain_over_frozen"], True)
    d = per_ds.assign(tg=per_ds.joint_tar - per_ds.joint_frozen).groupby("dataset")[["tg", "gain_vs_tar"]].mean()
    m["rhoFineTune"] = f"{spearmanr(d.tg, d.gain_vs_tar).statistic:.2f}"
    m["leakGap"], m["leakPos"], m["leakN"], m["leakDatasets"], m["leakP"] = (LEAK["gap"], LEAK["pos"], LEAK["n"],
                                                                         LEAK["datasets"], LEAK["p"])

    # ---- build-up (CatBoost, all full runs, vs Joint TAR of the same fold)
    fs = pd.read_csv(R("reports", "tables", "fold_scores.csv"))
    cb = fs[(fs.learner == "catboost") & fs.dataset.isin(piv.index)]
    stages = [("Joint Frozen (published)", None, "frozen"), ("our re-implementation of Joint TAR", "tar_repro", "tarRepro"),
              ("numeracy features only", "cte_num", "num"), ("neighbour expert only", "cte_nbr", "nbr"),
              ("lexical expert, \\emph{not} cross-fitted", "cte_lex_insample", "lexIn"),
              ("lexical expert only", "cte_lex", "lex"), ("all experts", "cte", "cte"),
              ("+ final-layer probe", "cte_probe", "probe"), ("+ depth probes (final method)", "cte_depth_top3", "depth")]
    rows = []
    for label, meth, key in stages:
        dd = cb[cb.method == ("cte" if meth is None else meth)]
        diff = ((dd.lb_joint_frozen if meth is None else dd.score) - dd.lb_joint_tar).groupby(dd.dataset).mean()
        rows.append((label, diff.mean(), int((diff > 0).sum()), wilcoxon(diff).pvalue))
        m[f"bu{key}"] = f3(diff.mean(), True)
    m["buLexInDrop"] = f3(rows[5][1] - rows[4][1])
    m["buProbeAdd"], m["buDepthAdd"] = f3(rows[7][1] - rows[6][1], True), f3(rows[8][1] - rows[7][1], True)
    tarr = [r for r in rows if r[0].startswith("our re-impl")][0]
    m["tarReproP"] = fp(tarr[3])
    tab = ["\\begin{tabular}{lrrr}", "\\toprule", "Text features (CatBoost) & $\\Delta$ vs.\\ Joint TAR & Wins & $p$ \\\\",
           "\\midrule"]
    for label, g, w, p in rows[2:]:  # the two baselines are reported in the text (Setup, Main result)
        tab.append(f"{label} & {f3(g, True)} & {w}/20 & {fp(p)} \\\\")
    tab += ["\\bottomrule", "\\end{tabular}"]
    open(join(GEN, "table_buildup.tex"), "w").write("\n".join(tab) + "\n")

    # ---- main table
    tab = ["\\begin{tabular}{lrrrrrrr}", "\\toprule",
           "Learner & $n$ & Joint Frozen & Joint TAR & Ours & Gain [95\\% CI] & Wins & $p$ (Holm) \\\\", "\\midrule"]
    for l in LEARNERS:
        s = summ.loc[l]
        tab.append(f"{l} & {int(s.datasets)} & {f3(s.joint_frozen)} & {f3(s.joint_tar)} & \\textbf{{{f3(s.ours)}}} & "
                   f"{f3(s.mean_gain, True)} [{f3(s.ci95_low)}, {f3(s.ci95_high)}] & {int(s.wins)} & "
                   f"{fp(s.wilcoxon_p)} ({fp(holm[HOLM_KEY[l]])}) \\\\")
    tab += ["\\bottomrule", "\\end{tabular}"]
    open(join(GEN, "table_main.tex"), "w").write("\n".join(tab) + "\n")

    # ---- depth profile (20 datasets, training-row OOF scores of the final CatBoost run)
    dp = pd.read_csv(R("reports", "figures", "fig7_depth_profile_20.csv"))
    hm = dp.pivot(index="dataset", columns="layer", values="delta")
    best = hm.idxmax(axis=1)
    m["depthInner"] = str(int((best < 12).sum()))
    m["depthMinLayer"], m["depthMaxLayer"] = str(int(best.min())), str(int(best.max()))
    top = hm.max(axis=1).sort_values(ascending=False)
    m["depthTopName"], m["depthTopGain"] = name(top.index[0]), f3(top.iloc[0], True)
    m["depthTopLayer"] = str(int(best[top.index[0]]))
    m["depthJigsaw"] = f3(top["BIN_TEXT_JIGSAW_TOXICITY"], True)

    # ---- 36 further datasets (C7), cost, checks
    c7 = json.load(open(R("reports", "p14", "p14_report.json")))["C7"]["cte_depth_top3|vs_joint_tar"]
    m["cSevenN"], m["cSevenWins"] = str(c7["n"]), str(c7["wins"])
    m["cSevenMedian"], m["cSevenP"] = f3(c7["median_diff"], True), fp(c7["p_value"])
    m["cSevenHolm"] = fp(holm["C7:cte_depth_top3|vs_joint_tar"])
    cc = pd.read_csv(R("results", "compute_cost.csv"))
    m["costOurs"], m["costTar"] = f"{cc.ours_seconds.median() / 60:.1f}", f"{cc.tar_seconds.median() / 60:.1f}"
    m["tarReproDiff"] = f3(rows[1][1], True)
    ls = pd.read_csv(R("reports", "checks", "label_shuffle.csv"))
    sh = ls[(ls.train_labels == "shuffled") & ~ls.dataset.str.startswith("REG")]
    m["shuffleMin"], m["shuffleMax"] = f"{sh.score.min():.2f}", f"{sh.score.max():.2f}"
    m["shuffleRtwo"] = f"{ls[(ls.train_labels == 'shuffled') & ls.dataset.str.startswith('REG')].score.max():+.2f}".replace("-", "$-$")
    cbal = pd.read_csv(R("results", "class_balance.csv"))
    mm = cbal.merge(mg.rename("g").reset_index(), on="dataset")
    r_ = spearmanr(mm.minority_share, mm.g)
    m["rhoImbalance"], m["pImbalance"] = f"{r_.statistic:.2f}", f"{r_.pvalue:.2f}"

    # ---- C12: other cross-fitting fold seeds (LightGBM), a replication of C10, Holm within its own family of 2
    import full_report as fr
    import p16_report as pr
    p17 = json.load(open(R("reports", "p17", "p17_report.json")))
    c12 = p17["C12"]
    m["seedGainA"], m["seedGainB"] = f3(c12["seed43"]["mean_diff"], True), f3(c12["seed44"]["mean_diff"], True)
    m["seedWinsA"], m["seedWinsB"] = str(c12["seed43"]["wins"]), str(c12["seed44"]["wins"])
    m["seedHolm"] = fp(max(c12["seed43"]["p_holm"], c12["seed44"]["p_holm"]))
    m["seedSd"] = f"{p17['seed_spread']['per_dataset_sd_over_seeds_median']:.3f}"
    m["seedSameSign"] = str(p17["seed_spread"]["datasets_same_sign_all_seeds"])
    raw = dict(pr.EARLIER_P, **{f"C11_{k}": c["p_value"] for k, c in p16["C11"].items()})
    holm19 = fr.holm(dict(raw, C12_seed43=c12["seed43"]["p_value"], C12_seed44=c12["seed44"]["p_value"]))
    m["holmNineteenPFNtwo"] = fp(holm19["C11_tabpfnv2"])
    m["nHolmPassNineteen"] = str(int(sum(holm19[HOLM_KEY[l]] < 0.05 for l in LEARNERS)))

    with open(join(GEN, "numbers.tex"), "w") as f:
        f.write("% generated by scripts/paper_assets.py from results/ and reports/ -- do not edit by hand\n")
        for k, v in m.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

    # ---- appendix: per-dataset gains and scores
    lines = ["\\begin{tabular}{l" + "r" * 10 + "}", "\\toprule",
             "& \\multicolumn{2}{c}{CatBoost} & \\multicolumn{2}{c}{LightGBM} & \\multicolumn{2}{c}{TabM} & "
             "\\multicolumn{2}{c}{TabPFN-v2} & \\multicolumn{2}{c}{TabPFN-2.5} \\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\\cmidrule(lr){10-11}",
             "Dataset" + " & TAR & Ours" * 5 + " \\\\", "\\midrule"]
    t_ = per_ds.pivot(index="dataset", columns="learner", values="joint_tar")
    o_ = per_ds.pivot(index="dataset", columns="learner", values="ours")
    for dsn in sorted(piv.index, key=lambda x: (x[:3], name(x))):
        cells = []
        for l in LEARNERS:
            if pd.isna(t_.loc[dsn, l]):
                cells += ["--", "--"]
            else:
                tv, ov = t_.loc[dsn, l], o_.loc[dsn, l]
                cells += [f3(tv), (f"\\textbf{{{f3(ov)}}}" if ov > tv else f3(ov))]
        task = {"BIN": "B", "MUL": "M", "REG": "R"}[dsn[:3]]
        lines.append(f"{name(dsn)} ({task}) & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(join(GEN, "table_per_dataset.tex"), "w").write("\n".join(lines) + "\n")

    # ---- appendix: all pre-registered tests (raw: computed above)
    what = {"C1": "Cross-fitted fine-tuned TAR head (CatBoost) vs.\\ Joint TAR",
            "C2": "Experts only (CatBoost) vs.\\ Joint TAR", "C3_nb": "Experts, Naive-Bayes lexical model vs.\\ Joint TAR",
            "C3_res": "Experts on residual targets vs.\\ Joint TAR",
            "C4": "Label-free variant (CatBoost) vs.\\ Joint Frozen", "C5": "Experts + probe (CatBoost) vs.\\ Joint TAR",
            "C6": "Experts + probe, e5 query prefix vs.\\ Joint TAR", "C8": "Experts + probe (LightGBM) vs.\\ Joint TAR",
            "C9": "\\textbf{Final method, CatBoost} vs.\\ Joint TAR",
            "C7:cte_probe|vs_joint_frozen": "36 datasets: experts + probe vs.\\ Joint Frozen",
            "C7:cte_probe|vs_joint_tar": "36 datasets: experts + probe vs.\\ Joint TAR",
            "C7:cte_depth_top3|vs_joint_frozen": "36 datasets: final method vs.\\ Joint Frozen",
            "C7:cte_depth_top3|vs_joint_tar": "36 datasets: final method vs.\\ Joint TAR",
            "C10_lgbm": "\\textbf{Final method, LightGBM} vs.\\ Joint TAR",
            "C10_tabm": "\\textbf{Final method, TabM} vs.\\ Joint TAR",
            "C11_tabpfnv2": "\\textbf{Final method, TabPFN-v2} vs.\\ Joint TAR",
            "C11_tabpfnv2p5": "\\textbf{Final method, TabPFN-2.5} vs.\\ Joint TAR"}
    order = ["C1", "C2", "C3_nb", "C3_res", "C4", "C5", "C6", "C8", "C9", "C7:cte_probe|vs_joint_frozen",
             "C7:cte_probe|vs_joint_tar", "C7:cte_depth_top3|vs_joint_frozen", "C7:cte_depth_top3|vs_joint_tar",
             "C10_lgbm", "C10_tabm", "C11_tabpfnv2", "C11_tabpfnv2p5"]
    assert set(order) == set(holm) == set(raw), (set(order) ^ set(holm), set(order) ^ set(raw))
    lines = ["\\begin{tabular}{llrr}", "\\toprule", "Test & Comparison & $p$ & Holm (all " + str(len(order)) + ") \\\\",
             "\\midrule"]
    c7_label = {k: f"C7{c}" for k, c in zip([o for o in order if o.startswith("C7:")], "abcd")}
    for k in order:
        label = c7_label.get(k) or k.replace("_nb", "a").replace("_res", "b").split("_")[0]
        lines.append(f"{label} & {what[k]} & {fp(raw[k])} & {fp(holm[k])} \\\\")
    lines.append("\\midrule")
    for s_, c_ in (("a", c12["seed43"]), ("b", c12["seed44"])):
        seed = {"a": 43, "b": 44}[s_]
        lines.append(f"C12{s_} & Final method, LightGBM, cross-fitting seed {seed} vs.\\ Joint TAR & {fp(c_['p_value'])} & "
                     f"{fp(c_['p_holm'])}$^\\dagger$ \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(join(GEN, "table_tests.tex"), "w").write("\n".join(lines) + "\n")

    figures(summ, hm)
    print(f"{len(m)} macros, 4 tables, 2 figures -> paper/generated/, paper/figures/")


def figures(summ: pd.DataFrame, hm: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # sized to their printed width in the paper (5.5 in text width: panel (a) 40%, panel (b) 58%), so fonts print as set
    plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    orange, grey = "#d95f02", "#555555"
    # (a) gain per learner with 95% CI
    fig, ax = plt.subplots(figsize=(2.15, 1.9))
    y = np.arange(len(LEARNERS))
    ax.errorbar(summ.mean_gain, y, xerr=[summ.mean_gain - summ.ci95_low, summ.ci95_high - summ.mean_gain],
                fmt="o", color=orange, ms=4, capsize=2.5, lw=1)
    for yi, (g, w, n) in enumerate(zip(summ.mean_gain, summ.wins, summ.datasets)):
        ax.text(summ.ci95_high.iloc[yi] + 0.002, yi, f"+{g:.3f}", va="center", fontsize=7.5, color=grey)
    ax.axvline(0, color="k", lw=0.7)
    ax.set_yticks(y, LEARNERS); ax.invert_yaxis(); ax.set_xlim(-0.004, 0.074)
    ax.set_xticks([0, 0.02, 0.04]); ax.set_xlabel("ours $-$ Joint TAR (95% CI)")
    fig.tight_layout(pad=0.3); fig.savefig(join(FIGS, "fig_gains.pdf"), bbox_inches="tight", pad_inches=0.02); plt.close(fig)
    # (b) depth profile heatmap
    best = hm.idxmax(axis=1)
    hm = hm.loc[best.sort_values(kind="stable").index]
    lim = 0.03
    fig, ax = plt.subplots(figsize=(3.15, 2.32))
    im = ax.imshow(hm.clip(-lim, lim), cmap="PuOr_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.scatter(hm.idxmax(axis=1).to_numpy() - 1, range(len(hm)), s=6, color="k")
    ax.set_xticks(range(12), range(1, 13)); ax.set_yticks(range(len(hm)), [name(d) for d in hm.index], fontsize=6.3)
    ax.set_xlabel("layer of frozen e5-small-v2"); ax.tick_params(length=1.5, labelsize=6.5)
    cb = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label("vs. final-layer probe", fontsize=7); cb.ax.tick_params(labelsize=6.5)
    fig.tight_layout(pad=0.3); fig.savefig(join(FIGS, "fig_depth.pdf"), bbox_inches="tight", pad_inches=0.02); plt.close(fig)


if __name__ == "__main__":
    main()
