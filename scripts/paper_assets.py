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
HOLM_KEY = {"CatBoost": "C13_catboost", "LightGBM": "C13_lgbm", "TabM": "C13_tabm", "TabPFN-v2": "C13_tabpfnv2",
            "TabPFN-2.5": "C13_tabpfnv2p5"}
SHORT = {"CatBoost": "catboost", "LightGBM": "lgbm", "TabM": "tabm", "TabPFN-v2": "tabpfnv2", "TabPFN-2.5": "tabpfnv2p5"}
FINAL, PREVIOUS = "cte_depth_top3_attn", "cte_depth_top3"  # the final method and the same method without the attention probe
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
    p18 = json.load(open(R("reports", "p18", "p18_report.json")))
    holm = p18["holm_all_preregistered"]  # Holm over all pre-registered tests, C1-C13
    m: dict[str, str] = {}

    # ---- headline per learner
    for l in LEARNERS:
        s, k = summ.loc[l], MACRO[l]
        m[f"n{k}"] = str(int(s.datasets))
        m[f"tar{k}"], m[f"ours{k}"], m[f"frozen{k}"] = f3(s.joint_tar), f3(s.ours), f3(s.joint_frozen)
        m[f"gain{k}"] = f3(s.mean_gain, True)
        m[f"wins{k}"] = str(int(s.wins))
        m[f"p{k}"] = fp(s.wilcoxon_p)
        m[f"holm{k}"] = fp(p18["C13"][SHORT[l]]["p_holm"])  # as registered for C13: Holm over the 5 learners
        m[f"holmAll{k}"] = fp(holm[HOLM_KEY[l]])  # the strictest reading: Holm over all pre-registered tests
    m["gainMin"], m["gainMax"] = f3(summ.mean_gain.min(), True), f3(summ.mean_gain.max(), True)
    m["pMax"] = fp(summ.wilcoxon_p.max())
    m["nHolmPass"] = str(int(sum(p18["C13"][SHORT[l]]["p_holm"] < 0.05 for l in LEARNERS)))
    fail = [l for l in LEARNERS if holm[HOLM_KEY[l]] >= 0.05]
    words = {0: "no", 1: "one", 2: "two", 3: "three", 4: "four", 5: "all five"}
    m["holmAllNote"] = ("all five pass" if not fail else
                        f"{words[5 - len(fail)]} of five pass (" + ", ".join(f"{l}: {fp(holm[HOLM_KEY[l]])}" for l in fail) + ")")
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
    r_ft = spearmanr(d.tg, d.gain_vs_tar)
    m["rhoFineTune"], m["pFineTune"] = f"{r_ft.statistic:.2f}", f"{r_ft.pvalue:.2f}"
    m["leakGap"], m["leakPos"], m["leakN"], m["leakDatasets"], m["leakP"] = (LEAK["gap"], LEAK["pos"], LEAK["n"],
                                                                         LEAK["datasets"], LEAK["p"])

    # ---- build-up (CatBoost, all full runs, vs Joint TAR of the same fold)
    fs = pd.read_csv(R("reports", "tables", "fold_scores.csv"))
    cb = fs[(fs.learner == "catboost") & fs.dataset.isin(piv.index)]
    stages = [("Joint Frozen (published)", None, "frozen"), ("our re-implementation of Joint TAR", "tar_repro", "tarRepro"),
              ("numeracy features only", "cte_num", "num"), ("neighbor expert only", "cte_nbr", "nbr"),
              ("lexical expert, \\emph{not} cross-fitted", "cte_lex_insample", "lexIn"),
              ("lexical expert only", "cte_lex", "lex"), ("all experts", "cte", "cte"),
              ("+ full-embedding probe", "cte_probe", "probe"), ("+ depth probes", PREVIOUS, "depth"),
              ("+ attention probe (final method)", FINAL, "attn")]
    rows = []
    for label, meth, key in stages:
        dd = cb[cb.method == ("cte" if meth is None else meth)]
        diff = ((dd.lb_joint_frozen if meth is None else dd.score) - dd.lb_joint_tar).groupby(dd.dataset).mean()
        rows.append((label, diff.mean(), int((diff > 0).sum()), wilcoxon(diff).pvalue))
        m[f"bu{key}"] = f3(diff.mean(), True)
    m["buLexInDrop"] = f3(rows[5][1] - rows[4][1])
    m["buProbeAdd"], m["buDepthAdd"] = f3(rows[7][1] - rows[6][1], True), f3(rows[8][1] - rows[7][1], True)
    m["buAttnAdd"] = f3(rows[9][1] - rows[8][1], True)
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
                   f"{fp(s.wilcoxon_p)} ({fp(p18['C13'][SHORT[l]]['p_holm'])}) \\\\")
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
    c7d = list(c7["per_dataset"].values())
    m["cSevenLosses"], m["cSevenBigLosses"] = str(sum(d < 0 for d in c7d)), str(sum(d < -0.07 for d in c7d))
    cc = pd.read_csv(R("results", "compute_cost.csv"))
    m["costOurs"], m["costTar"] = f"{cc.ours_seconds.median() / 60:.1f}", f"{cc.tar_seconds.median() / 60:.1f}"
    m["tarReproDiff"] = f3(rows[1][1], True)
    # label shuffle of the final method (E52: dev split 0 of fold 0, training labels permuted)
    ls = pd.concat([pd.read_csv(R("reports", "dev", "e52", f)) for f in ("label_shuffle.csv", "label_shuffle_jigsaw.csv")])
    ls = ls[ls.method == FINAL]
    sh = ls[(ls.train_labels == "shuffled") & ~ls.dataset.str.startswith("REG")]
    m["shuffleMin"], m["shuffleMax"] = f"{sh.score.min():.2f}", f"{sh.score.max():.2f}"
    m["shuffleRtwo"] = f"{ls[(ls.train_labels == 'shuffled') & ls.dataset.str.startswith('REG')].score.max():+.2f}".replace("-", "$-$")
    cbal = pd.read_csv(R("results", "class_balance.csv"))
    mm = cbal.merge(mg.rename("g").reset_index(), on="dataset")
    r_ = spearmanr(mm.minority_share, mm.g)
    m["rhoImbalance"], m["pImbalance"] = f"{r_.statistic:.2f}", f"{r_.pvalue:.2f}"

    # ---- the depth probes on their own: final method vs. the same method without them (CatBoost, 20 datasets)
    inc = json.load(open(R("reports", "p14", "p14_report.json")))["C9_vs_our_cte_probe"]
    m["depthIncWins"], m["depthIncP"] = str(inc["wins"]), fp(inc["p_value"])

    # ---- C12: other cross-fitting fold seeds (LightGBM), a replication of C10, Holm within its own family of 2
    p17 = json.load(open(R("reports", "p17", "p17_report.json")))
    c12 = p17["C12"]
    m["seedGainA"], m["seedGainB"] = f3(c12["seed43"]["mean_diff"], True), f3(c12["seed44"]["mean_diff"], True)
    m["seedWinsA"], m["seedWinsB"] = str(c12["seed43"]["wins"]), str(c12["seed44"]["wins"])
    m["seedHolm"] = fp(max(c12["seed43"]["p_holm"], c12["seed44"]["p_holm"]))
    m["seedSd"] = f"{p17['seed_spread']['per_dataset_sd_over_seeds_median']:.3f}"
    m["seedSameSign"] = str(p17["seed_spread"]["datasets_same_sign_all_seeds"])
    # ---- the attention probe on its own: final method vs. the same method without it (P18, paired, same folds)
    inc = p18["vs_current_method"]
    for l in LEARNERS:
        c_, k = inc[SHORT[l]], MACRO[l]
        m[f"attnInc{k}"], m[f"attnIncWins{k}"], m[f"attnIncP{k}"] = f3(c_["mean_diff"], True), str(c_["wins"]), fp(c_["p_value"])
    m["attnIncN"] = str(sum(inc[SHORT[l]]["mean_diff"] > 0 for l in LEARNERS))
    pos = [inc[SHORT[l]] for l in LEARNERS if inc[SHORT[l]]["mean_diff"] > 0]  # the learners it helps on average
    m["attnIncPosMin"], m["attnIncPosMax"] = f3(min(c_["mean_diff"] for c_ in pos), True), f3(max(c_["mean_diff"] for c_ in pos), True)
    m["attnIncPosPMax"] = fp(max(c_["p_value"] for c_ in pos))
    # Jigsaw Toxicity vs. Joint TAR, per learner, with and without the attention probe
    jig = fs[(fs.dataset == "BIN_TEXT_JIGSAW_TOXICITY") & fs.learner.isin(SHORT.values())]
    for meth, key in ((FINAL, "New"), (PREVIOUS, "Old")):
        g = (jig[jig.method == meth].score - jig[jig.method == meth].lb_joint_tar).groupby(jig.learner).mean()
        m[f"jigsaw{key}Min"], m[f"jigsaw{key}Max"] = f3(g.min(), True), f3(g.max(), True)
    # LightGBM on Spotify Genres: the one fold where LightGBM fails with the attention features
    sp = fs[(fs.dataset == "MUL_TEXT_SPOTIFY_GENRES") & (fs.method == FINAL)].set_index(["learner", "fold"]).score
    bad = sp["lgbm"].idxmin()
    m["spotifyFold"], m["spotifyBad"] = str(bad), f"{sp['lgbm'][bad]:.3f}"
    m["spotifyOtherMin"], m["spotifyOtherMax"] = f"{sp['lgbm'].drop(bad).min():.3f}", f"{sp['lgbm'].drop(bad).max():.3f}"
    m["spotifyCatBoost"], m["spotifyTabM"] = f"{sp['catboost'][bad]:.3f}", f"{sp['tabm'][bad]:.3f}"
    chk = pd.read_csv(R("reports", "p18", "a100_check.csv"))  # 19 P18 jobs re-run on an A100 from a clean commit (E55)
    m["spotifyAhundred"] = f"{chk[(chk.learner == 'lgbm') & (chk.dataset == 'MUL_TEXT_SPOTIFY_GENRES') & (chk.fold == bad)].score_a100.iloc[0]:.3f}"
    f0 = chk[chk.fold == 0].diff_a100_minus_t4
    m["gpuDiffMean"], m["gpuDiffMax"], m["gpuDiffN"] = f3(f0.mean(), True), f"{f0.abs().max():.2f}", str(len(f0))
    m["spotifyOld"] = f"{fs[(fs.dataset == 'MUL_TEXT_SPOTIFY_GENRES') & (fs.method == PREVIOUS) & (fs.learner == 'lgbm') & (fs.fold == bad)].score.iloc[0]:.3f}"
    m["spotifyLgbmDrop"] = f3((sp["lgbm"] - fs[(fs.dataset == "MUL_TEXT_SPOTIFY_GENRES") & (fs.method == PREVIOUS)].set_index(["learner", "fold"]).score["lgbm"]).mean(), True)
    m["attnIncLightGBMnoSpotify"] = f3(np.mean([v for d_, v in pd.Series(inc["lgbm"]["per_dataset"]).items() if d_ != "MUL_TEXT_SPOTIFY_GENRES"]), True) if "per_dataset" in inc["lgbm"] else "n/a"

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

    # ---- appendix: all pre-registered tests
    what = {"C1": "Cross-fitted fine-tuned TAR head (CatBoost) vs.\\ Joint TAR",
            "C2": "Experts only (CatBoost) vs.\\ Joint TAR", "C3_nb": "Experts, Naive-Bayes lexical model vs.\\ Joint TAR",
            "C3_res": "Experts on residual targets vs.\\ Joint TAR",
            "C4": "Label-free variant (CatBoost) vs.\\ Joint Frozen", "C5": "Experts + probe (CatBoost) vs.\\ Joint TAR",
            "C6": "Experts + probe, e5 query prefix vs.\\ Joint TAR", "C8": "Experts + probe (LightGBM) vs.\\ Joint TAR",
            "C9": "Depth-probe version, CatBoost vs.\\ Joint TAR",
            "C7:cte_probe|vs_joint_frozen": "36 datasets: experts + probe vs.\\ Joint Frozen",
            "C7:cte_probe|vs_joint_tar": "36 datasets: experts + probe vs.\\ Joint TAR",
            "C7:cte_depth_top3|vs_joint_frozen": "36 datasets: depth-probe version vs.\\ Joint Frozen",
            "C7:cte_depth_top3|vs_joint_tar": "36 datasets: depth-probe version vs.\\ Joint TAR",
            "C10_lgbm": "Depth-probe version, LightGBM vs.\\ Joint TAR",
            "C10_tabm": "Depth-probe version, TabM vs.\\ Joint TAR",
            "C11_tabpfnv2": "Depth-probe version, TabPFN-v2 vs.\\ Joint TAR",
            "C11_tabpfnv2p5": "Depth-probe version, TabPFN-2.5 vs.\\ Joint TAR",
            "C12_seed43": "Depth-probe version, LightGBM, cross-fitting seed 43 vs.\\ Joint TAR",
            "C12_seed44": "Depth-probe version, LightGBM, cross-fitting seed 44 vs.\\ Joint TAR",
            **{HOLM_KEY[l]: f"\\textbf{{Final method, {l}}} vs.\\ Joint TAR" for l in LEARNERS}}
    order = ["C1", "C2", "C3_nb", "C3_res", "C4", "C5", "C6", "C8", "C9", "C7:cte_probe|vs_joint_frozen",
             "C7:cte_probe|vs_joint_tar", "C7:cte_depth_top3|vs_joint_frozen", "C7:cte_depth_top3|vs_joint_tar",
             "C10_lgbm", "C10_tabm", "C11_tabpfnv2", "C11_tabpfnv2p5", "C12_seed43", "C12_seed44",
             *[HOLM_KEY[l] for l in LEARNERS]]
    from p16_report import EARLIER_P  # raw p-values of C1-C10, as in scripts/p18_report.py
    p16 = json.load(open(R("reports", "p16", "p16_report.json")))
    raw = dict(EARLIER_P, **{f"C11_{k}": c["p_value"] for k, c in p16["C11"].items()},
               C12_seed43=c12["seed43"]["p_value"], C12_seed44=c12["seed44"]["p_value"],
               **{HOLM_KEY[l]: p18["C13"][SHORT[l]]["p_value"] for l in LEARNERS})
    import full_report as fr
    assert all(abs(fr.holm(raw)[k] - holm[k]) < 1e-6 for k in holm)  # the Holm column is reproduced from these
    assert set(order) == set(holm) == set(raw), (set(order) ^ set(holm), set(order) ^ set(raw))
    lines = ["\\begin{tabular}{llrr}", "\\toprule", "Test & Comparison & $p$ & Holm (all " + str(len(order)) + ") \\\\",
             "\\midrule"]
    c7_label = {k: f"C7{c}" for k, c in zip([o for o in order if o.startswith("C7:")], "abcd")}
    c7_label.update({"C12_seed43": "C12a", "C12_seed44": "C12b"})
    for k in order:
        label = c7_label.get(k) or k.replace("_nb", "a").replace("_res", "b").split("_")[0]
        lines.append(f"{label} & {what[k]} & {fp(raw[k])} & {fp(holm[k])} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(join(GEN, "table_tests.tex"), "w").write("\n".join(lines) + "\n")

    figures(summ, hm, per_ds, rows[2:])
    print(f"{len(m)} macros, 4 tables, 2 figures -> paper/generated/, paper/figures/")


def figures(summ: pd.DataFrame, hm: pd.DataFrame, per_ds: pd.DataFrame, buildup: list) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # sized to their printed width in the paper (5.5 in text width: panel (a) 40%, panel (b) 58%), so fonts print as set
    plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    orange, grey = "#d95f02", "#555555"
    xlim, xt = (-0.004, 0.068), [0, 0.02, 0.04, 0.06]
    # results figure (a): gain per learner with its 95% interval
    fig, ax = plt.subplots(figsize=(2.6, 1.2))
    y = np.arange(len(LEARNERS))
    ax.errorbar(summ.mean_gain, y, xerr=[summ.mean_gain - summ.ci95_low, summ.ci95_high - summ.mean_gain],
                fmt="o", color=orange, ms=3.6, capsize=2, lw=0.9)
    for yi, l in enumerate(LEARNERS):
        ax.text(summ.ci95_high[l] + 0.002, yi, f"+{summ.mean_gain[l]:.3f}", va="center", fontsize=6.5, color=grey)
    ax.axvline(0, color="k", lw=0.7)
    ax.set_yticks(y, LEARNERS, fontsize=6.5); ax.set_ylim(len(LEARNERS) - 0.4, -0.6)  # inverted, padded
    ax.set_xlim(*xlim); ax.set_xticks(xt); ax.tick_params(labelsize=6.5, length=2)
    ax.set_xlabel("ours $-$ Joint TAR (95% CI)", fontsize=7)
    fig.tight_layout(pad=0.3); fig.savefig(join(FIGS, "fig_gains.pdf"), bbox_inches="tight", pad_inches=0.02); plt.close(fig)
    # results figure (b): mean gain per task type (dot: mean over learners; bar: lowest to highest learner)
    groups = [("REG", "regression (10)"), ("BIN", "binary (3)"), ("MUL", "multiclass (7)")]
    g = per_ds.assign(t=per_ds.dataset.str[:3]).groupby(["t", "learner"]).gain_vs_tar.mean()
    fig, ax = plt.subplots(figsize=(2.6, 1.2))
    for gi, (t, _) in enumerate(groups):
        v = g[t]
        ax.scatter(v.values, gi + np.linspace(-0.18, 0.18, len(v)), s=9, color=orange, alpha=0.55, lw=0)
        ax.scatter([v.mean()], [gi], s=30, color=orange, edgecolor="white", lw=0.6, zorder=3)
        ax.text(v.max() + 0.002, gi, f"+{v.mean():.3f}", va="center", fontsize=6.5, color=grey)
    ax.axvline(0, color="k", lw=0.7)
    ax.set_yticks(range(len(groups)), [lab for _, lab in groups], fontsize=6.5); ax.invert_yaxis()
    ax.set_ylim(len(groups) - 0.5, -0.5)
    ax.set_xlim(*xlim); ax.set_xticks(xt); ax.tick_params(labelsize=6.5, length=2)
    ax.set_xlabel("ours $-$ Joint TAR", fontsize=7)
    fig.tight_layout(pad=0.3); fig.savefig(join(FIGS, "fig_tasks.pdf"), bbox_inches="tight", pad_inches=0.02); plt.close(fig)
    # method figure (a): build-up of the method (CatBoost, 20 datasets; the rows of the appendix build-up table)
    short = {"numeracy features only": "numeracy only", "neighbor expert only": "neighbor only",
             "lexical expert, \\emph{not} cross-fitted": "lexical, not cross-fitted", "lexical expert only": "lexical only",
             "all experts": "all experts", "+ full-embedding probe": "+ full-embedding probe", "+ depth probes": "+ depth probes",
             "+ attention probe (final method)": "+ attention probe"}
    labels, gains = [short[r[0]] for r in buildup], [r[1] for r in buildup]
    fig, ax = plt.subplots(figsize=(2.3, 1.9))
    yb = np.arange(len(labels))
    ax.barh(yb, gains, height=0.62, color=[orange if v > 0 else "#7b3294" for v in gains])
    for yi, v in enumerate(gains):
        ax.text(max(v, 0) + 0.0012, yi, f"{v:+.3f}".replace("-", "$-$") if abs(v) >= 0.0005 else "0.000",
                va="center", ha="left", fontsize=6.3, color=grey)
    ax.axvline(0, color="k", lw=0.7)
    ax.set_yticks(yb, labels, fontsize=6.5); ax.invert_yaxis(); ax.set_xlim(-0.04, 0.032)
    ax.set_xticks([-0.03, 0, 0.03]); ax.tick_params(labelsize=6.5, length=2)
    ax.set_xlabel("score $-$ Joint TAR", fontsize=7)
    fig.tight_layout(pad=0.3); fig.savefig(join(FIGS, "fig_buildup.pdf"), bbox_inches="tight", pad_inches=0.02); plt.close(fig)
    # (c) depth profile heatmap
    best = hm.idxmax(axis=1)
    hm = hm.loc[best.sort_values(kind="stable").index]
    lim = 0.03
    fig, ax = plt.subplots(figsize=(3.15, 2.0))
    im = ax.imshow(hm.clip(-lim, lim), cmap="PuOr_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.scatter(hm.idxmax(axis=1).to_numpy() - 1, range(len(hm)), s=6, color="k")
    ax.set_xticks(range(12), range(1, 13)); ax.set_yticks(range(len(hm)), [name(d) for d in hm.index], fontsize=5.6)
    ax.set_xlabel("layer of frozen e5-small-v2"); ax.tick_params(length=1.5, labelsize=6.5)
    cb = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label("vs. full-embedding probe", fontsize=7); cb.ax.tick_params(labelsize=6.5)
    fig.tight_layout(pad=0.3); fig.savefig(join(FIGS, "fig_depth.pdf"), bbox_inches="tight", pad_inches=0.02); plt.close(fig)


if __name__ == "__main__":
    main()
