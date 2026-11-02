#!/usr/bin/env python3
"""F's full protocol (experiments/F_layer_sweep/preregistration.md, 6fc116d), as used by G3 / H2.

  run_full_f_protocol.py --model qwen2.5-1.5b --suite arith --task C --out experiments/H_arithmetic
"""

from __future__ import annotations

import argparse
import collections
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import f_common  # noqa: E402
from run_e_layer_sweep import bootstrap_ci  # noqa: E402
from run_gh import STOP, load_backend, task_items, task_prompts  # noqa: E402

SEEDS = range(5)


# ---------------------------------------------------------------- collection
def collect(model: str, suite: str, task: str, n: int, shifted: str = "qend5") -> dict:
    import torch

    b = load_backend(model)
    its = task_items(suite, task, n)
    cot_pre, ao_pre = task_prompts(suite, task)
    L = b.model.config.num_hidden_layers
    keys = ("last", "shifted", "mean_all", "mean_q")
    acts = {k: [] for k in keys}
    meta = []
    for i, it in enumerate(its):
        prompt = f_common.cot_prompt(cot_pre, it)
        ids = b.encode_raw(prompt)
        q_end = b.encode_raw(f_common.question_end_prefix(cot_pre, it)).shape[1] - 1
        q_start = b.encode_raw(cot_pre + "\n\nQ: ").shape[1]
        if shifted == "option_e":  # token where "(E" begins, located directly
            cut = it["input"].index("(E)") + 2
            s_idx = b.encode_raw(cot_pre + "\n\nQ: " + it["input"][:cut]).shape[1] - 1
        else:
            s_idx = q_end - 5
        with torch.no_grad():
            hs = b.model(ids, output_hidden_states=True).hidden_states[1:]  # L x [1, T, D]
        H = torch.stack([h[0] for h in hs]).float()  # [L, T, D]
        acts["last"].append(H[:, -1].cpu().numpy().astype(np.float16))
        acts["shifted"].append(H[:, s_idx].cpu().numpy().astype(np.float16))
        acts["mean_all"].append(H.mean(1).cpu().numpy().astype(np.float16))
        acts["mean_q"].append(H[:, q_start:q_end + 1].mean(1).cpu().numpy().astype(np.float16))
        cot_t = b.generate_text(prompt, max_new_tokens=384, stop_strings=STOP).text
        nc_t = b.generate_text(f_common.no_cot_prompt(ao_pre, it), max_new_tokens=2).text
        meta.append({"idx": it["idx"], "target": it["target"], "letters": it["letters"],
                     "prompt_len": int(ids.shape[1]), "q_end": int(q_end), "q_start": int(q_start),
                     "shifted_idx": int(s_idx), "shifted_mode": shifted, "group": it.get("group"),
                     "no_cot_text": nc_t, "cot_text": cot_t,
                     "no_cot": f_common.parse_no_cot(nc_t, it["letters"]),
                     "with_cot": f_common.parse_cot(cot_t, it["letters"])})
        if (i + 1) % 25 == 0:
            print(f"collect {i + 1}/{len(its)}", flush=True)
    b.shutdown()
    return {"acts": {k: np.stack(v) for k, v in acts.items()}, "meta": meta, "n_layers": L,
            "d_model": int(acts["last"][0].shape[1]), "device": b.device}


# ---------------------------------------------------------------- probing
def _splits(X, y, seed, groups):
    from sklearn.model_selection import KFold, StratifiedGroupKFold, StratifiedKFold

    _, counts = np.unique(y, return_counts=True)
    strat = counts.min() >= 5
    if groups is not None:
        sgk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        return sgk.split(X, y, groups), strat
    sp = (StratifiedKFold if strat else KFold)(n_splits=5, shuffle=True, random_state=seed)
    return sp.split(X, y), strat


def cv_correct(X, y, seed=0, groups=None):
    from cue_channels.probes.train import _make_pipeline

    splits, strat = _splits(X, y, seed, groups)
    c = np.zeros(len(y), dtype=bool)
    for tr, te in splits:
        if len(np.unique(y[tr])) < 2:
            c[te] = y[te] == y[tr][0]
            continue
        c[te] = _make_pipeline(C=0.1, seed=seed).fit(X[tr], y[tr]).predict(X[te]) == y[te]
    return c, strat


def cv_predict(X, y, seed=0, groups=None):
    from cue_channels.probes.train import _make_pipeline

    splits, _ = _splits(X, y, seed, groups)
    pred = np.empty(len(y), dtype=object)
    for tr, te in splits:
        if len(np.unique(y[tr])) < 2:
            pred[te] = y[tr][0]
            continue
        pred[te] = _make_pipeline(C=0.1, seed=seed).fit(X[tr], y[tr]).predict(X[te])
    return pred


def cell(X, y, groups=None):
    accs, strat_all = [], True
    for s in SEEDS:
        c, strat = cv_correct(X, y, s, groups)
        strat_all &= strat
        accs.append(float(c.mean()))
        if s == 0:
            c0 = c
    return {"acc": round(float(c0.mean()), 3), "ci95": bootstrap_ci(c0),
            "seed_mean": round(float(np.mean(accs)), 3), "seed_sd": round(float(np.std(accs)), 3),
            "seed_values": [round(a, 3) for a in accs], "stratified": strat_all}


def max_over_layers_perm(Xl, y, n_perm=100, groups=None):
    from joblib import Parallel, delayed

    def one(p):
        yp = np.random.default_rng(1000 + p).permutation(y)
        return max(cv_correct(Xl[:, lyr], yp, 0, groups)[0].mean() for lyr in range(Xl.shape[1]))

    return np.array(Parallel(n_jobs=-1)(delayed(one)(p) for p in range(n_perm)))


def analyze(blob: dict, n_perm: int = 100, grouped: bool = False) -> dict:
    meta, A = blob["meta"], blob["acts"]
    L, D = blob["n_layers"], blob["d_model"]
    K = len(meta[0]["letters"])
    out = {"n": len(meta), "n_layers": L, "d_model": D, "n_options": K, "chance": round(1 / K, 3)}
    truth = [m["target"] for m in meta]
    nca, wca = [m["no_cot"] for m in meta], [m["with_cot"] for m in meta]
    out["behavior"] = {
        "no_cot_acc": round(float(np.mean([a == t for a, t in zip(nca, truth, strict=True)])), 3),
        "no_cot_ci95": bootstrap_ci(np.array([a == t for a, t in zip(nca, truth, strict=True)])),
        "with_cot_acc": round(float(np.mean([a == t for a, t in zip(wca, truth, strict=True)])), 3),
        "with_cot_ci95": bootstrap_ci(np.array([a == t for a, t in zip(wca, truth, strict=True)])),
        "no_cot_dist": dict(collections.Counter(str(a) for a in nca)),
        "with_cot_dist": dict(collections.Counter(str(a) for a in wca)),
        "truth_dist": dict(collections.Counter(truth)),
        "no_cot_unparsed": nca.count(None), "with_cot_unparsed": wca.count(None),
        "A_eq_B_rate": round(float(np.mean([a == b for a, b in zip(nca, wca, strict=True)
                                            if a and b])), 3),
    }
    targets = {"A": nca, "B": wca}
    res = {}
    for name, lab in targets.items():
        keep = np.array([x is not None for x in lab])
        y = np.array([x for x in lab if x is not None])
        maj = max(collections.Counter(y.tolist()).values()) / len(y)
        Xl = A["last"][keep].astype(np.float32)  # [n, L, D]
        g = None
        if grouped:
            g = np.array([m["group"] for m, k in zip(meta, keep, strict=True) if k])
        by_layer = {}
        for lyr in range(L):
            by_layer[lyr] = cell(Xl[:, lyr], y, g)
            print(f"  target {name} layer {lyr}: {by_layer[lyr]['seed_mean']}", flush=True)
        gauss = []
        for s in SEEDS:
            G = np.random.default_rng(500 + s).standard_normal((len(y), D)).astype(np.float32)
            gauss.append(float(cv_correct(G, y, s, g)[0].mean()))
        peak = max(by_layer, key=lambda k: by_layer[k]["seed_mean"])
        t0 = time.time()
        perm = max_over_layers_perm(Xl, y, n_perm, g)
        pv = float((np.sum(perm >= by_layer[peak]["acc"]) + 1) / (n_perm + 1))
        print(f"  target {name}: peak L{peak}, perm {time.time() - t0:.0f}s p={pv:.3f}", flush=True)

        # positional checks at the peak layer
        meta_k = [m for m, k in zip(meta, keep, strict=True) if k]
        plen = np.array([m["prompt_len"] for m in meta_k])
        med = np.median(plen)
        lb = np.abs(plen - med) <= 2
        pos_feats = np.array([[m["prompt_len"], m["q_end"]] for m in meta_k], dtype=np.float32)
        checks = {
            "last_token": by_layer[peak],
            "length_balanced": {**(cell(Xl[lb, peak], y[lb], g[lb] if g is not None else None)
                                   if lb.sum() >= 10 and
                                   len(np.unique(y[lb])) > 1 else {"acc": None}),
                                "n": int(lb.sum()), "underpowered": bool(lb.sum() < 60),
                                "majority": round(max(collections.Counter(y[lb].tolist()).values())
                                                  / max(lb.sum(), 1), 3) if lb.sum() else None},
            "position_only": cell(pos_feats, y, g),
            "shifted_token": cell(A["shifted"][keep][:, peak].astype(np.float32), y, g),
            "mean_pooled_all": cell(A["mean_all"][keep][:, peak].astype(np.float32), y, g),
            "mean_pooled_question": cell(A["mean_q"][keep][:, peak].astype(np.float32), y, g),
        }
        pk = by_layer[peak]
        chance = 1 / K
        if name == "A":
            supported = pk["acc"] > maj + 0.05 and pk["ci95"][0] > maj
        else:
            supported = pk["acc"] > chance + 0.10 and pk["ci95"][0] > chance
        vs_major_B = pk["acc"] > maj + 0.05 and pk["ci95"][0] > maj

        def beats(other, checks=checks, pk=pk):
            o = checks[other]
            return (pk["acc"] - o["acc"] >= 0.10) and (pk["ci95"][0] > o["ci95"][1])

        pos_ok = {"vs_position_only": beats("position_only"),
                  "vs_shifted_token": beats("shifted_token")}
        verdict = "supported" if supported else "not supported"
        if supported and pv >= 0.05:
            verdict = "inconclusive"
        res[name] = {
            "n": int(len(y)), "excluded_unparsed": int((~keep).sum()),
            "majority": round(maj, 3), "label_dist": dict(collections.Counter(y.tolist())),
            "gaussian_baseline": {"mean": round(float(np.mean(gauss)), 3),
                                  "sd": round(float(np.std(gauss)), 3)},
            "by_layer": by_layer, "peak_layer": peak, "peak": pk, "layer_24": by_layer.get(24),
            "max_over_layers_perm": {"n_perm": n_perm, "p_value": round(pv, 4),
                                     "null_mean": round(float(perm.mean()), 3),
                                     "null_p95": round(float(np.percentile(perm, 95)), 3)},
            "positional_checks": checks,
            "rule_supported_uncorrected": bool(supported),
            "rule_B_vs_majority": bool(vs_major_B) if name == "B" else None,
            "positional_bias_ruled_out": bool(all(pos_ok.values())), "positional_detail": pos_ok,
            "verdict": verdict,
            "_peak_pred": cv_predict(Xl[:, peak], y, 0, g), "_keep": keep,
        }
    out["targets"] = res
    return out


# ---------------------------------------------------------------- plots
def plots(m: dict, fig_dir: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    L = m["n_layers"]
    xs = np.arange(L)
    fig, ax = plt.subplots(figsize=(8, 4.2), constrained_layout=True)
    style = {"A": ("-", "#1f5fa8", "Target A: no-CoT answer"),
             "B": ("--", "#c0392b", "Target B: with-CoT answer")}
    for name, (ls, col, lab) in style.items():
        t = m["targets"][name]
        mean = np.array([t["by_layer"][i]["seed_mean"] for i in xs])
        lo = np.array([t["by_layer"][i]["ci95"][0] for i in xs])
        hi = np.array([t["by_layer"][i]["ci95"][1] for i in xs])
        ax.plot(xs, mean, ls, color=col, linewidth=1.8, label=lab)
        ax.fill_between(xs, lo, hi, color=col, alpha=0.15)
        ax.axhline(t["majority"], color=col, linewidth=0.8, alpha=0.7)
        ax.axhline(t["gaussian_baseline"]["mean"], color=col, linewidth=0.6, ls="-.", alpha=0.6)
    ax.axhline(m["chance"], color="k", ls=":", linewidth=1, label=f"chance (1/{m['n_options']})")
    ax.plot([], [], color="grey", linewidth=0.8, label="majority class (per target)")
    ax.plot([], [], color="grey", linewidth=0.6, ls="-.", label="random-Gaussian probe")
    ax.set_xlabel("layer")
    ax.set_ylabel("probe accuracy (5-fold CV; 5-seed mean, 95% CI shaded)")
    ax.set_title(title, fontsize=10)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7.5, loc="lower right")
    fig.savefig(fig_dir / "1_layer_sweep.png", dpi=300)
    plt.close(fig)

    # 2: agreement of probe-A vs probe-B predictions (confusion heatmap), items with both labels
    ta, tb = m["targets"]["A"], m["targets"]["B"]
    both = ta["_keep"] & tb["_keep"]
    pa = dict(zip(np.where(ta["_keep"])[0], ta["_peak_pred"], strict=True))
    pb = dict(zip(np.where(tb["_keep"])[0], tb["_peak_pred"], strict=True))
    idx = np.where(both)[0]
    letters = sorted(set(ta["label_dist"]) | set(tb["label_dist"]))
    M = np.zeros((len(letters), len(letters)), dtype=int)
    for i in idx:
        M[letters.index(pa[i]), letters.index(pb[i])] += 1
    agree = np.trace(M) / max(M.sum(), 1)
    fig, ax = plt.subplots(figsize=(4.6, 4.2), constrained_layout=True)
    ax.imshow(M, cmap="Blues")
    for r in range(len(letters)):
        for c in range(len(letters)):
            ax.text(c, r, M[r, c], ha="center", va="center", fontsize=8)
    ax.plot([-0.5, len(letters) - 0.5], [-0.5, len(letters) - 0.5], "r--", linewidth=0.8)
    ax.set_xticks(range(len(letters)), letters)
    ax.set_yticks(range(len(letters)), letters)
    ax.set_xlabel(f"probe-B prediction (L{tb['peak_layer']})")
    ax.set_ylabel(f"probe-A prediction (L{ta['peak_layer']})")
    ax.set_title(f"Per-item probe predictions, A vs B\nagreement {agree:.2f} (n={M.sum()})",
                 fontsize=9)
    fig.savefig(fig_dir / "2_targetA_vs_targetB.png", dpi=300)
    plt.close(fig)

    # 3: positional checks
    conds = ["last_token", "length_balanced", "position_only", "shifted_token",
             "mean_pooled_all", "mean_pooled_question"]
    fig, ax = plt.subplots(figsize=(8, 3.8), constrained_layout=True)
    w = 0.38
    for j, (name, col) in enumerate((("A", "#1f5fa8"), ("B", "#c0392b"))):
        ch = m["targets"][name]["positional_checks"]
        vals = [ch[c]["acc"] if ch[c].get("acc") is not None else 0 for c in conds]
        errs = [[(v - ch[c]["ci95"][0]) if ch[c].get("acc") is not None else 0 for c, v in
                 zip(conds, vals, strict=True)],
                [(ch[c]["ci95"][1] - v) if ch[c].get("acc") is not None else 0 for c, v in
                 zip(conds, vals, strict=True)]]
        ax.bar(np.arange(len(conds)) + (j - 0.5) * w, vals, w, yerr=errs, capsize=2, color=col,
               label=f"Target {name} (L{m['targets'][name]['peak_layer']}), "
                     f"majority {m['targets'][name]['majority']:.2f}")
    ax.axhline(m["chance"], color="k", ls=":", linewidth=1)
    ax.set_xticks(range(len(conds)), [c.replace("_", "\n") for c in conds], fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("accuracy (95% CI)")
    ax.set_title("Positional sanity checks at each target's peak layer", fontsize=10)
    ax.legend(fontsize=7.5)
    fig.savefig(fig_dir / "3_positional_checks.png", dpi=300)
    plt.close(fig)

    # 4: no-CoT letter distribution
    beh = m["behavior"]
    letters = sorted(set(beh["truth_dist"]) | {k for k in beh["no_cot_dist"] if k != "None"})
    fig, ax = plt.subplots(figsize=(5, 3.4), constrained_layout=True)
    x = np.arange(len(letters))
    ax.bar(x - 0.2, [beh["no_cot_dist"].get(k, 0) for k in letters], 0.4, label="no-CoT answer",
           color="#1f5fa8")
    ax.bar(x + 0.2, [beh["truth_dist"].get(k, 0) for k in letters], 0.4, label="ground truth",
           color="#bbbbbb")
    ax.axhline(0.6 * m["n"], color="#c0392b", ls=":", linewidth=1, label="60% bias limit")
    ax.set_xticks(x, letters)
    ax.set_ylabel(f"items (of {m['n']})")
    ax.set_title("No-CoT letter distribution", fontsize=10)
    ax.legend(fontsize=7.5)
    fig.savefig(fig_dir / "4_no_cot_letter_distribution.png", dpi=300)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--suite", required=True, choices=["bbh", "arith", "i", "i3"])
    ap.add_argument("--task", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-perm", type=int, default=100)
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--shifted", choices=["qend5", "option_e"], default="qend5")
    ap.add_argument("--grouped", action="store_true", help="StratifiedGroupKFold by item group")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pkl = out / "full_activations.pkl"
    t0 = time.time()
    if a.reuse and pkl.exists():
        blob = pickle.load(open(pkl, "rb"))
    else:
        blob = collect(a.model, a.suite, a.task, a.n, a.shifted)
        blob["collect_seconds"] = round(time.time() - t0)
        pickle.dump(blob, open(pkl, "wb"))
    print(f"collected in {blob.get('collect_seconds')}s", flush=True)
    t1 = time.time()
    m = analyze(blob, a.n_perm, a.grouped)
    m["cv"] = "StratifiedGroupKFold by fact" if a.grouped else "StratifiedKFold"
    m["shifted_mode"] = a.shifted
    m["analysis_seconds"] = round(time.time() - t1)
    m["collect_seconds"] = blob.get("collect_seconds")
    m["model"], m["suite"], m["task"], m["device"] = a.model, a.suite, a.task, blob.get("device")
    plots(m, out / "figures", f"{a.model} · {a.suite}:{a.task} · last-prompt-token probe")
    for t in m["targets"].values():
        t.pop("_peak_pred")
        t.pop("_keep")
    (out / "metrics.json").write_text(json.dumps(m, indent=2, default=str) + "\n")
    with open(out / "generations.jsonl", "w") as f:
        for r in blob["meta"]:
            f.write(json.dumps(r) + "\n")
    summ = {k: {kk: v[kk] for kk in ("peak_layer", "peak", "majority", "gaussian_baseline",
                                       "max_over_layers_perm", "positional_bias_ruled_out",
                                       "positional_detail", "verdict")}
            for k, v in m["targets"].items()}
    print(json.dumps({"behavior": m["behavior"], "targets": summ}, indent=2, default=str))


if __name__ == "__main__":
    main()
