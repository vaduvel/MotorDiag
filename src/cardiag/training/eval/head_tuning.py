"""Head-tuning headroom probe: how much honest accuracy is left in the EXISTING
heads from decision-threshold selection and regularisation (C) tuning alone --
no new data, no re-embedding, no re-training of CLAP?

Why this exists
---------------
The heads already use ``class_weight=\"balanced\"`` (see ``scorecard.lr_head``),
but two cheap, standard levers are still on the table for the imbalanced heads
(knock, triage, cause) that Audit #1 flagged as scoring near/under their majority
baseline:

  1. **Decision threshold**: predicting the positive class at p>=0.5 is rarely
     the balanced-accuracy optimum on an imbalanced head. Selecting the
     threshold that maximises balanced accuracy is free.
  2. **Regularisation C**: the default ``C=1.0`` is a guess. A nested,
     by-video-grouped grid search picks C honestly.

Both are selected on the TRAIN fold only and scored on the held-out fold, so the
reported gains are leakage-free. Reuses the cached CLAP embeddings
(``data/training/corpus_emb.npz``) and the tested metric code in ``scorecard.py``.

Run:  python -m cardiag.training.eval.head_tuning
Writes docs/HEAD_TUNING.md + docs/head_tuning.json.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cardiag.training.eval.scorecard import (
    CHASSIS,
    ENGINE,
    L1_NORMAL,
    agg,
    embed_corpus,
)

C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]


def _make(C: float = 1.0):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced",
                           random_state=0, C=C),
    )


def _ns(y, groups, cap=5):
    y = np.asarray(y); groups = np.asarray(groups)
    return max(2, min(cap, min(len(set(groups[y == c])) for c in set(y))))


def heads(d: dict):
    """(name, X, y, groups, is_binary) for every head we tune."""
    X, VID = d["X"].astype(float), d["video"]
    KIND, L1, CAUSE = d["kind"], d["l1"], d["cause"]
    knock = np.array(["knock" if "knock" in x else ("normal_idle" if x == L1_NORMAL else "")
                      for x in L1])
    tri = np.array(["engine" if c in ENGINE else "chassis" if c in CHASSIS else ""
                    for c in CAUSE])
    out = []
    km = (KIND == "fault") | (KIND == "normal")
    out.append(("kind (fault vs normal, all sources)", X[km], KIND[km], VID[km], True))
    kk = knock != ""
    out.append(("knock (knock vs normal-idle)", X[kk], knock[kk], VID[kk], True))
    tm = (KIND == "fault") & (tri != "")
    out.append(("triage (engine vs running-gear)", X[tm], tri[tm], VID[tm], True))
    cm = CAUSE != ""
    out.append(("cause (part family)", X[cm], CAUSE[cm], VID[cm], False))
    return out


def _best_threshold(clf, Xtr, ytr, pos, neg):
    """Threshold on the positive-class probability that maximises balanced
    accuracy on the TRAIN fold."""
    p = clf.predict_proba(Xtr)[:, list(clf.classes_).index(pos)]
    best_t, best_s = 0.5, -1.0
    for t in np.linspace(0.1, 0.9, 33):
        pred = np.where(p >= t, pos, neg)
        s = balanced_accuracy_score(ytr, pred)
        if s > best_s:
            best_s, best_t = s, float(t)
    return best_t


def _best_C(Xtr, ytr, gtr):
    """Nested by-video-grouped grid search for C on the TRAIN fold. Falls back to
    C=1.0 if the inner split is infeasible (e.g. a class with too few groups)."""
    try:
        inner_ns = _ns(ytr, gtr, cap=4)
        inner = StratifiedGroupKFold(inner_ns, shuffle=True, random_state=0)
        gs = GridSearchCV(_make(1.0), {"logisticregression__C": C_GRID},
                          scoring="balanced_accuracy", cv=inner, n_jobs=None)
        gs.fit(Xtr, ytr, groups=gtr)
        return float(gs.best_params_["logisticregression__C"])
    except Exception:
        return 1.0


def evaluate(X, y, groups, is_binary, n_splits=5, repeats=5) -> dict:
    """Repeated by-video grouped CV. Per fold, all selection (threshold, C) is
    done on TRAIN only, then scored on the held-out fold."""
    X = np.asarray(X); y = np.asarray(y); groups = np.asarray(groups)
    if len(set(y)) < 2:
        return {"skipped": "single-class head"}
    classes = sorted(set(y))
    pos = classes[-1] if is_binary else None
    neg = classes[0] if is_binary else None
    ns = _ns(y, groups, cap=n_splits)
    default_ba, thr_ba, csel_ba = [], [], []
    default_mcc, csel_mcc = [], []
    picked_C, picked_T = [], []
    for rep in range(repeats):
        sgkf = StratifiedGroupKFold(ns, shuffle=True, random_state=rep)
        for tr, te in sgkf.split(X, y, groups):
            if len(set(y[tr])) < 2:
                continue
            assert not (set(groups[tr]) & set(groups[te])), "video leak across split!"
            Xtr, ytr, gtr = X[tr], y[tr], groups[tr]
            Xte, yte = X[te], y[te]
            # default C=1.0, threshold 0.5
            clf = _make(1.0).fit(Xtr, ytr)
            pred = clf.predict(Xte)
            default_ba.append(balanced_accuracy_score(yte, pred))
            default_mcc.append(matthews_corrcoef(yte, pred))
            # threshold-tuned (binary only)
            if is_binary:
                t = _best_threshold(clf, Xtr, ytr, pos, neg)
                p_te = clf.predict_proba(Xte)[:, list(clf.classes_).index(pos)]
                pred_t = np.where(p_te >= t, pos, neg)
                thr_ba.append(balanced_accuracy_score(yte, pred_t))
                picked_T.append(t)
            # C-tuned (default threshold)
            C = _best_C(Xtr, ytr, gtr)
            picked_C.append(C)
            clfc = _make(C).fit(Xtr, ytr)
            predc = clfc.predict(Xte)
            csel_ba.append(balanced_accuracy_score(yte, predc))
            csel_mcc.append(matthews_corrcoef(yte, predc))
    maj = max(Counter(y).values()) / len(y)

    def ms(v):
        return (float(np.mean(v)), float(np.std(v))) if v else (float("nan"), 0.0)

    out = {
        "n": int(len(y)),
        "classes": {str(k): int(v) for k, v in sorted(Counter(y).items())},
        "majority": round(float(maj), 3),
        "default": {"bal_acc": ms(default_ba), "mcc": ms(default_mcc)},
        "C_tuned": {"bal_acc": ms(csel_ba), "mcc": ms(csel_mcc),
                    "C_median": float(np.median(picked_C)) if picked_C else None},
    }
    if is_binary:
        out["threshold_tuned"] = {"bal_acc": ms(thr_ba),
                                  "threshold_median": float(np.median(picked_T)) if picked_T else None}
    return out


def _fmt(pair) -> str:
    if not isinstance(pair, (list, tuple)):
        if pair is None or (isinstance(pair, float) and math.isnan(pair)):
            return "\u2014"
        return f"{pair:.3f}"
    m, s = pair
    return "\u2014" if m is None or (isinstance(m, float) and math.isnan(m)) else f"{m:.3f}\u00b1{s:.2f}"


def run(out_md: Path | None = None) -> dict:
    d = embed_corpus()
    report: dict = {"heads": {}}
    for name, X, y, g, is_bin in heads(d):
        report["heads"][name] = evaluate(X, y, g, is_bin)

    L = [
        "# Head-tuning headroom -- threshold + C, no new data",
        "",
        "Generated by `python -m cardiag.training.eval.head_tuning`. By-video "
        "grouped 5x5 CV, class-weighted; every selection (decision threshold, "
        "regularisation C) is done on the TRAIN fold only and scored on the "
        "held-out fold, so the gains are leakage-free. Reuses the cached CLAP "
        "embeddings in `data/training/corpus_emb.npz`.",
        "",
        "| Head | n | majority | default balAcc | threshold-tuned balAcc | C-tuned balAcc | default MCC | C-tuned MCC |",
        "|---|--:|--:|---|---|---|---|---|",
    ]
    for name, h in report["heads"].items():
        if "skipped" in h:
            L.append(f"| {name} | \u2014 | \u2014 | _{h['skipped']}_ | \u2014 | \u2014 | \u2014 | \u2014 |")
            continue
        thr = h.get("threshold_tuned", {}).get("bal_acc")
        L.append(
            f"| {name} | {h['n']} | {h['majority']:.3f} | "
            f"{_fmt(h['default']['bal_acc'])} | {_fmt(thr)} | "
            f"{_fmt(h['C_tuned']['bal_acc'])} | {_fmt(h['default']['mcc'])} | "
            f"{_fmt(h['C_tuned']['mcc'])} |"
        )

    L += ["", "## Selected hyper-parameters (median across folds)", "",
          "| Head | median threshold | median C |", "|---|--:|--:|"]
    _mdash = '\u2014'
    for name, h in report["heads"].items():
        if "skipped" in h:
            continue
        t = h.get("threshold_tuned", {}).get("threshold_median")
        c = h.get("C_tuned", {}).get("C_median")
        L.append(f"| {name} | {('%.3f' % t) if t is not None else _mdash} | "
                 f"{('%.3g' % c) if c is not None else _mdash} |")

    L += [
        "",
        "## How to read this",
        "",
        "* A head where **threshold-tuned or C-tuned balAcc clears the default "
        "(and the majority baseline)** has free, honest headroom -- adopt that "
        "setting in the shipped head.",
        "* A head that **stays at/under majority even after tuning** (Audit #1 "
        "flagged triage and, on tiny n, knock) is data-limited, not "
        "hyper-parameter-limited -- tuning will not save it; more clips per class "
        "will.",
        "* Medians of the selected threshold/C tell you what to hard-code once the "
        "gain is confirmed.",
    ]

    repo_root = Path(__file__).resolve().parents[4]
    out_md = out_md or (repo_root / "docs" / "HEAD_TUNING.md")
    Path(out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(out_md).write_text("\n".join(L) + "\n")
    (Path(out_md).parent / "head_tuning.json").write_text(json.dumps(report, indent=2))
    print("\n".join(L))
    print(f"\nwrote {out_md} + head_tuning.json")
    return report


if __name__ == "__main__":
    run()
