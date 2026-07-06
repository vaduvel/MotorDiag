"""Confound-removal probe: can we recover cross-domain (leave-one-source-out)
fault/normal generalisation on the EXISTING corpus by neutralising the
recording-source shortcut -- without adding any new data?

Audit finding this targets
--------------------------
``external_scorecard.py`` showed the fault/normal head is REAL within a source
(permutation p~0.005) but collapses toward chance leave-one-source-out (LOSO),
while the recording *source* itself is linearly predictable at balAcc~0.67
(chance 0.33). That gap is the classic domain shortcut: the head partly encodes
"which dataset is this" instead of "is this a fault".

This probe measures whether two label-free domain-adaptation transforms on the
frozen CLAP embeddings close that gap, using ONLY the data already cached in
``data/training/corpus_emb.npz`` (no re-embedding, no new datasets):

  A. per-source standardisation (transductive CORAL-lite): subtract each
     source's own feature mean (and scale by its std). Uses only *unlabeled*
     per-domain statistics -- never the fault/normal labels -- so it is a
     legitimate, leakage-safe domain-adaptation step, not target leakage.
  B. source-nullspace projection (leakage-safe): fit a linear source classifier
     on the TRAINING sources only, then project embeddings onto the orthogonal
     complement of its weight row-space, deleting the (<= n_sources-1)
     directions that linearly encode source. Fit on train, applied to the
     held-out fold -- no leak.

For each transform it reports, side by side with the untouched baseline:
  * LOSO fault/normal            -- the number we want to move up
  * all-sources grouped-CV       -- must NOT collapse (in-domain sanity)
  * source predictability        -- must drop toward chance (shortcut gone)

Every transform is fit on the TRAIN fold only and applied to the held-out fold,
so no comparison is inflated by leakage.

Run:  python -m cardiag.training.eval.confound_probe
Writes docs/CONFOUND.md + docs/confound_probe.json.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cardiag.training.eval.scorecard import agg, ece, embed_corpus, lr_head

_EXTERNAL_SOURCES = ("db1", "car_engine", "ai_mechanic")


def source_of(clip_id: str, wav_src: str) -> str:
    """Recording source, external-aware (mirrors scorecard.source_of): scraped
    clips keep their platform; external clips are split into their sub-dataset by
    the clip_id prefix set in scripts/ingest_external_datasets.py (f\"{label}_{stem}\")."""
    if wav_src in ("youtube", "tiktok", "reddit"):
        return wav_src
    for s in _EXTERNAL_SOURCES:
        if clip_id.startswith(s + "_"):
            return s
    return "external"


def _kind_arrays(d: dict):
    """Aligned (X, y=fault/normal, video groups, source) for the kind head."""
    kind = d["kind"]
    m = (kind == "fault") | (kind == "normal")
    X = d["X"][m].astype(float)
    y = kind[m]
    vid = d["video"][m]
    src = np.array([source_of(str(c), str(s))
                    for c, s in zip(d["ids"][m], d["src"][m])])
    return X, y, vid, src


# ------------------------------------------------------------- transforms
# Each transform has signature (X_train, src_train, X_test, src_test) -> (Xtr', Xte')
# and is ALWAYS fit on the train fold only.

def tf_baseline(Xtr, srctr, Xte, srcte):
    return Xtr, Xte


def tf_per_source_standardize(Xtr, srctr, Xte, srcte, scale: bool = True):
    """A: subtract (and scale by) each source's own feature statistics. Train
    sources use train-fold stats; a source that appears only in the held-out
    fold (the LOSO case) uses its own *unlabeled* stats -- legitimate
    transductive domain adaptation, no label leakage."""
    Xtr2, Xte2 = Xtr.copy(), Xte.copy()
    stats: dict = {}
    for s in set(srctr):
        mth = srctr == s
        mu = Xtr[mth].mean(0)
        sd = Xtr[mth].std(0)
        sd[sd == 0] = 1.0
        stats[s] = (mu, sd)
        Xtr2[mth] = (Xtr[mth] - mu) / sd if scale else Xtr[mth] - mu
    for s in set(srcte):
        mth = srcte == s
        if s in stats:
            mu, sd = stats[s]
        else:                                  # unseen-in-train (LOSO held-out)
            mu = Xte[mth].mean(0)
            sd = Xte[mth].std(0)
            sd[sd == 0] = 1.0
        Xte2[mth] = (Xte[mth] - mu) / sd if scale else Xte[mth] - mu
    return Xtr2, Xte2


def _source_nullspace(Xtr, srctr):
    """Projector (dim x dim) onto the orthogonal complement of a linear source
    classifier's weight row-space, fit on TRAIN only. Removes up to
    n_sources-1 directions that linearly encode the recording source."""
    dim = Xtr.shape[1]
    srcs = sorted(set(srctr))
    if len(srcs) < 2:
        return np.eye(dim)
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced", random_state=0),
    ).fit(Xtr, srctr)
    lr = clf.named_steps["logisticregression"]
    scaler = clf.named_steps["standardscaler"]
    W = lr.coef_ / scaler.scale_               # (n_dirs, dim) in raw-feature space
    Q, _ = np.linalg.qr(W.T)                    # orthonormal basis of the row space
    return np.eye(dim) - Q @ Q.T                # project onto the complement


def tf_source_nullspace(Xtr, srctr, Xte, srcte):
    """B: delete the linear source directions (fit on train) from both folds."""
    P = _source_nullspace(Xtr, srctr)
    return Xtr @ P, Xte @ P


TRANSFORMS = {
    "baseline (raw embeddings)": tf_baseline,
    "A: per-source standardise": tf_per_source_standardize,
    "B: source-nullspace projection": tf_source_nullspace,
}


# --------------------------------------------------------------- evaluators
def _ns(y, groups, n_splits):
    return max(2, min(n_splits, min(len(set(np.asarray(groups)[np.asarray(y) == c]))
                                    for c in set(y))))


def all_sources_cv(X, y, vid, src, transform, n_splits=5, repeats=5) -> dict:
    """Repeated by-video grouped CV for fault/normal with a per-fold transform
    fit on train only. In-domain sanity: this must not collapse."""
    y = np.asarray(y); vid = np.asarray(vid); src = np.asarray(src)
    pos = sorted(set(y))[-1]
    ns = _ns(y, vid, n_splits)
    folds = []
    oof = np.full(len(y), np.nan)
    for rep in range(repeats):
        sgkf = StratifiedGroupKFold(ns, shuffle=True, random_state=rep)
        for tr, te in sgkf.split(X, y, vid):
            if len(set(y[tr])) < 2:
                continue
            assert not (set(vid[tr]) & set(vid[te])), "video leak across split!"
            Xtr, Xte = transform(X[tr], src[tr], X[te], src[te])
            clf = lr_head().fit(Xtr, y[tr])
            pr = clf.predict_proba(Xte)[:, list(clf.classes_).index(pos)]
            pred = clf.predict(Xte)
            oof[te] = pr
            yp = (y[te] == pos).astype(int)
            folds.append({"bal_acc": balanced_accuracy_score(y[te], pred),
                          "auroc": roc_auc_score(yp, pr) if len(set(yp)) == 2
                          else float("nan")})
    a = agg(folds, keys=("bal_acc", "auroc"))
    return {"bal_acc": a["bal_acc"], "auroc": a["auroc"],
            "ece": ece(oof, (y == pos))}


def loso(X, y, vid, src, transform) -> dict:
    """Leave-one-source-out fault/normal with the transform fit on the train
    sources only. The cross-domain number we want to move up."""
    y = np.asarray(y); src = np.asarray(src)
    pos = sorted(set(y))[-1]
    rows: dict = {}
    for held in sorted(set(src)):
        te = src == held
        tr = ~te
        if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
            rows[held] = {"n_test": int(te.sum()),
                          "skipped": "held-out or train side is single-class"}
            continue
        Xtr, Xte = transform(X[tr], src[tr], X[te], src[te])
        clf = lr_head().fit(Xtr, y[tr])
        pred = clf.predict(Xte)
        proba = clf.predict_proba(Xte)[:, list(clf.classes_).index(pos)]
        yp = (y[te] == pos).astype(int)
        rows[held] = {
            "n_test": int(te.sum()),
            "bal_acc": round(float(balanced_accuracy_score(y[te], pred)), 3),
            "auroc": round(float(roc_auc_score(yp, proba)), 3),
            "majority": round(float(max(Counter(y[te]).values()) / te.sum()), 3),
        }
    vals = [r["bal_acc"] for r in rows.values() if "bal_acc" in r]
    aur = [r["auroc"] for r in rows.values() if "auroc" in r]
    rows["_mean"] = {"bal_acc": round(float(np.mean(vals)), 3) if vals else None,
                     "auroc": round(float(np.mean(aur)), 3) if aur else None}
    return rows


def source_predictability(X, vid, src, transform, n_splits=5, repeats=3):
    """Grouped-CV balAcc of predicting the recording SOURCE, with the transform
    fit on train only. Should drop toward chance once the shortcut is removed."""
    src = np.asarray(src); vid = np.asarray(vid)
    if len(set(src)) < 2:
        return (float("nan"), 0.0), float("nan")
    ns = _ns(src, vid, n_splits)
    accs = []
    for rep in range(repeats):
        for tr, te in StratifiedGroupKFold(ns, shuffle=True, random_state=rep).split(X, src, vid):
            if len(set(src[tr])) < 2:
                continue
            Xtr, Xte = transform(X[tr], src[tr], X[te], src[te])
            clf = lr_head().fit(Xtr, src[tr])
            accs.append(balanced_accuracy_score(src[te], clf.predict(Xte)))
    return (float(np.mean(accs)), float(np.std(accs))), round(1 / len(set(src)), 3)


# ------------------------------------------------------------------- format
def _fmt(pair) -> str:
    if not isinstance(pair, (list, tuple)):
        if pair is None or (isinstance(pair, float) and math.isnan(pair)):
            return "\u2014"
        return f"{pair:.3f}"
    m, s = pair
    return "\u2014" if m is None or (isinstance(m, float) and math.isnan(m)) else f"{m:.3f}\u00b1{s:.2f}"


def run(out_md: Path | None = None) -> dict:
    d = embed_corpus()
    X, y, vid, src = _kind_arrays(d)
    if len(y) == 0:
        raise SystemExit("no fault/normal clips in the corpus -- nothing to score.")
    if len(set(src)) < 2:
        raise SystemExit("only one recording source -- confound probe needs >=2.")

    report: dict = {
        "corpus": {
            "n_kind_clips": int(len(y)),
            "n_videos": int(len(set(vid))),
            "by_source": {s: int(v) for s, v in sorted(Counter(src).items())},
            "by_kind": {str(k): int(v) for k, v in sorted(Counter(y).items())},
        },
        "transforms": {},
    }
    for name, tf in TRANSFORMS.items():
        report["transforms"][name] = {
            "all_sources_cv": all_sources_cv(X, y, vid, src, tf),
            "loso": loso(X, y, vid, src, tf),
            "source_predictability": {
                "bal_acc": source_predictability(X, vid, src, tf)[0],
                "chance": round(1 / len(set(src)), 3),
            },
        }

    c = report["corpus"]
    L = [
        "# Confound-removal probe -- can existing data alone recover LOSO?",
        "",
        "Generated by `python -m cardiag.training.eval.confound_probe`. Reuses the "
        "cached CLAP embeddings (`data/training/corpus_emb.npz`) and the metric "
        "code in `scorecard.py`. Every transform is fit on the TRAIN fold only and "
        "applied to the held-out fold -- no leakage. No new data is used.",
        "",
        f"**Corpus:** {c['n_kind_clips']} fault/normal clips, {c['n_videos']} "
        f"recordings, sources={c['by_source']}, kind={c['by_kind']}",
        "",
        "## fault/normal, by transform",
        "",
        "| Transform | all-sources balAcc | all-sources AUROC | LOSO balAcc (mean) | "
        "LOSO AUROC (mean) | source predictability |",
        "|---|---|---|---|---|---|",
    ]
    for name, r in report["transforms"].items():
        a = r["all_sources_cv"]
        lm = r["loso"]["_mean"]
        sp = r["source_predictability"]
        L.append(
            f"| {name} | {_fmt(a['bal_acc'])} | {_fmt(a['auroc'])} | "
            f"{_fmt(lm['bal_acc'])} | {_fmt(lm['auroc'])} | "
            f"{_fmt(sp['bal_acc'])} (chance {sp['chance']:.3f}) |"
        )

    L += [
        "",
        "## Per-held-out-source LOSO detail",
        "",
        "| Transform | held-out source | n_test | balAcc | AUROC | majority |",
        "|---|---|--:|--:|--:|--:|",
    ]
    for name, r in report["transforms"].items():
        for s, row in r["loso"].items():
            if s == "_mean":
                continue
            if "skipped" in row:
                L.append(f"| {name} | {s} | {row['n_test']} | \u2014 | \u2014 | _{row['skipped']}_ |")
            else:
                L.append(
                    f"| {name} | {s} | {row['n_test']} | {row['bal_acc']:.3f} | "
                    f"{row['auroc']:.3f} | {row['majority']:.3f} |"
                )

    L += [
        "",
        "## How to read this",
        "",
        "* **LOSO up + source predictability down** vs baseline => the transform "
        "removed a real recording-source shortcut and bought genuine cross-domain "
        "generalisation from data we already have.",
        "* **LOSO flat while source predictability drops** => the source signal and "
        "the fault signal are entangled in the same directions; you cannot separate "
        "them linearly with 3 sources, and a 4th independent domain is required.",
        "* **all-sources balAcc must not collapse** -- if it does, the transform "
        "destroyed the in-domain signal and is not worth keeping.",
    ]

    repo_root = Path(__file__).resolve().parents[4]
    out_md = out_md or (repo_root / "docs" / "CONFOUND.md")
    Path(out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(out_md).write_text("\n".join(L) + "\n")
    (Path(out_md).parent / "confound_probe.json").write_text(json.dumps(report, indent=2))
    print("\n".join(L))
    print(f"\nwrote {out_md} + confound_probe.json")
    return report


if __name__ == "__main__":
    run()
