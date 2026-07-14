"""External-corpus scorecard: honest, confound-controlled fault/normal metrics
for the DB1 / Car-Engine-Sounds / Ai_Mechanic datasets ingested into
``data/external/``.

Why this exists
---------------
``scorecard.py`` is the project's rigorous evaluation harness (by-video grouped
CV, AUROC/AUPRC/MCC/ECE, permutation null). But it was written for the scraped
YouTube/TikTok/Reddit corpus:

  * ``embed_corpus()`` labels every clip whose wav path is not under
    ``/youtube/``, ``/tiktok/`` or ``/reddit/`` as source ``"?"`` — so every
    externally-ingested clip is ``"?"``.
  * its honest "confound-free" cut is hard-coded to ``SRC == "youtube"``.

On an external-only clone (no scraped data) that cut is EMPTY, so the shipped
scorecard reports no honest fault/normal number at all — which is why NOTES.md
falls back to ``build._cv_report`` balanced accuracy (all sources mixed). That
number is *confound-inflated*: in the external corpus, "normal" vs "fault"
correlates strongly with which dataset/folder a clip came from, so a model can
score well by recognising a dataset's acoustics instead of the fault.

This driver reuses scorecard's rigorous, tested metric code but re-derives the
recording source from the clip_id prefix (``db1`` / ``car_engine`` /
``ai_mechanic``) and reports the fault/normal head three ways:

  1. all-sources           — confound-inflated (comparable to NOTES.md balAcc)
  2. within-source         — source held constant (per source that has both
                             classes): the honest in-domain number
  3. leave-one-source-out  — train on N-1 sources, test on the held-out one:
                             the honest cross-domain generalisation number

It also re-runs the Stage-0 source-confound probe (how predictable is the
source itself?) on the corrected source labels.

Run:  python -m cardiag.training.eval.external_scorecard
Writes docs/SCORECARD_EXTERNAL.md + docs/scorecard_external.json.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from cardiag import paths  # noqa: F401  (kept for parity / future output paths)
from cardiag.training.eval.scorecard import (
    agg,
    ece,
    embed_corpus,
    grouped_cv,
    lr_head,
)

_EXTERNAL_SOURCES = ("db1", "car_engine", "ai_mechanic")


def source_of(clip_id: str, wav_src: str) -> str:
    """Recording source, external-aware. Scraped clips keep their platform;
    external clips are split into their sub-dataset by clip_id prefix (set in
    scripts/ingest_external_datasets.py as ``f\"{label}_{stem}\"``)."""
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
    X = d["X"][m]
    y = kind[m]
    vid = d["video"][m]
    src = np.array([source_of(str(c), str(s))
                    for c, s in zip(d["ids"][m], d["src"][m])])
    return X, y, vid, src


def _cv_block(X, y, groups) -> dict:
    """Grouped-CV AUROC/balAcc/ECE for one binary cut."""
    folds, (oof, oofy, pos) = grouped_cv(X, y, groups)
    a = agg(folds)
    ec = ece(oof, (oofy == pos)) if pos is not None else float("nan")
    return {
        "n": int(len(y)),
        "bal_acc": a["bal_acc"],
        "auroc": a["auroc"],
        "auprc": a["auprc"],
        "mcc": a["mcc"],
        "ece": ec,
        "majority": round(float(max(Counter(y).values()) / len(y)), 3),
        "classes": {str(k): int(v) for k, v in sorted(Counter(y).items())},
    }


def loso_kind(X, y, vid, src, make=lr_head) -> dict:
    """Leave-one-source-out for fault/normal: train on every source but one,
    test on the held-out source (touched once). The honest cross-domain number.
    A source can only be a meaningful test fold if both it and the training side
    carry both classes."""
    y = np.asarray(y)
    src = np.asarray(src)
    pos = sorted(set(y))[-1]
    rows: dict = {}
    for held in sorted(set(src)):
        te = src == held
        tr = ~te
        if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
            rows[held] = {"n_test": int(te.sum()),
                          "skipped": "held-out or train side is single-class"}
            continue
        clf = make().fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        proba = clf.predict_proba(X[te])[:, list(clf.classes_).index(pos)]
        yp = (y[te] == pos).astype(int)
        rows[held] = {
            "n_test": int(te.sum()),
            "bal_acc": round(float(balanced_accuracy_score(y[te], pred)), 3),
            "auroc": round(float(roc_auc_score(yp, proba)), 3),
            "majority": round(float(max(Counter(y[te]).values()) / te.sum()), 3),
        }
    return rows


def _fmt(pair) -> str:
    """Format a scalar or an (mean, std) tuple; nan/None -> em dash."""
    if not isinstance(pair, (list, tuple)):
        if pair is None or (isinstance(pair, float) and math.isnan(pair)):
            return "—"
        return f"{pair:.3f}"
    m, s = pair
    return "—" if m is None or math.isnan(m) else f"{m:.3f}±{s:.2f}"


def run(out_md: Path | None = None) -> dict:
    d = embed_corpus()
    X, y, vid, src = _kind_arrays(d)
    if len(y) == 0:
        raise SystemExit("no fault/normal clips in the corpus — nothing to score.")

    report: dict = {
        "corpus": {
            "n_clips": int(len(d["X"])),
            "n_kind_clips": int(len(y)),
            "n_videos": int(len(set(vid))),
            "by_source": {s: int(v) for s, v in sorted(Counter(src).items())},
            "by_kind": {str(k): int(v) for k, v in sorted(Counter(y).items())},
        }
    }

    report["kind_all_sources"] = _cv_block(X, y, vid)
    within: dict = {}
    for s in sorted(set(src)):
        ms = src == s
        if len(set(y[ms])) < 2:
            continue
        within[s] = _cv_block(X[ms], y[ms], vid[ms])
    report["kind_within_source"] = within
    report["kind_loso"] = loso_kind(X, y, vid, src)

    if len(set(src)) > 1:
        folds, _ = grouped_cv(X, src, vid)
        a = agg(folds, keys=("bal_acc", "macro_f1", "mcc"))
        report["source_confound"] = {
            "predict_source_bal_acc": a["bal_acc"],
            "mcc": a["mcc"],
            "random": round(1 / len(set(src)), 3),
        }

    c = report["corpus"]
    L = [
        "# External scorecard — fault/normal, confound-controlled",
        "",
        "Generated by `python -m cardiag.training.eval.external_scorecard`. "
        "By-video grouped 5×5 CV (no clip leakage), class-weighted; reuses the "
        "metric code in `scorecard.py`. Reads the CLAP embeddings cached at "
        "`data/training/corpus_emb.npz`.",
        "",
        f"**Corpus:** {c['n_kind_clips']} fault/normal clips, {c['n_videos']} "
        f"source recordings, sources={c['by_source']}, kind={c['by_kind']}",
        "",
        "## fault/normal three ways",
        "",
        "| Cut | n | balAcc | AUROC | ECE | majority |",
        "|---|--:|---|---|--:|--:|",
    ]
    a = report["kind_all_sources"]
    L.append(
        f"| all sources (confound-inflated) | {a['n']} | {_fmt(a['bal_acc'])} | "
        f"{_fmt(a['auroc'])} | {_fmt(a['ece'])} | {a['majority']:.3f} |"
    )
    for s, b in within.items():
        L.append(
            f"| within {s} (source held constant) | {b['n']} | {_fmt(b['bal_acc'])} | "
            f"{_fmt(b['auroc'])} | {_fmt(b['ece'])} | {b['majority']:.3f} |"
        )

    L += [
        "",
        "## Leave-one-source-out (cross-domain generalisation)",
        "",
        "Train on every source but one, test on the held-out source. This is the "
        "number that says whether the model learned *faults* or just each "
        "dataset's recording signature.",
        "",
        "| Held-out source | n_test | balAcc | AUROC | majority |",
        "|---|--:|--:|--:|--:|",
    ]
    for s, r in report["kind_loso"].items():
        if "skipped" in r:
            L.append(f"| {s} | {r['n_test']} | — | — | _{r['skipped']}_ |")
        else:
            L.append(
                f"| {s} | {r['n_test']} | {r['bal_acc']:.3f} | {r['auroc']:.3f} | "
                f"{r['majority']:.3f} |"
            )

    if "source_confound" in report:
        sc = report["source_confound"]
        L += [
            "",
            "## Source confound (Stage 0)",
            "",
            f"Predicting the recording **source** from the embedding alone scores "
            f"balAcc {_fmt(sc['predict_source_bal_acc'])} (MCC {_fmt(sc['mcc'])}, "
            f"random {sc['random']:.3f}). The higher this is, the more the "
            f"all-sources fault/normal number above is a source shortcut rather "
            f"than a fault signal — trust the within-source and leave-one-source-out "
            f"rows instead.",
        ]

    repo_root = Path(__file__).resolve().parents[4]
    out_md = out_md or (repo_root / "docs" / "SCORECARD_EXTERNAL.md")
    Path(out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(out_md).write_text("\n".join(L) + "\n")
    (Path(out_md).parent / "scorecard_external.json").write_text(
        json.dumps(report, indent=2)
    )
    print("\n".join(L))
    print(f"\nwrote {out_md} + scorecard_external.json")
    return report


if __name__ == "__main__":
    run()
