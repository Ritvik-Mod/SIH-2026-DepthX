#!/usr/bin/env python3
"""
evaluate_batch.py — baseline vs refined on GAMUS prediction/ground-truth pairs.

This is the script that produces the numbers that matter.  Everything in
`validate_synthetic.py` is development evidence; this is the measurement.

Expected layout — the two bundles the pipeline already produces, from
`export_batch_threejs.py --source pred` and `--source gt`:

    pred_export/<scene>/heightmap.tif  texture.png  metadata.json
    gt_export/<scene>/heightmap.tif    metadata.json

Scenes are matched by directory name.  `metadata.json` in the GT bundle should
carry `field_source: ground_truth_lidar`; if the prediction bundle claims that
too, the scene is skipped, because that means the two bundles were exported
from the same source and any "improvement" measured would be meaningless.

Split discipline
----------------
    --split dev    discover relationships, tune, refit the classifier
    --split val    check that the tuning generalises
    --split test   run ONCE, with a frozen config, for the reported result

The script refuses to write a tuning artefact while pointed at the test split
(`--fit-classifier` + `--split test` is an error), because that is the specific
mistake that would invalidate the whole study.

Usage
-----
    # full ablation ladder on the development split
    python evaluate_batch.py --pred-root outputs/pred_export \\
        --gt-root outputs/gt_export --split dev \\
        --ablations A0 A1 A2 A3 A4 A5 FINAL --out results_dev.json

    # final, frozen, once
    python evaluate_batch.py --pred-root outputs/pred_export \\
        --gt-root outputs/gt_export --split test \\
        --config frozen_config.json --ablations A0 FINAL \\
        --out results_test.json --figures figs/

    # learn the region classifier weights from GT (dev only)
    python evaluate_batch.py ... --split dev --fit-classifier cls_weights.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dw_tda import metrics as M
from dw_tda import rasterio_io as io
from dw_tda import tda_core as T
from dw_tda.pipeline import refine, config_for_ablation


def find_scenes(pred_root: str, gt_root: str) -> List[str]:
    out = []
    for name in sorted(os.listdir(pred_root)):
        p = os.path.join(pred_root, name, "heightmap.tif")
        g = os.path.join(gt_root, name, "heightmap.tif")
        if os.path.isfile(p) and os.path.isfile(g):
            out.append(name)
    return out


def load_scene(pred_root, gt_root, name):
    pd = os.path.join(pred_root, name)
    gd = os.path.join(gt_root, name)
    pred, prof = io.read_heightmap(os.path.join(pd, "heightmap.tif"))
    gt, _ = io.read_heightmap(os.path.join(gd, "heightmap.tif"))
    pmeta = io.read_metadata(os.path.join(pd, "metadata.json")) if os.path.isfile(
        os.path.join(pd, "metadata.json")) else {}
    gmeta = io.read_metadata(os.path.join(gd, "metadata.json")) if os.path.isfile(
        os.path.join(gd, "metadata.json")) else {}
    tex = os.path.join(pd, "texture.png")
    rgb = io.read_texture(tex, pred.shape) if os.path.isfile(tex) else None
    return pred, gt, rgb, prof, pmeta, gmeta


def scene_split(pmeta: dict) -> str:
    return str(pmeta.get("split", "unknown")).lower()


# ---------------------------------------------------------------------------
# region-level classifier refit (the "discover from data" mechanism)
# ---------------------------------------------------------------------------


def fit_classifier(scenes, pred_root, gt_root, cfg, gsd_default=0.33) -> Dict:
    """Learn p(building-like) from the ground truth instead of guessing it.

    Label for each merge-tree region: does the region's height error behave the
    way a blurred plateau does — i.e. is the GT inside the region flatter than
    the prediction and is the GT edge sharper?  Concretely the target is
    ``1`` when refining that region with the full operator reduces its MAE
    against GT, and ``0`` otherwise.  That makes the classifier predict, from
    topology + geometry + RGB alone, exactly the thing it is used for at
    inference time: "will touching this region help".
    """
    from sklearn.linear_model import LogisticRegression

    X, y = [], []
    feat_order = ["z_log_area", "z_planarity", "z_rgb_texture", "z_solidity",
                  "z_rectangularity", "z_edge_alignment", "z_greenness",
                  "z_persistence"]
    for name in scenes:
        pred, gt, rgb, prof, pmeta, _ = load_scene(pred_root, gt_root, name)
        gsd = io.gsd_from_metadata(pmeta, gsd_default)
        H, _ = T.split_nodata(pred)
        labels = T.elevated_segmentation(H, cfg, gsd)
        if labels.max() == 0:
            continue
        feats = T.region_features(H, labels, rgb, gsd)
        res = refine(pred, rgb, gsd, cfg=cfg)
        z = {k: T._zscore(v) for k, v in {
            "z_log_area": np.log10(feats["area_m2"] + 1e-3),
            "z_planarity": feats["planarity"],
            "z_rgb_texture": feats["rgb_texture"],
            "z_solidity": feats["solidity"],
            "z_rectangularity": feats["rectangularity"],
            "z_edge_alignment": feats["edge_alignment"],
            "z_greenness": feats["greenness"],
            "z_persistence": feats["persistence"],
        }.items()}
        objs = __import__("scipy.ndimage", fromlist=["x"]).find_objects(labels)
        for i, sl in enumerate(objs, start=1):
            if sl is None:
                continue
            sub = labels[sl] == i
            if sub.sum() < 12:
                continue
            e0 = np.abs(pred[sl][sub] - gt[sl][sub]).mean()
            e1 = np.abs(res.height[sl][sub] - gt[sl][sub]).mean()
            X.append([z[k][i - 1] for k in feat_order])
            y.append(1 if e1 < e0 else 0)
    X = np.asarray(X)
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        raise SystemExit("classifier refit needs both classes; got only one")
    lr = LogisticRegression(max_iter=2000, C=1.0).fit(X, y)
    w = {"bias": float(lr.intercept_[0])}
    w.update({k: float(v) for k, v in zip(feat_order, lr.coef_[0])})
    return {"cls_weights": w, "n_regions": int(len(y)),
            "positive_rate": float(y.mean()),
            "train_accuracy": float(lr.score(X, y))}


# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-root", required=True)
    ap.add_argument("--gt-root", required=True)
    ap.add_argument("--split", default=None,
                    help="only scenes whose metadata.json says this split")
    ap.add_argument("--ablations", nargs="+", default=["A0", "FINAL"])
    ap.add_argument("--config", default=None, help="frozen RefineConfig JSON")
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--figures", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-topology", action="store_true",
                    help="skip persistence diagrams (they are the slow part)")
    ap.add_argument("--building-mask-height", type=float, default=2.0)
    ap.add_argument("--fit-classifier", default=None,
                    help="dev split only: refit region classifier weights and "
                         "write them here")
    args = ap.parse_args(argv)

    if args.fit_classifier and (args.split or "").lower() == "test":
        ap.error("refusing to fit anything on the test split (data leakage)")

    scenes = find_scenes(args.pred_root, args.gt_root)
    if not scenes:
        ap.error("no matching scene directories found")

    base_cfg = T.RefineConfig()
    if args.config:
        with open(args.config) as fh:
            base_cfg = replace(base_cfg, **json.load(fh))

    kept = []
    for name in scenes:
        pmeta = io.read_metadata(os.path.join(args.pred_root, name, "metadata.json")) \
            if os.path.isfile(os.path.join(args.pred_root, name, "metadata.json")) else {}
        if pmeta.get("field_source") == "ground_truth_lidar":
            print(f"skip {name}: prediction bundle is flagged as LiDAR truth")
            continue
        if args.split and scene_split(pmeta) != args.split.lower():
            continue
        kept.append(name)
    if args.limit:
        kept = kept[: args.limit]
    if not kept:
        ap.error(f"no scenes left after split filter '{args.split}'")
    print(f"{len(kept)} scenes, split={args.split or 'any'}")

    if args.fit_classifier:
        info = fit_classifier(kept, args.pred_root, args.gt_root, base_cfg)
        with open(args.fit_classifier, "w") as fh:
            json.dump(info, fh, indent=2)
        print(f"refit on {info['n_regions']} regions, "
              f"train acc {info['train_accuracy']:.3f}, "
              f"positive rate {info['positive_rate']:.3f} -> {args.fit_classifier}")
        return 0

    per_scene: Dict[str, Dict[str, Dict]] = {}
    for si, name in enumerate(kept, 1):
        pred, gt, rgb, prof, pmeta, _ = load_scene(args.pred_root, args.gt_root, name)
        gsd = io.gsd_from_metadata(pmeta)
        bmask = (gt >= args.building_mask_height)
        per_scene[name] = {}
        for ab in args.ablations:
            cfg, kw = config_for_ablation(ab, base_cfg)
            res = refine(pred, rgb, gsd, cfg=cfg, **kw)
            m = M.evaluate(res.height, gt, building_mask=bmask,
                           with_topology=not args.no_topology)
            m["stats"] = res.stats
            per_scene[name][ab] = m
            if args.figures and ab != "A0":
                os.makedirs(args.figures, exist_ok=True)
                try:
                    from dw_tda.viz import before_after_panel, cross_sections
                    before_after_panel(os.path.join(args.figures, f"{name}_{ab}.png"),
                                       pred, res, rgb, gsd, gt=gt)
                    cross_sections(os.path.join(args.figures, f"{name}_{ab}_sections.png"),
                                   pred, res.height, gsd, gt=gt)
                except Exception as exc:
                    print(f"  (figures skipped: {exc})")
        print(f"[{si}/{len(kept)}] {name}: " + "  ".join(
            f"{ab} bMAE {per_scene[name][ab]['building_mae']:.3f}"
            for ab in args.ablations))

    report_keys = ["overall_mae", "overall_rmse", "overall_r", "building_mae",
                   "building_rmse", "sharpness_ratio", "edge_abs_f1",
                   "edge_dist_px", "tv_ratio"]
    if not args.no_topology:
        report_keys += ["wasserstein1", "betti_l1", "n_feature_error"]

    summary = {}
    for ab in args.ablations:
        summary[ab] = {}
        for k in report_keys:
            vals = [per_scene[n][ab].get(k) for n in kept]
            vals = [v for v in vals if v is not None and np.isfinite(v)]
            summary[ab][k] = float(np.mean(vals)) if vals else float("nan")
            summary[ab][k + "_median"] = float(np.median(vals)) if vals else float("nan")

    print("\n=== summary (mean over scenes) ===")
    print(f"{'':>6} " + " ".join(f"{k[:11]:>12}" for k in report_keys))
    for ab in args.ablations:
        print(f"{ab:>6} " + " ".join(f"{summary[ab][k]:12.4f}" for k in report_keys))

    if "A0" in args.ablations:
        base = summary["A0"]
        print("\n=== change vs A0 ===")
        for ab in args.ablations:
            if ab == "A0":
                continue
            d = summary[ab]
            # Report the SIGNED CHANGE in the metric, with an explicit word.
            # The previous version printed (A0 - FINAL)/A0, so a metric that got
            # WORSE appeared as a negative percentage -- which reads as an
            # improvement. A regression could be, and was, screenshotted as a win.
            def _chg(lo_is_better, a, b):
                pct = 100 * (b - a) / a if a else float("nan")
                better = (b < a) if lo_is_better else (b > a)
                return f"{pct:+.2f}% {'better' if better else 'WORSE'}"

            print(
                f"{ab}:  building MAE {base['building_mae']:.4f} -> {d['building_mae']:.4f} m "
                f"({_chg(True, base['building_mae'], d['building_mae'])})\n"
                f"{' ' * len(ab)}   overall RMSE {base['overall_rmse']:.4f} -> {d['overall_rmse']:.4f} m "
                f"({_chg(True, base['overall_rmse'], d['overall_rmse'])})\n"
                f"{' ' * len(ab)}   sharpness    {base['sharpness_ratio']:.4f} -> {d['sharpness_ratio']:.4f} "
                f"({_chg(False, base['sharpness_ratio'], d['sharpness_ratio'])})"
            )

    with open(args.out, "w") as fh:
        json.dump({"split": args.split, "scenes": kept, "config": json.loads(base_cfg.to_json()),
                   "summary": summary, "per_scene": per_scene}, fh, indent=2, default=float)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
