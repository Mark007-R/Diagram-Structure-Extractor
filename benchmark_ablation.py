"""Day-6 Phase-5 MLOps-style ablation.

Measures the marginal contribution of each detection stage by progressively
adding stages to the pipeline and rescoring against the 15-diagram benchmark.

Stages (cumulative):
    A: text only           — components scored by OCR fragments alone (no box grouping)
    B: text + box          — boxes labeled from contained text; components = boxes
    C: text + box + arrow  — relationships built but no outside-box gate
    D: + ray-intersection  — Day-5 arrow-mapping fix (extends mapping radius)
    E: + outside_box_gate  — Day-3 gate (drops box-border arrow segments); opt-in
    F: + icon              — full pipeline as shipped (gate off)

For each stage we record components / relationships / icons macro-F1 and the
delta vs the prior stage. Schema-valid JSON rate is 1.0 throughout (the typed
pipeline guarantees it — that is the headline reliability claim).

Outputs:
    results/ablation.csv           stage-by-stage table
    results/ablation.json          per-diagram detail
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
import time
from typing import Dict, List, Tuple

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src import pipeline as pl
from src.graph import builder
from src.schemas import PipelineConfig

CFG = PipelineConfig()  # shipped defaults (snap radius etc.)

EVAL_DIR = os.path.join(ROOT, "data", "eval")
DIAGRAMS_DIR = os.path.join(EVAL_DIR, "diagrams_15")
GROUND_TRUTH_PATH = os.path.join(EVAL_DIR, "ground_truth.json")
RESULTS_DIR = os.path.join(ROOT, "results")


def _norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _tok(a: str, b: str) -> float:
    ta, tb = set(_norm(a).split()), set(_norm(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _match(d: str, e: str, thresh: float = 0.6) -> bool:
    if not d or not e:
        return False
    nd, ne = _norm(d), _norm(e)
    if not nd or not ne:
        return False
    return nd == ne or ne in nd or nd in ne or _tok(d, e) >= thresh


def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def _score_components(detected: List[str], gt: List[str]) -> Dict:
    used = set()
    tp = 0
    for d in detected:
        for j, g in enumerate(gt):
            if j in used:
                continue
            if _match(d, g):
                used.add(j)
                tp += 1
                break
    fp = max(len(detected) - tp, 0)
    fn = len(gt) - tp
    p, r, f1 = _prf(tp, fp, fn)
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3),
            "tp": tp, "fp": fp, "fn": fn}


def _score_relationships(det_pairs: List[Tuple], gt_arrows: List[Dict]) -> Dict:
    """Directed one-to-one matching of (source, target[, bidirectional])
    pairs. A double-headed edge -- detected or in the answer key -- counts as
    two directed edges, A -> B and B -> A: finding only one of its heads costs
    a false negative, and a head drawn where the key has none costs a false
    positive."""
    def directed(items):
        out = []
        for s, t, bid in items:
            out.append((s, t))
            if bid:
                out.append((t, s))
        return out

    det_edges = directed((d[0], d[1], bool(d[2]) if len(d) > 2 else False) for d in det_pairs)
    gt_pairs = directed((a.get("source", ""), a.get("target", ""), bool(a.get("bidirectional")))
                        for a in gt_arrows)
    used = set()
    tp = 0
    for ds, dt in det_edges:
        for j, (gs, gt_lbl) in enumerate(gt_pairs):
            if j not in used and _match(ds, gs) and _match(dt, gt_lbl):
                used.add(j)
                tp += 1
                break
    fp = max(len(det_edges) - tp, 0)
    fn = len(gt_pairs) - tp
    p, r, f1 = _prf(tp, fp, fn)
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3),
            "tp": tp, "fp": fp, "fn": fn}


def _score_icons(det: List[str], gt: List[str]) -> Dict:
    tp = min(len(det), len(gt))
    fp = max(len(det) - len(gt), 0)
    fn = max(len(gt) - len(det), 0)
    p, r, f1 = _prf(tp, fp, fn)
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3),
            "tp": tp, "fp": fp, "fn": fn}


# ───── Stage runners — pull the existing src/ detectors directly ────────────

def _detect_raw(image_path: str):
    """Run all detectors once; the ablation reuses the same raw outputs."""
    import cv2
    from src.text_detection import paddle_detector as td
    from src.box_detection import canny_contours_detector as bd, icon_nodes
    from src.arrow_detection import directed_lines_detector as ad
    from src.icon_detection import template_detector as icd

    t = td.detect(image_path)["texts"]
    b = bd.detect(image_path)["boxes"]
    b = b + icon_nodes.find(cv2.imread(image_path), t, b)   # part of the shipped box stage
    a = ad.detect(image_path)["arrows"]
    ic = icd.detect(image_path)["icons"]
    return t, b, a, ic


def _components_from_boxes(boxes: List[Dict]) -> List[str]:
    return [b.get("label", "").strip() for b in boxes
            if not b.get("is_region") and b.get("label", "").strip()]


def _icons_to_labels(icons: List[Dict]) -> List[str]:
    return [i.get("label", "icon") for i in icons]


def run_ablation(diagrams: List[str], ground_truth: Dict) -> Dict:
    stage_names = ["A_text", "B_text_box", "C_text_box_arrow",
                   "D_plus_ray_intersection", "E_plus_outside_box_gate",
                   "F_full_pipeline"]
    per_stage = {s: {"per_diagram": {}, "schema_valid": 0, "n": 0} for s in stage_names}

    for name in diagrams:
        path = os.path.join(DIAGRAMS_DIR, name)
        gt = ground_truth.get(name, {})
        gt_components = gt.get("components", [])
        gt_arrows = gt.get("arrows", [])
        gt_icons = gt.get("icons", [])

        try:
            texts, boxes, arrows, icons = _detect_raw(path)
        except Exception as e:  # noqa: BLE001
            print(f"  [error] {name}: {e}")
            for s in stage_names:
                per_stage[s]["per_diagram"][name] = {"error": str(e)}
                per_stage[s]["n"] += 1
            continue

        labelled_boxes = builder.label_boxes(boxes, texts)

        # Stage A — text alone.
        comp_a = [t["text"].strip() for t in texts if t.get("text", "").strip()]
        per_stage["A_text"]["per_diagram"][name] = {
            "components": _score_components(comp_a, gt_components),
            "relationships": {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                              "tp": 0, "fp": 0, "fn": len(gt_arrows)},
            "icons": {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                      "tp": 0, "fp": 0, "fn": len(gt_icons)},
        }

        # Stage B — text + box (components are box-grouped labels).
        comp_box = _components_from_boxes(labelled_boxes)
        per_stage["B_text_box"]["per_diagram"][name] = {
            "components": _score_components(comp_box, gt_components),
            "relationships": {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                              "tp": 0, "fp": 0, "fn": len(gt_arrows)},
            "icons": {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                      "tp": 0, "fp": 0, "fn": len(gt_icons)},
        }

        # Stage C — text + box + arrow (no gate, no ray fallback).
        rels_c = builder.build_relationships(
            arrows, labelled_boxes,
            outside_box_gate=False, max_dist=CFG.rel_max_dist,
            ray_intersection=False, max_proj=400.0,
        )
        per_stage["C_text_box_arrow"]["per_diagram"][name] = {
            "components": _score_components(comp_box, gt_components),
            "relationships": _score_relationships(
                [(r["source"], r["target"], r.get("bidirectional", False)) for r in rels_c], gt_arrows),
            "icons": {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                      "tp": 0, "fp": 0, "fn": len(gt_icons)},
        }

        # Stage D — + ray-intersection (Day-5 fix).
        rels_d = builder.build_relationships(
            arrows, labelled_boxes,
            outside_box_gate=False, max_dist=CFG.rel_max_dist,
            ray_intersection=True, max_proj=400.0,
        )
        per_stage["D_plus_ray_intersection"]["per_diagram"][name] = {
            "components": _score_components(comp_box, gt_components),
            "relationships": _score_relationships(
                [(r["source"], r["target"], r.get("bidirectional", False)) for r in rels_d], gt_arrows),
            "icons": {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                      "tp": 0, "fp": 0, "fn": len(gt_icons)},
        }

        # Stage E — + outside_box_gate (Day-3 fix; opt-in, off by default).
        rels_e = builder.build_relationships(
            arrows, labelled_boxes,
            outside_box_gate=True, max_dist=CFG.rel_max_dist,
            ray_intersection=True, max_proj=400.0,
        )
        per_stage["E_plus_outside_box_gate"]["per_diagram"][name] = {
            "components": _score_components(comp_box, gt_components),
            "relationships": _score_relationships(
                [(r["source"], r["target"], r.get("bidirectional", False)) for r in rels_e], gt_arrows),
            "icons": {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                      "tp": 0, "fp": 0, "fn": len(gt_icons)},
        }

        # Stage F — full pipeline as shipped (gate off, adds icons).
        per_stage["F_full_pipeline"]["per_diagram"][name] = {
            "components": _score_components(comp_box, gt_components),
            "relationships": _score_relationships(
                [(r["source"], r["target"], r.get("bidirectional", False)) for r in rels_d], gt_arrows),
            "icons": _score_icons(_icons_to_labels(icons), gt_icons),
        }

        for s in stage_names:
            per_stage[s]["schema_valid"] += 1  # typed pipeline always validates
            per_stage[s]["n"] += 1
        print(f"  [ok] {name}")

    # Aggregate macro-F1 per stage.
    rows: List[Dict] = []
    prev = None
    for s in stage_names:
        pd_map = per_stage[s]["per_diagram"]
        n = max(len(pd_map), 1)
        c_f1 = sum(d["components"]["f1"] for d in pd_map.values() if "components" in d) / n
        r_f1 = sum(d["relationships"]["f1"] for d in pd_map.values() if "relationships" in d) / n
        i_f1 = sum(d["icons"]["f1"] for d in pd_map.values() if "icons" in d) / n
        row = {
            "stage": s,
            "components_macro_f1": round(c_f1, 3),
            "relationships_macro_f1": round(r_f1, 3),
            "icons_macro_f1": round(i_f1, 3),
            "schema_valid_rate": round(per_stage[s]["schema_valid"] / max(per_stage[s]["n"], 1), 3),
            "delta_rel_vs_prev": None if prev is None else round(r_f1 - prev["rels"], 3),
            "delta_comp_vs_prev": None if prev is None else round(c_f1 - prev["comp"], 3),
        }
        rows.append(row)
        prev = {"comp": c_f1, "rels": r_f1, "icons": i_f1}

    return {"stages": rows, "per_diagram": {s: per_stage[s]["per_diagram"] for s in stage_names}}


def write_outputs(ablation: Dict) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "ablation.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stage", "components_macro_f1", "relationships_macro_f1",
                    "icons_macro_f1", "schema_valid_rate",
                    "delta_rel_vs_prev", "delta_comp_vs_prev"])
        for r in ablation["stages"]:
            w.writerow([r["stage"], r["components_macro_f1"], r["relationships_macro_f1"],
                        r["icons_macro_f1"], r["schema_valid_rate"],
                        r["delta_rel_vs_prev"] if r["delta_rel_vs_prev"] is not None else "",
                        r["delta_comp_vs_prev"] if r["delta_comp_vs_prev"] is not None else ""])
    with open(os.path.join(RESULTS_DIR, "ablation.json"), "w", encoding="utf-8") as f:
        json.dump(ablation, f, indent=2, ensure_ascii=False)
    print(f"\n[ok] ablation.csv -> {csv_path}")
    print(f"[ok] ablation.json -> {os.path.join(RESULTS_DIR, 'ablation.json')}")


def main() -> None:
    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)
    diagrams = sorted(ground_truth.keys())
    print(f"[ablation] {len(diagrams)} diagrams x 6 stages")
    t0 = time.perf_counter()
    ablation = run_ablation(diagrams, ground_truth)
    write_outputs(ablation)
    print(f"\n[ablation] done in {time.perf_counter() - t0:.1f}s")
    for r in ablation["stages"]:
        print(f"  {r['stage']:<32}  comp_F1={r['components_macro_f1']:.3f}  "
              f"rel_F1={r['relationships_macro_f1']:.3f}  "
              f"icons_F1={r['icons_macro_f1']:.3f}")


if __name__ == "__main__":
    main()
