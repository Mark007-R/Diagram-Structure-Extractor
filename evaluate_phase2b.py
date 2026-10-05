"""Phase 2b benchmark harness — Day 3 of the DiagraMine sprint.

Runs every arrow detector (directed_lines, pixel_scan, hough_lines, cnn) and
every icon detector (HSV, template, CLIP) against the 15-diagram benchmark.

Arrow scoring:
  Each detected arrow's two endpoints are snapped to the GT box whose border
  is nearest, within 25 px (using `ground_truth_boxes.json`). The resulting
  (src_label, tgt_label) is treated as an UNORDERED pair (only directed_lines
  recovers direction, so direction is not scored here; benchmark_ablation.py
  and benchmark_arrow_detectors.py score it). Detected pairs are bipartite-
  matched to GT arrows in `ground_truth.json`; a pair reported more than once
  counts as one TP plus FPs. TP / FP / FN follow, pooled over the diagrams.

  For `search_interview_test.png` we do NOT have machine-readable GT boxes
  (it's the original test image), so arrow scoring is skipped on that diagram.

Icon scoring (presence + label):
  GT icons are sparse: only `search_interview_test.png` has any (2 labels:
  ["docker", "MS SQL"]). On the 14 generated diagrams the icon GT is empty,
  so any detection is a false positive — and we report the FP rate per
  detector. On the test image we score label-match P/R/F1 with fuzzy ≥ 80
  on `label.lower()`. HSV returns generic `"icon"` so by-label P/R is zero
  on the test image; we ALSO report presence-only P/R for HSV as a fairer
  axis (did it find any region near a real icon location?).

Outputs:
  results/phase2b_arrow_icon.csv     (per-detector aggregate)
  results/phase2b_per_diagram.csv    (per-detector per-diagram)
  results/phase2b_detail.json        (raw matches)
  results/phase2b_schema_validity.json  (per-detector schema-valid output rate)
  (results/phase2_leaderboard.csv is built from these by results/_build_leaderboard.py)
  results/samples/arrow/<detector>__<diagram>.png   (annotated overlays)
  results/samples/icon/<detector>__<diagram>.png
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import cv2
from rapidfuzz import fuzz

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.arrow_detection import (
    cnn_detector, directed_lines_detector, hough_lines_detector, pixel_scan_detector,
)
from src.icon_detection import hsv_detector, template_detector, clip_detector

DIAGRAMS_DIR = os.path.join(ROOT, "data", "eval", "diagrams_15")
GT_LABELS_PATH = os.path.join(ROOT, "data", "eval", "ground_truth.json")
GT_BOXES_PATH = os.path.join(ROOT, "data", "eval", "ground_truth_boxes.json")
RESULTS_DIR = os.path.join(ROOT, "results")
SAMPLE_DIR_ARROW = os.path.join(RESULTS_DIR, "samples", "arrow")
SAMPLE_DIR_ICON = os.path.join(RESULTS_DIR, "samples", "icon")
os.makedirs(SAMPLE_DIR_ARROW, exist_ok=True)
os.makedirs(SAMPLE_DIR_ICON, exist_ok=True)

with open(GT_LABELS_PATH, encoding="utf-8") as f:
    GT_LABELS = json.load(f)
with open(GT_BOXES_PATH, encoding="utf-8") as f:
    GT_BOXES = json.load(f)

ARROW_DETECTORS = {
    "directed_lines": directed_lines_detector,
    "pixel_scan": pixel_scan_detector,
    "hough_lines": hough_lines_detector,
    "cnn_verified": cnn_detector,
}
ICON_DETECTORS = {
    "hsv": hsv_detector,
    "template_matching": template_detector,
    "clip_zero_shot": clip_detector,
}

LABEL_FUZZ_THRESHOLD = 60  # for snapping arrow endpoint -> GT box label
SAMPLE_DIAGRAMS = [
    "diagram_01.png",
    "diagram_05.png",
    "diagram_07.png",
    "diagram_10.png",
    "diagram_14.png",
]


# ---------------------------------------------------------------------------
# Arrow scoring
# ---------------------------------------------------------------------------

def _nearest_box(point: Tuple[int, int], gt_boxes: List[dict],
                 max_dist: int = 25) -> str | None:
    """Return the label of the GT box whose border is closest to `point`
    (0 inside the box), within `max_dist`; else None.

    Arrows end at box borders, but a detected line usually stops at the base
    of the arrowhead, short of the border, so `max_dist` allows about one
    arrowhead length. (This used to fall back to distance from the box
    centre, which only worked while the benchmark arrows stopped about 30 px
    from the box centres, inside the box.)"""
    x, y = point
    best, best_d = None, float(max_dist)
    for b in gt_boxes:
        dx = max(b["x"] - x, 0, x - (b["x"] + b["w"]))
        dy = max(b["y"] - y, 0, y - (b["y"] + b["h"]))
        d = (dx * dx + dy * dy) ** 0.5
        if d <= best_d:
            best_d = d
            best = b["label"]
    return best


def _snap_arrow_to_pair(a: dict, gt_boxes: List[dict]) -> Tuple[str, str] | None:
    src = _nearest_box((a["x1"], a["y1"]), gt_boxes)
    tgt = _nearest_box((a["x2"], a["y2"]), gt_boxes)
    if src is None or tgt is None or src == tgt:
        return None
    a, b = sorted([src, tgt])
    return (a, b)


def score_arrows(detected: List[dict], gt_arrows: List[dict],
                 gt_boxes: List[dict]) -> dict:
    """Bipartite match by unordered (src, tgt) pair equality.

    GT arrow pairs are normalised to (min, max) so we don't penalise the
    detector for missing direction.
    """
    if not gt_boxes:
        return {"detected_pairs": 0, "gt_pairs": len(gt_arrows),
                "tp": 0, "fp": 0, "fn": len(gt_arrows),
                "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "matches": [], "note": "no GT boxes — skipped"}

    det_pairs = []
    for a in detected:
        p = _snap_arrow_to_pair(a, gt_boxes)
        if p is not None:
            det_pairs.append(p)
    # Multiplicity within det_pairs is preserved (a detector that reports
    # the same pair 5×, gets 1 TP and 4 FP).
    gt_pairs = [tuple(sorted([g["source"], g["target"]])) for g in gt_arrows]

    # Bipartite match
    used_g = set()
    matches = []
    used_d_idx = set()
    for di, dp in enumerate(det_pairs):
        for gi, gp in enumerate(gt_pairs):
            if gi in used_g:
                continue
            if dp == gp:
                used_g.add(gi)
                used_d_idx.add(di)
                matches.append((dp, gi, di))
                break

    tp = len(matches)
    fp = len(det_pairs) - tp
    fn = len(gt_pairs) - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "detected_pairs": len(det_pairs),
        "gt_pairs": len(gt_pairs),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "matches": matches,
    }


# ---------------------------------------------------------------------------
# Icon scoring
# ---------------------------------------------------------------------------

def score_icons(detected: List[dict], gt_icon_labels: List[str]) -> dict:
    """Two axes:
      • by-label  : detector must produce the right label string (fuzzy ≥ 80).
                    Multiple detections of the same label collapse to one.
      • presence  : did the detector return ANY detection? (Fair to HSV which
                    can't classify.) If GT has icons and detector has at least
                    one, presence-TP=min(len(gt), len(det)) — capped.

    On diagrams with empty GT icons (the 14 generated ones), any detection is
    a false positive in both axes.
    """
    det_labels_raw = [d.get("label", "icon").lower() for d in detected]
    # collapse duplicate labels
    det_labels_unique = list(dict.fromkeys(det_labels_raw))
    gt = [g.lower() for g in gt_icon_labels]

    # ── By-label ──
    used_g = set()
    used_d = set()
    matches = []
    pairs = []
    for gi, gl in enumerate(gt):
        for di, dl in enumerate(det_labels_unique):
            pairs.append((fuzz.ratio(gl, dl), gi, di))
    pairs.sort(reverse=True)
    for r, gi, di in pairs:
        if r < 80:
            break
        if gi in used_g or di in used_d:
            continue
        used_g.add(gi)
        used_d.add(di)
        matches.append((gt[gi], det_labels_unique[di], r))
    bl_tp = len(matches)
    bl_fp = max(0, len(det_labels_unique) - bl_tp)
    bl_fn = len(gt) - bl_tp
    bl_p = bl_tp / (bl_tp + bl_fp) if (bl_tp + bl_fp) else 0.0
    bl_r = bl_tp / (bl_tp + bl_fn) if (bl_tp + bl_fn) else 0.0
    bl_f1 = 2 * bl_p * bl_r / (bl_p + bl_r) if (bl_p + bl_r) else 0.0

    # ── Presence ──
    # min(GT count, detection count) is a cap on presence TP.
    pr_tp = min(len(gt), len(detected))
    pr_fp = max(0, len(detected) - len(gt))
    pr_fn = max(0, len(gt) - len(detected))
    pr_p = pr_tp / (pr_tp + pr_fp) if (pr_tp + pr_fp) else 0.0
    pr_r = pr_tp / (pr_tp + pr_fn) if (pr_tp + pr_fn) else 0.0
    pr_f1 = 2 * pr_p * pr_r / (pr_p + pr_r) if (pr_p + pr_r) else 0.0

    return {
        "detected": len(detected),
        "detected_unique_labels": len(det_labels_unique),
        "gt": len(gt),
        "by_label": {"tp": bl_tp, "fp": bl_fp, "fn": bl_fn,
                     "precision": round(bl_p, 3), "recall": round(bl_r, 3),
                     "f1": round(bl_f1, 3), "matches": matches},
        "presence": {"tp": pr_tp, "fp": pr_fp, "fn": pr_fn,
                     "precision": round(pr_p, 3), "recall": round(pr_r, 3),
                     "f1": round(pr_f1, 3)},
    }


# ---------------------------------------------------------------------------
# Sample emitters
# ---------------------------------------------------------------------------

def emit_arrow_sample(name: str, fname: str, image_path: str,
                      detected: List[dict], gt_arrows: List[dict],
                      gt_boxes: List[dict], score: dict) -> None:
    img = cv2.imread(image_path)
    if img is None:
        return
    # GT arrows: green dashed lines between GT box centres
    label_to_box = {b["label"]: b for b in gt_boxes}
    for ga in gt_arrows:
        sb = label_to_box.get(ga["source"])
        tb = label_to_box.get(ga["target"])
        if not sb or not tb:
            continue
        sx = sb["x"] + sb["w"] // 2
        sy = sb["y"] + sb["h"] // 2
        tx = tb["x"] + tb["w"] // 2
        ty = tb["y"] + tb["h"] // 2
        cv2.line(img, (sx, sy), (tx, ty), (0, 180, 0), 2)
    # Detected: red endpoints + connecting line
    for d in detected:
        cv2.line(img, (d["x1"], d["y1"]), (d["x2"], d["y2"]),
                 (0, 0, 255), 1, cv2.LINE_AA)
        cv2.circle(img, (d["x1"], d["y1"]), 3, (0, 0, 255), -1)
        cv2.circle(img, (d["x2"], d["y2"]), 3, (0, 0, 255), -1)
    label = (f"{name} p={score['precision']} r={score['recall']} "
             f"f1={score['f1']} det_pairs={score['detected_pairs']} "
             f"gt_pairs={score['gt_pairs']}")
    cv2.putText(img, label, (10, img.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    out = os.path.join(SAMPLE_DIR_ARROW, f"{name}__{fname}")
    cv2.imwrite(out, img)


def emit_icon_sample(name: str, fname: str, image_path: str,
                     detected: List[dict], gt_labels: List[str],
                     score: dict) -> None:
    img = cv2.imread(image_path)
    if img is None:
        return
    for d in detected:
        cv2.rectangle(img, (d["x"], d["y"]),
                      (d["x"] + d["w"], d["y"] + d["h"]),
                      (0, 0, 255), 2)
        lbl = d.get("label", "icon")
        cv2.putText(img, lbl, (d["x"] + 2, d["y"] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1)
    label = (f"{name}  gt={','.join(gt_labels) or '(none)'}  "
             f"det_unique_labels={score['detected_unique_labels']} "
             f"by_label_f1={score['by_label']['f1']} "
             f"presence_f1={score['presence']['f1']}")
    cv2.putText(img, label, (10, img.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    out = os.path.join(SAMPLE_DIR_ICON, f"{name}__{fname}")
    cv2.imwrite(out, img)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def is_schema_valid_arrow(out: dict) -> bool:
    if not isinstance(out, dict): return False
    if "arrows" not in out or "runtime_seconds" not in out: return False
    if not isinstance(out["arrows"], list): return False
    for a in out["arrows"]:
        if not isinstance(a, dict): return False
        for k in ("x1", "y1", "x2", "y2"):
            if k not in a: return False
    try:
        json.dumps(out)
    except Exception:
        return False
    return True


def is_schema_valid_icon(out: dict) -> bool:
    if not isinstance(out, dict): return False
    if "icons" not in out or "runtime_seconds" not in out: return False
    if not isinstance(out["icons"], list): return False
    for ic in out["icons"]:
        if not isinstance(ic, dict): return False
        for k in ("x", "y", "w", "h"):
            if k not in ic: return False
    try:
        json.dumps(out)
    except Exception:
        return False
    return True


def run() -> None:
    detail: Dict[str, dict] = {"arrow": {}, "icon": {}}
    schema: Dict[str, dict] = {"arrow": {}, "icon": {}}
    per_diagram_rows: List[dict] = []
    diagram_files = sorted(GT_LABELS.keys())

    # ── ARROWS ──────────────────────────────────────────────────────────
    for det_name, mod in ARROW_DETECTORS.items():
        per_diag = {}
        agg = {"tp": 0, "fp": 0, "fn": 0, "runtime_sum": 0.0, "n_runs": 0}
        schema_runs = 0
        schema_valid = 0
        for fname in diagram_files:
            path = os.path.join(DIAGRAMS_DIR, fname)
            try:
                out = mod.detect(path)
            except Exception as exc:
                print(f"[arrow] {det_name} on {fname} FAILED: {exc!r}")
                continue
            schema_runs += 1
            if is_schema_valid_arrow(out):
                schema_valid += 1
            agg["runtime_sum"] += out["runtime_seconds"]
            agg["n_runs"] += 1
            # Scoring only on diagrams where we have GT boxes
            if fname not in GT_BOXES:
                per_diag[fname] = {"skipped": "no GT boxes",
                                    "runtime_seconds": out["runtime_seconds"],
                                    "detected": len(out["arrows"])}
                per_diagram_rows.append({
                    "stage": "arrow", "detector": det_name, "diagram": fname,
                    "detected": len(out["arrows"]), "gt": "",
                    "tp": "", "fp": "", "fn": "", "precision": "",
                    "recall": "", "f1": "",
                    "runtime_seconds": out["runtime_seconds"],
                    "note": "no GT boxes (search_interview_test.png)",
                })
                if fname in SAMPLE_DIAGRAMS:
                    pass  # no GT to overlay
                continue
            gt_arrows = GT_LABELS[fname]["arrows"]
            gt_boxes = GT_BOXES[fname]["boxes"]
            score = score_arrows(out["arrows"], gt_arrows, gt_boxes)
            per_diag[fname] = {**score, "runtime_seconds": out["runtime_seconds"]}
            agg["tp"] += score["tp"]; agg["fp"] += score["fp"]; agg["fn"] += score["fn"]
            per_diagram_rows.append({
                "stage": "arrow", "detector": det_name, "diagram": fname,
                "detected": score["detected_pairs"], "gt": score["gt_pairs"],
                "tp": score["tp"], "fp": score["fp"], "fn": score["fn"],
                "precision": score["precision"], "recall": score["recall"],
                "f1": score["f1"],
                "runtime_seconds": out["runtime_seconds"], "note": "",
            })
            if fname in SAMPLE_DIAGRAMS:
                emit_arrow_sample(det_name, fname, path, out["arrows"],
                                  gt_arrows, gt_boxes, score)
        tp, fp, fn = agg["tp"], agg["fp"], agg["fn"]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        avg_rt = agg["runtime_sum"] / agg["n_runs"] if agg["n_runs"] else 0.0
        detail["arrow"][det_name] = {
            "per_diagram": per_diag,
            "aggregate": {
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1": round(f1, 3),
                "avg_runtime_seconds": round(avg_rt, 3),
                "scored_diagrams": agg["n_runs"] - 1,  # minus the skipped one
            },
        }
        schema["arrow"][det_name] = {"runs": schema_runs,
                                      "valid": schema_valid,
                                      "rate": round(schema_valid / schema_runs, 3)
                                      if schema_runs else 0.0}
        print(f"[arrow] {det_name:<13} p={precision:.3f} r={recall:.3f} "
              f"f1={f1:.3f} avg_rt={avg_rt:.2f}s "
              f"schema={schema['arrow'][det_name]['valid']}/{schema['arrow'][det_name]['runs']}")

    # ── ICONS ──────────────────────────────────────────────────────────
    for det_name, mod in ICON_DETECTORS.items():
        per_diag = {}
        agg = {"bl_tp": 0, "bl_fp": 0, "bl_fn": 0,
               "pr_tp": 0, "pr_fp": 0, "pr_fn": 0,
               "fp_count": 0, "runtime_sum": 0.0, "n_runs": 0}
        schema_runs = 0
        schema_valid = 0
        for fname in diagram_files:
            path = os.path.join(DIAGRAMS_DIR, fname)
            try:
                out = mod.detect(path)
            except Exception as exc:
                print(f"[icon] {det_name} on {fname} FAILED: {exc!r}")
                continue
            schema_runs += 1
            if is_schema_valid_icon(out):
                schema_valid += 1
            gt = GT_LABELS[fname].get("icons", [])
            score = score_icons(out["icons"], gt)
            per_diag[fname] = {**score, "runtime_seconds": out["runtime_seconds"]}
            agg["bl_tp"] += score["by_label"]["tp"]
            agg["bl_fp"] += score["by_label"]["fp"]
            agg["bl_fn"] += score["by_label"]["fn"]
            agg["pr_tp"] += score["presence"]["tp"]
            agg["pr_fp"] += score["presence"]["fp"]
            agg["pr_fn"] += score["presence"]["fn"]
            agg["fp_count"] += len(out["icons"]) if not gt else 0
            agg["runtime_sum"] += out["runtime_seconds"]
            agg["n_runs"] += 1
            per_diagram_rows.append({
                "stage": "icon", "detector": det_name, "diagram": fname,
                "detected": score["detected"],
                "gt": score["gt"],
                "by_label_p": score["by_label"]["precision"],
                "by_label_r": score["by_label"]["recall"],
                "by_label_f1": score["by_label"]["f1"],
                "presence_p": score["presence"]["precision"],
                "presence_r": score["presence"]["recall"],
                "presence_f1": score["presence"]["f1"],
                "runtime_seconds": out["runtime_seconds"],
            })
            if fname in SAMPLE_DIAGRAMS or fname == "search_interview_test.png":
                emit_icon_sample(det_name, fname, path, out["icons"], gt, score)

        # Aggregates
        def _prf(tp, fp, fn):
            p = tp / (tp + fp) if (tp + fp) else 0.0
            r = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) else 0.0
            return round(p, 3), round(r, 3), round(f1, 3)

        bl = _prf(agg["bl_tp"], agg["bl_fp"], agg["bl_fn"])
        pr = _prf(agg["pr_tp"], agg["pr_fp"], agg["pr_fn"])
        avg_rt = agg["runtime_sum"] / agg["n_runs"] if agg["n_runs"] else 0.0
        detail["icon"][det_name] = {
            "per_diagram": per_diag,
            "aggregate": {
                "by_label": {"precision": bl[0], "recall": bl[1], "f1": bl[2]},
                "presence": {"precision": pr[0], "recall": pr[1], "f1": pr[2]},
                "no_gt_fp_count": agg["fp_count"],
                "avg_runtime_seconds": round(avg_rt, 3),
            },
        }
        schema["icon"][det_name] = {"runs": schema_runs,
                                     "valid": schema_valid,
                                     "rate": round(schema_valid / schema_runs, 3)
                                     if schema_runs else 0.0}
        print(f"[icon]  {det_name:<17} "
              f"by_label_f1={bl[2]:.3f}  presence_f1={pr[2]:.3f}  "
              f"FPs_on_empty_gt={agg['fp_count']}  avg_rt={avg_rt:.2f}s")

    # ── Aggregate CSV ─────────────────────────────────────────────────
    arrow_icon_path = os.path.join(RESULTS_DIR, "phase2b_arrow_icon.csv")
    with open(arrow_icon_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stage", "detector",
                    "precision", "recall", "f1",
                    "secondary", "avg_runtime_seconds", "notes"])
        for det_name, info in detail["arrow"].items():
            agg = info["aggregate"]
            w.writerow(["arrow", det_name, agg["precision"], agg["recall"],
                        agg["f1"], "",
                        agg["avg_runtime_seconds"],
                        f"unordered src/tgt pairs, {agg['scored_diagrams']} scored diagrams"])
        for det_name, info in detail["icon"].items():
            agg = info["aggregate"]
            w.writerow(["icon", det_name,
                        agg["by_label"]["precision"],
                        agg["by_label"]["recall"],
                        agg["by_label"]["f1"],
                        f"presence_f1={agg['presence']['f1']} "
                        f"fps_empty_gt={agg['no_gt_fp_count']}",
                        agg["avg_runtime_seconds"],
                        "by-label fuzzy>=80; GT icons only present on search_interview_test.png"])

    # ── Per-diagram CSV ───────────────────────────────────────────────
    per_diag_path = os.path.join(RESULTS_DIR, "phase2b_per_diagram.csv")
    with open(per_diag_path, "w", newline="", encoding="utf-8") as f:
        if per_diagram_rows:
            all_keys: List[str] = []
            seen = set()
            for r in per_diagram_rows:
                for k in r.keys():
                    if k not in seen:
                        seen.add(k); all_keys.append(k)
            w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            w.writeheader()
            for row in per_diagram_rows:
                w.writerow({k: row.get(k, "") for k in all_keys})

    # ── Detail JSON ───────────────────────────────────────────────────
    # strip large match payloads
    for det_name, info in detail["arrow"].items():
        for fname, pd in info["per_diagram"].items():
            pd.pop("matches", None)
    for det_name, info in detail["icon"].items():
        for fname, pd in info["per_diagram"].items():
            if "by_label" in pd and "matches" in pd["by_label"]:
                pd["by_label"] = {k: v for k, v in pd["by_label"].items()
                                  if k != "matches"}
    detail_path = os.path.join(RESULTS_DIR, "phase2b_detail.json")
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(detail, f, indent=2)

    # ── Schema-valid audit ────────────────────────────────────────────
    total_runs = sum(s["runs"] for s in schema["arrow"].values()) + \
                 sum(s["runs"] for s in schema["icon"].values())
    total_valid = sum(s["valid"] for s in schema["arrow"].values()) + \
                  sum(s["valid"] for s in schema["icon"].values())
    schema["aggregate"] = {"runs": total_runs, "valid": total_valid,
                            "rate": round(total_valid / total_runs, 3)
                            if total_runs else 0.0}
    schema_path = os.path.join(RESULTS_DIR, "phase2b_schema_validity.json")
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    print("\nAggregate CSV     ->", arrow_icon_path)
    print("Per-diagram CSV   ->", per_diag_path)
    print("Detail JSON       ->", detail_path)
    print("Schema validity   ->", schema_path)
    print(f"\nSchema-valid JSON output rate (Phase 2b): "
          f"{total_valid}/{total_runs} = {schema['aggregate']['rate']:.3f}")


if __name__ == "__main__":
    run()
