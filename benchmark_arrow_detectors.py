"""Arrow-detector comparison: hough_lines vs directed_lines.

Scores relationships (directed source -> target pairs, macro-F1 with the same
matching as benchmark_ablation.py) for each arrow detector with the
outside-box gate on and off, in two settings:

  full pipeline   PaddleOCR text + Canny boxes + the arrow detector, over all
                  15 diagrams, also split into the 14 synthetic diagrams and
                  the one real-world diagram (search_interview_test.png).
  arrow stage     the answer-key boxes from ground_truth_boxes.json (derived
                  from the generator by data/eval/derive_ground_truth_boxes.py) fed to
                  the graph builder instead of detected boxes, isolating the
                  arrow stage from OCR and box errors (14 synthetic diagrams;
                  the real diagram has no box answer key).

Outputs:
    results/arrow_detector_comparison.csv
"""
from __future__ import annotations

import csv
import json
import os
import sys
from typing import Dict, List

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import benchmark_ablation as ba  # noqa: E402  (shared scoring + paths)
from src.graph import builder  # noqa: E402

REAL = "search_interview_test.png"
GT_BOXES_PATH = os.path.join(ba.EVAL_DIR, "ground_truth_boxes.json")


def _macro(f1s: List[float]) -> float:
    return round(sum(f1s) / len(f1s), 3) if f1s else 0.0


def main() -> None:
    from src.arrow_detection import directed_lines_detector, hough_lines_detector
    from src.box_detection import canny_contours_detector
    from src.text_detection import paddle_detector

    with open(ba.GROUND_TRUTH_PATH, encoding="utf-8") as f:
        gt = json.load(f)
    with open(GT_BOXES_PATH, encoding="utf-8") as f:
        gt_boxes = json.load(f)
    names = sorted(gt)

    detectors = {"hough_lines": hough_lines_detector, "directed_lines": directed_lines_detector}
    texts, boxes, arrows = {}, {}, {d: {} for d in detectors}
    for n in names:
        path = os.path.join(ba.DIAGRAMS_DIR, n)
        texts[n] = paddle_detector.detect(path)["texts"]
        boxes[n] = canny_contours_detector.detect(path)["boxes"]
        for d, mod in detectors.items():
            arrows[d][n] = mod.detect(path)["arrows"]
        print(f"  [ok] {n}")

    def rel_f1(arrow_list: List[Dict], box_list: List[Dict], name: str, gate: bool) -> float:
        rels = builder.build_relationships(
            arrow_list, box_list, outside_box_gate=gate,
            max_dist=160.0, ray_intersection=True, max_proj=400.0,
        )
        return ba._score_relationships(
            [(r["source"], r["target"]) for r in rels], gt[name]["arrows"])["f1"]

    rows = []
    for d in detectors:
        for gate in (True, False):
            full = {n: rel_f1(arrows[d][n], builder.label_boxes([dict(b) for b in boxes[n]], texts[n]), n, gate)
                    for n in names}
            stage = [rel_f1(arrows[d][n], [dict(b) for b in gt_boxes[n]["boxes"]], n, gate)
                     for n in names if n in gt_boxes]
            rows.append({
                "arrow_detector": d,
                "outside_box_gate": gate,
                "full_pipeline_15": _macro(list(full.values())),
                "full_pipeline_synthetic_14": _macro([v for n, v in full.items() if n != REAL]),
                "full_pipeline_real_1": full[REAL],
                "arrow_stage_gt_boxes_14": _macro(stage),
            })

    os.makedirs(ba.RESULTS_DIR, exist_ok=True)
    out = os.path.join(ba.RESULTS_DIR, "arrow_detector_comparison.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[ok] {out}")
    for r in rows:
        print("  ", r)


if __name__ == "__main__":
    main()
