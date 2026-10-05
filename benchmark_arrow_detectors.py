"""Arrow-detector comparison: hough_lines vs directed_lines.

Scores relationships (directed source -> target pairs, macro-F1 with the same
matching as benchmark_ablation.py; a double-headed edge counts as one edge each
way) for each arrow detector with the outside-box gate on and off. Everything else
is the shipped pipeline (PaddleOCR text, Canny boxes plus icon nodes, the
graph builder with its default settings), so only the arrow detector differs.

Sets (each scored where it exists):
  benchmark      data/eval/diagrams_15: 14 synthetic diagrams + the real
                 draw.io export (search_interview_test.png)
  drawio_dev     data/eval/drawio_style: 8 draw.io-style diagrams, used to
                 tune the draw.io handling
  drawio_heldout data/eval/drawio_heldout: draw.io-style diagrams built
                 independently, first scored after tuning was frozen

Two settings per set:
  full pipeline  detected text and boxes
  arrow stage    the answer-key boxes (ground_truth_boxes.json) fed to the
                 graph builder instead, isolating the arrow stage from OCR
                 and box errors (the real diagram has no box key)

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

import cv2  # noqa: E402

import benchmark_ablation as ba  # noqa: E402  (shared scoring + paths)
from benchmark_ablation import CFG  # noqa: E402  (shipped defaults)
from src.graph import builder  # noqa: E402

REAL = "search_interview_test.png"
SETS = {
    "benchmark": os.path.join(ba.EVAL_DIR),
    "drawio_dev": os.path.join(ba.EVAL_DIR, "drawio_style"),
    "drawio_heldout": os.path.join(ba.EVAL_DIR, "drawio_heldout"),
}


def _images_dir(name: str, root: str) -> str:
    return os.path.join(root, "diagrams_15" if name == "benchmark" else "diagrams")


def _macro(f1s: List[float]) -> float:
    return round(sum(f1s) / len(f1s), 3) if f1s else 0.0


def main() -> None:
    from src.arrow_detection import directed_lines_detector, hough_lines_detector
    from src.box_detection import canny_contours_detector, icon_nodes
    from src.text_detection import paddle_detector

    detectors = {"hough_lines": hough_lines_detector, "directed_lines": directed_lines_detector}
    data = {}
    for set_name, root in SETS.items():
        if not os.path.exists(os.path.join(root, "ground_truth.json")):
            continue
        with open(os.path.join(root, "ground_truth.json"), encoding="utf-8") as f:
            gt = json.load(f)
        with open(os.path.join(root, "ground_truth_boxes.json"), encoding="utf-8") as f:
            gt_boxes = json.load(f)
        items = {}
        for n in sorted(gt):
            path = os.path.join(_images_dir(set_name, root), n)
            texts = paddle_detector.detect(path)["texts"]
            boxes = canny_contours_detector.detect(path)["boxes"]
            boxes = boxes + icon_nodes.find(cv2.imread(path), texts, boxes)
            items[n] = {"texts": texts, "boxes": boxes, "gt": gt[n]["arrows"],
                        "gt_boxes": gt_boxes.get(n, {}).get("boxes"),
                        "arrows": {d: m.detect(path)["arrows"] for d, m in detectors.items()}}
            print(f"  [ok] {set_name}/{n}")
        data[set_name] = items

    def rel_f1(arrow_list: List[Dict], box_list: List[Dict], gt_arrows: List[Dict], gate: bool) -> float:
        rels = builder.build_relationships(
            arrow_list, box_list, outside_box_gate=gate, max_dist=CFG.rel_max_dist,
            ray_intersection=CFG.rel_ray_intersection, max_proj=CFG.rel_max_proj,
        )
        return ba._score_relationships(
            [(r["source"], r["target"], r.get("bidirectional", False)) for r in rels], gt_arrows)["f1"]

    rows = []
    for d in detectors:
        for gate in (True, False):
            row = {"arrow_detector": d, "outside_box_gate": gate}
            for set_name, items in data.items():
                full = {n: rel_f1(it["arrows"][d], builder.label_boxes([dict(b) for b in it["boxes"]], it["texts"]),
                                  it["gt"], gate) for n, it in items.items()}
                stage = [rel_f1(it["arrows"][d], [dict(b) for b in it["gt_boxes"]], it["gt"], gate)
                         for it in items.values() if it["gt_boxes"]]
                if set_name == "benchmark":
                    row["full_pipeline_15"] = _macro(list(full.values()))
                    row["full_pipeline_synthetic_14"] = _macro([v for n, v in full.items() if n != REAL])
                    row["full_pipeline_real_1"] = full[REAL]
                    row["arrow_stage_gt_boxes_14"] = _macro(stage)
                else:
                    row[f"full_pipeline_{set_name}"] = _macro(list(full.values()))
                    row[f"arrow_stage_{set_name}"] = _macro(stage)
            rows.append(row)

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
