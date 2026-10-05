"""DiagraMine pipeline orchestrator.

Wires the Day-3 Phase-2 champions into one configurable, data-driven pipeline:

    text  -> EasyOCR           (champion: F1 0.949)
    box   -> Canny + contours  (champion: F1 0.808)
    arrow -> directed lines    (any-angle Hough, arrowhead-oriented)
    icon  -> template matching (champion: F1 1.000 on curated library)

Each stage is selectable via `PipelineConfig`, and "none" skips it. Every stage
is wrapped so that a failure degrades to an empty result rather than a crash — that is what keeps the
final `ExtractionResult` schema-valid 100% of the time (the reliability metric
DiagraMine beats Claude Vision on).

Usage:
    python -m src.pipeline <image_path> [--out DIR] [--text easyocr]
                           [--box canny_contours] [--arrow directed_lines]
                           [--icon template_matching] [--gate]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from typing import Callable, Dict

import cv2
import numpy as np

from src.graph import builder
from src.schemas import (
    ArrowElement,
    BoxElement,
    ExtractionResult,
    IconElement,
    PipelineConfig,
    Relationship,
    StageRuntimes,
    TextElement,
)


# ── Lazy detector registry ──────────────────────────────────────────────────
# Importing detector modules eagerly would load EasyOCR/CLIP/torch weights on
# every `import pipeline`. Resolve them lazily so callers only pay for the
# detectors they actually select.

def _skip(image_path: str) -> dict:
    """The "none" detector: a skipped stage yields no items (see `_safe`)."""
    return {}


def _text_detect(name: str) -> Callable[[str], dict]:
    if name == "none":
        return _skip
    if name == "easyocr":
        from src.text_detection import easyocr_detector as m
    elif name == "paddleocr":
        from src.text_detection import paddle_detector as m
    elif name == "tesseract":
        from src.text_detection import tesseract_detector as m
    else:
        raise ValueError(f"unknown text detector: {name}")
    return m.detect


def _box_detect(name: str) -> Callable[[str], dict]:
    if name == "none":
        return _skip
    if name == "canny_contours":
        from src.box_detection import canny_contours_detector as m
    elif name == "hough":
        from src.box_detection import hough_detector as m
    elif name == "yolo":
        from src.box_detection import yolo_detector as m
    else:
        raise ValueError(f"unknown box detector: {name}")
    return m.detect


def _arrow_detect(name: str) -> Callable[[str], dict]:
    if name == "none":
        return _skip
    if name == "directed_lines":
        from src.arrow_detection import directed_lines_detector as m
    elif name == "hough_lines":
        from src.arrow_detection import hough_lines_detector as m
    elif name == "pixel_scan":
        from src.arrow_detection import pixel_scan_detector as m
    elif name == "cnn":
        from src.arrow_detection import cnn_detector as m
    else:
        raise ValueError(f"unknown arrow detector: {name}")
    return m.detect


def _icon_detect(name: str) -> Callable[[str], dict]:
    if name == "none":
        return _skip
    if name == "template_matching":
        from src.icon_detection import template_detector as m
    elif name == "clip":
        from src.icon_detection import clip_detector as m
    elif name == "hsv":
        from src.icon_detection import hsv_detector as m
    else:
        raise ValueError(f"unknown icon detector: {name}")
    return m.detect


def _safe(fn: Callable[[str], dict], image_path: str, key: str) -> tuple[list, float, str | None]:
    """Run a detector, returning (items, runtime, error). On failure return an
    empty list so the overall result stays schema-valid."""
    try:
        out = fn(image_path)
        return out.get(key, []), float(out.get("runtime_seconds", 0.0)), None
    except Exception as e:  # noqa: BLE001 — reliability over strictness here
        return [], 0.0, f"{type(e).__name__}: {e}"


def extract(image_path: str, config: PipelineConfig | None = None) -> ExtractionResult:
    """Run the full pipeline on one image and return a validated ExtractionResult."""
    config = config or PipelineConfig()
    if not os.path.isfile(image_path):
        raise FileNotFoundError(image_path)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"cv2 could not read image: {image_path}")
    h_img, w_img = img.shape[:2]

    t0 = time.perf_counter()
    raw_texts, rt_text, err_text = _safe(_text_detect(config.text_detector), image_path, "texts")
    raw_boxes, rt_box, err_box = _safe(_box_detect(config.box_detector), image_path, "boxes")
    raw_arrows, rt_arrow, err_arrow = _safe(_arrow_detect(config.arrow_detector), image_path, "arrows")
    raw_icons, rt_icon, err_icon = _safe(_icon_detect(config.icon_detector), image_path, "icons")

    # Label boxes from contained text, then build relationships data-driven.
    g0 = time.perf_counter()
    raw_boxes = builder.label_boxes(raw_boxes, raw_texts)
    raw_rels = builder.build_relationships(
        raw_arrows, raw_boxes,
        outside_box_gate=config.outside_box_gate,
        max_dist=config.rel_max_dist,
        ray_intersection=config.rel_ray_intersection,
        max_proj=config.rel_max_proj,
    )
    rt_graph = time.perf_counter() - g0

    regions = [b for b in raw_boxes if b.get("is_region")]
    components = [b for b in raw_boxes if not b.get("is_region")]

    result = ExtractionResult(
        image=os.path.basename(image_path),
        width=int(w_img),
        height=int(h_img),
        detectors={
            "text": config.text_detector,
            "box": config.box_detector,
            "arrow": config.arrow_detector,
            "icon": config.icon_detector,
            "outside_box_gate": config.outside_box_gate,
            "rel_max_dist": config.rel_max_dist,
            "rel_ray_intersection": config.rel_ray_intersection,
            "errors": {k: v for k, v in {
                "text": err_text, "box": err_box,
                "arrow": err_arrow, "icon": err_icon,
            }.items() if v},
        },
        texts=[TextElement(**t) for t in raw_texts],
        boxes=[BoxElement(**b) for b in components],
        regions=[BoxElement(**b) for b in regions],
        arrows=[ArrowElement(**a) for a in raw_arrows],
        icons=[IconElement(**i) for i in raw_icons],
        relationships=[Relationship(**r) for r in raw_rels],
        runtimes=StageRuntimes(text=rt_text, box=rt_box, arrow=rt_arrow,
                               icon=rt_icon, graph=round(rt_graph, 3)),
        runtime_seconds=round(time.perf_counter() - t0, 3),
    )
    return result


# ── Rendering / output writers ───────────────────────────────────────────────

def annotate(image_path: str, result: ExtractionResult) -> np.ndarray:
    """Draw the detection overlay: regions=orange, boxes=green, text=blue,
    arrows=red, icons=magenta."""
    img = cv2.imread(image_path)
    for r in result.regions:
        cv2.rectangle(img, (r.x, r.y), (r.x + r.w, r.y + r.h), (0, 165, 255), 3)
    for b in result.boxes:
        cv2.rectangle(img, (b.x, b.y), (b.x + b.w, b.y + b.h), (0, 200, 0), 2)
        if b.label:
            cv2.putText(img, b.label[:30], (b.x + 3, b.y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 0), 1, cv2.LINE_AA)
    for t in result.texts:
        cv2.rectangle(img, (t.x, t.y), (t.x + t.w, t.y + t.h), (255, 100, 0), 1)
    for a in result.arrows:
        cv2.line(img, (a.x1, a.y1), (a.x2, a.y2), (0, 0, 255), 2)
        cv2.circle(img, (a.x1, a.y1), 5, (0, 0, 255), -1)
        cv2.circle(img, (a.x2, a.y2), 5, (0, 0, 255), -1)
    for ic in result.icons:
        cv2.rectangle(img, (ic.x, ic.y), (ic.x + ic.w, ic.y + ic.h), (255, 0, 255), 2)
        cv2.putText(img, ic.label[:20], (ic.x, ic.y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1, cv2.LINE_AA)
    return img


def write_csv(result: ExtractionResult, output_path: str) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["=== TEXT LABELS ==="])
        w.writerow(["Text", "X", "Y", "Width", "Height", "Confidence"])
        for t in result.texts:
            w.writerow([t.text, t.x, t.y, t.w, t.h, round(t.conf, 3)])
        w.writerow([])
        w.writerow(["=== BOXES ==="])
        w.writerow(["Label", "X", "Y", "Width", "Height", "Entity Type"])
        for b in result.boxes:
            w.writerow([b.label, b.x, b.y, b.w, b.h, b.entity_type])
        w.writerow([])
        w.writerow(["=== RELATIONSHIPS ==="])
        w.writerow(["Source", "Target", "Line Style", "Direction", "Relationship", "Detected"])
        for r in result.relationships:
            w.writerow([r.source, r.target, r.line_style, r.direction, r.relationship, r.detected])


def run(image_path: str, output_dir: str, config: PipelineConfig | None = None) -> ExtractionResult:
    """Full run: extract + write annotated PNG, relationship graph, JSON, CSV."""
    config = config or PipelineConfig()
    os.makedirs(output_dir, exist_ok=True)
    result = extract(image_path, config)
    stem = os.path.splitext(os.path.basename(image_path))[0]

    cv2.imwrite(os.path.join(output_dir, f"{stem}_annotated.png"), annotate(image_path, result))
    builder.draw_graph([r.model_dump() for r in result.relationships],
                       os.path.join(output_dir, f"{stem}_graph.png"),
                       layout=config.graph_layout)
    with open(os.path.join(output_dir, f"{stem}_structure.json"), "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, indent=2, ensure_ascii=False)
    write_csv(result, os.path.join(output_dir, f"{stem}_data.csv"))
    return result


def _build_config(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        text_detector=args.text,
        box_detector=args.box,
        arrow_detector=args.arrow,
        icon_detector=args.icon,
        outside_box_gate=args.gate,
        graph_layout=args.layout,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="DiagraMine modular extraction pipeline")
    p.add_argument("image", help="path to the diagram image")
    p.add_argument("--out", default=".", help="output directory")
    p.add_argument("--text", default="easyocr", choices=["easyocr", "paddleocr", "tesseract", "none"])
    p.add_argument("--box", default="canny_contours", choices=["canny_contours", "hough", "yolo", "none"])
    p.add_argument("--arrow", default="directed_lines",
                   choices=["directed_lines", "hough_lines", "pixel_scan", "cnn", "none"])
    p.add_argument("--icon", default="template_matching", choices=["template_matching", "clip", "hsv", "none"])
    p.add_argument("--gate", action="store_true", help="enable the outside-box arrow gate (off by default)")
    p.add_argument("--layout", default="kamada_kawai", choices=["kamada_kawai", "spring"])
    args = p.parse_args()

    result = run(args.image, args.out, _build_config(args))
    print(json.dumps(result.summary(), indent=2))
    print(f"[pipeline] detectors={result.detectors} runtime={result.runtime_seconds}s")
    print(f"[pipeline] outputs written to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
