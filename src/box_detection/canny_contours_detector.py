"""Canny + RETR_TREE contour box detection — equivalent to the current
diagram_analysis.detect_boxes_and_regions but stripped down to just the
box-detection portion (no text labeling, no region/parent nesting). Returns
the same (x, y, w, h) the legacy pipeline would call a 'box'."""
from __future__ import annotations

import time
from typing import List

import cv2


def _dedup_boxes(boxes: List[dict], iou_thresh: float = 0.6) -> List[dict]:
    keep = []
    used = set()
    order = sorted(range(len(boxes)), key=lambda i: boxes[i]["w"] * boxes[i]["h"], reverse=True)
    for i in order:
        if i in used:
            continue
        keep.append(boxes[i])
        for j in order:
            if j in used or j == i:
                continue
            a, b = boxes[i], boxes[j]
            ix1 = max(a["x"], b["x"])
            iy1 = max(a["y"], b["y"])
            ix2 = min(a["x"] + a["w"], b["x"] + b["w"])
            iy2 = min(a["y"] + a["h"], b["y"] + b["h"])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = a["w"] * a["h"] + b["w"] * b["h"] - inter
            if union > 0 and inter / union > iou_thresh:
                used.add(j)
        used.add(i)
    return keep


# Champion config (Day-5 Optuna-tuned; the tuning script has since been
# retired). The defaults below are the live parameters the pipeline uses; the
# constants are documented so the tuning provenance is auditable.
#
# Day-5 Phase-4 sweep (40 TPE trials, IoU>=0.5 micro-F1 over the 14 generated
# diagrams): the original hardcoded values [canny 30/120, min_area 1200,
# dilate 3x3, rect 0.3, dedup 0.6] scored F1=0.808 (P=0.709, R=0.938) — the
# loss was precision, RETR_TREE nested/duplicate contours surviving dedup. The
# tuned config below raises min_area and the rectangularity floor (killing the
# tiny nested contours) and tightens dedup, scoring F1=0.992 (P=1.000,
# R=0.985), fp 25 -> 0. NOTE: tuned and evaluated on the same synthetic family,
# so 0.992 is an in-sample optimum; on the real search_interview_test diagram
# it still drops box false positives 19 -> 16 (recall preserved).
# `region_area` is NOT a Canny knob — it is the component/region split. The
# Day-5 error analysis found the original 25000 threshold mislabeled 27 genuine
# component boxes as "regions" (their detected area drifts above 25k once Canny
# + dilation pad the contour), pulling them out of the relationship candidate
# pool and the schema's `boxes` field. Raising it to 60000 keeps real component
# boxes as components while still flagging true large containers (e.g. the
# search_interview platform box at ~204k/105k/82k). This single change lifted
# relationship micro-F1 0.171 -> 0.250 at the time.
DEFAULTS = dict(
    canny_low=39,
    canny_high=69,
    min_area=2300,
    min_w=40,
    min_h=20,
    dilate_ksize=2,
    dilate_iters=1,
    rectangularity_min=0.6,
    dedup_iou=0.5,
    region_area=60000,
)


def detect(
    image_path: str,
    *,
    canny_low: int = DEFAULTS["canny_low"],
    canny_high: int = DEFAULTS["canny_high"],
    min_area: int = DEFAULTS["min_area"],
    min_w: int = DEFAULTS["min_w"],
    min_h: int = DEFAULTS["min_h"],
    dilate_ksize: int = DEFAULTS["dilate_ksize"],
    dilate_iters: int = DEFAULTS["dilate_iters"],
    rectangularity_min: float = DEFAULTS["rectangularity_min"],
    dedup_iou: float = DEFAULTS["dedup_iou"],
    region_area: int = DEFAULTS["region_area"],
) -> dict:
    """Canny + RETR_TREE contour box detection.

    All thresholds are keyword-tunable (Day-5 Phase-4 sweep) with defaults
    equal to the original hardcoded values, so existing callers — the pipeline
    and the Phase-2a harness — are byte-for-byte unaffected when called as
    ``detect(path)``.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    h_img, w_img = img.shape[:2]

    start = time.perf_counter()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)
    ksize = max(1, int(dilate_ksize))
    if ksize > 1 and dilate_iters > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        dilated = cv2.dilate(edges, kernel, iterations=int(dilate_iters))
    else:
        dilated = edges
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    boxes: List[dict] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < min_area or w < min_w or h < min_h:
            continue
        rect_area = w * h
        if rect_area == 0:
            continue
        rectangularity = cv2.contourArea(cnt) / rect_area
        if rectangularity < rectangularity_min:
            continue
        if w > w_img * 0.95 and h > h_img * 0.95:
            continue
        # The legacy split: REGIONS (area > region_area) vs BOXES.
        # For Phase 2a evaluation we want all rectangular objects, so we
        # keep both, but flag them.
        boxes.append(
            {
                "x": int(x), "y": int(y),
                "w": int(w), "h": int(h),
                "area": int(area),
                "is_region": area > region_area,
            }
        )

    boxes = _dedup_boxes(boxes, iou_thresh=dedup_iou)
    elapsed = time.perf_counter() - start
    return {"boxes": boxes, "runtime_seconds": round(elapsed, 3)}
