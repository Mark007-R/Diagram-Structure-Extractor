"""Icon nodes found from their captions.

draw.io and cloud-architecture diagrams draw many components as an icon (a
database cylinder, a service logo) with the caption underneath, not as a
labelled box. Contour box detection often misses them: a connector touching
the icon merges with its outline. So look for them from the caption side: a
text that sits outside every component box, with a compact graphic directly
above it, is an icon node.

Line labels ("SFTP", "REST") and container titles don't qualify: above them
is empty space, a 1-px line, or a long border, which the size and aspect
checks reject.
"""
from __future__ import annotations

from typing import List

import cv2
import numpy as np

INK_THRESHOLD = 200       # grey level below which a pixel is ink
SATURATION_MIN = 60       # or a clearly coloured pixel (blue icons are often light)
MAX_ICON = 120            # px; largest icon side considered
MIN_ICON = 14             # px; smallest


def _inside(t: dict, b: dict) -> bool:
    cx, cy = t["x"] + t["w"] / 2, t["y"] + t["h"] / 2
    return b["x"] <= cx <= b["x"] + b["w"] and b["y"] <= cy <= b["y"] + b["h"]


def _next_line(t: dict, texts: List[dict]) -> dict | None:
    """The caption line directly under `t` (a two-line caption), if any."""
    cx = t["x"] + t["w"] / 2
    for o in texts:
        gap = o["y"] - (t["y"] + t["h"])
        if (o is not t and o.get("text", "").strip() and -2 <= gap <= 0.6 * t["h"]
                and abs(o["x"] + o["w"] / 2 - cx) <= max(t["w"], o["w"]) / 2):
            return o
    return None


def _clear_glyphs(ink: np.ndarray, texts: List[dict]) -> None:
    """Remove the letters of every OCR'd text from `ink`: a shape that lies
    entirely within a text box (grown by 2 px). Letters are never part of an
    icon -- without this, the first line of a two-line caption would pass for
    an icon above the second -- but an icon outline that a padded OCR box
    overlaps extends beyond it and stays."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
    for t in texts:
        x0, y0, x1, y1 = t["x"] - 2, t["y"] - 2, t["x"] + t["w"] + 2, t["y"] + t["h"] + 2
        inside = ((st[:, 0] >= x0) & (st[:, 1] >= y0) & (st[:, 0] + st[:, 2] <= x1)
                  & (st[:, 1] + st[:, 3] <= y1))
        inside[0] = False
        if inside.any():
            sl = (slice(max(0, y0), max(0, y1)), slice(max(0, x0), max(0, x1)))
            ink[sl] &= ~np.isin(lab[sl], np.nonzero(inside)[0])


def find(image_bgr: np.ndarray, texts: List[dict], boxes: List[dict]) -> List[dict]:
    """Return new component boxes for icon nodes, labelled with their caption.
    Each records the caption lines it used in `caption_boxes`, so the graph
    builder doesn't hand them to another box."""
    gray = image_bgr.mean(axis=2)
    sat = image_bgr.max(axis=2).astype(np.int16) - image_bgr.min(axis=2).astype(np.int16)
    ink = (gray < INK_THRESHOLD) | (sat > SATURATION_MIN)
    _clear_glyphs(ink, texts)
    h_img, w_img = gray.shape
    components = [b for b in boxes if not b.get("is_region")]
    found: List[dict] = []
    for t in texts:
        if not t.get("text", "").strip() or any(_inside(t, b) for b in components):
            continue
        cx = t["x"] + t["w"] / 2
        # A caption right under an existing box is that box's caption.
        if any(0 <= t["y"] - (b["y"] + b["h"]) <= 14 and b["x"] <= cx <= b["x"] + b["w"]
               for b in components):
            continue
        half = max(t["w"], 40) * 0.8
        x0, x1 = int(max(0, cx - half)), int(min(w_img, cx + half))
        y_top = int(max(0, t["y"] - MAX_ICON))
        y_end = int(t["y"]) - 1
        if y_end <= y_top:                  # caption at the top edge: nothing above it
            continue
        window = ink[y_top:y_end, x0:x1].astype(np.uint8)
        if window.size == 0:
            continue
        # The icon is made of the connected shapes whose bottom sits just above
        # the caption and which overlap its centre column; thin long shapes
        # are connector lines passing by, not part of the icon.
        n, lab, st, _ = cv2.connectedComponentsWithStats(window, 8)
        bottom_row, ccol = window.shape[0] - 1, cx - x0
        parts = [i for i in range(1, n)
                 if bottom_row - (st[i, cv2.CC_STAT_TOP] + st[i, cv2.CC_STAT_HEIGHT] - 1) <= 12
                 and not (min(st[i, 2], st[i, 3]) <= 3 and max(st[i, 2], st[i, 3]) > 20)
                 and st[i, 0] - 0.5 * MIN_ICON <= ccol <= st[i, 0] + st[i, 2] + 0.5 * MIN_ICON]
        if not parts:
            continue
        bx0 = x0 + int(min(st[i, 0] for i in parts))
        bx1 = x0 + int(max(st[i, 0] + st[i, 2] for i in parts)) - 1
        by0 = y_top + int(min(st[i, 1] for i in parts))
        by1 = y_top + int(max(st[i, 1] + st[i, 3] for i in parts)) - 1
        w, h = bx1 - bx0 + 1, by1 - by0 + 1
        if not (MIN_ICON <= w <= MAX_ICON and MIN_ICON <= h <= MAX_ICON and 0.4 <= w / h <= 2.5):
            continue
        fill = sum(int(st[i, cv2.CC_STAT_AREA]) for i in parts) / float(w * h)
        if fill < 0.08:            # a sparse frame or line crossing, not an icon
            continue
        lines = [t]
        second = _next_line(t, texts)
        if second is not None:
            lines.append(second)
        box = {"x": bx0, "y": by0, "w": w, "h": h, "area": w * h, "is_region": False,
               "label": " ".join(o["text"].strip() for o in lines), "entity_type": "icon",
               "caption_boxes": [(o["x"], o["y"], o["w"], o["h"]) for o in lines]}

        if any(min(b["x"] + b["w"], bx1) - max(b["x"], bx0) > 0.5 * w and
               min(b["y"] + b["h"], by1) - max(b["y"], by0) > 0.5 * h for b in components):
            continue                # already a box there
        clash = [b for b in found
                 if min(b["x"] + b["w"], bx1 + 1) - max(b["x"], bx0) > 0.5 * min(w, b["w"])
                 and min(b["y"] + b["h"], by1 + 1) - max(b["y"], by0) > 0.5 * min(h, b["h"])]
        # Two captions found the same icon: keep the box that contains the
        # other -- a text inside the icon only sees the part above it.
        if clash and all(w * h > b["w"] * b["h"] for b in clash):
            found = [b for b in found if b not in clash]
        elif clash:
            continue
        found.append(box)
    return found
