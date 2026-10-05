"""Directed-lines arrow detector (any angle, arrowhead-oriented).

The Hough-lines detector this replaces has three gaps:

  * it drops every diagonal segment, but many connectors run diagonally
    between boxes;
  * it never looks for the arrowhead: every line is reported left to right or
    top to bottom, which is only right when the arrow happens to point that
    way (true of every axis-aligned arrow in the synthetic benchmark);
  * on the original (centre-to-centre) benchmark renders, a row of chained
    arrows A -> B -> C merged into one A -> C line.

Pipeline:
  1. Threshold, then `cv2.HoughLinesP` at all angles. `maxLineGap` bridges
     the gaps of dashed lines.
  2. Greedily merge near-collinear segments (angle, perpendicular offset and
     along-line gap all within tolerance) into whole connectors.
  3. Split each connector wherever a solid blob sits strictly inside it: an
     arrowhead mid-line means two chained arrows, not one.
  4. Orient: the end with more solid ink (what survives an erosion that wipes
     out 2-px lines) is the arrowhead, so it becomes `(x2, y2)` â€” the target.

Box borders are not masked here: a border segment has both endpoints on the
same box, which `graph.builder.build_relationships` already discards.
"""
from __future__ import annotations

import math
import time
from typing import List

import cv2
import numpy as np

BINARY_THRESHOLD = 200
HOUGH_THRESHOLD = 15
HOUGH_MIN_LENGTH = 20
HOUGH_MAX_GAP = 20          # bridges dash gaps; 10 px already breaks dashed lines
MERGE_ANGLE_TOL = 4.0       # degrees
MERGE_PERP_TOL = 6.0        # px off the line
MERGE_GAP = 30.0            # px along the line
MIN_ARROW_LENGTH = 40.0
HEAD_ERODE = 5              # erosion kernel that removes lines but keeps arrowheads
HEAD_RADIUS = 12
HEAD_BACK = 6               # look slightly behind the tip, where the head is widest
SPLIT_RADIUS = 5
SPLIT_MARGIN = 15
SPLIT_MIN_INK = 12


def _try_join(a: List[float], b: List[float]):
    """Merge segment b into a if they are near-collinear and close; else None."""
    la = math.hypot(a[2] - a[0], a[3] - a[1])
    lb = math.hypot(b[2] - b[0], b[3] - b[1])
    if la < lb:
        a, b, la, lb = b, a, lb, la
    if la < 1e-6:
        return None
    ax1, ay1 = a[0], a[1]
    ux, uy = (a[2] - ax1) / la, (a[3] - ay1) / la
    th_a = math.degrees(math.atan2(uy, ux)) % 180
    th_b = math.degrees(math.atan2(b[3] - b[1], b[2] - b[0])) % 180
    d = abs(th_a - th_b)
    if lb > 8 and min(d, 180 - d) > MERGE_ANGLE_TOL:
        return None
    for x, y in ((b[0], b[1]), (b[2], b[3])):
        if abs(-(x - ax1) * uy + (y - ay1) * ux) > MERGE_PERP_TOL:
            return None
    tb = [(b[0] - ax1) * ux + (b[1] - ay1) * uy, (b[2] - ax1) * ux + (b[3] - ay1) * uy]
    if max(min(tb) - la, -max(tb), 0.0) > MERGE_GAP:
        return None
    t0, t1 = min(0.0, *tb), max(la, *tb)
    return [ax1 + ux * t0, ay1 + uy * t0, ax1 + ux * t1, ay1 + uy * t1]


def _merge_collinear(segments: List[List[float]]) -> List[List[float]]:
    segs = [list(map(float, s)) for s in segments]
    changed = True
    while changed:
        changed = False
        out: List[List[float]] = []
        used = [False] * len(segs)
        for i in range(len(segs)):
            if used[i]:
                continue
            a = segs[i]
            for j in range(i + 1, len(segs)):
                if not used[j]:
                    m = _try_join(a, segs[j])
                    if m is not None:
                        a, used[j], changed = m, True, True
            out.append(a)
        segs = out
    return segs


def _split_at_heads(solid: np.ndarray, seg: List[float]) -> List[List[float]]:
    """Cut a connector at every solid blob strictly inside it, so chained
    arrows A -> B -> C come out as two lines instead of one A -> C."""
    x1, y1, x2, y2 = seg
    length = math.hypot(x2 - x1, y2 - y1)
    if length < 2 * MIN_ARROW_LENGTH:
        return [seg]
    ux, uy = (x2 - x1) / length, (y2 - y1) / length
    h, w = solid.shape
    r = SPLIT_RADIUS
    runs: List[List[float]] = []
    t = float(SPLIT_MARGIN)
    while t < length - SPLIT_MARGIN:
        cx, cy = int(round(x1 + ux * t)), int(round(y1 + uy * t))
        win = solid[max(cy - r, 0):min(cy + r + 1, h), max(cx - r, 0):min(cx + r + 1, w)]
        if np.count_nonzero(win) >= SPLIT_MIN_INK:
            if runs and t - runs[-1][1] <= 6:
                runs[-1][1] = t
            else:
                runs.append([t, t])
        t += 2
    cuts = [0.0] + [(a + b) / 2 for a, b in runs] + [length]
    pieces = [[x1 + ux * a, y1 + uy * a, x1 + ux * b, y1 + uy * b]
              for a, b in zip(cuts, cuts[1:]) if b - a >= MIN_ARROW_LENGTH]
    return pieces or [seg]


def _head_ink(solid: np.ndarray, ex: float, ey: float, ox: float, oy: float) -> int:
    """Solid ink in a disc just behind end (ex, ey); (ox, oy) is the other end."""
    length = math.hypot(ex - ox, ey - oy) or 1.0
    cx = ex - (ex - ox) / length * HEAD_BACK
    cy = ey - (ey - oy) / length * HEAD_BACK
    r = HEAD_RADIUS
    h, w = solid.shape
    x0, x1 = max(int(cx - r), 0), min(int(cx + r) + 1, w)
    y0, y1 = max(int(cy - r), 0), min(int(cy + r) + 1, h)
    if x0 >= x1 or y0 >= y1:
        return 0
    yy, xx = np.mgrid[y0:y1, x0:x1]
    disc = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    return int(np.count_nonzero(solid[y0:y1, x0:x1][disc]))


def detect(image_path: str) -> dict:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    start = time.perf_counter()

    _, inv = cv2.threshold(gray, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    # No thinning: it shrinks short dashes to a few pixels and HoughLinesP then
    # returns fragments. The parallel duplicates a 2-px stroke yields are folded
    # together by the collinear merge below.
    lines = cv2.HoughLinesP(inv, rho=1, theta=np.pi / 180, threshold=HOUGH_THRESHOLD,
                            minLineLength=HOUGH_MIN_LENGTH, maxLineGap=HOUGH_MAX_GAP)
    segments = [line[0] for line in lines] if lines is not None else []

    solid = cv2.erode(inv, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (HEAD_ERODE, HEAD_ERODE)))
    connectors = [piece for seg in _merge_collinear(segments)
                  for piece in _split_at_heads(solid, seg)]

    out_arrows = []
    for x1, y1, x2, y2 in connectors:
        if math.hypot(x2 - x1, y2 - y1) < MIN_ARROW_LENGTH:
            continue
        # The arrowhead end becomes (x2, y2): build_relationships reads x1 -> x2
        # as source -> target.
        if _head_ink(solid, x1, y1, x2, y2) > _head_ink(solid, x2, y2, x1, y1):
            x1, y1, x2, y2 = x2, y2, x1, y1
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        direction = "horizontal" if dy <= 3 else "vertical" if dx <= 3 else "diagonal"
        out_arrows.append({"x1": int(round(x1)), "y1": int(round(y1)),
                           "x2": int(round(x2)), "y2": int(round(y2)),
                           "direction": direction})

    elapsed = time.perf_counter() - start
    return {"arrows": out_arrows, "runtime_seconds": round(elapsed, 3)}
