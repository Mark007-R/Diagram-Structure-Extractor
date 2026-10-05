"""Directed-lines arrow detector (any angle, elbows, arrowhead-oriented).

Finds connectors and the arrowheads on them, in both clean renders and real
draw.io exports (thin 1-px dashed lines, small filled heads, orthogonal elbow
routing, double-headed connectors, pale rounded containers).

Pipeline:
  1. Ink mask, and the typical stroke width (median skeleton thickness, over
     stroke-sized values only).
  2. Drop ink that can never be a connector: compact blobs (letters, judged by
     their tightest rotated rectangle so diagonal dashes survive) and the
     solid cores of filled shapes (logos, icons, filled boxes) -- not the
     connectors drawn up to them.
  3. Bridge dash gaps along each component's own axis, then `cv2.HoughLinesP`
     at all angles.
  4. Drop segments that run along a closed outline (box borders, containers,
     icons). Outlines are found on the ink plus faint anti-aliased pixels, so
     the 1-px rounded corners of pale containers still close; a connector that
     merely crosses a border keeps its other pixels and survives.
  5. Merge near-collinear segments (and across a line label, never across a
     box outline), split where an arrowhead or filled icon sits mid-line
     (chained arrows A -> B -> C, or two connectors meeting at an icon), and
     join segments meeting at a right angle into elbow polylines -- never at
     an end carrying a head.
  6. Arrowheads are small solid blobs that survive an erosion scaled to the
     width of the line they end. A head pointing back into a line is not its
     head, and each head ends one connector. An end with a head is extended
     to the head's tip and becomes the target; two convincing heads make the
     connector bidirectional.

Output arrows carry `points` (tail -> head polyline), `has_head` and
`bidirectional`.
"""
from __future__ import annotations

import math
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

BINARY_THRESHOLD = 200
HOUGH_THRESHOLD = 15
HOUGH_MIN_LENGTH = 12
HOUGH_MAX_GAP = 20          # bridges dash gaps; 10 px already breaks dashed lines
DASH_BRIDGE = 11            # axis-aligned closing that turns 1-px dashes into lines
MERGE_ANGLE_TOL = 4.0       # degrees
MERGE_PERP_TOL = 6.0        # px off the line
MERGE_GAP = 30.0            # px along the line
LABEL_GAP = 100.0           # px; widest gap a line label can cut into a connector
MIN_ARROW_LENGTH = 40.0
MIN_HEADED_LENGTH = 24.0    # a connector with an arrowhead may be this short
GLYPH_MAX = 40              # px; compact components up to this size are letters
BLOB_FILL = 0.3             # bbox fill above which a thick component is a solid shape
FAINT_DELTA = 12            # grey levels darker than the local median = faint line ink
MIN_OUTLINE_AREA = 150      # px^2; smaller enclosed regions are letter counters
OUTLINE_FRACTION = 0.6      # a segment this much inside an outline band is a border
OUTLINE_CONVEXITY = 0.9     # region area / convex hull area for a shape interior
OUTLINE_RECTANGULARITY = 0.75  # region area / rotated bounding rectangle area
                               # (an ellipse, e.g. a database icon, fills pi/4)
ELBOW_JOIN = 28.0           # px between segment ends at a rounded elbow corner
ELBOW_MIN_ANGLE = 55.0      # degrees between the two legs
SPLIT_RADIUS = 5
SPLIT_MARGIN = 15
SPLIT_MIN_INK = 12
MAX_STROKE = 15             # px; thicker "strokes" are the medial axes of filled shapes
TIP_SKEW = 0.2              # head-blob skewness beyond which its direction is clear
END_SKIP = 12.0             # px from each end left out when measuring a line's width

Point = Tuple[float, float]


def _distance(inv: np.ndarray) -> np.ndarray:
    """Distance to the nearest background pixel. Padded with background, so
    an image with no pixel brighter than the threshold stays finite."""
    padded = cv2.copyMakeBorder(inv, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    return cv2.distanceTransform(padded, cv2.DIST_L2, 3)[1:-1, 1:-1]


def _stroke_width(inv: np.ndarray, dt: np.ndarray) -> int:
    """Typical line thickness: median of 2*distance-1 at skeleton pixels. The
    medial axis of a filled shape is not a stroke, so values above
    MAX_STROKE are left out while any stroke-sized ones remain."""
    values = 2 * dt[cv2.ximgproc.thinning(inv) > 0].astype(np.float64) - 1
    strokes = values[values <= MAX_STROKE]
    values = strokes if strokes.size else values
    return max(1, min(MAX_STROKE, int(round(float(np.median(values)))))) if values.size else 1


def _line_width(dt: np.ndarray, pts: List[Point], default: int) -> int:
    """Thickness of one connector, sampled along its legs away from both
    ends (where the arrowheads are). Boxes and filled shapes can make the
    image-wide stroke width a poor guide to a single line. Each sample takes
    the deepest point within 3 px across the leg: a Hough segment can run
    along a thick stroke's edge rather than its centre."""
    total = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))
    skip = min(END_SKIP, total / 4.0)          # short lines still get samples
    h, w = dt.shape
    across = np.arange(-3, 4)
    vals, run = [], 0.0
    for a, b in zip(pts, pts[1:]):
        seg = math.hypot(b[0] - a[0], b[1] - a[1])
        if seg < 1e-6:
            continue
        nx, ny = -(b[1] - a[1]) / seg, (b[0] - a[0]) / seg
        for t in np.arange(0.0, seg, 3.0):
            pos = run + t
            if skip <= pos <= total - skip:
                cx = a[0] + (b[0] - a[0]) * t / seg
                cy = a[1] + (b[1] - a[1]) * t / seg
                xs = np.clip(np.round(cx + nx * across).astype(int), 0, w - 1)
                ys = np.clip(np.round(cy + ny * across).astype(int), 0, h - 1)
                deepest = float(dt[ys, xs].max())
                if deepest > 0:
                    vals.append(2 * deepest - 1)
        run += seg
    if len(vals) < 3:
        return default
    med = float(np.median(vals))
    # Over a filled shape the samples measure the shape, not the line: they
    # run past any stroke width, or disagree with each other.
    if med > MAX_STROKE or np.mean(np.abs(np.array(vals) - med) <= 1.0) < 0.6:
        return default
    return max(1, int(round(med)))


def _connector_ink(inv: np.ndarray, sw: int) -> Tuple[np.ndarray, np.ndarray]:
    """Ink minus letters and solid shapes, which are never connectors.
    Returns (clean, cores): `cores` is the solid interior of filled shapes
    found with an opening wider than any arrowhead, so a head touching a
    filled box stays outside it; heads are looked for on `inv - cores`."""
    clean = inv.copy()
    cores = np.zeros_like(inv)
    n, lab, st, _ = cv2.connectedComponentsWithStats(clean, 8)
    thin = sw + 1
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if max(w, h) <= GLYPH_MAX and min(w, h) > thin:
            # Compact by its axis-aligned box -- but so is a diagonal dash.
            # Judge thickness from the tightest rotated rectangle instead.
            pts = cv2.findNonZero((lab[y:y + h, x:x + w] == i).astype(np.uint8))
            (_, _), (rw, rh), _ = cv2.minAreaRect(pts)
            if min(rw, rh) > thin + 1:
                clean[lab == i] = 0

    # Solid shapes (filled boxes, icons, logos) are what survives an opening
    # wider than any connector. Judged piece by piece, not by connected
    # component: connectors touching a filled icon and the outlined boxes
    # around it make one sparse component. Each shape is grown back a few px
    # within the stroke-sized opening (recovering its rounded corners, not a
    # line it touches) and only that is erased, so a connector drawn up to a
    # filled box survives.
    ellipse = cv2.getStructuringElement
    open_stroke = ellipse(cv2.MORPH_ELLIPSE, (2 * thin + 1, 2 * thin + 1))
    open_solid = ellipse(cv2.MORPH_ELLIPSE, (4 * thin + 1, 4 * thin + 1))
    head_max = 6 * sw + 14                 # the largest blob _head_blobs accepts
    open_wide = ellipse(cv2.MORPH_ELLIPSE, (head_max + 1, head_max + 1))
    margin, step = np.ones((5, 5), np.uint8), np.ones((3, 3), np.uint8)
    no_strokes = cv2.morphologyEx(inv, cv2.MORPH_OPEN, open_stroke)
    solids = cv2.morphologyEx(inv, cv2.MORPH_OPEN, open_solid)
    m, plab, pst, _ = cv2.connectedComponentsWithStats(solids, 8)
    rows, cols = inv.shape
    pad = 2 * thin + 3
    for j in range(1, m):
        x, y, w, h, area = pst[j]
        if min(w, h) <= 3 * thin or area / float(w * h) <= BLOB_FILL:
            continue
        x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(cols, x + w + pad), min(rows, y + h + pad)
        piece = (plab[y0:y1, x0:x1] == j).astype(np.uint8)
        core, allowed = piece, no_strokes[y0:y1, x0:x1] > 0
        for _ in range(2 * thin):
            core = (cv2.dilate(core, step) > 0).astype(np.uint8) & allowed
        ink = inv[y0:y1, x0:x1] > 0
        clean[y0:y1, x0:x1][(cv2.dilate(core, margin) > 0) & ink] = 0
        wide = cv2.morphologyEx(piece, cv2.MORPH_OPEN, open_wide)
        cores[y0:y1, x0:x1][(cv2.dilate(wide, margin) > 0) & ink] = 255
    return clean, cores


def _outline_band(gray: np.ndarray, inv: np.ndarray, sw: int, strict: bool = True) -> np.ndarray:
    """A band along closed outlines: box borders, containers, icons. With
    `strict`, only around convex, rectangle-like regions (shape interiors);
    otherwise around every enclosed region, which also catches outlines that
    other shapes are drawn across -- and solid connectors that close a region."""
    med = cv2.medianBlur(gray, 9)
    faint = ((med.astype(np.int16) - gray.astype(np.int16)) > FAINT_DELTA).astype(np.uint8) * 255
    # A 1-px dilation closes the 1-2 px gaps where an outline meets a shape
    # drawn over it; dash gaps (about 4 px and up) stay open.
    lines = cv2.dilate(cv2.bitwise_or(inv, faint), np.ones((3, 3), np.uint8))
    band = np.zeros_like(inv)
    cnts, hier = cv2.findContours(lines, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hier is not None:
        thickness = 2 * sw + 5
        for c, h in zip(cnts, hier[0]):
            area = cv2.contourArea(c)
            if h[3] == -1 or area < MIN_OUTLINE_AREA:
                continue
            # Box, container and icon interiors are convex and rectangle-like.
            # A region closed off by a solid connector together with the boxes
            # and borders it touches usually is not -- keep that connector.
            (_, _), (rw, rh), _ = cv2.minAreaRect(c)
            hull = cv2.contourArea(cv2.convexHull(c))
            shape_like = (hull > 0 and area / hull >= OUTLINE_CONVEXITY
                          and rw * rh > 0 and area / (rw * rh) >= OUTLINE_RECTANGULARITY)
            if shape_like or not strict:
                cv2.drawContours(band, [c], -1, 255, thickness)
    return band


def _fraction_on(seg, band: np.ndarray) -> float:
    x1, y1, x2, y2 = seg
    n = max(2, int(math.hypot(x2 - x1, y2 - y1)))
    xs = np.clip(np.round(np.linspace(x1, x2, n)).astype(int), 0, band.shape[1] - 1)
    ys = np.clip(np.round(np.linspace(y1, y2, n)).astype(int), 0, band.shape[0] - 1)
    return float(np.count_nonzero(band[ys, xs])) / n


def _bridge_dashes(clean: np.ndarray) -> np.ndarray:
    """Close dash gaps along each component's own axis. HoughLinesP samples
    thin 1-px dashes unreliably; bridging both ways on everything would also
    fill the gap between two stacked horizontal borders."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(clean, 8)
    horiz = np.where(st[:, cv2.CC_STAT_WIDTH] >= st[:, cv2.CC_STAT_HEIGHT])[0]
    is_h = np.isin(lab, horiz[horiz > 0])
    h_part = np.where(is_h, clean, 0).astype(np.uint8)
    v_part = np.where(~is_h, clean, 0).astype(np.uint8)
    rect = cv2.getStructuringElement
    bridged = cv2.bitwise_or(
        cv2.morphologyEx(h_part, cv2.MORPH_CLOSE, rect(cv2.MORPH_RECT, (DASH_BRIDGE, 1))),
        cv2.morphologyEx(v_part, cv2.MORPH_CLOSE, rect(cv2.MORPH_RECT, (1, DASH_BRIDGE))))
    return cv2.bitwise_or(clean, bridged)


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


def _bridge_line_labels(segs: List[List[float]], glyphs: np.ndarray, band: np.ndarray) -> List[List[float]]:
    """Join two collinear pieces of one connector separated by a line label:
    draw.io paints edge labels ("VPN", "HTTPS") on a white background that
    cuts the line, often by more than MERGE_GAP. Joined only when the gap
    holds letter ink and crosses no shape outline -- the label of a small box
    sitting between two connectors is not a line label."""
    segs = [list(map(float, s)) for s in segs]
    changed = True
    while changed:
        changed = False
        for i in range(len(segs)):
            for j in range(len(segs)):
                if i == j:
                    continue
                a, b = segs[i], segs[j]
                la = math.hypot(a[2] - a[0], a[3] - a[1])
                if la < 1e-6:
                    continue
                ux, uy = (a[2] - a[0]) / la, (a[3] - a[1]) / la
                da = abs(_angle(a) - _angle(b))
                if min(da, 180 - da) > MERGE_ANGLE_TOL:
                    continue
                if any(abs(-(x - a[0]) * uy + (y - a[1]) * ux) > MERGE_PERP_TOL for x, y in ((b[0], b[1]), (b[2], b[3]))):
                    continue
                tb = sorted([(b[0] - a[0]) * ux + (b[1] - a[1]) * uy, (b[2] - a[0]) * ux + (b[3] - a[1]) * uy])
                gap = tb[0] - la
                if not MERGE_GAP < gap <= LABEL_GAP:
                    continue
                n = int(gap)
                xs = np.clip(np.round(a[2] + ux * np.arange(n)).astype(int), 0, glyphs.shape[1] - 1)
                ys = np.clip(np.round(a[3] + uy * np.arange(n)).astype(int), 0, glyphs.shape[0] - 1)
                ink = sum(int(np.count_nonzero(glyphs[max(0, y - 6):y + 7, max(0, x - 6):x + 7] > 0)) > 0
                          for x, y in zip(xs, ys))
                crosses_outline = np.count_nonzero(band[ys, xs]) >= 3
                if ink >= 0.4 * n and not crosses_outline:   # letters along most of the gap
                    segs[i] = [a[0], a[1], a[0] + ux * tb[1], a[1] + uy * tb[1]]
                    del segs[j]
                    changed = True
                    break
            if changed:
                break
    return segs


def _split_at_heads(solid: np.ndarray, seg: List[float]) -> List[List[float]]:
    """Cut a connector at every solid blob strictly inside it (a head, or a
    filled icon between two connectors), so chained arrows A -> B -> C come
    out as two lines instead of one A -> C. A solid run along most of the
    line is the line itself, thicker than the erosion, and is not cut."""
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
    cuts = [0.0] + [(a + b) / 2 for a, b in runs if b - a < 0.5 * length] + [length]
    pieces = [[x1 + ux * a, y1 + uy * a, x1 + ux * b, y1 + uy * b]
              for a, b in zip(cuts, cuts[1:]) if b - a >= MIN_ARROW_LENGTH]
    return pieces or [seg]


def _split_at_head_blobs(seg: List[float], heads) -> List[List[float]]:
    """Cut a connector wherever a detected arrowhead lies on its interior: two
    collinear connectors meeting at a node (or crossing an icon the box stage
    did not see) otherwise merge into one line. Complements _split_at_heads,
    whose fixed ink threshold misses the small heads of 1-px draw.io lines."""
    x1, y1, x2, y2 = seg
    length = math.hypot(x2 - x1, y2 - y1)
    if length < 2 * MIN_ARROW_LENGTH:
        return [seg]
    ux, uy = (x2 - x1) / length, (y2 - y1) / length
    cuts = []
    for hd in heads:
        hx, hy, hr = hd[:3]
        along = (hx - x1) * ux + (hy - y1) * uy
        perp = abs(-(hx - x1) * uy + (hy - y1) * ux)
        if perp <= hr + 2 and SPLIT_MARGIN < along < length - SPLIT_MARGIN:
            cuts.append(along)
    if not cuts:
        return [seg]
    bounds = [0.0] + sorted(cuts) + [length]
    pieces = [[x1 + ux * a, y1 + uy * a, x1 + ux * b, y1 + uy * b]
              for a, b in zip(bounds, bounds[1:]) if b - a >= 12]
    return pieces or [seg]


def _angle(s) -> float:
    return math.degrees(math.atan2(s[3] - s[1], s[2] - s[0])) % 180


def _join_elbows(segs: List[List[float]], has_head=lambda i, e: False) -> List[List[Point]]:
    """Join segment ends that meet at a corner (roughly perpendicular legs)
    into polylines; every segment ends up in exactly one polyline. An end
    with an arrowhead is a tip, not a corner: arrows converging on a small
    node or a box corner stay two arrows."""
    ends = [(i, e, s[2 * e], s[2 * e + 1]) for i, s in enumerate(segs) for e in (0, 1)]
    ends = [t for t in ends if not has_head(t[0], t[1])]
    pairs = []
    for a in range(len(ends)):
        for b in range(a + 1, len(ends)):
            i, ei, x1, y1 = ends[a]
            j, ej, x2, y2 = ends[b]
            if i == j:
                continue
            d = math.hypot(x1 - x2, y1 - y2)
            da = abs(_angle(segs[i]) - _angle(segs[j]))
            if d <= ELBOW_JOIN and min(da, 180 - da) >= ELBOW_MIN_ANGLE:
                pairs.append((d, (i, ei), (j, ej)))
    link = {}
    for _, A, B in sorted(pairs):
        if A not in link and B not in link:
            link[A], link[B] = B, A
    seen, paths = set(), []
    for start in range(len(segs)):
        if start in seen:
            continue
        cur, end, visited = start, 0, {start}          # walk back to a free end
        while (cur, end) in link and link[(cur, end)][0] not in visited:
            cur, e2 = link[(cur, end)]
            visited.add(cur)
            end = 1 - e2
        pts: List[Point] = []
        visited = set()
        while True:                                     # walk forward from it
            visited.add(cur)
            s = segs[cur]
            a, b = (s[0], s[1]), (s[2], s[3])
            if end == 1:
                a, b = b, a
            if not pts:
                pts.append(a)
            pts.append(b)
            nxt = link.get((cur, 1 - end))
            if nxt is None or nxt[0] in visited:
                break
            cur, end = nxt
        seen |= visited
        paths.append(pts)
    return paths


def _head_blobs(solid: np.ndarray, sw: int) -> List[tuple]:
    """Arrowhead candidates: small solid blobs that survive an erosion which
    removes lines and letter strokes. Returns (cx, cy, radius, moments), the
    blob's second and third central moments, from which _tip_skew reads the
    way it points."""
    n, lab, st, cen = cv2.connectedComponentsWithStats(solid, 8)
    max_side = 6 * sw + 14
    min_side = max(3, sw + 2)
    out = []
    for i in range(1, n):
        x, y, w, h, area = st[i]
        # A head is roughly as wide as long; letter strokes that survive the
        # erosion are tall and thin, and a line crossing a border leaves a
        # 1-2 px speck.
        if (3 <= area <= 40 * sw * sw + 120 and min_side <= min(w, h) and max(w, h) <= max_side
                and min(w, h) >= 0.45 * max(w, h)):
            m = cv2.moments((lab[y:y + h, x:x + w] == i).astype(np.uint8), binaryImage=True)
            moments = (m["m00"], m["mu20"], m["mu11"], m["mu02"], m["mu30"], m["mu21"], m["mu12"], m["mu03"])
            out.append((float(cen[i][0]), float(cen[i][1]), max(w, h) / 2.0 + sw, moments))
    return out


def _tip_skew(head, ux: float, uy: float) -> float:
    """Skewness of the head blob's pixels along (ux, uy). A triangle's pixels
    pile up at its base and thin out towards its tip, so the skew is positive
    when the tip points along (ux, uy), negative when it points back, and
    near zero for a symmetric blob (a diamond, a dot)."""
    m00, mu20, mu11, mu02, mu30, mu21, mu12, mu03 = head[3]
    var = (mu20 * ux * ux + 2 * mu11 * ux * uy + mu02 * uy * uy) / m00
    if m00 <= 0 or var <= 1e-6:
        return 0.0
    third = (mu30 * ux ** 3 + 3 * mu21 * ux * ux * uy + 3 * mu12 * ux * uy * uy + mu03 * uy ** 3) / m00
    return third / var ** 1.5


def _head_at(heads, end: Point, prev: Point, reach: float) -> Optional[Tuple[float, Point, tuple]]:
    """Nearest head blob within `reach` of this end (not behind it, and not
    pointing back into the line): returns (distance, tip, blob). The tip is
    the far side of the blob along the line -- an arrow ends at its tip, not
    at the base of its head.

    A head whose tip points back into the line belongs to another connector:
    A -> icon -> C merges into one line that is cut at A's head, and the
    piece towards C starts right at that head; or another arrow arrives at
    the box this line leaves from."""
    ex, ey = end
    length = math.hypot(ex - prev[0], ey - prev[1]) or 1.0
    ux, uy = (ex - prev[0]) / length, (ey - prev[1]) / length
    best = None
    for hd in heads:
        hx, hy, hr = hd[:3]
        d = math.hypot(hx - ex, hy - ey)
        along = (hx - ex) * ux + (hy - ey) * uy
        off_line = abs(-(hx - ex) * uy + (hy - ey) * ux)
        if (d <= reach and along >= -0.8 * reach and off_line <= hr + 1
                and (best is None or d < best[0])
                and _tip_skew(hd, ux, uy) >= -TIP_SKEW):
            ahead = max(0.0, along) + hr
            best = (d, (ex + ux * ahead, ey + uy * ahead), hd)
    return best


def _polyline_length(a: dict) -> float:
    return sum(math.hypot(q[0] - p[0], q[1] - p[1]) for p, q in zip(a["points"], a["points"][1:]))


def _runs_along_another(a: dict, arrows: List[dict], sw: int) -> bool:
    """A headless line lying (80%+) along an arrow with a head, or along a
    longer headless line, is a duplicate of it: Hough can return a thick
    shaft twice, a little apart in angle. Of two equal copies the first is
    kept."""
    pts = a["points"]
    samples = [(p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t)
               for p, q in zip(pts, pts[1:]) for t in np.linspace(0.0, 1.0, 12)]
    tol = sw + 4.0
    own = _polyline_length(a)
    for k, o in enumerate(arrows):
        if o is a:
            continue
        other = _polyline_length(o)
        if not (o["has_head"] or other > own or (other == own and k < arrows.index(a))):
            continue
        legs = list(zip(o["points"], o["points"][1:]))

        def near(x, y):
            for (x1, y1), (x2, y2) in legs:
                dx, dy = x2 - x1, y2 - y1
                seg2 = dx * dx + dy * dy
                t = 0.0 if seg2 == 0 else max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / seg2))
                if math.hypot(x - (x1 + t * dx), y - (y1 + t * dy)) <= tol:
                    return True
            return False

        if sum(near(x, y) for x, y in samples) >= 0.8 * len(samples):
            return True
    return False


def detect(image_path: str) -> dict:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    start = time.perf_counter()

    _, inv = cv2.threshold(gray, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    dt = _distance(inv)
    sw = _stroke_width(inv, dt)
    clean, cores = _connector_ink(inv, sw)
    band = _outline_band(gray, inv, sw, strict=True)
    loose_band = _outline_band(gray, inv, sw, strict=False)
    lines = cv2.HoughLinesP(_bridge_dashes(clean), rho=1, theta=np.pi / 180,
                            threshold=HOUGH_THRESHOLD, minLineLength=HOUGH_MIN_LENGTH,
                            maxLineGap=HOUGH_MAX_GAP)
    segments = [line[0] for line in lines] if lines is not None else []
    segments = [s for s in segments if _fraction_on(s, band) <= OUTLINE_FRACTION]

    def erode(ink, width):
        k = max(3, 2 * width + 1)
        return cv2.erode(ink, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))

    # Heads are looked for on the ink minus the cores of filled shapes, so a
    # head touching a filled box is still a separate blob; and at the scale
    # of the line they end, which the image-wide stroke width may not match.
    head_ink = cv2.subtract(inv, cores)
    head_cache = {}

    def heads_for(width):
        if width not in head_cache:
            head_cache[width] = _head_blobs(erode(head_ink, width), width)
        return head_cache[width]

    def reach_for(width):
        return max(14.0, 5.0 * width + 10.0)

    solid_cache = {}

    def solid_for(width):
        if width not in solid_cache:
            solid_cache[width] = erode(inv, width)
        return solid_cache[width]

    def widths(pts):
        # Heads are looked for at the line's own width first, then at half of
        # it (heads on thick lines are relatively smaller), then image-wide.
        lw = _line_width(dt, pts, sw)
        return list(dict.fromkeys((lw, (lw + 1) // 2, sw))), lw

    glyphs = cv2.subtract(inv, clean)          # the letter/blob ink _connector_ink removed
    merged = _bridge_line_labels(_merge_collinear(segments), glyphs, band)
    pieces = []
    for seg in merged:
        # Eroded at the width of a line thicker than the image's strokes, its
        # shaft vanishes and only heads and filled icons along it remain to cut
        # at. Never finer than image-wide: along a whole line, a finer erosion
        # leaves pieces of the shaft that look like heads.
        ws, lw = widths([(seg[0], seg[1]), (seg[2], seg[3])])
        seg_heads = [hd for wd in ws if wd >= sw for hd in heads_for(wd)]
        pieces += [q for p in _split_at_heads(solid_for(max(sw, lw)), seg)
                   for q in _split_at_head_blobs(p, seg_heads)]
    pieces = [p for p in pieces if math.hypot(p[2] - p[0], p[3] - p[1]) >= 12]

    def end_has_head(i, e):
        s = pieces[i]
        end, prev = ((s[0], s[1]), (s[2], s[3])) if e == 0 else ((s[2], s[3]), (s[0], s[1]))
        for wd in widths([prev, end])[0]:
            found = _head_at(heads_for(wd), end, prev, reach_for(wd))
            # The smallest blob _head_blobs accepts is too weak to outweigh a
            # clean corner: a rounded corner of a 2-px dashed line leaves one.
            if found is not None and found[2][2] > max(3, wd + 2) / 2.0 + wd:
                return True
        return False

    candidates = []
    for pts in _join_elbows(pieces, end_has_head):
        length = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))
        for width in widths(pts)[0]:
            line_heads, reach = heads_for(width), reach_for(width)
            h_tail = _head_at(line_heads, pts[0], pts[1], reach)
            h_tip = _head_at(line_heads, pts[-1], pts[-2], reach)
            if h_tail or h_tip:
                break
        if h_tail is not None and h_tip is not None:
            # Two heads make a double-headed connector only if both are
            # convincing -- clearly pointing out of the line and above the
            # minimum blob size; otherwise the weaker one is a stray blob (a
            # letter, a box corner) at the tail.
            weakest = max(3, width + 2) / 2.0 + width

            def clear(found, end, prev):
                length = math.hypot(end[0] - prev[0], end[1] - prev[1]) or 1.0
                skew = _tip_skew(found[2], (end[0] - prev[0]) / length, (end[1] - prev[1]) / length)
                return skew >= TIP_SKEW and found[2][2] > weakest

            tail_ok, tip_ok = clear(h_tail, pts[0], pts[1]), clear(h_tip, pts[-1], pts[-2])
            if not (tail_ok and tip_ok):
                if tail_ok or (not tip_ok and h_tail[0] < h_tip[0]):
                    h_tip = None
                else:
                    h_tail = None
        candidates.append((length, pts, h_tail, h_tip))

    # Each head ends one connector. Longer lines claim heads first: a short
    # stray segment between two arrows' heads (where they converge) otherwise
    # takes both and becomes a double-headed connector of its own.
    claimed = set()
    for i in sorted(range(len(candidates)), key=lambda i: -candidates[i][0]):
        length, pts, h_tail, h_tip = candidates[i]
        ends = []
        for h in (h_tail, h_tip):
            key = None if h is None else (round(h[2][0], 1), round(h[2][1], 1))
            ends.append(None if key is None or key in claimed else h)
            if key is not None:
                claimed.add(key)
        candidates[i] = (length, pts, ends[0], ends[1])

    out_arrows = []
    for length, pts, h_tail, h_tip in candidates:
        # A short line with an arrowhead is still a connector (two boxes drawn
        # close together); without one it is more likely a stray stroke.
        extended = length + sum(math.hypot(h[1][0] - p[0], h[1][1] - p[1])
                                for h, p in ((h_tail, pts[0]), (h_tip, pts[-1])) if h is not None)
        if extended < (MIN_HEADED_LENGTH if (h_tail or h_tip) else MIN_ARROW_LENGTH):
            continue
        if h_tail is None and h_tip is None:
            # No arrowhead: keep it only if it doesn't run along any enclosed
            # region (a border of a shape something else is drawn across).
            on = sum(_fraction_on([a[0], a[1], b[0], b[1]], loose_band) * math.hypot(b[0] - a[0], b[1] - a[1])
                     for a, b in zip(pts, pts[1:]))
            if on / length > OUTLINE_FRACTION:
                continue
        if h_tail is not None:
            pts = [h_tail[1]] + pts[1:]
        if h_tip is not None:
            pts = pts[:-1] + [h_tip[1]]
        # The head end becomes the target: build_relationships reads the
        # first point as the source and the last as the target.
        if h_tail is not None and (h_tip is None or h_tail[0] < h_tip[0]):
            pts = pts[::-1]
            h_tail, h_tip = h_tip, h_tail
        points = [[int(round(x)), int(round(y))] for x, y in pts]
        (x1, y1), (x2, y2) = points[0], points[-1]
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        direction = ("elbow" if len(points) > 2 else
                     "horizontal" if dy <= 3 else "vertical" if dx <= 3 else "diagonal")
        out_arrows.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "direction": direction,
                           "points": points, "bidirectional": h_tail is not None and h_tip is not None,
                           "has_head": h_tip is not None})

    out_arrows = [a for a in out_arrows if a["has_head"] or not _runs_along_another(a, out_arrows, sw)]
    elapsed = time.perf_counter() - start
    return {"arrows": out_arrows, "runtime_seconds": round(elapsed, 3)}
