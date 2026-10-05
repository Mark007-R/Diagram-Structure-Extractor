"""Data-driven graph construction.

This module is the heart of the Day-4 de-hardcoding. The legacy monolith built
its relationship graph by *merging in* `_known_connections()` — 8 edges hand-
coded for the "Plant An App - AWS" diagram — so the graph looked complete even
when arrow detection found nothing. That baked the answer key into the code.

Here, EVERY relationship is derived from a detected arrow:
  1. `label_boxes`   — assign each box a label from the text contained in it.
  2. `build_relationships` — map each arrow's endpoints to the nearest boxes,
     gated so segments lying inside/on a box border (the Day-3 snap-to-adjacency
     artifact) are rejected.
  3. `draw_graph`    — render with a general `kamada_kawai` / `spring` layout.
     No hardcoded node coordinates.
"""
from __future__ import annotations

from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def label_boxes(boxes: List[dict], texts: List[dict]) -> List[dict]:
    """Assign each box a text label from the OCR fragments whose centre falls
    inside it (smallest enclosing box wins, mirroring the legacy
    associate_elements). Fragments are joined in reading order (top-to-bottom,
    left-to-right). A box that contains no text keeps the label it came with
    (an icon node is labelled from its caption) or an empty one."""
    contained: dict = {i: [] for i in range(len(boxes))}
    for t in texts:
        tcx = t["x"] + t["w"] // 2
        tcy = t["y"] + t["h"] // 2
        best_box = None
        best_area = float("inf")
        for bi, b in enumerate(boxes):
            if b["x"] <= tcx <= b["x"] + b["w"] and b["y"] <= tcy <= b["y"] + b["h"]:
                area = b["w"] * b["h"]
                if area < best_area:
                    best_area = area
                    best_box = bi
        if best_box is not None:
            contained[best_box].append(t)

    for bi, b in enumerate(boxes):
        frags = sorted(contained[bi], key=lambda t: (t["y"], t["x"]))
        label = " ".join(f["text"].strip() for f in frags if f.get("text", "").strip())
        b["label"] = label.strip() or (b.get("label") or "").strip()
    return _label_from_captions(boxes, texts)


def _caption_gap(t: dict, b: dict) -> Optional[float]:
    """Distance from box `b` to text `t` if `t` sits just below or above it
    (a caption or a title), else None."""
    cx = t["x"] + t["w"] / 2
    if not b["x"] - 0.25 * b["w"] <= cx <= b["x"] + 1.25 * b["w"]:
        return None
    gaps = [g for g in (t["y"] - (b["y"] + b["h"]), b["y"] - (t["y"] + t["h"]))   # below, above
            if -4 <= g <= max(14.0, 0.45 * b["h"])]
    return min(abs(g) for g in gaps) if gaps else None


def _label_from_captions(boxes: List[dict], texts: List[dict]) -> List[dict]:
    """An unlabelled component box takes the nearest free text just below or
    above it: icons (a database cylinder) carry their caption underneath, and
    titled frames (e.g. "Back-End") carry it on top. Text inside any component
    box is never free, nor is a caption line an icon node was found and
    labelled by (its `caption_boxes`); containers (regions) don't claim text
    here. Captions go to the closest box first, so a farther box cannot take a
    nearer box's caption."""
    def centre_inside(t: dict, b: dict) -> bool:
        cx, cy = t["x"] + t["w"] / 2, t["y"] + t["h"] / 2
        return b["x"] <= cx <= b["x"] + b["w"] and b["y"] <= cy <= b["y"] + b["h"]

    components = [b for b in boxes if not b.get("is_region")]
    used = {tuple(c) for b in components for c in (b.get("caption_boxes") or [])}
    free = [t for t in texts if t.get("text", "").strip()
            and not any(centre_inside(t, b) for b in components)
            and (t["x"], t["y"], t["w"], t["h"]) not in used]
    pairs = sorted((gap, bi, ti) for bi, b in enumerate(components) if not b.get("label")
                   for ti, t in enumerate(free) for gap in [_caption_gap(t, b)] if gap is not None)
    used_boxes, used_texts = set(), set()
    for _, bi, ti in pairs:
        if bi not in used_boxes and ti not in used_texts:
            components[bi]["label"] = free[ti]["text"].strip()
            used_boxes.add(bi)
            used_texts.add(ti)
    return boxes


def _point_in_box(x: int, y: int, b: dict) -> bool:
    return b["x"] <= x <= b["x"] + b["w"] and b["y"] <= y <= b["y"] + b["h"]


def _nearest_box(x: int, y: int, boxes: List[dict], max_dist: float = 100.0,
                 tie_px: float = 8.0) -> Optional[int]:
    """Index of the box nearest to (x, y) within max_dist, by min(centre, edge)
    distance. Among boxes within `tie_px` of the nearest, the smallest wins: an
    arrow ending at an inner box is also inside (distance 0) the frame around
    it, only a few px from the inner box. Returns None if nothing is close."""
    cands = []
    for i, b in enumerate(boxes):
        bcx = b["x"] + b["w"] / 2
        bcy = b["y"] + b["h"] / 2
        d_centre = float(np.hypot(x - bcx, y - bcy))
        dx = max(b["x"] - x, 0, x - (b["x"] + b["w"]))
        dy = max(b["y"] - y, 0, y - (b["y"] + b["h"]))
        d = min(d_centre, float(np.hypot(dx, dy)))
        if d <= max_dist:
            cands.append((d, b["w"] * b["h"], i))
    if not cands:
        return None
    nearest = min(c[0] for c in cands)
    return min((c for c in cands if c[0] <= nearest + tie_px), key=lambda c: c[1])[2]


def _arrow_points(a: dict) -> List[List[float]]:
    return a.get("points") or [[a["x1"], a["y1"]], [a["x2"], a["y2"]]]


def _along_box_edge(a: dict, boxes: List[dict], tol: float = 4.0, frac: float = 0.7) -> bool:
    """True if the arrow mostly runs along some box's or container's edge.
    Box borders are never connectors, but an outline the detector could not
    see as closed (e.g. a container edge cut by a white edge label) still
    reaches here as a line."""
    total = covered = 0.0
    pts = _arrow_points(a)
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        total += float(np.hypot(x2 - x1, y2 - y1))
        horizontal, vertical = abs(y2 - y1) <= tol, abs(x2 - x1) <= tol
        best = 0.0
        for b in boxes:
            if horizontal and min(abs((y1 + y2) / 2 - b["y"]), abs((y1 + y2) / 2 - b["y"] - b["h"])) <= tol:
                best = max(best, min(max(x1, x2), b["x"] + b["w"]) - max(min(x1, x2), b["x"]))
            elif vertical and min(abs((x1 + x2) / 2 - b["x"]), abs((x1 + x2) / 2 - b["x"] - b["w"])) <= tol:
                best = max(best, min(max(y1, y2), b["y"] + b["h"]) - max(min(y1, y2), b["y"]))
        covered += max(0.0, best)
    return total > 0 and covered / total >= frac


def _branch_point(px: float, py: float, path: List[List[float]], tol: float, end_margin: float):
    """Where (px, py) lies on the interior of polyline `path`: (leg index,
    foot point), or None."""
    total = sum(float(np.hypot(q[0] - p[0], q[1] - p[1])) for p, q in zip(path, path[1:]))
    run = 0.0
    for i, ((x1, y1), (x2, y2)) in enumerate(zip(path, path[1:])):
        seg = float(np.hypot(x2 - x1, y2 - y1))
        t = 0.0 if seg == 0 else max(0.0, min(1.0, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / seg ** 2))
        fx, fy = x1 + t * (x2 - x1), y1 + t * (y2 - y1)
        if float(np.hypot(px - fx, py - fy)) <= tol and end_margin <= run + t * seg <= total - end_margin:
            return i, [fx, fy]
        run += seg
    return None


def _inherit_trunks(arrows: List[dict], tol: float = 5.0, end_margin: float = 15.0) -> List[dict]:
    """A branch whose tail lies on another connector's interior starts where
    that connector starts: draw.io routes connectors that leave the same port
    along one shared first leg, so only one of them is seen from the port.
    The trunk's path up to the branch point is spliced in, so the tail end
    still points along a real leg, and a branch off a branch resolves to the
    first trunk's tail. Both need an arrowhead: a leftover border line is
    never a trunk, and a headless line has no known tail -- the end touching
    the trunk may be where another connector merges in. A double-headed
    trunk has no tail either."""
    paths = [[list(p) for p in _arrow_points(a)] for a in arrows]
    resolved: dict = {}

    def resolve(k: int, visiting: frozenset) -> List[List[float]]:
        if k in resolved:
            return resolved[k]
        pts = paths[k]
        if arrows[k].get("has_head"):
            for j, o in enumerate(arrows):
                if j == k or j in visiting or not o.get("has_head") or o.get("bidirectional"):
                    continue
                hit = _branch_point(pts[0][0], pts[0][1], paths[j], tol, end_margin)
                if hit is not None:
                    trunk = resolve(j, visiting | {k})
                    # The trunk may itself have been extended back; find the
                    # branch point on its resolved path.
                    leg = _branch_point(pts[0][0], pts[0][1], trunk, tol, 0.0) or hit
                    pts = [list(p) for p in trunk[:leg[0] + 1]] + [leg[1]] + pts[1:]
                    break
        resolved[k] = pts
        return pts

    out = []
    for k, a in enumerate(arrows):
        pts = resolve(k, frozenset())
        b = dict(a, points=pts)
        b["x1"], b["y1"], b["x2"], b["y2"] = pts[0][0], pts[0][1], pts[-1][0], pts[-1][1]
        out.append(b)
    return out


def _midpoint_inside_any_box(arrow: dict, boxes: List[dict]) -> bool:
    """The Day-3 'outside any box bbox' gate. A genuine connector runs through
    whitespace BETWEEN boxes, so its midpoint is not inside a box. A box-border
    / internal-edge segment (the snap-to-adjacency artifact that inflated CNN-
    verified recall on Day 3) has its midpoint on or inside a box bbox."""
    mx = (arrow["x1"] + arrow["x2"]) // 2
    my = (arrow["y1"] + arrow["y2"]) // 2
    return any(_point_in_box(mx, my, b) for b in boxes)


def _ray_aabb_t(ox: float, oy: float, dx: float, dy: float, b: dict) -> Optional[float]:
    """Slab ray/AABB intersection. Returns the entry distance t>=0 along the
    unit ray (ox,oy)+t*(dx,dy) at which the ray enters box `b`, or None if it
    never does. Handles axis-aligned rays (dx or dy == 0)."""
    minx, miny = b["x"], b["y"]
    maxx, maxy = b["x"] + b["w"], b["y"] + b["h"]
    tmin, tmax = 0.0, float("inf")
    for o, d, lo, hi in ((ox, dx, minx, maxx), (oy, dy, miny, maxy)):
        if abs(d) < 1e-9:
            if o < lo or o > hi:
                return None
        else:
            t1, t2 = (lo - o) / d, (hi - o) / d
            if t1 > t2:
                t1, t2 = t2, t1
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
            if tmin > tmax:
                return None
    return tmin if tmax >= 0 else None


def _box_along_ray(x: float, y: float, dx: float, dy: float,
                   boxes: List[dict], max_proj: float) -> Optional[int]:
    """Index of the first box the outward ray from (x,y) along (dx,dy) enters,
    within `max_proj` px. This is the 'intersection-with-box' recovery: a short
    detected segment that stops in whitespace still *points at* its true target
    box, so we project the line and snap to the box it would hit."""
    norm = float(np.hypot(dx, dy))
    if norm < 1e-9:
        return None
    ux, uy = dx / norm, dy / norm
    best_i, best_t = None, max_proj
    for i, b in enumerate(boxes):
        t = _ray_aabb_t(x, y, ux, uy, b)
        if t is not None and 0.0 <= t < best_t:
            best_t, best_i = t, i
    return best_i


def _assign_endpoint(x: float, y: float, ox: float, oy: float,
                     boxes: List[dict], max_dist: float,
                     ray_intersection: bool, max_proj: float) -> Optional[int]:
    """Map an arrow endpoint (x,y) to a box. (ox,oy) is the *other* endpoint,
    used to compute the outward direction for the ray fallback. Try the nearest
    box within `max_dist` first (precise when the endpoint sits on/near a box);
    if nothing is close enough and `ray_intersection` is on, project the line
    outward and snap to the box it enters."""
    i = _nearest_box(x, y, boxes, max_dist)
    if i is not None:
        return i
    if ray_intersection:
        return _box_along_ray(x, y, x - ox, y - oy, boxes, max_proj)
    return None


def build_relationships(
    arrows: List[dict],
    boxes: List[dict],
    outside_box_gate: bool = True,
    max_dist: float = 120.0,
    ray_intersection: bool = False,
    max_proj: float = 400.0,
    edge_filter: bool = True,
    trunks: bool = True,
) -> List[dict]:
    """Build relationships purely from detected arrows. No hardcoded edges.

    For each arrow, snap its first point (tail) and last point (head) to a
    labelled box and emit a de-duplicated source -> target edge, flagged
    `bidirectional` for a double-headed connector. With `outside_box_gate`,
    arrows whose midpoint lies inside/on a box are dropped (border artifacts).
    With `edge_filter`, arrows that run along a box or container edge are
    dropped. With `trunks`, a branch that leaves another connector's shared
    first leg is traced back to that connector's tail.

    Day-5 arrow-mapping fix (`ray_intersection`): error analysis found the
    dominant failure mode is arrow-mapping, and within it ~40% of missed
    connections are *mis-mapped* short segments whose endpoints stop in
    whitespace beyond `max_dist`. With `ray_intersection` on, an endpoint that
    has no box within `max_dist` is projected outward along the segment's
    direction and snapped to the first box the ray enters (within `max_proj`).
    """
    # Endpoints map to component boxes (skip region/container boxes and
    # platform background boxes — they swallow endpoints and create spurious
    # parent->child edges).
    candidates = [
        b for b in boxes
        if not b.get("is_region") and b.get("entity_type") != "platform"
    ]
    if len(candidates) < 2:
        candidates = [b for b in boxes if not b.get("is_region")]
    if len(candidates) < 2:
        candidates = boxes
    gate_boxes = candidates

    def labelled(b: dict) -> bool:
        return bool((b.get("label") or "").strip())

    def within(inner: dict, outer: dict, slack: float = 2.0) -> bool:
        return (inner["x"] >= outer["x"] - slack and inner["y"] >= outer["y"] - slack
                and inner["x"] + inner["w"] <= outer["x"] + outer["w"] + slack
                and inner["y"] + inner["h"] <= outer["y"] + outer["h"] + slack)

    # An unlabelled box can never give an edge, but an end that reaches one
    # (an icon with no caption, a node whose text OCR missed) stops there --
    # it must not carry on to a labelled box nearby. It must not outbid the
    # labelled box around it either: a glyph inside a node is not a blocker.
    named = [b for b in candidates if labelled(b)]
    blockers = []
    if named:
        blockers = [b for b in candidates if not labelled(b) and not any(within(b, n) for n in named)]
        candidates = named
    # Likewise only labelled boxes and containers have edges worth filtering
    # against: a spurious unlabelled contour can have a connector as a side.
    edge_boxes = [b for b in boxes if labelled(b) or b.get("is_region")]

    relationships: List[dict] = []
    seen: dict = {}
    if trunks:
        arrows = _inherit_trunks(arrows)
    for a in arrows:
        if outside_box_gate and _midpoint_inside_any_box(a, gate_boxes):
            continue
        if edge_filter and _along_box_edge(a, edge_boxes):
            continue
        # Each end projects along its own leg, which for an elbow is not the
        # line to the other end.
        pts = _arrow_points(a)
        (sx, sy), (sx2, sy2) = pts[0], pts[1]
        (tx, ty), (tx2, ty2) = pts[-1], pts[-2]
        # A contour around the whole connector (e.g. split along it) blocks
        # nothing.
        pool = candidates + [u for u in blockers
                             if not (_point_in_box(sx, sy, u) and _point_in_box(tx, ty, u))]
        si = _assign_endpoint(sx, sy, sx2, sy2, pool, max_dist, ray_intersection, max_proj)
        ti = _assign_endpoint(tx, ty, tx2, ty2, pool, max_dist, ray_intersection, max_proj)
        if si is None or ti is None or si == ti or si >= len(candidates) or ti >= len(candidates):
            continue
        src = candidates[si].get("label", "").strip()
        tgt = candidates[ti].get("label", "").strip()
        if not src or not tgt or src == tgt:
            continue
        bidirectional = bool(a.get("bidirectional"))
        if (src, tgt) in seen:
            if bidirectional:      # a second detection of the same connector saw both heads
                relationships[seen[(src, tgt)]]["bidirectional"] = True
            continue
        reverse = seen.get((tgt, src))
        if reverse is not None and (bidirectional or relationships[reverse]["bidirectional"]):
            relationships[reverse]["bidirectional"] = True   # already covered both ways
            continue
        seen[(src, tgt)] = len(relationships)
        relationships.append({
            "source": src,
            "target": tgt,
            "line_style": a.get("line_style", "solid"),
            "direction": a.get("direction", "unknown"),
            "relationship": "connects_to",
            "bidirectional": bidirectional,
            "detected": True,
        })
    # A one-way row seen before the double-headed detection of the same pair
    # is covered by it.
    both_ways = {(r["source"], r["target"]) for r in relationships if r["bidirectional"]}
    return [r for r in relationships if r["bidirectional"] or (r["target"], r["source"]) not in both_ways]


def build_graph(relationships: List[dict]) -> nx.DiGraph:
    G = nx.DiGraph()
    for r in relationships:
        src, tgt = (r.get("source") or "").strip(), (r.get("target") or "").strip()
        if src and tgt and src != tgt:
            G.add_edge(src, tgt,
                       line_style=r.get("line_style", "solid"),
                       relationship=r.get("relationship", ""))
    return G


def _short_label(s: str, max_chars: int = 18) -> str:
    s = (s or "").strip()
    if len(s) <= max_chars:
        return s
    words, line1, line2 = s.split(), "", ""
    for w in words:
        if len(line1) + len(w) + 1 <= max_chars or not line1:
            line1 = (line1 + " " + w).strip()
        else:
            line2 = (line2 + " " + w).strip()
    return line1 + ("\n" + line2 if line2 else "")


def draw_graph(relationships: List[dict], output_path: str, layout: str = "kamada_kawai") -> bool:
    """Render the relationship DiGraph with a general, data-driven layout.

    No hardcoded `pos = {...}`: node positions come from the graph structure, so
    this works on any diagram. Returns False if there is nothing to draw."""
    G = nx.DiGraph()
    for r in relationships:
        src = _short_label(r.get("source", ""))
        tgt = _short_label(r.get("target", ""))
        if src and tgt and src != tgt:
            G.add_edge(src, tgt,
                       line_style=r.get("line_style", "solid"),
                       relationship=r.get("relationship", ""),
                       bidirectional=bool(r.get("bidirectional")))
    if len(G.nodes) == 0:
        return False

    if layout == "spring":
        pos = nx.spring_layout(G, seed=42, k=1.2)
    else:
        try:
            pos = nx.kamada_kawai_layout(G)
        except Exception:
            pos = nx.spring_layout(G, seed=42, k=1.2)

    fig, ax = plt.subplots(figsize=(16, 9))
    node_size, node_shape = 3600, "s"
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color="#D5E8F0",
                           edgecolors="#333", linewidths=2, node_shape=node_shape, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold", ax=ax)

    # Edges are told the node size and shape so they stop at the squares;
    # otherwise the heads are drawn underneath them.
    for both in (False, True):
        heads = "<|-|>" if both else "-|>"   # double-headed connectors get both heads
        edges = [(u, v, d) for u, v, d in G.edges(data=True) if bool(d.get("bidirectional")) == both]
        solid = [(u, v) for u, v, d in edges if d.get("line_style") == "solid"]
        dashed = [(u, v) for u, v, d in edges if d.get("line_style") != "solid"]
        nx.draw_networkx_edges(G, pos, edgelist=solid, edge_color="#333", width=2.0,
                               arrows=True, arrowstyle=heads, arrowsize=18, ax=ax,
                               node_size=node_size, node_shape=node_shape,
                               connectionstyle="arc3,rad=0.08")
        nx.draw_networkx_edges(G, pos, edgelist=dashed, edge_color="#666", width=1.5,
                               style="dashed", arrows=True, arrowstyle=heads, arrowsize=18, ax=ax,
                               node_size=node_size, node_shape=node_shape,
                               connectionstyle="arc3,rad=0.08")
    edge_labels = {(u, v): d.get("relationship", "")
                   for u, v, d in G.edges(data=True) if d.get("relationship")}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                 font_size=7, font_color="#444", ax=ax)
    ax.set_title("Relationship Graph (data-driven)", fontsize=15, fontweight="bold", pad=18)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True
