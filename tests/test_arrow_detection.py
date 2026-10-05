"""Tests for arrow detection."""
from __future__ import annotations

import pytest


def test_hough_lines_returns_schema_valid_dict(test_image: str):
    from src.arrow_detection import hough_lines_detector as det
    out = det.detect(test_image)
    assert set(out.keys()) >= {"arrows", "runtime_seconds"}
    assert isinstance(out["arrows"], list)


def test_hough_lines_arrows_have_endpoint_fields(test_image: str):
    """Every arrow has x1/y1/x2/y2 — the (source endpoint, target endpoint)
    pair that downstream graph-builder.build_relationships consumes."""
    from src.arrow_detection import hough_lines_detector as det
    out = det.detect(test_image)
    for a in out["arrows"]:
        assert all(k in a for k in ("x1", "y1", "x2", "y2"))


def _draw_arrow(img, tail, tip, dashed=False):
    """Black connector from tail to tip with a filled matplotlib-style head."""
    import cv2
    import numpy as np
    tail, tip = np.array(tail, float), np.array(tip, float)
    u = (tip - tail) / np.linalg.norm(tip - tail)
    n = np.array([-u[1], u[0]])
    base = tip - u * 16
    if dashed:
        length = np.linalg.norm(base - tail)
        for t in np.arange(0, length, 14):
            a, b = tail + u * t, tail + u * min(t + 8, length)
            cv2.line(img, tuple(map(int, a)), tuple(map(int, b)), (0, 0, 0), 2)
    else:
        cv2.line(img, tuple(map(int, tail)), tuple(map(int, base)), (0, 0, 0), 2)
    head = np.array([tip, base + n * 7, base - n * 7], np.int32)
    cv2.fillPoly(img, [head], (0, 0, 0))


@pytest.fixture()
def arrow_image(tmp_path):
    """Three boxes: A -> B horizontal (dashed), A -> C diagonal (pointing up-left
    from C's side, i.e. drawn tip-first in reading order)."""
    import cv2
    import numpy as np
    img = np.full((500, 800, 3), 255, np.uint8)
    for x, y in ((50, 50), (550, 50), (550, 350)):
        cv2.rectangle(img, (x, y), (x + 200, y + 100), (0, 0, 0), 2)
    _draw_arrow(img, (250, 100), (550, 100), dashed=True)   # A -> B
    _draw_arrow(img, (550, 400), (250, 140))                # C -> A, diagonal
    path = str(tmp_path / "arrows.png")
    cv2.imwrite(path, img)
    return path


def _endpoint_box(x, y):
    boxes = {"A": (50, 50), "B": (550, 50), "C": (550, 350)}
    for name, (bx, by) in boxes.items():
        if bx - 20 <= x <= bx + 220 and by - 20 <= y <= by + 120:
            return name
    return None


def test_directed_lines_schema_valid(test_image: str):
    from src.arrow_detection import directed_lines_detector as det
    out = det.detect(test_image)
    assert set(out.keys()) >= {"arrows", "runtime_seconds"}
    for a in out["arrows"]:
        assert all(k in a for k in ("x1", "y1", "x2", "y2", "direction"))


def test_directed_lines_finds_diagonals_and_orients_by_arrowhead(arrow_image: str):
    """Diagonal connectors are kept, and (x1,y1)->(x2,y2) runs tail -> head,
    which is the source -> target order build_relationships relies on."""
    from src.arrow_detection import directed_lines_detector as det
    edges = {(_endpoint_box(a["x1"], a["y1"]), _endpoint_box(a["x2"], a["y2"]))
             for a in det.detect(arrow_image)["arrows"]}
    assert ("A", "B") in edges
    assert ("C", "A") in edges
    assert ("B", "A") not in edges and ("A", "C") not in edges


def test_directed_lines_splits_chained_arrows(tmp_path):
    """A row A -> B -> C must not collapse into one A -> C line."""
    import cv2
    import numpy as np
    img = np.full((300, 1000, 3), 255, np.uint8)
    for x in (50, 400, 750):
        cv2.rectangle(img, (x, 100), (x + 200, 200), (0, 0, 0), 2)
    _draw_arrow(img, (180, 150), (450, 150), dashed=True)
    _draw_arrow(img, (530, 150), (800, 150), dashed=True)
    path = str(tmp_path / "chain.png")
    cv2.imwrite(path, img)

    from src.arrow_detection import directed_lines_detector as det
    from src.graph import builder
    boxes = [{"x": x, "y": 100, "w": 200, "h": 100, "label": lbl}
             for x, lbl in ((50, "A"), (400, "B"), (750, "C"))]
    rels = builder.build_relationships(det.detect(path)["arrows"], boxes)
    pairs = {(r["source"], r["target"]) for r in rels}
    assert pairs == {("A", "B"), ("B", "C")}


def test_pixel_scan_detector_schema_valid(test_image: str):
    """The legacy pixel_scan detector still satisfies the schema (the only
    requirement enforced project-wide)."""
    from src.arrow_detection import pixel_scan_detector as det
    out = det.detect(test_image)
    assert set(out.keys()) >= {"arrows", "runtime_seconds"}
    assert isinstance(out["arrows"], list)
