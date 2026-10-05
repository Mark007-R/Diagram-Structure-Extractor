"""draw.io-style connectors: thin dashed lines, small filled heads, elbows,
double-headed connectors, line labels, icon nodes with captions -- and the
graph-builder rules that map them to boxes."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

BLACK = (0, 0, 0)


def _dashed(img, p0, p1, dash=5, gap=4):
    p0, p1 = np.array(p0, float), np.array(p1, float)
    length = np.linalg.norm(p1 - p0)
    u = (p1 - p0) / length
    t = 0.0
    while t < length:
        a, b = p0 + u * t, p0 + u * min(t + dash, length)
        cv2.line(img, tuple(map(int, a)), tuple(map(int, b)), BLACK, 1)
        t += dash + gap


def _head(img, tip, frm, size=8):
    """Small filled 'classic' head pointing at `tip`, coming from `frm`."""
    tip, frm = np.array(tip, float), np.array(frm, float)
    u = (tip - frm) / np.linalg.norm(tip - frm)
    n = np.array([-u[1], u[0]])
    base = tip - u * size
    tri = np.array([tip, base + n * size * 0.45, base - n * size * 0.45], np.int32)
    cv2.fillPoly(img, [tri], BLACK)


def _box(img, x, y, w, h):
    cv2.rectangle(img, (x, y), (x + w, y + h), BLACK, 1)


@pytest.fixture()
def drawio_image(tmp_path):
    """A (left) -> B (right, lower) through a dashed elbow; A <-> C dashed
    straight and double-headed; C -> D cut by a white-backed line label."""
    img = np.full((420, 900, 3), 255, np.uint8)
    for x, y in ((60, 60), (520, 260), (60, 260), (60, 360)):
        _box(img, x, y, 160, 50)
    # A -> B: right from A, down, right into B (elbow, 2 bends).
    _dashed(img, (221, 85), (400, 85))
    _dashed(img, (400, 85), (400, 285))
    _dashed(img, (400, 285), (510, 285))
    _head(img, (519, 285), (400, 285))
    # A <-> C: vertical, heads at both ends.
    _dashed(img, (140, 121), (140, 249))
    _head(img, (140, 111), (140, 200))
    _head(img, (140, 259), (140, 150))
    # C -> D? No: a separate row. E -> F label-cut connector on its own line.
    _dashed(img, (230, 390), (700, 390))
    cv2.rectangle(img, (420, 380), (500, 400), (255, 255, 255), -1)
    cv2.putText(img, "HTTPS", (424, 396), cv2.FONT_HERSHEY_SIMPLEX, 0.45, BLACK, 1)
    _box(img, 710, 365, 150, 50)
    _head(img, (709, 390), (600, 390))
    path = str(tmp_path / "drawio.png")
    cv2.imwrite(path, img)
    return path


def _boxes():
    return [{"x": 60, "y": 60, "w": 160, "h": 50, "label": "A"},
            {"x": 520, "y": 260, "w": 160, "h": 50, "label": "B"},
            {"x": 60, "y": 260, "w": 160, "h": 50, "label": "C"},
            {"x": 60, "y": 360, "w": 160, "h": 50, "label": "E"},
            {"x": 710, "y": 365, "w": 150, "h": 50, "label": "F"}]


def test_detector_finds_elbow_double_headed_and_label_cut_connectors(drawio_image):
    from src.arrow_detection import directed_lines_detector as det
    from src.graph import builder
    arrows = det.detect(drawio_image)["arrows"]
    assert any(len(a["points"]) >= 3 for a in arrows), "the A->B elbow should keep its bends"
    assert any(a["bidirectional"] for a in arrows)
    rels = builder.build_relationships(arrows, _boxes(), outside_box_gate=False, max_dist=60.0,
                                       ray_intersection=True)
    got = {(r["source"], r["target"], r["bidirectional"]) for r in rels}
    assert ("A", "B", False) in got
    assert ("A", "C", True) in got or ("C", "A", True) in got
    assert ("E", "F", False) in got        # one connector across the "HTTPS" label
    assert len(got) == 3, got


def test_caption_labels_an_icon_box_and_a_titled_frame():
    from src.graph import builder
    boxes = [{"x": 100, "y": 100, "w": 60, "h": 60},            # icon, caption below
             {"x": 300, "y": 120, "w": 160, "h": 120},          # frame, title above
             {"x": 320, "y": 160, "w": 100, "h": 40}]           # inner box with its own text
    texts = [{"text": "Database", "x": 95, "y": 166, "w": 70, "h": 14},
             {"text": "Back-End", "x": 340, "y": 98, "w": 80, "h": 16},
             {"text": "Client", "x": 340, "y": 172, "w": 50, "h": 14}]
    labels = [b["label"] for b in builder.label_boxes(boxes, texts)]
    assert labels == ["Database", "Back-End", "Client"]


def test_endpoint_prefers_the_innermost_box():
    from src.graph import builder
    outer = {"x": 0, "y": 0, "w": 300, "h": 200, "label": "Frame"}
    inner = {"x": 100, "y": 50, "w": 120, "h": 60, "label": "Inner"}
    other = {"x": 500, "y": 50, "w": 100, "h": 60, "label": "Other"}
    # The tail sits 3 px right of Inner's border, still inside Frame.
    arrows = [{"x1": 223, "y1": 80, "x2": 498, "y2": 80, "points": [[223, 80], [498, 80]], "has_head": True}]
    rels = builder.build_relationships(arrows, [outer, inner, other], outside_box_gate=False, max_dist=60.0)
    assert [(r["source"], r["target"]) for r in rels] == [("Inner", "Other")]


def test_lines_along_a_box_edge_are_not_connectors():
    from src.graph import builder
    a = {"x": 0, "y": 0, "w": 200, "h": 100, "label": "A"}
    b = {"x": 0, "y": 140, "w": 200, "h": 100, "label": "B"}
    border = {"x1": 0, "y1": 101, "x2": 0, "y2": 139, "points": [[2, 10], [2, 90]]}
    assert builder._along_box_edge(border, [a, b])
    connector = {"points": [[100, 101], [100, 139]]}
    assert not builder._along_box_edge(connector, [a, b])


def test_branch_inherits_its_trunks_tail():
    from src.graph import builder
    src = {"x": 0, "y": 0, "w": 100, "h": 60, "label": "Src"}
    t1 = {"x": 400, "y": 0, "w": 100, "h": 60, "label": "T1"}
    t2 = {"x": 160, "y": 200, "w": 100, "h": 60, "label": "T2"}
    trunk = {"points": [[101, 30], [399, 30]], "x1": 101, "y1": 30, "x2": 399, "y2": 30, "has_head": True}
    # The branch leaves the trunk's shared first leg at x=210, ending at T2.
    branch = {"points": [[210, 31], [210, 199]], "x1": 210, "y1": 31, "x2": 210, "y2": 199, "has_head": True}
    rels = builder.build_relationships([trunk, branch], [src, t1, t2], outside_box_gate=False, max_dist=60.0)
    assert {(r["source"], r["target"]) for r in rels} == {("Src", "T1"), ("Src", "T2")}


def test_icon_node_found_from_its_caption(tmp_path):
    from src.box_detection import icon_nodes
    img = np.full((300, 400, 3), 255, np.uint8)
    cv2.ellipse(img, (200, 110), (28, 10), 0, 0, 360, (200, 120, 30), 2)       # cylinder top
    cv2.ellipse(img, (200, 170), (28, 10), 0, 0, 180, (200, 120, 30), 2)       # bottom
    cv2.line(img, (172, 110), (172, 170), (200, 120, 30), 2)
    cv2.line(img, (228, 110), (228, 170), (200, 120, 30), 2)
    texts = [{"text": "Orders DB", "x": 165, "y": 190, "w": 72, "h": 14}]
    nodes = icon_nodes.find(img, texts, [])
    assert len(nodes) == 1 and nodes[0]["label"] == "Orders DB"
    n = nodes[0]
    assert 165 <= n["x"] <= 175 and 95 <= n["y"] <= 105 and n["y"] + n["h"] <= 185


# --- review regressions ----------------------------------------------------

def _b(x, y, w, h, label):
    return {"x": x, "y": y, "w": w, "h": h, "label": label}


def _rels(img, tmp_path, boxes, name="scene.png"):
    from src.arrow_detection import directed_lines_detector as det
    from src.graph import builder
    path = str(tmp_path / name)
    cv2.imwrite(path, img)
    arrows = det.detect(path)["arrows"]
    rels = builder.build_relationships(arrows, [dict(b) for b in boxes], outside_box_gate=False,
                                       max_dist=60.0, ray_intersection=True)
    return {(r["source"], r["target"], r["bidirectional"]) for r in rels}


def _labelled_box(img, x, y, w, h, text):
    _box(img, x, y, w, h)
    cv2.putText(img, text, (x + 8, y + h // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 1, cv2.LINE_AA)


def test_connector_touching_filled_boxes_keeps_its_direction(tmp_path):
    """A connector drawn up to dark-filled boxes is one component with them;
    it must not be erased with them, and its head must still be found."""
    img = np.full((240, 820, 3), 255, np.uint8)
    for x, t in ((60, "S0"), (620, "S1")):
        cv2.rectangle(img, (x, 90), (x + 140, 150), (226, 161, 27), -1)
        cv2.putText(img, t, (x + 55, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.line(img, (619, 120), (209, 120), BLACK, 1)
    _head(img, (201, 120), (619, 120))
    assert _rels(img, tmp_path, [_b(60, 90, 140, 60, "S0"), _b(620, 90, 140, 60, "S1")]) == {("S1", "S0", False)}


def test_line_label_bridge_stops_at_a_small_box(tmp_path):
    """Two connectors meeting a small box's sides are not one connector cut
    by a line label, even though the box's label fills the gap."""
    img = np.full((240, 700, 3), 255, np.uint8)
    _labelled_box(img, 40, 90, 120, 60, "Alpha")
    _labelled_box(img, 310, 90, 80, 60, "Queue")
    _labelled_box(img, 540, 90, 120, 60, "Gamma")
    cv2.line(img, (161, 120), (309, 120), BLACK, 1)
    cv2.line(img, (391, 120), (539, 120), BLACK, 1)
    got = _rels(img, tmp_path, [_b(40, 90, 120, 60, "A"), _b(310, 90, 80, 60, "MID"), _b(540, 90, 120, 60, "C")])
    assert {frozenset(r[:2]) for r in got} == {frozenset(("A", "MID")), frozenset(("MID", "C"))}


def test_head_found_on_a_line_thicker_than_the_borders(tmp_path):
    img = np.full((240, 640, 3), 255, np.uint8)
    _labelled_box(img, 60, 90, 140, 60, "Alpha")
    _labelled_box(img, 420, 90, 140, 60, "Beta")
    img[120:123, 213:420] = 0                      # 3-px line, 12-px head
    _head(img, (201, 121), (419, 121), size=12)
    assert _rels(img, tmp_path, [_b(60, 90, 140, 60, "Alpha"), _b(420, 90, 140, 60, "Beta")]) == {("Beta", "Alpha", False)}


def test_chain_through_a_small_icon_is_two_one_way_arrows(tmp_path):
    """A -> icon -> C on one row merges into one line that is cut at the first
    head; the second piece must not claim that head as its own tail head."""
    img = np.full((240, 700, 3), 255, np.uint8)
    _labelled_box(img, 40, 90, 120, 60, "Alpha")
    _labelled_box(img, 540, 90, 120, 60, "Gamma")
    _box(img, 340, 110, 20, 20)
    cv2.line(img, (161, 120), (331, 120), BLACK, 1)
    _head(img, (339, 120), (161, 120))
    cv2.line(img, (361, 120), (531, 120), BLACK, 1)
    _head(img, (539, 120), (361, 120))
    boxes = [_b(40, 90, 120, 60, "A"), _b(340, 110, 20, 20, "ICON"), _b(540, 90, 120, 60, "C")]
    assert _rels(img, tmp_path, boxes) == {("A", "ICON", False), ("ICON", "C", False)}


def test_arrows_converging_on_a_small_node_stay_two_arrows(tmp_path):
    img = np.full((340, 460, 3), 255, np.uint8)
    _box(img, 220, 180, 26, 26)
    _box(img, 20, 150, 100, 60)
    _box(img, 183, 20, 100, 50)
    cv2.line(img, (121, 193), (211, 193), BLACK, 1)
    _head(img, (219, 193), (121, 193))
    cv2.line(img, (233, 71), (233, 171), BLACK, 1)
    _head(img, (233, 179), (233, 71))
    boxes = [_b(20, 150, 100, 60, "L"), _b(183, 20, 100, 50, "T"), _b(220, 180, 26, 26, "ICON")]
    assert _rels(img, tmp_path, boxes) == {("L", "ICON", False), ("T", "ICON", False)}


def test_image_without_background_does_not_crash(tmp_path):
    from src.arrow_detection import directed_lines_detector as det
    img = np.full((300, 600, 3), 190, np.uint8)      # grey paper: no pixel above the ink threshold
    cv2.rectangle(img, (40, 100), (180, 160), BLACK, 2)
    cv2.line(img, (180, 130), (390, 130), BLACK, 2)
    path = str(tmp_path / "grey.png")
    cv2.imwrite(path, img)
    assert isinstance(det.detect(path)["arrows"], list)


def test_two_line_caption_gives_one_icon_node():
    from src.box_detection import icon_nodes
    img = np.full((360, 480, 3), 255, np.uint8)
    cv2.rectangle(img, (200, 60), (280, 140), (22, 161, 122), -1)
    texts = []
    for i, word in enumerate(("Payment", "Gateway")):
        (tw, th), base = cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        x, y = 240 - tw // 2, 168 + i * (th + base + 6)
        cv2.putText(img, word, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, BLACK, 2)
        texts.append({"text": word, "x": x - 1, "y": y - th - 1, "w": tw + 2, "h": th + base + 2})
    nodes = icon_nodes.find(img, texts, [])
    assert [(n["label"], n["w"] >= 60) for n in nodes] == [("Payment Gateway", True)]


def test_caption_on_the_top_edge_is_not_searched_from_the_bottom():
    from src.box_detection import icon_nodes
    img = np.full((295, 400, 3), 255, np.uint8)
    cv2.circle(img, (200, 275), 18, (200, 120, 30), -1)       # a logo at the bottom
    title = {"text": "System Overview", "x": 140, "y": 0, "w": 120, "h": 14}
    assert icon_nodes.find(img, [title], []) == []


def test_builder_keeps_icon_captions_and_merges_double_heads():
    from src.graph import builder
    # The icon node was found from this caption; the box under the caption is
    # even closer to it, but must not take it.
    caption = {"text": "Orders DB", "x": 95, "y": 152, "w": 60, "h": 14}
    icon = {"x": 100, "y": 100, "w": 40, "h": 40, "label": "Orders DB", "entity_type": "icon",
            "caption_boxes": [(95, 152, 60, 14)]}
    below = {"x": 80, "y": 168, "w": 100, "h": 60}
    labels = [b.get("label") for b in builder.label_boxes([below, icon], [caption])]
    assert labels == ["", "Orders DB"]
    a, b = _b(0, 0, 100, 60, "A"), _b(300, 0, 100, 60, "B")
    one_way = {"points": [[101, 30], [299, 30]], "x1": 101, "y1": 30, "x2": 299, "y2": 30, "has_head": True}
    both = dict(one_way, bidirectional=True)
    rels = builder.build_relationships([one_way, both], [a, b], outside_box_gate=False, max_dist=60.0)
    assert [(r["source"], r["target"], r["bidirectional"]) for r in rels] == [("A", "B", True)]


def test_unlabelled_boxes_neither_take_endpoints_nor_filter_connectors():
    from src.graph import builder
    host, glyph = _b(368, 497, 66, 71, "Bastion Host"), {"x": 384, "y": 504, "w": 41, "h": 57}
    app = _b(700, 497, 90, 71, "App Server")
    arrow = {"points": [[378, 532], [699, 532]], "x1": 378, "y1": 532, "x2": 699, "y2": 532, "has_head": True}
    rels = builder.build_relationships([arrow], [host, glyph, app], outside_box_gate=False, max_dist=60.0)
    assert [(r["source"], r["target"]) for r in rels] == [("Bastion Host", "App Server")]
    split = {"x": 141, "y": 224, "w": 303, "h": 66}                 # a contour split along the connector
    web, gw = _b(141, 120, 150, 100, "Web App"), _b(380, 120, 120, 100, "API Gateway")
    conn = {"points": [[297, 227], [379, 227]], "x1": 297, "y1": 227, "x2": 379, "y2": 227, "has_head": True}
    assert not builder._along_box_edge(conn, [web, gw]) and builder._along_box_edge(conn, [split])
    rels = builder.build_relationships([conn], [web, gw, split], outside_box_gate=False, max_dist=60.0)
    assert [(r["source"], r["target"]) for r in rels] == [("Web App", "API Gateway")]


def test_headless_branch_is_not_traced_to_a_trunk():
    from src.graph import builder
    s1, s2, t = _b(0, 0, 100, 60, "S1"), _b(200, 200, 100, 60, "S2"), _b(400, 0, 100, 60, "T")
    trunk = {"points": [[101, 30], [399, 30]], "x1": 101, "y1": 30, "x2": 399, "y2": 30, "has_head": True}
    merge_in = {"points": [[250, 31], [250, 199]], "x1": 250, "y1": 31, "x2": 250, "y2": 199, "has_head": False}
    rels = builder.build_relationships([trunk, merge_in], [s1, s2, t], outside_box_gate=False, max_dist=60.0)
    assert ("S1", "S2") not in {(r["source"], r["target"]) for r in rels}


# --- review round 2 --------------------------------------------------------

def _hline(img, x0, x1, y, lw):
    img[y - lw // 2:y - lw // 2 + lw, min(x0, x1):max(x0, x1) + 1] = 0


def _vline(img, y0, y1, x, lw):
    img[min(y0, y1):max(y0, y1) + 1, x - lw // 2:x - lw // 2 + lw] = 0


def test_filled_icon_between_outlined_boxes_is_not_read_as_lines(tmp_path):
    """Connectors touching a filled icon and the outlined boxes around it make
    one sparse component; the icon is still a solid shape, not 20 lines."""
    img = np.full((260, 900, 3), 255, np.uint8)
    _labelled_box(img, 40, 100, 120, 60, "Client")
    cv2.rectangle(img, (201, 98), (265, 162), (0, 113, 237), -1)
    cv2.rectangle(img, (217, 112), (249, 126), (255, 255, 255), 2)
    cv2.rectangle(img, (217, 132), (249, 146), (255, 255, 255), 2)
    _labelled_box(img, 306, 100, 120, 60, "Backend")
    cv2.line(img, (161, 130), (192, 130), BLACK, 1)
    _head(img, (200, 130), (161, 130), size=9)
    cv2.line(img, (266, 130), (296, 130), BLACK, 1)
    _head(img, (305, 130), (266, 130), size=9)
    boxes = [_b(40, 100, 120, 60, "Client"), _b(201, 98, 64, 64, "Lambda"), _b(306, 100, 120, 60, "Backend")]
    assert _rels(img, tmp_path, boxes) == {("Client", "Lambda", False), ("Lambda", "Backend", False)}


@pytest.mark.parametrize("size", [14, 16, 18])
def test_large_head_touching_a_filled_box_is_kept(tmp_path, size):
    img = np.full((240, 820, 3), 255, np.uint8)
    cv2.rectangle(img, (60, 90), (200, 150), (0, 155, 215), -1)
    cv2.putText(img, "Queue", (95, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    _labelled_box(img, 281, 90, 140, 60, "Producer")
    _hline(img, 201 + size - 2, 281, 120, 2)
    _head(img, (201, 120), (281, 120), size=size)
    assert _rels(img, tmp_path, [_b(60, 90, 140, 60, "Q"), _b(281, 90, 140, 60, "P")]) == {("P", "Q", False)}


@pytest.mark.parametrize("size,lw", [(12, 3), (16, 4)])
def test_thick_arrows_meeting_at_a_box_corner_stay_two_arrows(tmp_path, size, lw):
    img = np.full((400, 500, 3), 255, np.uint8)
    _labelled_box(img, 250, 200, 140, 70, "Target")
    _labelled_box(img, 20, 160, 110, 60, "Left")
    _labelled_box(img, 190, 20, 110, 50, "Top")
    _hline(img, 131, 249 - size + 2, 205, lw)
    _head(img, (249, 205), (131, 205), size=size)
    _vline(img, 71, 199 - size + 2, 256, lw)
    _head(img, (256, 199), (256, 71), size=size)
    boxes = [_b(250, 200, 140, 70, "T"), _b(20, 160, 110, 60, "L"), _b(190, 20, 110, 50, "U")]
    assert _rels(img, tmp_path, boxes) == {("L", "T", False), ("U", "T", False)}


@pytest.mark.parametrize("lw,size", [(3, 14), (4, 16)])
def test_thick_chain_through_an_icon(tmp_path, lw, size):
    """Thick shafts must neither be cut mid-way nor let the second piece
    claim the first arrow's head."""
    img = np.full((240, 700, 3), 255, np.uint8)
    _labelled_box(img, 40, 90, 120, 60, "Alpha")
    _labelled_box(img, 540, 90, 120, 60, "Gamma")
    _box(img, 340, 110, 20, 20)
    _hline(img, 161, 339 - size + 1, 120, lw)
    _head(img, (339, 120), (161, 120), size=size)
    _hline(img, 361, 539 - size + 1, 120, lw)
    _head(img, (539, 120), (361, 120), size=size)
    boxes = [_b(40, 90, 120, 60, "A"), _b(340, 110, 20, 20, "ICON"), _b(540, 90, 120, 60, "C")]
    assert _rels(img, tmp_path, boxes) == {("A", "ICON", False), ("ICON", "C", False)}


@pytest.mark.parametrize("gap", [5, 6, 7, 8])
def test_dash_gap_behind_a_head_touching_a_thick_border(tmp_path, gap):
    img = np.full((240, 700, 3), 255, np.uint8)
    _box(img, 40, 90, 120, 60)
    cv2.rectangle(img, (40, 90), (160, 150), BLACK, 2)
    cv2.rectangle(img, (540, 90), (660, 150), BLACK, 2)
    _head(img, (162, 120), (539, 120), size=9)
    x = 162 + 9 + gap
    while x < 538:
        img[120, x:min(538, x + 8)] = 0
        x += 16
    boxes = [_b(40, 90, 120, 60, "Alpha"), _b(540, 90, 120, 60, "Gamma")]
    assert _rels(img, tmp_path, boxes) == {("Gamma", "Alpha", False)}


def test_stray_segment_between_two_heads_is_not_a_connector(tmp_path):
    """Two arrows converging on a box corner: a short line between their
    heads must not take both and become a double-headed connector."""
    img = np.full((400, 500, 3), 255, np.uint8)
    _labelled_box(img, 250, 200, 140, 70, "Target")
    _labelled_box(img, 20, 160, 110, 60, "Left")
    _labelled_box(img, 190, 20, 110, 50, "Top")
    cv2.line(img, (131, 205), (241, 205), BLACK, 1)
    _head(img, (249, 205), (131, 205), size=8)
    cv2.line(img, (256, 71), (256, 191), BLACK, 1)
    _head(img, (256, 199), (256, 71), size=8)
    boxes = [_b(250, 200, 140, 70, "T"), _b(20, 160, 110, 60, "L"), _b(190, 20, 110, 50, "U")]
    assert _rels(img, tmp_path, boxes) == {("L", "T", False), ("U", "T", False)}


def _txt(img, word, cx, y, scale=0.5, thick=1, color=BLACK, pad=1):
    (tw, th), base = cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x = int(cx - tw / 2)
    cv2.putText(img, word, (x, y + th), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
    return {"text": word, "x": x - pad, "y": y - pad, "w": tw + 2 * pad, "h": th + base + 2 * pad}


@pytest.mark.parametrize("reverse", [False, True])
def test_icon_with_text_inside_is_found_from_its_caption(reverse):
    """Text written inside an icon is not its caption, and a padded OCR box
    must not cut the icon's outline away."""
    from src.box_detection import icon_nodes
    blue = (200, 120, 30)
    img = np.full((300, 400, 3), 255, np.uint8)
    cv2.ellipse(img, (200, 100), (28, 10), 0, 0, 360, blue, 2)
    cv2.ellipse(img, (200, 160), (28, 10), 0, 0, 180, blue, 2)
    cv2.line(img, (172, 100), (172, 160), blue, 2)
    cv2.line(img, (228, 100), (228, 160), blue, 2)
    texts = [_txt(img, "SQL", 200, 135, 0.6, 2, blue), _txt(img, "Orders DB", 200, 180)]
    nodes = icon_nodes.find(img, texts[::-1] if reverse else texts, [])
    assert [(n["label"], n["h"] >= 75) for n in nodes] == [("Orders DB", True)]
    img = np.full((300, 400, 3), 255, np.uint8)
    cv2.rectangle(img, (172, 80), (228, 140), blue, 2)
    texts = [_txt(img, "KAFKA", 200, 104, color=blue, pad=7), _txt(img, "Event Bus", 200, 150, pad=3)]
    nodes = icon_nodes.find(img, texts[::-1] if reverse else texts, [])
    assert [(n["label"], n["h"] >= 55) for n in nodes] == [("Event Bus", True)]


def test_arrow_ending_at_an_unlabelled_node_stops_there():
    from src.graph import builder
    a, c = _b(0, 0, 100, 60, "A"), _b(380, 0, 100, 60, "C")
    logo = {"x": 200, "y": 0, "w": 80, "h": 60}                    # no caption, no text
    arrow = {"points": [[101, 30], [199, 30]], "x1": 101, "y1": 30, "x2": 199, "y2": 30, "has_head": True}
    assert builder.build_relationships([arrow], [a, logo, c], outside_box_gate=False, max_dist=60.0,
                                       ray_intersection=True) == []


def test_captions_are_kept_by_exact_fragment_not_by_substring():
    from src.box_detection import icon_nodes
    from src.graph import builder
    img = np.full((360, 480, 3), 255, np.uint8)
    cv2.rectangle(img, (220, 60), (261, 101), (22, 161, 122), -1)
    texts = [_txt(img, "Payment", 240, 105, 0.6), _txt(img, "Gateway", 240, 126, 0.6)]
    nodes = icon_nodes.find(img, texts, [])
    below = {"x": 180, "y": texts[1]["y"] + texts[1]["h"] + 8, "w": 120, "h": 60}
    assert [b.get("label", "") for b in builder.label_boxes([below] + nodes, texts)] == ["", "Payment Gateway"]
    web, small = {"x": 100, "y": 0, "w": 160, "h": 100}, {"x": 165, "y": 108, "w": 30, "h": 30}
    texts = [{"text": "Web App", "x": 140, "y": 40, "w": 80, "h": 16}, {"text": "App", "x": 166, "y": 143, "w": 28, "h": 14}]
    assert [b["label"] for b in builder.label_boxes([web, small], texts)] == ["Web App", "App"]


def test_branches_splice_the_trunk_path_and_resolve_chains():
    from src.graph import builder

    def arrow(pts, **kw):
        return dict({"points": pts, "x1": pts[0][0], "y1": pts[0][1], "x2": pts[-1][0], "y2": pts[-1][1],
                     "has_head": True}, **kw)

    def rels(arrows, boxes):
        return {(r["source"], r["target"]) for r in builder.build_relationships(
            arrows, boxes, outside_box_gate=False, max_dist=60.0, ray_intersection=True)}

    src, t1, t2, t3 = _b(0, 0, 100, 60, "Src"), _b(400, 0, 100, 60, "T1"), _b(160, 200, 100, 60, "T2"), _b(330, 120, 100, 60, "T3")
    # The trunk stops short of its source; the branch still reaches Src along the trunk's leg.
    assert rels([arrow([[170, 30], [399, 30]]), arrow([[210, 31], [210, 199]])], [src, t1, t2]) == {("Src", "T1"), ("Src", "T2")}
    # A branch off a branch.
    assert rels([arrow([[101, 30], [399, 30]]), arrow([[210, 31], [210, 199]]), arrow([[211, 150], [329, 150]])],
                [src, t1, t2, t3]) == {("Src", "T1"), ("Src", "T2"), ("Src", "T3")}
    # A double-headed trunk has no tail to inherit.
    got = rels([arrow([[399, 30], [101, 30]], bidirectional=True), arrow([[210, 31], [210, 199]])], [src, t1, t2])
    assert not {("T1", "T2"), ("Src", "T2")} & got


def test_double_headed_detection_covers_earlier_one_way_rows():
    from src.graph import builder
    a, b = _b(0, 0, 100, 60, "A"), _b(300, 0, 100, 60, "B")
    ab = {"points": [[101, 30], [299, 30]], "x1": 101, "y1": 30, "x2": 299, "y2": 30, "has_head": True}
    ba = dict(ab, points=[[299, 30], [101, 30]], x1=299, x2=101)
    both = dict(ab, bidirectional=True)
    rels = builder.build_relationships([ab, ba, both], [a, b], outside_box_gate=False, max_dist=60.0)
    assert [(r["source"], r["target"], r["bidirectional"]) for r in rels] == [("A", "B", True)]


def test_scoring_counts_each_head_of_a_double_headed_edge():
    import benchmark_ablation as ba
    key = [{"source": "A", "target": "B", "bidirectional": True}, {"source": "C", "target": "D"}]
    # One head of A <-> B found, C -> D reversed: 1 of 3 directed key edges.
    s = ba._score_relationships([("B", "A", False), ("D", "C", False)], key)
    assert (s["tp"], s["fp"], s["fn"]) == (1, 1, 2)
    # Both heads found, and a false second head on C -> D costs a false positive.
    s = ba._score_relationships([("A", "B", True), ("D", "C", True)], key)
    assert (s["tp"], s["fp"], s["fn"]) == (3, 1, 0)
