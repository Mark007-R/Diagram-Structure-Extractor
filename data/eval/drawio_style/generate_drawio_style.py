#!/usr/bin/env python
"""Deterministic generator for the draw.io-style DEVELOPMENT set (8 synthetic diagrams).

Built as a held-out set, independently of the detector code; it was then used
to tune the draw.io handling (2026-10-05), so it is a development set now.
data/eval/drawio_heldout/ is the untouched held-out set.

Matches draw.io's default export look (see search_interview_test.png): 1 px black outlines,
regular-weight 12 px Helvetica/Arial text, white or pale-filled rectangles, rounded pale containers
(draw.io rounding: r = 0.15 * min(w, h)), blue database-cylinder icons with the caption BELOW the icon,
1 px solid / dashed connectors, draw.io "classic" filled arrowheads (~9 px long like the reference),
orthogonal elbow connectors (1-2 bends, some with rounded bends), double-headed connectors and
connector labels drawn on a white label background.

No randomness. Every coordinate below is in canvas pixels (origin top-left, y down).  DPI is a power
of two so that pixel <-> inch conversions are exact and the cropped PNG maps 1:1 onto canvas pixels.

Outputs (next to this file):
  diagrams/drawio_XX.png, ground_truth.json, ground_truth_boxes.json, spec_resolved.json
(spec_resolved.json -- resolved connector paths -- is not kept in the repo.)

Needs Arial. Without it matplotlib falls back to DejaVu Sans, the text extents
change, the crops and box key shift and a self-check fails -- so the script
stops before writing anything, and the committed PNGs and keys stay the
reference.
"""
import json
import math
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.path import Path  # noqa: E402
from matplotlib.patches import PathPatch, Polygon  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "diagrams")

DPI = 128                      # power of two -> exact px/inch arithmetic
PT = 72.0 / DPI                # points per pixel
FONT = ["Arial", "DejaVu Sans"]   # draw.io "Helvetica" renders as Arial-like
matplotlib.rcParams.update({
    "lines.scale_dashes": False,
    "font.family": FONT,
    "path.simplify": False,
    "text.antialiased": True,
})

BLACK = "#000000"
ICON_BLUE = "#0080F0"          # sampled from the reference database icon (0,128,240)
PALETTE = {                    # draw.io's built-in pale fill / stroke pairs
    "blue": ("#dae8fc", "#6c8ebf"),
    "green": ("#d5e8d4", "#82b366"),
    "gray": ("#f5f5f5", "#666666"),
    "orange": ("#ffe6cc", "#d79b00"),
    "yellow": ("#fff2cc", "#d6b656"),
    "purple": ("#e1d5e7", "#9673a6"),
    "red": ("#f8cecc", "#b85450"),
}
DASH_PX = (8, 8)               # measured on the reference export (~8-9 px dash, ~8 px gap)
EDGE_LABEL_FS = 12           # reference: "VPN"/"DB" ~12 px Arial-metric text
DEFAULT_FS = 13              # reference box labels measure ~13 px Arial ("ELSER Model" 81 px wide)
ARROW_SIZE = 7.0               # draw.io classic marker; reference heads measure ~9 px long
EDGE_SW = 1.0
BEND_RADIUS = 8.0


# ----------------------------------------------------------------------------------------------
# spec helpers
# ----------------------------------------------------------------------------------------------
def box(id, label, x, y, w, h, parent=None, fill="#ffffff", fs=None):
    return dict(id=id, kind="box", label=label, x=x, y=y, w=w, h=h, parent=parent, fill=fill,
                stroke=BLACK, fs=fs)


def rbox(id, label, x, y, w, h, parent=None, fill="#ffffff", fs=None):
    d = box(id, label, x, y, w, h, parent, fill, fs)
    d["kind"] = "rounded_box"
    return d


def db(id, label, x, y, w=52, h=64, parent=None, fs=None, cap_bg=True):
    return dict(id=id, kind="icon_db", label=label, x=x, y=y, w=w, h=h, parent=parent, fs=fs,
                cap_bg=cap_bg)


def cont(id, title, x, y, w, h, color="blue", parent=None, align="center", fs=None, arc=0.15):
    fill, stroke = PALETTE[color]
    return dict(id=id, kind="container", shape="rounded", label=title, x=x, y=y, w=w, h=h,
                parent=parent, fill=fill, stroke=stroke, align=align, fs=fs, arc=arc)


def group(id, title, x, y, w, h, parent=None, fs=17, fill=None):
    """Sharp-cornered titled group (like 'Front-End' in the reference) -- still a container."""
    return dict(id=id, kind="container", shape="rect", label=title, x=x, y=y, w=w, h=h,
                parent=parent, fill=fill, stroke=BLACK, align="center", fs=fs, arc=0.0)


def conn(frm, to, style="solid", heads="to", src="r", dst="l", route="straight", mid=None,
         wp=None, label=None, label_pos=0.5, rounded=False):
    return dict(frm=frm, to=to, style=style, heads=heads, src=src, dst=dst, route=route, mid=mid,
                wp=wp, label=label, label_pos=label_pos, rounded=rounded)


# ----------------------------------------------------------------------------------------------
# the eight diagrams
# ----------------------------------------------------------------------------------------------
DIAGRAMS = [
    dict(file="drawio_01.png", name="Web shop on AWS", W=1200, H=550, crisp=False, border=12,
         nodes=[
             cont("aws", "AWS Cloud", 300, 30, 600, 490, "blue", align="left"),
             cont("priv", "Private Subnet", 560, 105, 310, 385, "gray", parent="aws"),
             box("browser", "Browser", 40, 240, 120, 60),
             box("lb", "Load Balancer", 340, 240, 130, 60, parent="aws"),
             box("api", "API Server", 600, 155, 130, 60, parent="priv"),
             rbox("worker", "Worker", 600, 380, 130, 60, parent="priv"),
             db("odb", "Orders DB", 790, 156, 52, 58, parent="priv", fs=15),
             box("pay", "Payment\nGateway", 985, 240, 140, 60),
         ],
         conns=[
             conn("browser", "lb", "solid", src="r", dst="l", label="HTTPS", label_pos=0.45),
             conn("lb", "api", "solid", src="r", dst="l", route="hvh", mid=530, rounded=True),
             conn("lb", "worker", "dashed", src="b", dst="l", route="vh"),
             conn("api", "odb", "dashed", "both", src="r", dst="l"),
             conn("api", "pay", "solid", src="t", dst="t", route="vhv", mid=75),
             conn("pay", "worker", "dashed", src="b", dst="r", route="vh", label="Webhook"),
         ]),
    dict(file="drawio_02.png", name="Streaming data platform", W=1250, H=600, crisp=False, border=10, fs=12,
         nodes=[
             cont("ing", "Ingestion", 30, 40, 330, 520, "green"),
             cont("proc", "Processing", 440, 40, 360, 520, "orange"),
             cont("stor", "Storage", 880, 40, 340, 520, "purple"),
             box("app", "Mobile App", 110, 120, 150, 60, parent="ing"),
             box("iapi", "Ingest API", 110, 360, 150, 60, parent="ing"),
             box("sp", "Stream\nProcessor", 540, 120, 160, 60, parent="proc"),
             rbox("batch", "Batch Jobs", 540, 358, 160, 64, parent="proc"),
             db("lake", "Data Lake", 1000, 118, 56, 64, parent="stor"),
             db("wh", "Warehouse", 1000, 358, 56, 64, parent="stor"),
         ],
         conns=[
             conn("app", "iapi", "solid", src="b", dst="t", label="REST"),
             conn("app", "sp", "solid", src="r", dst="l", label="gRPC"),
             conn("iapi", "batch", "dashed", src="r", dst="l"),
             conn("sp", "lake", "dashed", "both", src="r", dst="l"),
             conn("batch", "wh", "solid", src="r", dst="l", label="JDBC"),
             conn("sp", "batch", "solid", "both", src="b", dst="t"),
             conn("batch", "lake", "dashed", src=("r", -15), dst=("l", 16), route="hvh", mid=840,
                  rounded=True),
         ]),
    dict(file="drawio_03.png", name="Kubernetes microservices", W=1250, H=650, crisp=True, border=12,
         nodes=[
             cont("k8s", "Kubernetes Cluster", 290, 30, 650, 600, "blue", align="left", arc=0.08),
             group("fe", "Frontend", 330, 100, 260, 200, parent="k8s"),
             group("be", "Backend", 640, 100, 270, 500, parent="k8s"),
             box("client", "Client App", 40, 170, 140, 60),
             box("web", "Web UI", 375, 170, 170, 60, parent="fe"),
             box("auth", "Auth Service", 685, 170, 180, 60, parent="be"),
             box("order", "Order Service", 685, 320, 180, 60, parent="be"),
             box("inv", "Inventory\nService", 685, 470, 180, 60, parent="be"),
             db("redis", "Redis Cache", 1050, 168, 52, 64),
             db("pg", "Postgres", 1050, 318, 52, 64),
         ],
         conns=[
             conn("client", "web", "solid", src="r", dst="l", label="HTTPS", label_pos=0.3),
             conn("web", "auth", "dashed", "both", src="r", dst="l"),
             conn("web", "order", "solid", src="b", dst="l", route="vh"),
             conn("auth", "redis", "dashed", src="r", dst="l", label="TLS", label_pos=0.6),
             conn("order", "pg", "solid", src="r", dst="l", label="SQL", label_pos=0.6),
             conn("order", "inv", "solid", src="b", dst="t"),
             conn("order", "auth", "dashed", src="t", dst="b"),
             conn("inv", "pg", "dashed", src="r", dst=("l", 16), route="hvh", mid=1005,
                  rounded=True),
         ]),
    dict(file="drawio_04.png", name="Hybrid on-prem to Azure connectivity", W=1200, H=500,
         crisp=False, border=14,
         nodes=[
             cont("onprem", "On-Premises Data Center", 30, 50, 420, 400, "green"),
             cont("azure", "Azure VNet", 720, 50, 450, 400, "blue"),
             box("erp", "ERP Server", 80, 140, 140, 60, parent="onprem"),
             db("ldb", "Legacy DB", 300, 138, 52, 64, parent="onprem"),
             box("dc", "Domain\nController", 80, 320, 140, 60, parent="onprem"),
             box("agw", "App Gateway", 780, 140, 140, 60, parent="azure"),
             box("webapp", "Web App", 1000, 140, 140, 60, parent="azure"),
             db("sqlmi", "SQL MI", 1044, 310, 52, 64, parent="azure"),
         ],
         conns=[
             conn("erp", "ldb", "solid", "both", src="r", dst="l"),
             conn("dc", "erp", "dashed", src="t", dst="b"),
             conn("dc", "agw", "dashed", src="r", dst="l", route="hvh", mid=585, label="VPN",
                  label_pos=0.86, rounded=True),
             conn("agw", "webapp", "solid", src="r", dst="l", label="HTTPS"),
             conn("webapp", "sqlmi", "dashed", "both", src="b", dst="t"),
             conn("agw", "sqlmi", "solid", src="b", dst="l", route="vh"),
         ]),
    dict(file="drawio_05.png", name="CI/CD delivery pipeline", W=1300, H=525, crisp=True, border=12, fs=12,
         nodes=[
             cont("tgt", "Deploy Targets", 760, 110, 420, 250, "yellow"),
             box("dev", "Developer", 30, 200, 120, 60),
             rbox("git", "Git Repo", 220, 200, 120, 60),
             box("ci", "CI Runner", 410, 200, 120, 60),
             db("art", "Artifacts", 615, 198, 52, 64),
             box("stg", "Staging", 800, 200, 120, 60, parent="tgt"),
             box("prod", "Production", 1010, 200, 130, 60, parent="tgt"),
             box("mon", "Monitoring", 1010, 420, 130, 60),
         ],
         conns=[
             conn("dev", "git", "solid", src="r", dst="l", label="push"),
             conn("git", "ci", "solid", src="r", dst="l"),
             conn("ci", "art", "dashed", "both", src="r", dst="l"),
             conn("art", "stg", "solid", src="r", dst="l"),
             conn("stg", "prod", "solid", src="r", dst="l"),
             conn("prod", "mon", "dashed", src="b", dst="t", label="metrics", label_pos=0.4),
             conn("mon", "dev", "dashed", src="l", dst="b", route="hv", label="alerts"),
             conn("ci", "stg", "dashed", src="t", dst=("t", -20), route="vhv", mid=60,
                  label="deploy", rounded=True),
         ]),
    dict(file="drawio_06.png", name="IoT telemetry", W=1100, H=650, crisp=False, border=12,
         nodes=[
             cont("edge", "Edge Site", 40, 30, 440, 280, "green", align="left"),
             cont("cloud", "Cloud Platform", 40, 370, 1020, 260, "blue", align="left", arc=0.1),
             box("sa", "Sensor A", 80, 90, 120, 50, parent="edge"),
             box("sb", "Sensor B", 300, 90, 120, 50, parent="edge"),
             rbox("gw", "Edge\nGateway", 190, 200, 140, 60, parent="edge"),
             box("hub", "IoT Hub", 190, 450, 140, 60, parent="cloud"),
             box("rules", "Rules Engine", 480, 450, 140, 60, parent="cloud"),
             db("tdb", "Telemetry DB", 723, 448, 54, 64, parent="cloud"),
             box("dash", "Dashboard", 880, 450, 140, 60, parent="cloud"),
             box("ops", "Ops Team", 700, 120, 140, 60),
         ],
         conns=[
             conn("sa", "gw", "solid", src="b", dst="l", route="vh"),
             conn("sb", "gw", "solid", src="b", dst="r", route="vh", rounded=True),
             conn("gw", "hub", "dashed", src="b", dst="t", label="MQTT", label_pos=0.38),
             conn("hub", "rules", "solid", src="r", dst="l"),
             conn("rules", "tdb", "dashed", "both", src="r", dst="l"),
             conn("tdb", "dash", "solid", src="r", dst="l"),
             conn("rules", "ops", "solid", src="t", dst="l", route="vh", label="alert"),
             conn("ops", "dash", "dashed", src="r", dst="t", route="hv", rounded=True),
         ]),
    dict(file="drawio_07.png", name="OAuth token flow", W=1200, H=450, crisp=True, border=12, fs=14,
         nodes=[
             cont("zone", "Trust Boundary", 300, 30, 260, 390, "gray"),
             rbox("client", "Client", 40, 190, 130, 60, fill=PALETTE["blue"][0]),
             box("idp", "Identity\nProvider", 340, 80, 180, 60, parent="zone",
                 fill=PALETTE["yellow"][0]),
             box("ra", "Resource API", 340, 300, 180, 60, parent="zone",
                 fill=PALETTE["green"][0]),
             db("cache", "Token Cache", 820, 78, 52, 64),
             box("audit", "Audit Log", 800, 300, 150, 60, fill=PALETTE["red"][0]),
             box("siem", "SIEM", 1040, 300, 120, 60),
         ],
         conns=[
             conn("client", "idp", "solid", "both", src="t", dst="l", route="vh", label="OIDC",
                  rounded=True),
             conn("client", "ra", "solid", src="b", dst="l", route="vh", label="Bearer",
                  label_pos=0.6),
             conn("ra", "idp", "dashed", src="t", dst="b"),
             conn("idp", "cache", "dashed", "both", src="r", dst="l"),
             conn("ra", "audit", "solid", src="r", dst="l"),
             conn("audit", "siem", "dashed", src="r", dst="l", label="syslog"),
         ]),
    dict(file="drawio_08.png", name="Analytics on Google Cloud", W=1300, H=725, crisp=False,
         border=12, fs=12, efs=11,
         nodes=[
             cont("gcp", "Google Cloud", 240, 30, 1010, 660, "blue", align="left", arc=0.1),
             cont("dz", "Data Zone", 285, 95, 560, 330, "orange", parent="gcp"),
             group("serv", "Serving", 285, 470, 440, 190, parent="gcp"),
             box("sftp", "Partner SFTP", 30, 185, 140, 60),
             db("gcs", "Cloud Storage", 330, 183, 54, 64, parent="dz"),
             box("flow", "Dataflow", 480, 185, 140, 60, parent="dz"),
             db("bq", "BigQuery", 700, 183, 54, 64, parent="dz"),
             box("looker", "Looker", 960, 185, 140, 60, parent="gcp"),
             box("mapi", "Model API", 330, 545, 150, 60, parent="serv"),
             db("fstore", "Feature Store", 580, 543, 54, 64, parent="serv"),
             box("vertex", "Vertex AI", 960, 545, 140, 60, parent="gcp"),
         ],
         conns=[
             conn("sftp", "gcs", "solid", src="r", dst="l", label="SFTP", label_pos=0.25),
             conn("gcs", "flow", "solid", src="r", dst="l"),
             conn("flow", "bq", "solid", src="r", dst="l"),
             conn("bq", "looker", "dashed", src="r", dst="l"),
             conn("flow", "mapi", "dashed", src="b", dst="t", route="vhv", mid=510, rounded=True),
             conn("mapi", "fstore", "solid", "both", src="r", dst="l"),
             conn("vertex", "looker", "solid", src="t", dst="b", label="REST"),
             conn("fstore", "vertex", "dashed", src="r", dst="l"),
         ]),
]


# ----------------------------------------------------------------------------------------------
# geometry
# ----------------------------------------------------------------------------------------------
def is_component(n):
    return n["kind"] in ("box", "rounded_box", "icon_db")


def node_rect(n):
    """Pixel-edge rectangle (x0, y0, x1, y1): the outline covers pixel columns x0..x1-1."""
    return (n["x"], n["y"], n["x"] + n["w"], n["y"] + n["h"])


def corner_radius(n):
    if n["kind"] == "rounded_box":
        return 0.15 * min(n["w"], n["h"])
    if n["kind"] == "container" and n["shape"] == "rounded":
        return n["arc"] * min(n["w"], n["h"])
    return 0.0


def snap_along(v, crisp):
    return math.floor(v) + 0.5 if crisp else float(math.floor(v + 0.5))


def side_and_offset(s):
    return (s, 0.0) if isinstance(s, str) else (s[0], float(s[1]))


def port(n, side, off, crisp):
    x, y, w, h = n["x"], n["y"], n["w"], n["h"]
    inset = 1.0 if n["kind"] == "icon_db" else 0.5      # centre of the outline stroke
    if n["kind"] == "icon_db" and side == "b":
        raise ValueError("icon bottom is reserved for its caption")
    if side in ("l", "r"):
        v = snap_along(y + h / 2.0 + off, crisp)
        return np.array([x + inset if side == "l" else x + w - inset, v])
    v = snap_along(x + w / 2.0 + off, crisp)
    return np.array([v, y + inset if side == "t" else y + h - inset])


def route_points(c, ps, pd, crisp):
    r = c["route"]
    m = snap_along(c["mid"], crisp) if c.get("mid") is not None else None
    if r == "straight":
        return [ps, pd]
    if r == "hv":
        return [ps, np.array([pd[0], ps[1]]), pd]
    if r == "vh":
        return [ps, np.array([ps[0], pd[1]]), pd]
    if r == "hvh":
        return [ps, np.array([m, ps[1]]), np.array([m, pd[1]]), pd]
    if r == "vhv":
        return [ps, np.array([ps[0], m]), np.array([pd[0], m]), pd]
    if r == "pts":
        return [ps] + [np.array(p, float) for p in c["wp"]] + [pd]
    raise ValueError(r)


def classic_head(pe, u, sw=EDGE_SW, size=ARROW_SIZE):
    """draw.io mxMarker 'classic' (widthFactor 2): returns polygon and where the line must stop."""
    L = size + sw
    tip = pe - u * 1.118 * sw          # the miter of the stroked tip reaches pe exactly
    nrm = np.array([-u[1], u[0]])
    poly = [tip, tip - u * L + nrm * L / 2.0, tip - u * L * 0.75, tip - u * L - nrm * L / 2.0]
    line_end = pe - u * (L * 0.75 + 1.118 * sw)
    return np.array(poly), line_end


def unit(v):
    return v / np.linalg.norm(v)


def round_corners(P, radius):
    out = [P[0]]
    for i in range(1, len(P) - 1):
        a, b, c = P[i - 1], P[i], P[i + 1]
        r = min(radius, np.linalg.norm(b - a) / 2.0, np.linalg.norm(c - b) / 2.0)
        p1 = b + unit(a - b) * r
        p2 = b + unit(c - b) * r
        for t in np.linspace(0, 1, 9):
            out.append((1 - t) ** 2 * p1 + 2 * (1 - t) * t * b + t ** 2 * p2)
    out.append(P[-1])
    return out


def point_at(P, frac):
    seg = [np.linalg.norm(P[i + 1] - P[i]) for i in range(len(P) - 1)]
    target = frac * sum(seg)
    for i, s in enumerate(seg):
        if target <= s or i == len(seg) - 1:
            return P[i] + (P[i + 1] - P[i]) * (target / s), i, target, s
        target -= s


def densify(P, step=0.5):
    pts = []
    for i in range(len(P) - 1):
        a, b = np.asarray(P[i], float), np.asarray(P[i + 1], float)
        n = max(1, int(math.ceil(np.linalg.norm(b - a) / step)))
        t = np.linspace(0, 1, n + 1)[:-1] if i < len(P) - 2 else np.linspace(0, 1, n + 1)
        pts.append(a[None, :] + (b - a)[None, :] * t[:, None])
    return np.vstack(pts)


def resolve_connector(c, nodes, crisp):
    s, d = nodes[c["frm"]], nodes[c["to"]]
    ss, so = side_and_offset(c["src"])
    ds, do = side_and_offset(c["dst"])
    ps = port(s, ss, so, crisp)
    pd = port(d, ds, do, crisp)
    if c["route"] == "straight":
        if ss in "lr" and ds in "lr":
            pd[1] = ps[1]
        elif ss in "tb" and ds in "tb":
            pd[0] = ps[0]
    P = route_points(c, ps, pd, crisp)
    heads_end = True
    heads_start = c["heads"] == "both"
    polys = []
    line = [p.copy() for p in P]
    if heads_end:
        poly, le = classic_head(P[-1], unit(P[-1] - P[-2]))
        polys.append(poly)
        line[-1] = le
    if heads_start:
        poly, le = classic_head(P[0], unit(P[0] - P[1]))
        polys.append(poly)
        line[0] = le
    if c["rounded"] and len(line) > 2:
        line = round_corners(line, BEND_RADIUS)
    lab = None
    if c.get("label"):
        lp, seg_i, along, seg_len = point_at(P, c["label_pos"])
        lab = dict(text=c["label"], pos=lp, seg=seg_i, along=along, seg_len=seg_len)
    return dict(P=P, line=line, polys=polys, label=lab, ps=ps, pd=pd, ss=ss, ds=ds)


def cylinder(n):
    x, y, w, h = n["x"], n["y"], n["w"], n["h"]
    lw = 2.0
    L, R, T, B = x + lw / 2, x + w - lw / 2, y + lw / 2, y + h - lw / 2
    cx, rx, ry = (L + R) / 2.0, (R - L) / 2.0, round(0.12 * h, 1)
    cyT, cyB = T + ry, B - ry

    def arc(cy, t0, t1, k=48):
        t = np.linspace(t0, t1, k)
        return np.column_stack([cx + rx * np.cos(t), cy + ry * np.sin(t)])

    body = np.vstack([arc(cyT, math.pi, 2 * math.pi), [[R, cyB]], arc(cyB, 0, math.pi), [[L, cyT]]])
    strokes = [arc(cyT, 0, 2 * math.pi, 96), np.array([[L, cyT], [L, cyB]]),
               np.array([[R, cyT], [R, cyB]])]
    for k in (1, 2, 3):
        strokes.append(arc(cyT + k * (cyB - cyT) / 3.0, 0, math.pi))
    return body, strokes


def polyline_path(polys):
    verts, codes = [], []
    for p in polys:
        verts.extend(p.tolist())
        codes.extend([Path.MOVETO] + [Path.LINETO] * (len(p) - 1))
    return Path(verts, codes)


def rect_path(x0, y0, x1, y1, r):
    if r <= 0:
        return Path([(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)],
                    [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY])
    r = min(r, (x1 - x0) / 2.0, (y1 - y0) / 2.0)
    V = [(x0 + r, y0), (x1 - r, y0), (x1, y0), (x1, y0 + r), (x1, y1 - r), (x1, y1), (x1 - r, y1),
         (x0 + r, y1), (x0, y1), (x0, y1 - r), (x0, y0 + r), (x0, y0), (x0 + r, y0), (x0 + r, y0)]
    C = [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CURVE3, Path.CURVE3,
         Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CURVE3, Path.CURVE3,
         Path.CLOSEPOLY]
    return Path(V, C)


def outline_path(n):
    """Stroke-centre path: a 1 px outline exactly covering pixel columns x..x+w-1."""
    x0, y0, x1, y1 = node_rect(n)
    return rect_path(x0 + 0.5, y0 + 0.5, x1 - 0.5, y1 - 0.5, corner_radius(n))


def gt_label(n):
    return " ".join(n["label"].split("\n"))


# ----------------------------------------------------------------------------------------------
# rendering
# ----------------------------------------------------------------------------------------------
def new_canvas(W, H):
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_axis_off()
    return fig, ax


def depth(n, nodes):
    d = 0
    while n.get("parent"):
        n = nodes[n["parent"]]
        d += 1
    return d


def render(D):
    W, H, crisp = D["W"], D["H"], D["crisp"]
    for n in D["nodes"]:
        if n.get("fs") is None:
            n["fs"] = D.get("fs", DEFAULT_FS)
    efs = D.get("efs", EDGE_LABEL_FS)
    nodes = {n["id"]: n for n in D["nodes"]}
    fig, ax = new_canvas(W, H)
    texts = []        # (kind, owner, artist)

    # containers (outer first)
    for n in sorted([n for n in D["nodes"] if n["kind"] == "container"], key=lambda n: depth(n, nodes)):
        z = 1 + 0.1 * depth(n, nodes)
        ax.add_patch(PathPatch(outline_path(n), facecolor=n["fill"] or "none", edgecolor=n["stroke"],
                               linewidth=PT, joinstyle="miter", snap=False, zorder=z))
        r = corner_radius(n)
        if n["align"] == "left":
            tx, ha = n["x"] + max(10.0, 0.8 * r), "left"
        else:
            tx, ha = n["x"] + n["w"] / 2.0, "center"
        t = ax.text(tx, n["y"] + 6, n["label"], fontsize=n["fs"] * PT, ha=ha, va="top", color=BLACK,
                    zorder=z + 0.05)
        texts.append(("title", n["id"], t))

    # connectors
    resolved = []
    for c in D["conns"]:
        rc = resolve_connector(c, nodes, crisp)
        resolved.append(rc)
        line = np.array(rc["line"])
        ls = (0, (DASH_PX[0] * PT, DASH_PX[1] * PT)) if c["style"] == "dashed" else "-"
        ax.add_line(Line2D(line[:, 0], line[:, 1], linewidth=EDGE_SW * PT, color=BLACK, linestyle=ls,
                           solid_capstyle="butt", dash_capstyle="butt", solid_joinstyle="miter",
                           snap=False, zorder=3))
        for poly in rc["polys"]:
            ax.add_patch(Polygon(poly, closed=True, facecolor=BLACK, edgecolor=BLACK,
                                 linewidth=EDGE_SW * PT, joinstyle="miter", snap=False, zorder=3.1))
        if rc["label"]:
            lp = rc["label"]["pos"]
            t = ax.text(lp[0], lp[1], rc["label"]["text"], fontsize=efs * PT, ha="center",
                        va="center", color=BLACK, zorder=6,
                        bbox=dict(boxstyle="square,pad=0.12", facecolor="white", edgecolor="none"))
            texts.append(("edge_label", len(resolved) - 1, t))

    # component nodes
    for n in D["nodes"]:
        if n["kind"] in ("box", "rounded_box"):
            ax.add_patch(PathPatch(outline_path(n), facecolor=n["fill"], edgecolor=n["stroke"],
                                   linewidth=PT, joinstyle="miter", snap=False, zorder=4))
            t = ax.text(n["x"] + n["w"] / 2.0, n["y"] + n["h"] / 2.0, n["label"],
                        fontsize=n["fs"] * PT, ha="center", va="center", multialignment="center",
                        linespacing=1.2, color=BLACK, zorder=5)
            texts.append(("node", n["id"], t))
        elif n["kind"] == "icon_db":
            body, strokes = cylinder(n)
            ax.add_patch(Polygon(body, closed=True, facecolor="white", edgecolor="none", zorder=4))
            ax.add_patch(PathPatch(polyline_path(strokes), facecolor="none", edgecolor=ICON_BLUE,
                                   linewidth=2 * PT, capstyle="round", joinstyle="round", snap=False,
                                   zorder=4.1))
            kw = dict(bbox=dict(boxstyle="square,pad=0.1", facecolor="white", edgecolor="none")) \
                if n.get("cap_bg") else {}
            t = ax.text(n["x"] + n["w"] / 2.0, n["y"] + n["h"] + 4, n["label"], fontsize=n["fs"] * PT,
                        ha="center", va="top", color=BLACK, zorder=5, **kw)
            texts.append(("caption", n["id"], t))

    fig.canvas.draw()
    rend = fig.canvas.get_renderer()

    def canvas_bb(bb):
        return (bb.x0, H - bb.y1, bb.x1, H - bb.y0)

    text_info = []
    for kind, owner, t in texts:
        tb = canvas_bb(t.get_window_extent(rend))
        pb = canvas_bb(t.get_bbox_patch().get_window_extent(rend)) if t.get_bbox_patch() else tb
        text_info.append(dict(kind=kind, owner=owner, text=t.get_text(), bbox=tb, outer=pb))

    # tight crop to the content + border (what draw.io's PNG export does)
    xs, ys = [], []
    for n in D["nodes"]:
        x0, y0, x1, y1 = node_rect(n)
        xs += [x0, x1]
        ys += [y0, y1]
    for ti in text_info:
        xs += [ti["outer"][0], ti["outer"][2]]
        ys += [ti["outer"][1], ti["outer"][3]]
    for rc in resolved:
        pts = np.vstack([np.array(rc["line"])] + rc["polys"])
        xs += [pts[:, 0].min() - 0.5, pts[:, 0].max() + 0.5]
        ys += [pts[:, 1].min() - 0.5, pts[:, 1].max() + 0.5]
    B = D["border"]
    cx0, cy0 = max(0, int(math.floor(min(xs))) - B), max(0, int(math.floor(min(ys))) - B)
    cx1, cy1 = min(W, int(math.ceil(max(xs))) + B), min(H, int(math.ceil(max(ys))) + B)
    crop = (cx0, cy0, cx1, cy1)
    # bbox_inches is given in inches in display space (y up): exact because DPI is a power of 2
    bbox_in = Bbox.from_extents(cx0 / DPI, (H - cy1) / DPI, cx1 / DPI, (H - cy0) / DPI)
    out = os.path.join(OUT_DIR, D["file"])
    fig.savefig(out, dpi=DPI, bbox_inches=bbox_in, facecolor="white")
    plt.close(fig)
    return nodes, resolved, text_info, crop


def render_mask(D, crop):
    """Renderer-side verification: every component in a unique flat colour, same transform+crop."""
    W, H = D["W"], D["H"]
    fig, ax = new_canvas(W, H)
    comps = [n for n in D["nodes"] if is_component(n)]
    colors = {}
    for i, n in enumerate(comps):
        col = ((i + 1) * 20 / 255.0, (200 - i * 10) / 255.0, 0.5)
        colors[n["id"]] = tuple(int(round(v * 255)) for v in col)
        if n["kind"] == "icon_db":
            body, strokes = cylinder(n)
            ax.add_patch(Polygon(body, closed=True, facecolor=col, edgecolor="none",
                                 antialiased=False))
            ax.add_patch(PathPatch(polyline_path(strokes), facecolor="none", edgecolor=col,
                                   linewidth=2 * PT, capstyle="round", joinstyle="round", snap=False,
                                   antialiased=False))
        else:
            ax.add_patch(PathPatch(outline_path(n), facecolor=col, edgecolor=col, linewidth=PT,
                                   joinstyle="miter", snap=False, antialiased=False))
    cx0, cy0, cx1, cy1 = crop
    bbox_in = Bbox.from_extents(cx0 / DPI, (H - cy1) / DPI, cx1 / DPI, (H - cy0) / DPI)
    tmp = os.path.join(HERE, "_mask_tmp.png")
    fig.savefig(tmp, dpi=DPI, bbox_inches=bbox_in, facecolor="white")
    plt.close(fig)
    img = plt.imread(tmp)[:, :, :3]
    os.remove(tmp)
    img = np.round(img * 255).astype(int)
    res = {}
    for n in comps:
        m = np.all(img == np.array(colors[n["id"]]), axis=2)
        yy, xx = np.nonzero(m)
        res[n["id"]] = (int(xx.min()), int(yy.min()), int(xx.max() - xx.min() + 1),
                        int(yy.max() - yy.min() + 1)) if len(xx) else None
    return res, (img.shape[1], img.shape[0])


# ----------------------------------------------------------------------------------------------
# self-check (geometry level, canvas coordinates)
# ----------------------------------------------------------------------------------------------
def inside(pts, r, m=0.0):
    x0, y0, x1, y1 = r
    return (pts[:, 0] > x0 - m) & (pts[:, 0] < x1 + m) & (pts[:, 1] > y0 - m) & (pts[:, 1] < y1 + m)


def rects_overlap(a, b, m=0.0):
    return not (a[2] + m <= b[0] or b[2] + m <= a[0] or a[3] + m <= b[1] or b[3] + m <= a[1])


def rect_inside(a, b, m=0.0):
    return a[0] >= b[0] + m and a[1] >= b[1] + m and a[2] <= b[2] - m and a[3] <= b[3] - m


def inside_rounded(pt, c, margin):
    """point inside the rounded container shape, at least `margin` px from its outline."""
    x0, y0, x1, y1 = node_rect(c)
    if not (x0 + margin <= pt[0] <= x1 - margin and y0 + margin <= pt[1] <= y1 - margin):
        return False
    r = corner_radius(c)
    if r <= 0:
        return True
    cx = min(max(pt[0], x0 + r), x1 - r)
    cy = min(max(pt[1], y0 + r), y1 - r)
    return math.hypot(pt[0] - cx, pt[1] - cy) <= r - margin


def rect_corners(r):
    return [(r[0], r[1]), (r[2], r[1]), (r[0], r[3]), (r[2], r[3])]


def seg_intersect(p1, p2, p3, p4):
    d = (p2[0] - p1[0]) * (p4[1] - p3[1]) - (p2[1] - p1[1]) * (p4[0] - p3[0])
    if abs(d) < 1e-9:
        return False
    t = ((p3[0] - p1[0]) * (p4[1] - p3[1]) - (p3[1] - p1[1]) * (p4[0] - p3[0])) / d
    u = ((p3[0] - p1[0]) * (p2[1] - p1[1]) - (p3[1] - p1[1]) * (p2[0] - p1[0])) / d
    return 0.0 < t < 1.0 and 0.0 < u < 1.0


def self_check(D, nodes, resolved, text_info, crop):
    errs, warns = [], []
    comps = [n for n in D["nodes"] if is_component(n)]
    conts = [n for n in D["nodes"] if n["kind"] == "container"]
    samples = []
    for rc in resolved:
        pts = [densify(rc["line"])] + [densify(list(p) + [p[0]]) for p in rc["polys"]]
        samples.append(np.vstack(pts))
    texts_by = lambda k: [t for t in text_info if t["kind"] == k]  # noqa: E731

    # 1. text fits: node labels inside their box, all text inside the crop
    for t in text_info:
        if not rect_inside(t["outer"], crop, 2):
            errs.append(f"text '{t['text']}' clipped by image crop")
        if t["kind"] == "node":
            n = nodes[t["owner"]]
            m = 3 + (corner_radius(n) * 0.3)
            if not rect_inside(t["bbox"], node_rect(n), m):
                errs.append(f"label '{t['text']}' does not fit inside its box")
    # 2. captions / titles / edge labels must not overlap any other node or text
    for i, t in enumerate(text_info):
        if t["kind"] == "node":
            continue
        for n in comps:
            if t["kind"] == "caption" and t["owner"] == n["id"]:
                continue
            if rects_overlap(t["outer"], node_rect(n), 3):
                errs.append(f"{t['kind']} '{t['text']}' overlaps node {n['id']}")
        for j, u in enumerate(text_info):
            if j != i and rects_overlap(t["outer"], u["outer"], 2):
                errs.append(f"text '{t['text']}' overlaps text '{u['text']}'")
    # 3. connectors never pass through a box / label they do not connect
    for ci, (c, rc, S) in enumerate(zip(D["conns"], resolved, samples)):
        tag = f"conn#{ci + 1} {c['frm']}->{c['to']}"
        for n in comps:
            r = node_rect(n)
            if n["id"] in (c["frm"], c["to"]):
                p = rc["ps"] if n["id"] == c["frm"] else rc["pd"]
                far = np.hypot(S[:, 0] - p[0], S[:, 1] - p[1]) > 1.6
                if np.any(inside(S[far], (r[0] + 1, r[1] + 1, r[2] - 1, r[3] - 1))):
                    errs.append(f"{tag} re-enters its own endpoint {n['id']}")
            elif np.any(inside(S, r, 4)):
                errs.append(f"{tag} passes through node {n['id']}")
        for t in text_info:
            if t["kind"] == "edge_label" and t["owner"] == ci:
                continue
            if np.any(inside(S, t["outer"], 3)):
                errs.append(f"{tag} passes through text '{t['text']}'")
        # orthogonality / segment lengths
        P = rc["P"]
        for k in range(len(P) - 1):
            dv = P[k + 1] - P[k]
            if abs(dv[0]) > 1e-6 and abs(dv[1]) > 1e-6:
                errs.append(f"{tag} segment {k} is not axis-aligned")
            if np.linalg.norm(dv) < 18:
                errs.append(f"{tag} segment {k} too short ({np.linalg.norm(dv):.1f}px)")
        if np.linalg.norm(P[-1] - P[-2]) < 24 or (c["heads"] == "both" and np.linalg.norm(P[1] - P[0]) < 24):
            errs.append(f"{tag} arrow segment too short")
        bends = len(P) - 2
        if bends > 2:
            errs.append(f"{tag} has {bends} bends")
        # port sits on the side of the box, away from corners
        for nid, p, side in ((c["frm"], rc["ps"], rc["ss"]), (c["to"], rc["pd"], rc["ds"])):
            n = nodes[nid]
            x0, y0, x1, y1 = node_rect(n)
            m = 6 + corner_radius(n)
            if side in "lr" and not (y0 + m <= p[1] <= y1 - m):
                errs.append(f"{tag} port off the {side} side of {nid}")
            if side in "tb" and not (x0 + m <= p[0] <= x1 - m):
                errs.append(f"{tag} port off the {side} side of {nid}")
            # first/last segment must leave the box outward
        out_dir = {"l": (-1, 0), "r": (1, 0), "t": (0, -1), "b": (0, 1)}
        if np.dot(unit(P[1] - P[0]), out_dir[rc["ss"]]) < 0.99:
            errs.append(f"{tag} does not leave source side {rc['ss']} outward")
        if np.dot(unit(P[-2] - P[-1]), out_dir[rc["ds"]]) < 0.99:
            errs.append(f"{tag} does not enter target side {rc['ds']} from outside")
        # must not run along a container border
        for cn in conts:
            x0, y0, x1, y1 = node_rect(cn)
            for (ax0, ay0, ax1, ay1) in ((x0, y0, x1, y0), (x0, y1, x1, y1), (x0, y0, x0, y1), (x1, y0, x1, y1)):
                near = (np.abs(S[:, 1] - ay0) < 3) & (S[:, 0] >= ax0) & (S[:, 0] <= ax1) if ay0 == ay1 \
                    else (np.abs(S[:, 0] - ax0) < 3) & (S[:, 1] >= ay0) & (S[:, 1] <= ay1)
                if near.sum() * 0.5 > 12:
                    errs.append(f"{tag} runs along the border of container {cn['id']}")
        # edge label: on a straight stretch, clear of bends, off container borders
        if rc["label"]:
            lab = [t for t in text_info if t["kind"] == "edge_label" and t["owner"] == ci][0]
            half = max(lab["outer"][2] - lab["outer"][0], lab["outer"][3] - lab["outer"][1]) / 2.0
            a, s = rc["label"]["along"], rc["label"]["seg_len"]
            if a < half + 6 or s - a < half + 6:
                errs.append(f"{tag} label '{c['label']}' too close to a bend/end")
            for cn in conts:
                x0, y0, x1, y1 = node_rect(cn)
                ob = lab["outer"]
                for edge in ((x0, y0, x1, y0), (x0, y1, x1, y1), (x0, y0, x0, y1), (x1, y0, x1, y1)):
                    if rects_overlap(ob, (edge[0], edge[1], edge[2] + 0.01, edge[3] + 0.01), 3):
                        errs.append(f"{tag} label '{c['label']}' sits on container {cn['id']} border")
            for cj, S2 in enumerate(samples):
                if cj != ci and np.any(inside(S2, lab["outer"], 2)):
                    errs.append(f"{tag} label '{c['label']}' is crossed by conn#{cj + 1}")
    # 4. connector vs connector: no overlapping runs, report crossings
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            ci, cj = D["conns"][i], D["conns"][j]
            shared = {ci["frm"], ci["to"]} & {cj["frm"], cj["to"]}
            A, Bs = samples[i], samples[j]
            keep = np.ones(len(A), bool)
            for nid in shared:
                keep &= ~inside(A, node_rect(nodes[nid]), 16)
            A = A[keep]
            if len(A) and len(Bs):
                d = np.min(np.hypot(A[:, None, 0] - Bs[None, ::4, 0], A[:, None, 1] - Bs[None, ::4, 1]), axis=1)
                if (d < 4).sum() * 0.5 > 10:
                    errs.append(f"conn#{i + 1} and conn#{j + 1} overlap")
            Pi, Pj = resolved[i]["P"], resolved[j]["P"]
            for a in range(len(Pi) - 1):
                for b in range(len(Pj) - 1):
                    if seg_intersect(Pi[a], Pi[a + 1], Pj[b], Pj[b + 1]):
                        warns.append(f"conn#{i + 1} crosses conn#{j + 1}")
    # 5. layout: nodes inside parents, no node overlaps, containers nest properly
    for n in D["nodes"]:
        r = node_rect(n)
        if n["kind"] == "icon_db":
            cap = [t for t in text_info if t["kind"] == "caption" and t["owner"] == n["id"]][0]
            r = (min(r[0], cap["outer"][0]), r[1], max(r[2], cap["outer"][2]), cap["outer"][3])
        if n.get("parent"):
            p = nodes[n["parent"]]
            if not all(inside_rounded(pt, p, 6) for pt in rect_corners(r)):
                errs.append(f"{n['id']} not inside its container {p['id']}")
        for c2 in conts:
            if c2["id"] == n["id"] or c2["id"] == n.get("parent"):
                continue
            # anything that is not a descendant must not overlap a container
            q, anc = n, set()
            while q.get("parent"):
                anc.add(q["parent"])
                q = nodes[q["parent"]]
            if c2["id"] in anc:
                continue
            d2, q = set(), c2
            while q.get("parent"):
                d2.add(q["parent"])
                q = nodes[q["parent"]]
            if n["kind"] == "container" and n["id"] in d2:
                continue
            if rects_overlap(r, node_rect(c2), 4):
                errs.append(f"{n['id']} overlaps unrelated container {c2['id']}")
    for i, a in enumerate(comps):
        for b in comps[i + 1:]:
            if rects_overlap(node_rect(a), node_rect(b), 10):
                errs.append(f"nodes {a['id']} and {b['id']} overlap")
    for t in texts_by("title"):
        c = nodes[t["owner"]]
        if not all(inside_rounded(pt, c, 3) for pt in rect_corners(t["outer"])):
            errs.append(f"title '{t['text']}' leaves the container shape")
    return errs, warns


# ----------------------------------------------------------------------------------------------
def main():
    from matplotlib import font_manager
    try:
        font_manager.findfont("Arial", fallback_to_default=False)
    except ValueError:
        print("Arial is not installed: the renders would not match the committed set. Nothing written.")
        return 1
    os.makedirs(OUT_DIR, exist_ok=True)
    gt, gt_boxes, resolved_out = {}, {}, {}
    all_ok = True
    for D in DIAGRAMS:
        nodes, resolved, text_info, crop = render(D)
        cx0, cy0, cx1, cy1 = crop
        size = [cx1 - cx0, cy1 - cy0]
        png = plt.imread(os.path.join(OUT_DIR, D["file"]))
        if [png.shape[1], png.shape[0]] != size:
            print(f"!! {D['file']}: saved size {png.shape[1]}x{png.shape[0]} != expected {size}")
            all_ok = False
        comps = [n for n in D["nodes"] if is_component(n)]
        labels = [gt_label(n) for n in comps]
        assert len(set(labels)) == len(labels), "component labels must be unique"
        gt[D["file"]] = {
            "name": D["name"],
            "source": "synthetic-drawio-style",
            "components": labels,
            "arrows": [{"source": gt_label(nodes[c["frm"]]), "target": gt_label(nodes[c["to"]]),
                        "style": c["style"], "bidirectional": c["heads"] == "both"}
                       for c in D["conns"]],
            "icons": [],
        }
        boxes = []
        for n in comps:
            boxes.append({"label": gt_label(n), "x": n["x"] - cx0, "y": n["y"] - cy0,
                          "w": n["w"], "h": n["h"]})
        gt_boxes[D["file"]] = {"image_size": size, "boxes": boxes}

        # renderer-side verification of the boxes
        mask, msize = render_mask(D, crop)
        worst = 0
        for n, b in zip(comps, boxes):
            mb = mask[n["id"]]
            if mb is None:
                print(f"!! {D['file']}: {n['id']} not found in mask")
                all_ok = False
                continue
            diff = max(abs(mb[0] - b["x"]), abs(mb[1] - b["y"]), abs(mb[2] - b["w"]), abs(mb[3] - b["h"]))
            worst = max(worst, diff)
            if diff > 1:
                print(f"!! {D['file']}: {n['id']} GT {b} vs rendered {mb}")
                all_ok = False

        errs, warns = self_check(D, nodes, resolved, text_info, crop)
        all_ok &= not errs
        def P2(p):  # noqa: E306
            return [round(float(p[0]) - cx0, 2), round(float(p[1]) - cy0, 2)]
        resolved_out[D["file"]] = {
            "image_size": size,
            "crop_offset_in_canvas": [cx0, cy0],
            "components": [{"id": n["id"], "kind": n["kind"], "label": gt_label(n)} for n in comps],
            "containers": [{"id": n["id"], "title": n["label"], "shape": n["shape"],
                            "x": n["x"] - cx0, "y": n["y"] - cy0, "w": n["w"], "h": n["h"],
                            "parent": n.get("parent")} for n in D["nodes"] if n["kind"] == "container"],
            "connectors": [{"source": gt_label(nodes[c["frm"]]), "target": gt_label(nodes[c["to"]]),
                            "style": c["style"], "heads": c["heads"], "route": c["route"],
                            "bends": len(rc["P"]) - 2, "rounded_bends": bool(c["rounded"]),
                            "label": c.get("label"),
                            "path": [P2(p) for p in rc["P"]]}
                           for c, rc in zip(D["conns"], resolved)],
            "texts": [{"kind": t["kind"], "text": t["text"],
                       "bbox": [round(t["outer"][0] - cx0, 1), round(t["outer"][1] - cy0, 1),
                                round(t["outer"][2] - cx0, 1), round(t["outer"][3] - cy0, 1)]}
                      for t in text_info],
        }
        n_el = sum(1 for rc in resolved if len(rc["P"]) > 2)
        print(f"{D['file']}: {size[0]}x{size[1]}  comps={len(comps)} conns={len(D['conns'])} "
              f"elbows={n_el} mask_max_diff={worst}px errors={len(errs)} warnings={len(warns)}")
        for e in errs:
            print("   ERROR", e)
        for w in warns:
            print("   warn ", w)

    with open(os.path.join(HERE, "ground_truth.json"), "w", encoding="utf-8") as f:
        json.dump(gt, f, indent=2)
    with open(os.path.join(HERE, "ground_truth_boxes.json"), "w", encoding="utf-8") as f:
        json.dump(gt_boxes, f, indent=2)
    with open(os.path.join(HERE, "spec_resolved.json"), "w", encoding="utf-8") as f:
        json.dump(resolved_out, f, indent=1)
    print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
