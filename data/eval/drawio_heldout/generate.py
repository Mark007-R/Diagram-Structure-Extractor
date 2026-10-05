#!/usr/bin/env python
"""Deterministic generator for HELD-OUT draw.io-style architecture diagrams (set 2).

Built independently of the detector code and of data/eval/drawio_style/, with
deliberately different styles, then scored ONCE after the draw.io handling was
frozen (2026-10-05). The fixes from the code review that followed were checked
against it too, so it is no longer untouched; the README keeps the frozen
score. Keep it out of tuning, or it stops being held out at all.
Uses Windows system fonts (Arial, Segoe UI, Verdana, Tahoma, Calibri,
Trebuchet MS); on other machines the regenerated pixels differ, so the
committed PNGs are the reference.

Every diagram is an explicit, hand-written spec (no randomness).  Rendering is
matplotlib Agg at 100 dpi on a figure whose single axes covers the whole canvas
with data units == pixels and y pointing down, saved WITHOUT bbox_inches="tight"
so the saved PNG is exactly W x H.  Component boxes are pushed through
ax.transData and the display->image mapping of the saved file, then verified
by re-rendering each component on its own and measuring its ink bbox.

Outputs (next to this file):
  diagrams/heldout_XX.png
  overlays/heldout_XX_overlay.png   (GT boxes + connectors drawn for review)
  ground_truth.json                 (components, arrows, icons)
  ground_truth_boxes.json           (pixel boxes of every component)
  ground_truth_details.json         (routes, styles, labels, containers; diagnostics --
                                     not kept in the repo, nor are the overlays)

Run:  python generate.py            exit code 1 if any layout/verification check fails
"""
import json
import math
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, PathPatch, Polygon, Rectangle  # noqa: E402
from matplotlib.path import Path  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DIAG_DIR = os.path.join(HERE, "diagrams")
OVL_DIR = os.path.join(HERE, "overlays")
SOURCE = "synthetic-drawio-style-heldout-2"
DPI = 100
PT = 72.0 / DPI  # points per pixel

matplotlib.rcParams.update({
    "lines.scale_dashes": False,
    "path.simplify": False,
    "text.antialiased": True,
    "lines.antialiased": True,
    "patch.antialiased": True,
})

BOX_KINDS = ("rect", "round")
ICON_KINDS = ("db", "cloud", "user", "server", "queue")


# ----------------------------------------------------------------------------
# spec helpers
# ----------------------------------------------------------------------------
def B(id, label, x, y, w, h, kind="rect", **kw):
    d = dict(id=id, label=label, kind=kind, x=x, y=y, w=w, h=h)
    d.update(kw)
    return d


def I(id, label, kind, cx, cy, w, h=None, **kw):
    d = dict(id=id, label=label, kind=kind, cx=cx, cy=cy, w=w, h=h)
    d.update(kw)
    return d


def G(title, x, y, w, h, **kw):
    d = dict(title=title, x=x, y=y, w=w, h=h)
    d.update(kw)
    return d


def E(s, t, **kw):
    d = dict(s=s, t=t)
    d.update(kw)
    return d


def snapc(v, lw):
    """Pixel-snap a coordinate so strokes of width lw render crisp."""
    if int(round(lw)) % 2 == 1:
        return math.floor(v) + 0.5
    return float(math.floor(v + 0.5))


def gt_label(lbl):
    return " ".join(lbl.split())


# ----------------------------------------------------------------------------
# nodes
# ----------------------------------------------------------------------------
CLOUD_CIRCLES = [(0.20, 0.40, 0.18), (0.45, 0.27, 0.25), (0.70, 0.33, 0.19), (0.82, 0.42, 0.16)]
CLOUD_RECT = (0.20, 0.30, 0.82, 0.58)  # u0, v0, u1, v1   (union bbox u 0.02..0.98, v 0.02..0.58)


class Node:
    def __init__(self, sp, D):
        self.sp = sp
        self.id = sp["id"]
        self.label = sp["label"]
        self.kind = k = sp["kind"]
        self.shape_artists = []
        self.text_artist = None
        self.D = D
        if k in BOX_KINDS:
            lw = sp.get("lw", D.get("node_lw", 1))
            x0, y0 = snapc(sp["x"], lw), snapc(sp["y"], lw)
            self.g = (x0, y0, x0 + sp["w"], y0 + sp["h"])
            self.lw = lw
            self.r = sp.get("r", D.get("round_r", 8)) if k == "round" else 0.0
            self.pix = (self.g[0] - lw / 2, self.g[1] - lw / 2, self.g[2] + lw / 2, self.g[3] + lw / 2)
        elif k == "db":
            w, h = sp["w"], sp["h"]
            self.lw = lw = sp.get("lw", 2)
            cx, cy = sp["cx"], sp["cy"]
            self.g = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            self.eh = min(0.28 * w, 0.22 * h)
            self.pix = (self.g[0] - lw / 2, self.g[1] - lw / 2, self.g[2] + lw / 2, self.g[3] + lw / 2)
        elif k == "cloud":
            w = sp["w"]
            self.s = s = w / 0.96
            self.lw = lw = sp.get("lw", 2)
            cx, cy = sp["cx"], sp["cy"]
            self.g = (cx - 0.48 * s, cy - 0.28 * s, cx + 0.48 * s, cy + 0.28 * s)
            self.pix = (self.g[0] - lw, self.g[1] - lw, self.g[2] + lw, self.g[3] + lw)
        elif k == "user":
            h = sp["h"]
            w = sp.get("w") or 0.72 * h
            cx, cy = sp["cx"], sp["cy"]
            self.lw = 0
            self.g = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            self.hr = 0.19 * h
            self.torso_top = self.g[1] + 2 * self.hr + 0.05 * h
            self.torso_h = self.g[3] - self.torso_top
            self.rr = min(0.32 * w, 0.42 * self.torso_h)
            self.pix = self.g
        elif k == "server":
            w, h = sp["w"], sp["h"]
            self.lw = lw = sp.get("lw", 2)
            cx, cy = sp["cx"], sp["cy"]
            self.g = (snapc(cx - w / 2, lw), snapc(cy - h / 2, lw), 0, 0)
            self.g = (self.g[0], self.g[1], self.g[0] + w, self.g[1] + h)
            self.pix = (self.g[0] - lw / 2, self.g[1] - lw / 2, self.g[2] + lw / 2, self.g[3] + lw / 2)
        elif k == "queue":
            w, h = sp["w"], sp["h"]
            cx, cy = sp["cx"], sp["cy"]
            self.lw = 0
            self.g = (round(cx - w / 2), round(cy - h / 2), round(cx - w / 2) + w, round(cy - h / 2) + h)
            self.gap = 4
            self.bh = (h - 2 * self.gap) / 3.0
            self.pix = self.g
        else:
            raise ValueError(k)

    # centre of geometry
    @property
    def cx(self):
        return (self.g[0] + self.g[2]) / 2

    @property
    def cy(self):
        return (self.g[1] + self.g[3]) / 2

    def side_range(self, side):
        """Allowed (min,max) offsets from the default anchor along the side."""
        x0, y0, x1, y1 = self.g
        k = self.kind
        if k in BOX_KINDS or k == "server":
            r = self.r if k == "round" else (4 if k == "server" else 0)
            half = ((y1 - y0) if side in "lr" else (x1 - x0)) / 2 - r - 2
            return (-half, half)
        if k == "db":
            if side in "lr":
                half = (y1 - y0) / 2 - self.eh / 2 - 2
                return (-half, half)
            return (-1, 1)
        if k == "user":
            if side in "lr":
                half = self.torso_h / 2 - self.rr - 1
                return (-half, half)
            return (-1, 1)
        if k == "queue":
            if side in "lr":
                return (-(self.bh / 2 - 2), self.bh / 2 - 2)
            half = (x1 - x0) / 2 - 4
            return (-half, half)
        if k == "cloud":
            return (-0.5, 0.5)
        return (0, 0)

    def anchor(self, side, off=0.0):
        """Point on the node border, plus the default along-side coordinate."""
        x0, y0, x1, y1 = self.g
        k = self.kind
        cx, cy = self.cx, self.cy
        if k == "user":
            my = self.torso_top + self.torso_h / 2
            base = {"l": (x0, my), "r": (x1, my), "t": (cx, y0), "b": (cx, y1)}[side]
        elif k == "cloud":
            s, lw = self.s, self.lw
            base = {"l": (x0 - lw / 2, cy + 0.10 * s), "r": (x1 + lw / 2, cy + 0.12 * s),
                    "t": (cx - 0.05 * s, y0 - lw / 2), "b": (cx + 0.01 * s, y1 + lw / 2)}[side]
        else:
            base = {"l": (x0, cy), "r": (x1, cy), "t": (cx, y0), "b": (cx, y1)}[side]
        if side in "lr":
            return (base[0], base[1] + off)
        return (base[0] + off, base[1])

    # ---------------------------------------------------------------- draw
    def draw(self, ax, D):
        sp, k = self.sp, self.kind
        fs = sp.get("fs", D["fs"])
        font = D["font"]
        z = 4
        if k in BOX_KINDS:
            x0, y0, x1, y1 = self.g
            fc = sp.get("fc", D.get("node_fc", "white"))
            ec = sp.get("ec", D.get("node_ec", "black"))
            if k == "rect":
                p = Rectangle((x0, y0), x1 - x0, y1 - y0, fc=fc, ec=ec, lw=self.lw * PT,
                              joinstyle="miter", snap=False, zorder=z)
            else:
                p = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                   boxstyle="round,pad=0,rounding_size=%g" % self.r,
                                   fc=fc, ec=ec, lw=self.lw * PT, snap=False, zorder=z)
            ax.add_patch(p)
            self.shape_artists.append(p)
            t = ax.text(self.cx, self.cy, self.label, ha="center", va="center", fontsize=fs,
                        family=font, color=sp.get("fcol", "black"), multialignment="center",
                        zorder=z + 1, linespacing=1.25)
            self.text_artist = t
            return
        if k == "db":
            self._draw_db(ax, z)
        elif k == "cloud":
            self._draw_cloud(ax, z)
        elif k == "user":
            self._draw_user(ax, z)
        elif k == "server":
            self._draw_server(ax, z)
        elif k == "queue":
            self._draw_queue(ax, z)
        # caption below icon
        gap = sp.get("cap_gap", 6)
        t = ax.text(self.cx, self.pix[3] + gap, self.label, ha="center", va="top", fontsize=fs,
                    family=font, color="black", multialignment="center", zorder=z + 1)
        self.text_artist = t

    def _draw_db(self, ax, z):
        sp = self.sp
        x0, y0, x1, y1 = self.g
        cx = self.cx
        a, b = (x1 - x0) / 2, self.eh / 2
        ec = sp.get("ec", "#1a73e8")
        fc = sp.get("fc", "white")
        lw = self.lw * PT

        def arc(yc, t0, t1, n=48):
            ts = np.linspace(t0, t1, n)
            return list(zip(cx + a * np.cos(ts), yc + b * np.sin(ts)))

        top_c, bot_c = y0 + b, y1 - b
        verts = [(x0, top_c), (x0, bot_c)] + arc(bot_c, math.pi, 0) + [(x1, top_c)] + arc(top_c, 0, -math.pi)
        codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 1)
        verts.append(verts[0])
        codes.append(Path.CLOSEPOLY)
        p = PathPatch(Path(verts, codes), fc=fc, ec=ec, lw=lw, joinstyle="round", snap=False, zorder=z)
        ax.add_patch(p)
        self.shape_artists.append(p)
        body = (bot_c - top_c)
        for yc in (top_c, top_c + body / 3.0, top_c + 2 * body / 3.0):
            pts = arc(yc, math.pi, 0)
            ln = Line2D([q[0] for q in pts], [q[1] for q in pts], color=ec, lw=lw, snap=False,
                        solid_capstyle="butt", zorder=z + 0.1)
            ax.add_line(ln)
            self.shape_artists.append(ln)

    def _draw_cloud(self, ax, z):
        sp = self.sp
        s, lw = self.s, self.lw
        cx, cy = self.cx, self.cy
        ec = sp.get("ec", "#4d4d4d")
        fc = sp.get("fc", "#f5f5f5")

        def P(u, v):
            return (cx + (u - 0.5) * s, cy + (v - 0.30) * s)

        shapes = []
        for (u, v, r) in CLOUD_CIRCLES:
            shapes.append(("c", P(u, v), r * s))
        u0, v0, u1, v1 = CLOUD_RECT
        shapes.append(("r", P(u0, v0), P(u1, v1)))
        for layer in (0, 1):
            for sh in shapes:
                kw = dict(snap=False, zorder=z + 0.01 * layer)
                if layer == 0:
                    kw.update(fc=ec, ec=ec, lw=2 * lw * PT)
                else:
                    kw.update(fc=fc, ec="none", lw=0)
                if sh[0] == "c":
                    p = Circle(sh[1], sh[2], **kw)
                else:
                    (ax0, ay0), (ax1, ay1) = sh[1], sh[2]
                    p = Rectangle((ax0, ay0), ax1 - ax0, ay1 - ay0, joinstyle="miter", **kw)
                ax.add_patch(p)
                self.shape_artists.append(p)

    def _draw_user(self, ax, z):
        sp = self.sp
        col = sp.get("fc", "#4a5d73")
        x0, y0, x1, y1 = self.g
        head = Circle((self.cx, y0 + self.hr), self.hr, fc=col, ec="none", lw=0, snap=False, zorder=z)
        torso = FancyBboxPatch((x0, self.torso_top), x1 - x0, self.torso_h,
                               boxstyle="round,pad=0,rounding_size=%g" % self.rr,
                               fc=col, ec="none", lw=0, snap=False, zorder=z)
        for p in (head, torso):
            ax.add_patch(p)
            self.shape_artists.append(p)

    def _draw_server(self, ax, z):
        sp = self.sp
        x0, y0, x1, y1 = self.g
        ec = sp.get("ec", "#3c3c3c")
        fc = sp.get("fc", "#eeeeee")
        outer = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=4",
                               fc=fc, ec=ec, lw=self.lw * PT, snap=False, zorder=z)
        ax.add_patch(outer)
        self.shape_artists.append(outer)
        inset, gap = 6, 4
        uh = ((y1 - y0) - 2 * inset - 2 * gap) / 3.0
        for i in range(3):
            uy = y0 + inset + i * (uh + gap)
            u = Rectangle((math.floor(x0 + inset) + 0.5, math.floor(uy) + 0.5), (x1 - x0) - 2 * inset - 1, round(uh),
                          fc="white", ec=ec, lw=1 * PT, snap=False, zorder=z + 0.1)
            ax.add_patch(u)
            self.shape_artists.append(u)
            ymid = math.floor(uy) + 0.5 + round(uh) / 2
            for j, col in enumerate((sp.get("led", "#2e9e44"), "#9e9e9e")):
                c = Circle((x1 - inset - 5 - j * 6, ymid), 1.8, fc=col, ec="none", snap=False, zorder=z + 0.2)
                ax.add_patch(c)
                self.shape_artists.append(c)
            ln = Line2D([x0 + inset + 4, x0 + inset + 4 + (x1 - x0) * 0.35], [ymid, ymid], color=ec,
                        lw=1.5 * PT, snap=False, solid_capstyle="butt", zorder=z + 0.2)
            ax.add_line(ln)
            self.shape_artists.append(ln)

    def _draw_queue(self, ax, z):
        sp = self.sp
        x0, y0, x1, y1 = self.g
        col = sp.get("fc", "#d79b00")
        for i in range(3):
            by = y0 + i * (self.bh + self.gap)
            p = FancyBboxPatch((x0, by), x1 - x0, self.bh, boxstyle="round,pad=0,rounding_size=2",
                               fc=col, ec="none", lw=0, snap=False, zorder=z)
            ax.add_patch(p)
            self.shape_artists.append(p)


# ----------------------------------------------------------------------------
# edges
# ----------------------------------------------------------------------------
def parse_end(e):
    nid, side = e[0], e[1]
    off = e[2] if len(e) > 2 else 0.0
    return nid, side, off


def edge_route(es, nodes, st):
    sid, sside, soff = parse_end(es["s"])
    tid, tside, toff = parse_end(es["t"])
    S, T = nodes[sid], nodes[tid]
    if soff == "align" and toff == "align":
        raise ValueError("both ends align")
    if soff == "align":
        p1 = T.anchor(tside, toff)
        base = S.anchor(sside, 0)
        soff = (p1[1] - base[1]) if sside in "lr" else (p1[0] - base[0])
    if toff == "align":
        p0 = S.anchor(sside, soff)
        base = T.anchor(tside, 0)
        toff = (p0[1] - base[1]) if tside in "lr" else (p0[0] - base[0])
    p0 = S.anchor(sside, soff)
    p1 = T.anchor(tside, toff)
    lw = st["lw"]
    p0 = (snapc(p0[0], lw), snapc(p0[1], lw))
    p1 = (snapc(p1[0], lw), snapc(p1[1], lw))
    pts = [p0]
    for (ax_, v) in es.get("via", []):
        cur = pts[-1]
        if ax_ == "x":
            nv = p1[0] if v == "T" else snapc(v, lw)
            pts.append((nv, cur[1]))
        else:
            nv = p1[1] if v == "T" else snapc(v, lw)
            pts.append((cur[0], nv))
    pts.append(p1)
    out = [pts[0]]
    for q in pts[1:]:
        if abs(q[0] - out[-1][0]) > 1e-6 or abs(q[1] - out[-1][1]) > 1e-6:
            out.append(q)
    return out, (sid, sside, soff), (tid, tside, toff)


def round_path(pts, r, n=10):
    if r <= 0 or len(pts) < 3:
        return list(pts)
    out = [pts[0]]
    for i in range(1, len(pts) - 1):
        a, v, b = np.array(pts[i - 1]), np.array(pts[i]), np.array(pts[i + 1])
        lin, lout = np.linalg.norm(v - a), np.linalg.norm(b - v)
        rr = min(r, lin / 2 if i > 1 else lin - 2, lout / 2 if i < len(pts) - 2 else lout - 2)
        da, db = (v - a) / lin, (b - v) / lout
        P, Q = v - da * rr, v + db * rr
        for t in np.linspace(0, 1, n):
            q = (1 - t) ** 2 * P + 2 * (1 - t) * t * v + t ** 2 * Q
            out.append((float(q[0]), float(q[1])))
    out.append(pts[-1])
    return out


def head_poly(tip, u, kind, L):
    tip = np.array(tip, float)
    u = np.array(u, float)
    n = np.array([-u[1], u[0]])
    if kind == "classic":
        wd = 0.85 * L
        pts = [tip, tip - L * u + wd / 2 * n, tip - 0.72 * L * u, tip - L * u - wd / 2 * n]
        trim = 0.62 * L
    elif kind == "block":
        wd = 0.8 * L
        pts = [tip, tip - L * u + wd / 2 * n, tip - L * u - wd / 2 * n]
        trim = 0.88 * L
    else:
        raise ValueError(kind)
    return [tuple(map(float, p)) for p in pts], trim


EDGE_KEYS = ("color", "lw", "dash", "head", "hs", "both", "r", "lab_fs")


class Edge:
    def __init__(self, es, nodes, D):
        self.es = es
        st = {k: D["edge"][k] for k in EDGE_KEYS if k in D["edge"]}
        for k in EDGE_KEYS:
            if k in es:
                st[k] = es[k]
        st.setdefault("both", False)
        st.setdefault("r", 0)
        st.setdefault("dash", None)
        self.st = st
        self.raw, self.send, self.tend = edge_route(es, nodes, st)
        self.src, self.dst = self.send[0], self.tend[0]
        self.path = round_path(self.raw, st["r"])
        self.label = es.get("label")
        self.label_artist = None
        self.artists = []
        self.heads = []

    def draw(self, ax, D):
        st = self.st
        path = [np.array(p, float) for p in self.path]
        L = st["hs"]
        # end head
        u_end = path[-1] - path[-2]
        u_end /= np.linalg.norm(u_end)
        hp, trim = head_poly(path[-1], u_end, st["head"], L)
        self.heads.append(hp)
        path[-1] = path[-1] - u_end * trim
        if st["both"]:
            u_st = path[0] - path[1]
            u_st /= np.linalg.norm(u_st)
            hp2, trim2 = head_poly(path[0], u_st, st["head"], L)
            self.heads.append(hp2)
            path[0] = path[0] - u_st * trim2
        ls = "-" if not st["dash"] else (0, (st["dash"][0] * PT, st["dash"][1] * PT))
        ln = Line2D([p[0] for p in path], [p[1] for p in path], color=st["color"], lw=st["lw"] * PT,
                    linestyle=ls, solid_capstyle="butt", dash_capstyle="butt", solid_joinstyle="miter",
                    dash_joinstyle="miter", snap=False, zorder=4.5)
        ax.add_line(ln)
        self.artists.append(ln)
        for hp in self.heads:
            pg = Polygon(hp, closed=True, fc=st["color"], ec=st["color"], lw=0.5 * PT, joinstyle="miter",
                         snap=False, zorder=4.6)
            ax.add_patch(pg)
            self.artists.append(pg)
        if self.label:
            seg, frac = self.es.get("lab", (0, 0.5))
            a, b = np.array(self.raw[seg], float), np.array(self.raw[seg + 1], float)
            q = a + frac * (b - a)
            self.label_pos = (float(q[0]), float(q[1]))
            fs = st.get("lab_fs", D["fs"] - 1)
            t = ax.text(q[0], q[1], self.label, ha="center", va="center", fontsize=fs, family=D["font"],
                        color=self.es.get("lab_col", "black"), zorder=6,
                        bbox=dict(boxstyle="square,pad=0.18", fc="white", ec="none"))
            self.label_artist = t
            self.label_pad = 0.18 * fs / PT


# ----------------------------------------------------------------------------
# containers / frame
# ----------------------------------------------------------------------------
def draw_container(ax, c, D):
    arts = []
    lw = c.get("lw", 1)
    x, y = snapc(c["x"], lw), snapc(c["y"], lw)
    r = c.get("r", 14)
    ls = "-" if not c.get("dash") else (0, (c["dash"][0] * PT, c["dash"][1] * PT))
    p = FancyBboxPatch((x, y), c["w"], c["h"], boxstyle="round,pad=0,rounding_size=%g" % r,
                       fc=c.get("fc", "#f5f5f5"), ec=c.get("ec", "#666666"), lw=lw * PT, linestyle=ls,
                       snap=False, zorder=c.get("z", 1))
    ax.add_patch(p)
    arts.append(p)
    pos = c.get("tpos", "tl")
    fs = c.get("fs", D.get("title_fs", D["fs"] + 1))
    ty = y + c.get("tdy", 8)
    if pos == "tl":
        tx, ha = x + 12, "left"
    elif pos == "tr":
        tx, ha = x + c["w"] - 12, "right"
    else:
        tx, ha = x + c["w"] / 2, "center"
    t = ax.text(tx, ty, c["title"], ha=ha, va="top", fontsize=fs, family=D["font"],
                color=c.get("tcol", "black"), zorder=c.get("z", 1) + 0.5)
    return p, t


def draw_frame(ax, f, D):
    x, y, w, h = snapc(f["x"], 1), snapc(f["y"], 1), f["w"], f["h"]
    p = Rectangle((x, y), w, h, fc="none", ec=f.get("ec", "black"), lw=1 * PT, snap=False, zorder=1)
    ax.add_patch(p)
    tw, th = f["tab_w"], f["tab_h"]
    tab = Polygon([(x, y), (x + tw, y), (x + tw, y + th - 8), (x + tw - 8, y + th), (x, y + th)], closed=True,
                  fc=f.get("tab_fc", "white"), ec=f.get("ec", "black"), lw=1 * PT, joinstyle="miter",
                  snap=False, zorder=1.2)
    ax.add_patch(tab)
    t = ax.text(x + 8, y + th / 2, f["title"], ha="left", va="center", fontsize=f.get("fs", D["fs"]),
                family=D["font"], color="black", zorder=1.5)
    return p, tab, t


# ----------------------------------------------------------------------------
# DIAGRAM SPECS (explicit, deterministic)
# ----------------------------------------------------------------------------
SPECS = []

# 01 -- plain rectangles, black 1px solid, classic 8px, sharp elbows ------------
SPECS.append(dict(
    file="heldout_01.png", name="Online checkout", W=1100, H=560, margin=15, font="Arial", fs=10,
    edge=dict(color="black", lw=1, head="classic", hs=8),
    containers=[
        G("Public edge", 180, 220, 390, 125, fc="#f5f5f5", ec="#666666"),
        G("Kubernetes cluster", 630, 70, 210, 420, fc="#d5e8d4", ec="#82b366"),
    ],
    nodes=[
        I("U", "Customer", "user", 85, 270, None, 60),
        B("WA", "Web App", 200, 255, 130, 54),
        B("GW", "API Gateway", 420, 255, 130, 54),
        B("OS", "Orders Service", 660, 110, 150, 54),
        B("PS", "Payments Service", 660, 405, 150, 54),
        I("DB", "Orders DB", "db", 950, 137, 52, 66),
        I("PP", "Payment Provider", "cloud", 960, 417, 120),
    ],
    edges=[
        E(("U", "r"), ("WA", "l", "align")),
        E(("WA", "r"), ("GW", "l"), label="HTTPS"),
        E(("GW", "t"), ("OS", "l"), via=[("y", "T")]),
        E(("GW", "b"), ("PS", "l"), via=[("y", "T")]),
        E(("OS", "r"), ("DB", "l", "align")),
        E(("PS", "r", "align"), ("PP", "l")),
        E(("OS", "b"), ("PS", "t"), dash=(6, 4), label="events"),
    ],
))

# 02 -- reference-like: dark grey dashed 1px, classic 7, rounded elbows, double heads
SPECS.append(dict(
    file="heldout_02.png", name="Search platform", W=1300, H=620, margin=10, font="DejaVu Sans", fs=10,
    edge=dict(color="#333333", lw=1, head="classic", hs=7, dash=(6, 4), r=6),
    containers=[
        G("Client side", 30, 80, 360, 500, fc="#dae8fc", ec="#6c8ebf", r=22, tpos="tc"),
        G("Cloud backend", 450, 80, 820, 500, fc="#d5e8d4", ec="#82b366", r=22, tpos="tc"),
    ],
    nodes=[
        B("BR", "Browser", 110, 170, 150, 56),
        B("MA", "Mobile App", 110, 400, 150, 56),
        B("LB", "Load Balancer", 510, 280, 150, 60),
        B("AU", "Auth Service", 760, 150, 160, 56),
        B("SS", "Search Service", 760, 410, 160, 56),
        B("SC", "Session Cache", 1060, 150, 160, 56),
        I("SI", "Search Index", "db", 1140, 438, 54, 70, ec="#007fff"),
    ],
    edges=[
        E(("BR", "r"), ("LB", "l", -12), via=[("x", 420), ("y", "T")], label="HTTPS", lab=(0, 0.5)),
        E(("MA", "r"), ("LB", "l", 12), via=[("x", 435), ("y", "T")], label="REST", lab=(0, 0.5)),
        E(("LB", "t"), ("AU", "l"), via=[("y", "T")]),
        E(("LB", "b"), ("SS", "l"), via=[("y", "T")]),
        E(("AU", "r"), ("SC", "l"), both=True),
        E(("SS", "r"), ("SI", "l", "align"), label="query"),
        E(("AU", "b"), ("SS", "t"), dash=None, color="black"),
    ],
))

# 03 -- hub & spoke, diagonal straight, dark blue 2px, block heads 11, coloured fills
SPECS.append(dict(
    file="heldout_03.png", name="Order event routing", W=1200, H=680, margin=16, font="Segoe UI", fs=11,
    edge=dict(color="#1f3a93", lw=2, head="block", hs=11),
    containers=[],
    nodes=[
        B("ER", "Event Router", 520, 305, 160, 70, kind="round", fc="#fff2cc", ec="#d6b656"),
        B("OI", "Order Intake", 110, 90, 160, 60, kind="round", fc="#dae8fc", ec="#6c8ebf"),
        B("IV", "Inventory", 930, 90, 160, 60, kind="round", fc="#d5e8d4", ec="#82b366"),
        B("BL", "Billing", 110, 540, 160, 60, kind="round", fc="#f8cecc", ec="#b85450"),
        B("SH", "Shipping", 930, 540, 160, 60, kind="round", fc="#e1d5e7", ec="#9673a6"),
        I("AL", "Audit Log", "queue", 1010, 340, 80, 46, fc="#d79b00"),
        I("EM", "Email Gateway", "cloud", 200, 324, 130, fc="#ffffff", ec="#1f3a93"),
    ],
    edges=[
        E(("OI", "r"), ("ER", "t", -40), label="orders", lab=(0, 0.45)),
        E(("ER", "t", 40), ("IV", "l"), both=True),
        E(("ER", "b", -40), ("BL", "r")),
        E(("ER", "b", 40), ("SH", "l")),
        E(("ER", "r"), ("AL", "l", "align"), label="audit"),
        E(("ER", "l", "align"), ("EM", "r")),
        E(("IV", "b", 60), ("SH", "t", 60), dash=(8, 4), label="restock", lab=(0, 0.25)),
    ],
))

# 04 -- nested containers (region > VPC > subnets), black 1px, classic 9, rounded r=10
SPECS.append(dict(
    file="heldout_04.png", name="AWS web tier", W=1350, H=760, margin=12, font="Verdana", fs=9,
    edge=dict(color="black", lw=1, head="classic", hs=9, r=10),
    containers=[
        G("AWS Region eu-west-1", 230, 40, 1095, 690, fc="#f5f5f5", ec="#666666", r=12),
        G("VPC 10.0.0.0/16", 260, 90, 1035, 615, fc="#ffffff", ec="#82b366", lw=2, r=12, z=1.1),
        G("Public subnet", 290, 145, 300, 530, fc="#dae8fc", ec="#6c8ebf", dash=(6, 3), r=10, z=1.2),
        G("Private subnet", 630, 145, 640, 530, fc="#d5e8d4", ec="#82b366", dash=(6, 3), r=10, z=1.2),
    ],
    nodes=[
        I("INT", "Internet", "cloud", 110, 343, 120),
        I("ADM", "Admin", "user", 110, 555, None, 60, fc="#5b5b5b"),
        B("LB", "Load Balancer", 360, 330, 160, 56),
        I("BH", "Bastion Host", "server", 440, 560, 50, 66),
        I("APP", "App Server", "server", 770, 358, 54, 70),
        I("PDB", "Primary DB", "db", 1150, 358, 56, 72),
        B("RC", "Redis Cache", 1060, 520, 170, 56, kind="round", r=6),
    ],
    edges=[
        E(("INT", "r"), ("LB", "l", "align")),
        E(("ADM", "r"), ("BH", "l", "align"), dash=(4, 3), label="SSH", lab=(0, 0.2)),
        E(("LB", "r"), ("APP", "l", "align"), label="HTTP", lab=(0, 0.75)),
        E(("APP", "r", -18), ("PDB", "l", "align"), label="SQL"),
        E(("APP", "r", 18), ("RC", "l"), via=[("x", 900), ("y", "T")]),
        E(("BH", "r"), ("APP", "l", 22), via=[("x", 690), ("y", "T")], dash=(3, 3)),
    ],
))

# 05 -- titled frame (UML-like tab), Tahoma, black 1px, classic 6 (small), solid + dotted
SPECS.append(dict(
    file="heldout_05.png", name="Payment subsystem", W=1000, H=600, margin=10, font="Tahoma", fs=10,
    edge=dict(color="black", lw=1, head="classic", hs=6),
    containers=[],
    frame=dict(title="sd Payment subsystem", x=200, y=40, w=770, h=520, tab_w=180, tab_h=26),
    nodes=[
        B("MP", "Merchant Portal", 30, 230, 130, 60),
        B("PA", "Payment API", 260, 110, 150, 56),
        B("FC", "Fraud Check", 520, 110, 150, 56),
        B("LG", "Ledger", 790, 110, 140, 56),
        B("RR", "Risk Rules", 520, 290, 150, 56),
        B("SJ", "Settlement Job", 790, 430, 140, 56),
        B("NT", "Notifier", 260, 430, 150, 56),
    ],
    edges=[
        E(("MP", "r"), ("PA", "l", 14), via=[("x", 228), ("y", "T")]),
        E(("PA", "r"), ("FC", "l")),
        E(("FC", "b"), ("RR", "t"), dash=(2, 3), label="lookup"),
        E(("FC", "r"), ("LG", "l")),
        E(("LG", "b"), ("SJ", "t")),
        E(("SJ", "l"), ("NT", "r"), label="events"),
        E(("PA", "b"), ("NT", "t"), dash=(2, 3), both=True),
        E(("LG", "b", -40), ("RR", "r"), via=[("y", 220), ("x", 720), ("y", "T")], label="history",
          lab=(1, 0.5)),
    ],
))

# 06 -- data pipeline, Calibri 12, black 2px, classic 12, rounded elbows r=8
SPECS.append(dict(
    file="heldout_06.png", name="Streaming analytics pipeline", W=1400, H=560, margin=10, font="Calibri", fs=12,
    edge=dict(color="black", lw=2, head="classic", hs=12, r=8),
    containers=[],
    nodes=[
        I("WC", "Web Clients", "user", 70, 110, None, 60, fc="#3d5a80"),
        I("IOT", "IoT Devices", "server", 70, 390, 48, 64),
        B("IN", "Ingest API", 200, 230, 140, 60),
        I("EQ", "Event Queue", "queue", 480, 260, 84, 48, fc="#9673a6"),
        B("SP", "Stream Processor", 610, 230, 180, 60, kind="round", r=10),
        I("DL", "Data Lake", "db", 960, 110, 56, 72, ec="#2b6cb0"),
        I("WH", "Warehouse", "db", 960, 410, 56, 72, ec="#2b6cb0"),
        B("BI", "BI Dashboard", 1160, 230, 160, 60),
    ],
    edges=[
        E(("WC", "r"), ("IN", "l", -12), via=[("x", 150), ("y", "T")]),
        E(("IOT", "r"), ("IN", "l", 12), via=[("x", 165), ("y", "T")]),
        E(("IN", "r"), ("EQ", "l", "align"), label="events"),
        E(("EQ", "r"), ("SP", "l", "align")),
        E(("SP", "t"), ("DL", "l"), via=[("y", "T")]),
        E(("SP", "b"), ("WH", "l"), via=[("y", "T")]),
        E(("WH", "r"), ("BI", "b"), via=[("x", "T")]),
        E(("DL", "r"), ("BI", "t"), via=[("x", "T")]),
        E(("SP", "b", -45), ("IN", "b"), via=[("y", 500), ("x", "T")], dash=(8, 4), label="replay", lab=(1, 0.5)),
    ],
))

# 07 -- vertical layout, Trebuchet, dark grey dashed (5,3), classic 10, rounded r=12, double heads
SPECS.append(dict(
    file="heldout_07.png", name="Web app with background jobs", W=1100, H=720, margin=12, font="Trebuchet MS",
    fs=11,
    edge=dict(color="#4d4d4d", lw=1, head="classic", hs=10, dash=(5, 3), r=12),
    containers=[],
    nodes=[
        I("VI", "Visitor", "user", 95, 90, None, 62),
        B("EP", "Edge Proxy", 380, 75, 190, 56, kind="round", r=10),
        B("AS", "App Server", 380, 240, 190, 60, kind="round", r=10),
        B("SS", "Session Store", 60, 400, 180, 56, kind="round", r=10),
        I("JQ", "Job Queue", "queue", 850, 420, 80, 48, fc="#6c8ebf"),
        B("WK", "Worker", 770, 600, 160, 56, kind="round", r=10),
        I("DB", "Primary DB", "db", 475, 615, 58, 74),
    ],
    edges=[
        E(("VI", "r"), ("EP", "l", "align"), label="HTTPS"),
        E(("EP", "b"), ("AS", "t"), both=True),
        E(("AS", "l"), ("SS", "t"), via=[("x", "T")]),
        E(("AS", "r"), ("JQ", "t"), via=[("x", "T")]),
        E(("AS", "b"), ("DB", "t", "align"), both=True, label="SQL"),
        E(("JQ", "r"), ("WK", "r"), via=[("x", 975), ("y", "T")]),
        E(("WK", "l"), ("DB", "r", "align")),
        E(("SS", "b"), ("DB", "l"), via=[("y", "T")]),
    ],
))

# 08 -- coloured fills, block heads 8, black 1px, diagonals + straight
SPECS.append(dict(
    file="heldout_08.png", name="ML platform", W=1250, H=640, margin=14, font="Arial", fs=12,
    edge=dict(color="black", lw=1, head="block", hs=8),
    containers=[
        G("ML Platform", 30, 250, 1000, 170, fc="#f5f5f5", ec="#666666", r=14, tpos="tr"),
    ],
    nodes=[
        B("FS", "Feature Store", 60, 90, 170, 60, fc="#dae8fc", ec="#6c8ebf"),
        B("TP", "Training Pipeline", 60, 300, 170, 60, fc="#d5e8d4", ec="#82b366"),
        B("MR", "Model Registry", 420, 300, 170, 60, fc="#fff2cc", ec="#d6b656"),
        B("SA", "Serving API", 800, 300, 170, 60, fc="#f8cecc", ec="#b85450"),
        I("CA", "Client App", "user", 1150, 317, None, 62, fc="#666666"),
        B("MO", "Monitoring", 800, 490, 170, 60, fc="#e1d5e7", ec="#9673a6"),
        I("MD", "Metrics DB", "db", 480, 520, 54, 70, ec="#9673a6"),
    ],
    edges=[
        E(("FS", "b"), ("TP", "t")),
        E(("FS", "r"), ("MR", "t", -30)),
        E(("TP", "r"), ("MR", "l")),
        E(("MR", "r"), ("SA", "l"), label="deploy"),
        E(("SA", "r", "align"), ("CA", "l"), both=True),
        E(("SA", "b"), ("MO", "t"), dash=(5, 5), color="#555555"),
        E(("MO", "l"), ("MD", "r", "align")),
        E(("MD", "l"), ("TP", "b"), label="drift"),
    ],
))

# 09 -- large: cluster with two namespaces, 3-bend elbow, mixed styles
SPECS.append(dict(
    file="heldout_09.png", name="Orders and billing on Kubernetes", W=1400, H=800, margin=12, font="DejaVu Sans", fs=9,
    edge=dict(color="black", lw=1, head="classic", hs=8, r=10),
    containers=[
        G("Kubernetes cluster", 300, 50, 770, 720, fc="#dae8fc", ec="#6c8ebf", r=16, tpos="tc"),
        G("namespace: orders", 330, 100, 360, 640, fc="#ffffff", ec="#999999", dash=(4, 3), r=10, z=1.1),
        G("namespace: billing", 715, 100, 325, 640, fc="#ffffff", ec="#999999", dash=(4, 3), r=10, z=1.1),
    ],
    nodes=[
        I("CU", "Customer", "user", 140, 175, None, 60),
        B("AC", "Admin Console", 60, 470, 160, 60),
        B("OA", "Orders API", 410, 170, 200, 56),
        B("OW", "Orders Worker", 410, 360, 200, 56),
        I("OD", "Orders DB", "db", 510, 590, 56, 72),
        B("BA", "Billing API", 790, 170, 180, 56),
        I("IQ", "Invoice Queue", "queue", 880, 388, 84, 48, fc="#e07b39"),
        I("ST", "Stripe", "cloud", 1250, 186, 120, fc="#ffffff", ec="#6c8ebf"),
        I("DW", "Data Warehouse", "db", 1250, 585, 60, 76),
    ],
    edges=[
        E(("CU", "r"), ("OA", "l", "align")),
        E(("AC", "t"), ("OA", "l", 14), via=[("y", 320), ("x", 370), ("y", "T")]),
        E(("OA", "b"), ("OW", "t"), color="#0b3d91", lw=2, dash=(7, 4), head="block", hs=9, label="events"),
        E(("OW", "b"), ("OD", "t", "align")),
        E(("OA", "r"), ("BA", "l"), both=True, label="gRPC", lab=(0, 0.25)),
        E(("BA", "r", "align"), ("ST", "l"), label="HTTPS", lab=(0, 0.6)),
        E(("BA", "b"), ("IQ", "t", "align")),
        E(("IQ", "l"), ("OW", "r", "align"), color="#0b3d91", lw=2, dash=(7, 4), head="block", hs=9),
        E(("IQ", "r"), ("DW", "t"), via=[("x", "T")]),
        E(("OD", "r", -5), ("DW", "l", "align"), dash=(6, 4), color="#555555", label="nightly ETL"),
    ],
))

# 10 -- small: icons, dark blue dashed 2px (10,5), classic 9, rounded elbows
SPECS.append(dict(
    file="heldout_10.png", name="Mobile login", W=1000, H=560, margin=17, font="Arial", fs=13,
    edge=dict(color="#0b3d91", lw=2, head="classic", hs=9, dash=(10, 5), r=10),
    containers=[],
    nodes=[
        I("MU", "Mobile User", "user", 70, 230, None, 64, fc="#0b3d91"),
        B("AS", "Auth Server", 250, 209, 190, 70, kind="round", r=12),
        I("IP", "Identity Provider", "cloud", 760, 110, 150),
        I("TS", "Token Store", "db", 760, 420, 60, 78),
    ],
    edges=[
        E(("MU", "r"), ("AS", "l", "align"), label="login"),
        E(("AS", "r", -15), ("IP", "l"), via=[("x", 560), ("y", "T")], label="OIDC", lab=(2, 0.45)),
        E(("AS", "b"), ("TS", "l"), via=[("y", "T")], both=True),
        E(("IP", "r"), ("TS", "r"), via=[("x", 890), ("y", "T")]),
    ],
))


# ----------------------------------------------------------------------------
# geometry checks
# ----------------------------------------------------------------------------
def seg_hits_rect(p, q, r):
    x0, y0, x1, y1 = r
    dx, dy = q[0] - p[0], q[1] - p[1]
    t0, t1 = 0.0, 1.0
    for pp, qq in ((-dx, p[0] - x0), (dx, x1 - p[0]), (-dy, p[1] - y0), (dy, y1 - p[1])):
        if abs(pp) < 1e-12:
            if qq < 0:
                return False
        else:
            t = qq / pp
            if pp < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
            if t0 > t1:
                return False
    return True


def pt_seg_dist(p, a, b):
    p, a, b = np.array(p, float), np.array(a, float), np.array(b, float)
    ab = b - a
    L2 = float(ab @ ab)
    if L2 == 0:
        return float(np.linalg.norm(p - a))
    t = max(0.0, min(1.0, float((p - a) @ ab) / L2))
    return float(np.linalg.norm(p - (a + t * ab)))


def seg_seg_dist(a, b, c, d):
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) - (B[1] - A[1]) * (C[0] - A[0])
    d1, d2, d3, d4 = ccw(c, d, a), ccw(c, d, b), ccw(a, b, c), ccw(a, b, d)
    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)) and d1 != 0 and d2 != 0 and d3 != 0 and d4 != 0:
        return 0.0
    return min(pt_seg_dist(a, c, d), pt_seg_dist(b, c, d), pt_seg_dist(c, a, b), pt_seg_dist(d, a, b))


def expand(r, m):
    return (r[0] - m, r[1] - m, r[2] + m, r[3] + m)


def rects_overlap(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def inside(a, b, m=0):
    return a[0] >= b[0] + m and a[1] >= b[1] + m and a[2] <= b[2] - m and a[3] <= b[3] - m


def segments(path):
    return [(path[i], path[i + 1]) for i in range(len(path) - 1)]


# ----------------------------------------------------------------------------
# render one diagram
# ----------------------------------------------------------------------------
def shifted(spec, dx, dy):
    """Copy of spec with every absolute coordinate moved by integer (dx, dy)."""
    s = json.loads(json.dumps(spec))
    for n in s["nodes"]:
        if "x" in n:
            n["x"] += dx
            n["y"] += dy
        else:
            n["cx"] += dx
            n["cy"] += dy
    for c in s.get("containers", []):
        c["x"] += dx
        c["y"] += dy
    if s.get("frame"):
        s["frame"]["x"] += dx
        s["frame"]["y"] += dy
    for e in s["edges"]:
        e["s"], e["t"] = tuple(e["s"]), tuple(e["t"])
        if "via" in e:
            e["via"] = [(a, v if v == "T" else v + (dx if a == "x" else dy)) for a, v in e["via"]]
        for k in ("dash", "lab"):
            if k in e and e[k] is not None:
                e[k] = tuple(e[k])
    for k in ("dash",):
        if s["edge"].get(k):
            s["edge"][k] = tuple(s["edge"][k])
    return s


def tight_spec(spec):
    """Measure the rendered ink on a roomy canvas, then crop like a draw.io export:
    content + a uniform per-diagram margin."""
    x0, y0, x1, y1 = render(spec, measure_only=True)
    if x0 <= 0 or y0 <= 0 or x1 >= spec["W"] or y1 >= spec["H"]:
        raise RuntimeError("%s: content touches the measuring canvas" % spec["file"])
    m = spec.get("margin", 12)
    s = shifted(spec, m - x0, m - y0)
    s["W"] = (x1 - x0) + 2 * m
    s["H"] = (y1 - y0) + 2 * m
    return s


def render(spec, measure_only=False):
    W, H = spec["W"], spec["H"]
    D = spec
    # +1e-7 inch guards against float truncation of W/DPI*DPI (4.6*100 -> 459.999.. -> 459 px)
    # while keeping the data->pixel scale exact to ~1e-5 px.
    fig = plt.figure(figsize=(W / DPI + 1e-7, H / DPI + 1e-7), dpi=DPI, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    ax.set_facecolor("white")

    containers = []
    for c in spec.get("containers", []):
        p, t = draw_container(ax, c, D)
        containers.append(dict(spec=c, patch=p, title=t))
    frame = None
    if spec.get("frame"):
        fp, ftab, ft = draw_frame(ax, spec["frame"], D)
        frame = dict(spec=spec["frame"], patch=fp, tab=ftab, title=ft)

    nodes = {}
    for ns in spec["nodes"]:
        n = Node(ns, D)
        n.draw(ax, D)
        nodes[n.id] = n
    edges = []
    for es in spec["edges"]:
        e = Edge(es, nodes, D)
        e.draw(ax, D)
        edges.append(e)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    if measure_only:
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        ys, xs = np.nonzero((buf < 250).any(axis=2))
        plt.close(fig)
        return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

    # display -> saved-image pixel mapping.  We save with bbox_inches=None, so the saved
    # PNG is the full canvas: offset (0,0) and image height == canvas height.
    fw, fh = fig.canvas.get_width_height()
    assert (fw, fh) == (W, H), (fw, fh)
    off_x, off_y, img_h = 0.0, 0.0, float(fh)

    def disp2img(xd, yd):
        return xd - off_x, img_h - (yd - off_y)

    def data_box_to_img(r):
        (ax0, ay0), (ax1, ay1) = ax.transData.transform([(r[0], r[1]), (r[2], r[3])])
        X0, Y0 = disp2img(ax0, ay0)
        X1, Y1 = disp2img(ax1, ay1)
        return (min(X0, X1), min(Y0, Y1), max(X0, X1), max(Y0, Y1))

    def data_pt_to_img(p):
        xd, yd = ax.transData.transform([p])[0]
        return disp2img(xd, yd)

    def text_box(t, pad=0.0):
        bb = t.get_window_extent(renderer)
        X0, Y0 = disp2img(bb.x0, bb.y0)
        X1, Y1 = disp2img(bb.x1, bb.y1)
        return (min(X0, X1) - pad, min(Y0, Y1) - pad, max(X0, X1) + pad, max(Y0, Y1) + pad)

    # ---- GT boxes from transforms
    boxes = {}
    for n in nodes.values():
        r = data_box_to_img(n.pix)
        x, y = math.floor(r[0] + 0.01), math.floor(r[1] + 0.01)
        boxes[n.id] = dict(label=gt_label(n.label), x=int(x), y=int(y),
                           w=int(math.ceil(r[2] - 0.01) - x), h=int(math.ceil(r[3] - 0.01) - y))

    # ---- save
    out_png = os.path.join(DIAG_DIR, spec["file"])
    fig.savefig(out_png, dpi=DPI, facecolor="white", bbox_inches=None, pad_inches=0,
                metadata={"Software": None})
    im = Image.open(out_png)
    assert im.size == (W, H), im.size

    errors = []
    tag = spec["file"]

    # ---- verification render: each component alone, measure ink bbox
    all_artists = list(ax.patches) + list(ax.lines) + list(ax.texts)
    vis0 = {id(a): a.get_visible() for a in all_artists}
    max_dev = 0.0
    for n in nodes.values():
        keep = set(id(a) for a in n.shape_artists)
        for a in all_artists:
            a.set_visible(id(a) in keep)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        ink = (buf < 245).any(axis=2)
        ys, xs = np.nonzero(ink)
        b = boxes[n.id]
        rx0, ry0, rx1, ry1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        dev = max(abs(rx0 - b["x"]), abs(ry0 - b["y"]), abs(rx1 - (b["x"] + b["w"])), abs(ry1 - (b["y"] + b["h"])))
        max_dev = max(max_dev, dev)
        if dev > 1:
            errors.append("%s: box of %s deviates %d px from rendered ink (%s vs %s)" % (
                tag, n.id, dev, (b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]), (rx0, ry0, rx1, ry1)))
    for a in all_artists:
        a.set_visible(vis0[id(a)])
    fig.canvas.draw()

    # ---- collect texts
    texts = []  # (kind, owner, rect)
    for n in nodes.values():
        texts.append(("node" if n.kind in BOX_KINDS else "caption", n.id, text_box(n.text_artist)))
    for c in containers:
        texts.append(("ctitle", c["spec"]["title"], text_box(c["title"])))
    if frame:
        texts.append(("ftitle", frame["spec"]["title"], text_box(frame["title"])))
    for i, e in enumerate(edges):
        if e.label_artist is not None:
            texts.append(("elabel", i, text_box(e.label_artist, e.label_pad)))

    nbox = {k: (b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]) for k, b in boxes.items()}

    # canvas margins
    for k, r in list(nbox.items()) + [(t[1], t[2]) for t in texts]:
        if r[0] < 4 or r[1] < 4 or r[2] > W - 4 or r[3] > H - 4:
            errors.append("%s: %s too close to canvas edge %s" % (tag, k, r))
    arr = np.asarray(im.convert("RGB"))
    border = np.concatenate([arr[:3].reshape(-1, 3), arr[-3:].reshape(-1, 3), arr[:, :3].reshape(-1, 3),
                             arr[:, -3:].reshape(-1, 3)])
    if (border < 250).any():
        errors.append("%s: non-white pixels in 3px border (clipping risk)" % tag)

    # node-node spacing
    ids = list(nodes)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if rects_overlap(expand(nbox[ids[i]], 10), nbox[ids[j]]):
                errors.append("%s: nodes %s and %s too close" % (tag, ids[i], ids[j]))

    # nodes vs containers
    crects = []
    for c in containers:
        cs = c["spec"]
        crects.append((cs["title"], (cs["x"], cs["y"], cs["x"] + cs["w"], cs["y"] + cs["h"])))
    if frame:
        fs_ = frame["spec"]
        crects.append((fs_["title"], (fs_["x"], fs_["y"], fs_["x"] + fs_["w"], fs_["y"] + fs_["h"])))
    for nid, r in nbox.items():
        for cname, cr in crects:
            if not (inside(r, cr, 8) or not rects_overlap(expand(r, 8), cr)):
                errors.append("%s: node %s straddles container '%s'" % (tag, nid, cname))

    # texts
    for a in range(len(texts)):
        for b in range(a + 1, len(texts)):
            if rects_overlap(expand(texts[a][2], 2), texts[b][2]):
                errors.append("%s: text overlap %s/%s" % (tag, texts[a][:2], texts[b][:2]))
    for kind, owner, tr in texts:
        for nid, r in nbox.items():
            if kind == "node" and owner == nid:
                if not inside(tr, r, 3):
                    errors.append("%s: label of %s does not fit its box" % (tag, nid))
                continue
            if rects_overlap(expand(tr, 2), r):
                errors.append("%s: text %s/%s overlaps node %s" % (tag, kind, owner, nid))
    if frame:
        ft = [t for t in texts if t[0] == "ftitle"][0][2]
        fsp = frame["spec"]
        tab = (fsp["x"], fsp["y"], fsp["x"] + fsp["tab_w"] - 8, fsp["y"] + fsp["tab_h"])
        if not inside(ft, tab, 1):
            errors.append("%s: frame title does not fit tab" % tag)

    # edges
    endpoints = {}
    for i, e in enumerate(edges):
        raw = [data_pt_to_img(p) for p in e.raw]
        path = [data_pt_to_img(p) for p in e.path]
        heads = [[data_pt_to_img(p) for p in hp] for hp in e.heads]
        e.img_raw, e.img_path, e.img_heads = raw, path, heads
        L = e.st["hs"]
        # orientation of raw segments
        for (p, q) in segments(raw):
            dx, dy = q[0] - p[0], q[1] - p[1]
            if abs(dx) > 1e-6 and abs(dy) > 1e-6:
                ang = math.degrees(math.atan2(abs(dy), abs(dx)))
                if ang < 10 or ang > 80:
                    errors.append("%s: edge %d has a near-axis diagonal (%.1f deg)" % (tag, i, ang))
        # anchor ranges and exit directions
        for (nid, side, off), pt, nxt in ((e.send[0:3], raw[0], raw[1]), (e.tend[0:3], raw[-1], raw[-2])):
            lo, hi = nodes[nid].side_range(side)
            if not (lo - 0.6 <= off <= hi + 0.6):
                errors.append("%s: edge %d anchor offset %.1f outside %s side %s range (%.1f,%.1f)" % (
                    tag, i, off, nid, side, lo, hi))
            nrm = {"l": (-1, 0), "r": (1, 0), "t": (0, -1), "b": (0, 1)}[side]
            d = np.array(nxt) - np.array(pt)
            d /= np.linalg.norm(d)
            if d[0] * nrm[0] + d[1] * nrm[1] < 0.3:
                errors.append("%s: edge %d leaves %s side %s inward/parallel" % (tag, i, nid, side))
            endpoints.setdefault(nid, []).append((i, pt))
        # straight run before heads
        last = math.dist(raw[-1], raw[-2])
        need = L + (e.st["r"] if len(raw) > 2 else 0) + 4
        if last < need:
            errors.append("%s: edge %d last segment %.0f < %.0f" % (tag, i, last, need))
        if e.st["both"]:
            first = math.dist(raw[0], raw[1])
            if first < need:
                errors.append("%s: edge %d first segment %.0f < %.0f" % (tag, i, first, need))
        segs = segments(path)
        hsegs = [s for hp in heads for s in segments(hp + [hp[0]])]
        for nid, r in nbox.items():
            if nid in (e.src, e.dst):
                g = data_box_to_img(nodes[nid].g)
                inner = expand(g, -1.5)
                for (p, q) in segs:
                    if seg_hits_rect(p, q, inner):
                        errors.append("%s: edge %d enters its endpoint %s" % (tag, i, nid))
                        break
                continue
            for (p, q) in segs + hsegs:
                if seg_hits_rect(p, q, expand(r, 3)):
                    errors.append("%s: edge %d crosses node %s" % (tag, i, nid))
                    break
        for kind, owner, tr in texts:
            if kind == "elabel" and owner == i:
                if not any(seg_hits_rect(p, q, tr) for p, q in segs):
                    errors.append("%s: edge %d label not on its line" % (tag, i))
                continue
            for (p, q) in segs + hsegs:
                if seg_hits_rect(p, q, expand(tr, 2)):
                    errors.append("%s: edge %d crosses text %s/%s" % (tag, i, kind, owner))
                    break
        # running along container borders
        for cname, cr in crects:
            sides = [((cr[0], cr[1]), (cr[2], cr[1])), ((cr[0], cr[3]), (cr[2], cr[3])),
                     ((cr[0], cr[1]), (cr[0], cr[3])), ((cr[2], cr[1]), (cr[2], cr[3]))]
            for (p, q) in segs:
                for (a, b) in sides:
                    if abs(p[1] - q[1]) < 1e-6 and abs(a[1] - b[1]) < 1e-6 and abs(p[1] - a[1]) < 5:
                        ov = min(max(p[0], q[0]), max(a[0], b[0])) - max(min(p[0], q[0]), min(a[0], b[0]))
                        if ov > 4:
                            errors.append("%s: edge %d runs along border of '%s'" % (tag, i, cname))
                    if abs(p[0] - q[0]) < 1e-6 and abs(a[0] - b[0]) < 1e-6 and abs(p[0] - a[0]) < 5:
                        ov = min(max(p[1], q[1]), max(a[1], b[1])) - max(min(p[1], q[1]), min(a[1], b[1]))
                        if ov > 4:
                            errors.append("%s: edge %d runs along border of '%s'" % (tag, i, cname))
    # edge-edge
    for i in range(len(edges)):
        for j in range(i + 1, len(edges)):
            si = segments(edges[i].img_path) + [s for hp in edges[i].img_heads for s in segments(hp + [hp[0]])]
            sj = segments(edges[j].img_path) + [s for hp in edges[j].img_heads for s in segments(hp + [hp[0]])]
            dmin = min(seg_seg_dist(a, b, c, d) for (a, b) in si for (c, d) in sj)
            if dmin < 6:
                errors.append("%s: edges %d and %d come within %.1f px" % (tag, i, j, dmin))
    for nid, lst in endpoints.items():
        for a in range(len(lst)):
            for b in range(a + 1, len(lst)):
                if math.dist(lst[a][1], lst[b][1]) < 10:
                    errors.append("%s: endpoints of edges %d/%d on %s too close" % (tag, lst[a][0], lst[b][0], nid))

    # ---- ground truth
    gt = dict(name=spec["name"], source=SOURCE,
              components=[gt_label(n.label) for n in nodes.values()],
              arrows=[dict(source=gt_label(nodes[e.src].label), target=gt_label(nodes[e.dst].label),
                           style="dashed" if e.st["dash"] else "solid", bidirectional=bool(e.st["both"]))
                      for e in edges],
              icons=[])
    gtb = dict(image_size=[W, H], boxes=[boxes[n.id] for n in nodes.values()])
    det = dict(
        name=spec["name"], font=spec["font"], font_pt=spec["fs"],
        nodes=[dict(label=gt_label(n.label), kind=n.kind) for n in nodes.values()],
        containers=[dict(title=c["spec"]["title"], rect=list(r), kind="container") for c, (_, r) in
                    zip(containers, crects)] + ([dict(title=frame["spec"]["title"], rect=list(crects[-1][1]),
                                                     kind="frame")] if frame else []),
        connectors=[dict(source=gt_label(nodes[e.src].label), target=gt_label(nodes[e.dst].label),
                         bidirectional=bool(e.st["both"]), color=e.st["color"], width_px=e.st["lw"],
                         dash_px=list(e.st["dash"]) if e.st["dash"] else None, head=e.st["head"],
                         head_px=e.st["hs"], corner_radius=e.st["r"], bends=len(e.img_raw) - 2,
                         label=e.label, route=[[round(p[0], 1), round(p[1], 1)] for p in e.img_raw])
                    for e in edges],
    )
    overlay(out_png, spec, nodes, edges, boxes)
    plt.close(fig)
    return gt, gtb, det, errors, max_dev


# ----------------------------------------------------------------------------
# overlay for review
# ----------------------------------------------------------------------------
def _font(sz):
    for f in ("C:/Windows/Fonts/arial.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, sz)
        except Exception:
            pass
    return ImageFont.load_default()


def overlay(png, spec, nodes, edges, boxes):
    base = Image.open(png).convert("RGBA")
    W, H = base.size
    lay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    f = _font(11)
    for i, e in enumerate(edges):
        pts = [tuple(p) for p in e.img_raw]
        d.line(pts, fill=(0, 190, 0, 150), width=3)
        tx, ty = pts[-1]
        d.ellipse((tx - 4, ty - 4, tx + 4, ty + 4), fill=(230, 0, 200, 230))
        sx, sy = pts[0]
        if e.st["both"]:
            d.ellipse((sx - 4, sy - 4, sx + 4, sy + 4), fill=(230, 0, 200, 230))
        else:
            d.rectangle((sx - 3, sy - 3, sx + 3, sy + 3), outline=(0, 90, 255, 255), width=2)
        # index tag at middle of longest raw segment
        segs = segments(pts)
        k = max(range(len(segs)), key=lambda s: math.dist(*segs[s]))
        mx = (segs[k][0][0] + segs[k][1][0]) / 2
        my = (segs[k][0][1] + segs[k][1][1]) / 2
        d.rectangle((mx + 4, my + 3, mx + 24, my + 17), fill=(255, 255, 255, 220))
        d.text((mx + 6, my + 4), "e%d" % (i + 1), fill=(0, 140, 0, 255), font=f)
    for nid, b in boxes.items():
        d.rectangle((b["x"], b["y"], b["x"] + b["w"] - 1, b["y"] + b["h"] - 1), outline=(255, 0, 0, 255), width=1)
    img = Image.alpha_composite(base, lay).convert("RGB")
    # legend panel
    lines = ["%s  (%dx%d)   red = GT component boxes, green = connector routes, blue square = tail (source), "
             "magenta dot = head (target)" % (spec["file"], W, H)]
    for i, e in enumerate(edges):
        lines.append("e%d: %s %s %s   [%s%s, %s %dpx%s]" % (
            i + 1, gt_label(nodes[e.src].label), "<->" if e.st["both"] else "->", gt_label(nodes[e.dst].label),
            "dashed" if e.st["dash"] else "solid", (" %s" % (tuple(e.st["dash"]),)) if e.st["dash"] else "",
            e.st["head"], e.st["hs"], (", label '%s'" % e.label) if e.label else ""))
    lh = 15
    cols = 2
    rows = int(math.ceil((len(lines) - 1) / cols)) + 1
    panel = Image.new("RGB", (W, rows * lh + 10), (250, 250, 235))
    pd = ImageDraw.Draw(panel)
    pd.text((6, 4), lines[0], fill=(0, 0, 0), font=f)
    for k, ln in enumerate(lines[1:]):
        c, r = k % cols, k // cols
        pd.text((6 + c * (W // cols), 4 + (r + 1) * lh), ln, fill=(0, 0, 0), font=f)
    out = Image.new("RGB", (W, H + panel.size[1]), (255, 255, 255))
    out.paste(img, (0, 0))
    out.paste(panel, (0, H))
    out.save(os.path.join(OVL_DIR, spec["file"].replace(".png", "_overlay.png")))


# ----------------------------------------------------------------------------
def main():
    os.makedirs(DIAG_DIR, exist_ok=True)
    os.makedirs(OVL_DIR, exist_ok=True)
    GT, GTB, DET = {}, {}, {}
    all_err = []
    for spec0 in SPECS:
        spec = tight_spec(spec0)
        gt, gtb, det, errs, dev = render(spec)
        if not (880 <= spec["W"] <= 1420 and 440 <= spec["H"] <= 820):
            errs.append("%s: final size %dx%d outside the target range" % (spec["file"], spec["W"], spec["H"]))
        GT[spec["file"]] = gt
        GTB[spec["file"]] = gtb
        DET[spec["file"]] = det
        all_err += errs
        print("%s  %dx%d  components=%d connectors=%d  max box/ink deviation=%dpx  issues=%d" % (
            spec["file"], spec["W"], spec["H"], len(gt["components"]), len(gt["arrows"]), dev, len(errs)))
    with open(os.path.join(HERE, "ground_truth.json"), "w", encoding="utf-8") as fh:
        json.dump(GT, fh, indent=2)
    with open(os.path.join(HERE, "ground_truth_boxes.json"), "w", encoding="utf-8") as fh:
        json.dump(GTB, fh, indent=2)
    with open(os.path.join(HERE, "ground_truth_details.json"), "w", encoding="utf-8") as fh:
        json.dump(DET, fh, indent=2)
    for e in all_err:
        print("ISSUE:", e)
    print("TOTAL ISSUES:", len(all_err))
    return 1 if all_err else 0


if __name__ == "__main__":
    sys.exit(main())
