"""Render assets/architecture.png.

Pillow, drawn at 2x and downsampled. Dark card with light text so it reads on
both the GitHub light and dark themes.

Run:  python assets/make_architecture.py
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

S = 2
W, H = 980 * S, 600 * S
OUT = Path(__file__).with_name("architecture.png")

BG, FG, MUTED, LINE = (13, 17, 23), (201, 209, 217), (139, 148, 158), (110, 118, 129)
ACCENT, GREEN, AMBER = (188, 140, 255), (63, 185, 80), (210, 153, 34)
FONTS = r"C:\Windows\Fonts"


def font(n, s):
    return ImageFont.truetype(f"{FONTS}\\{n}", s * S)


f_title, f_head = font("seguisb.ttf", 15), font("seguisb.ttf", 12)
f_small, f_lbl = font("segoeui.ttf", 10), font("segoeuii.ttf", 9)

img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)


def box(x, y, w, h, c=LINE, width=2):
    d.rounded_rectangle([x * S, y * S, (x + w) * S, (y + h) * S],
                        radius=6 * S, outline=c, width=int(width * S))


def text(x, y, s, f=f_small, fill=MUTED, anchor="mm"):
    d.text((x * S, y * S), s, font=f, fill=fill, anchor=anchor)


def _head(p0, p1, c, size=6):
    (x0, y0), (x1, y1) = p0, p1
    dx, dy = x1 - x0, y1 - y0
    dist = max((dx * dx + dy * dy) ** .5, 1e-6)
    ux, uy = dx / dist, dy / dist
    px, py = -uy, ux
    s = size * S
    d.polygon([(x1, y1),
               (x1 - ux * s + px * s * .5, y1 - uy * s + py * s * .5),
               (x1 - ux * s - px * s * .5, y1 - uy * s - py * s * .5)], fill=c)


def arrow(pts, c=LINE, w=1.5):
    pts = [(x * S, y * S) for x, y in pts]
    for i in range(len(pts) - 1):
        d.line([pts[i], pts[i + 1]], fill=c, width=int(w * S))
    _head(pts[-2], pts[-1], c)


text(490, 26, "Diagram-Structure-Extractor — four detectors into one schema-valid JSON object",
     f_title, FG)

box(340, 52, 300, 50)
text(490, 68, "Upload — PNG / JPG / WebP", f_head, FG)
text(490, 87, "CLI · FastAPI · Streamlit")

arrow([(490, 102), (490, 130)])
box(30, 132, 920, 40, LINE)
text(52, 152, "src/pipeline.py — orchestrator: each stage is wrapped so a failure yields an empty list, never a crash",
     f_small, FG, anchor="lm")

# ── detectors ───────────────────────────────────────────────────────────────
cols = [
    (30, "Text", "PaddleOCR", "F1 0.955", "EasyOCR · Tesseract", GREEN),
    (265, "Boxes", "Canny + contours", "F1 1.000", "Hough · YOLOv8", GREEN),
    (500, "Arrows", "Directed lines", "F1 0.962", "CNN · Hough · pixel scan", GREEN),
    (735, "Icons", "Template match", "F1 1.000", "CLIP · HSV", GREEN),
]
for x, t, champ, f1, alts, c in cols:
    arrow([(x + 107, 172), (x + 107, 194)])
    box(x, 196, 215, 86, c)
    text(x + 107, 216, t, f_head, FG)
    text(x + 107, 236, champ, f_small, FG)
    text(x + 107, 255, f1)
    text(x + 107, 272, alts, f_lbl)

text(607, 294, "real draw.io diagram still weak — 0.31 rel-F1", f_lbl, AMBER, anchor="mm")

# ── graph builder ───────────────────────────────────────────────────────────
for x in (137, 372, 607, 842):
    arrow([(x, 282), (x, 308), (490, 308), (490, 320)])
box(240, 322, 500, 62, ACCENT)
text(490, 344, "src/graph/builder.py", f_head, FG)
text(490, 363, "label boxes · build relationships from geometry, no hardcoded edges")

# ── output ──────────────────────────────────────────────────────────────────
arrow([(490, 384), (490, 410)])
box(240, 412, 500, 74, GREEN)
text(490, 434, "Pydantic v2  ExtractionResult", f_head, FG)
text(490, 453, "typed and validated by construction — schema-valid on 15/15 benchmark diagrams")
text(490, 471, "texts · boxes · regions · arrows · icons · relationships · per-stage runtimes")

box(30, 412, 190, 74)
text(125, 434, "Benchmark", f_head, FG)
text(125, 453, "14 synthetic + 1 real")
text(125, 471, "+ ground truth JSON")
arrow([(220, 449), (238, 449)], GREEN)

box(760, 412, 190, 74)
text(855, 434, "Outputs", f_head, FG)
text(855, 453, "JSON · annotated PNG")
text(855, 471, "graph PNG · edge CSV")

text(30, 522, "The reliability claim is structural: outputs are typed models, so every call returns parseable JSON",
     f_lbl, GREEN, anchor="lm")
text(30, 542, "The Claude Vision comparison in the README is a literature-based PROJECTION — that run had no API key",
     f_lbl, AMBER, anchor="lm")
text(30, 562, "Docker · 35 pytest tests including a no-hardcoding AST regression",
     f_lbl, MUTED, anchor="lm")

img.resize((W // S, H // S), Image.LANCZOS).save(OUT, "PNG", optimize=True)
print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")
