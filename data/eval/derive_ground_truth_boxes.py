"""
Derive pixel-level bounding boxes for diagram_01..14 by re-running the same
matplotlib render that produced them (canvas, dpi=120, bbox_inches="tight",
pad_inches=0.3) and projecting the box (x, y, w, h) figure-coordinates through
``ax.transData.transform`` into the saved PNG's pixel coordinates.

The 15th diagram (search_interview_test.png) is a real published architecture
diagram with no machine-readable box coordinates, so it is excluded from
pixel-level IoU evaluation. Its component labels remain in ground_truth.json.

Output: data/eval/ground_truth_boxes.json
  {
    "diagram_01.png": {
        "image_size": [w_px, h_px],
        "boxes": [
            {"label": "Browser", "x": 78, "y": 75, "w": 240, "h": 144},
            ...
        ]
    },
    ...
  }
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# Reuse the specs and the exact figure builder from generate_diagrams.py.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from generate_diagrams import SPECS, build_figure  # type: ignore


PAD_INCHES = 0.3
DPI = 120


def derive_pixel_boxes(spec: dict, target_png: str) -> dict:
    """Re-render the spec with generate_diagrams.build_figure() and
    map each box's figure-coordinate corners into pixel coordinates of the
    saved PNG. Verified by checking that the re-rendered figure's saved size
    matches the on-disk PNG dimensions for this diagram.
    """
    boxes = spec["boxes"]
    fig, ax = build_figure(spec)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Figure.get_tightbbox returns a bbox in INCHES (matplotlib quirk).
    # Convert to figure pixels using DPI.
    tight_in = fig.get_tightbbox(renderer)
    pad_px = PAD_INCHES * DPI
    crop_x0 = tight_in.x0 * DPI - pad_px
    crop_y0 = tight_in.y0 * DPI - pad_px
    saved_w = int(round((tight_in.x1 - tight_in.x0 + 2 * PAD_INCHES) * DPI))
    saved_h = int(round((tight_in.y1 - tight_in.y0 + 2 * PAD_INCHES) * DPI))

    pixel_boxes: List[dict] = []
    for box in boxes:
        x, y, w, h, label = box
        bl = ax.transData.transform((x, y))  # bottom-left in figure-display px
        tr = ax.transData.transform((x + w, y + h))
        bl_x = bl[0] - crop_x0
        tr_x = tr[0] - crop_x0
        bl_y = bl[1] - crop_y0
        tr_y = tr[1] - crop_y0
        # Flip y to PIL (origin top-left)
        px_x = int(round(min(bl_x, tr_x)))
        px_x2 = int(round(max(bl_x, tr_x)))
        px_y = int(round(saved_h - max(bl_y, tr_y)))
        px_y2 = int(round(saved_h - min(bl_y, tr_y)))
        pixel_boxes.append(
            {
                "label": label,
                "x": px_x,
                "y": px_y,
                "w": px_x2 - px_x,
                "h": px_y2 - px_y,
            }
        )

    plt.close(fig)

    # Sanity: compare with on-disk PNG size
    on_disk = Image.open(target_png).size
    return {
        "image_size": [saved_w, saved_h],
        "on_disk_size": [on_disk[0], on_disk[1]],
        "boxes": pixel_boxes,
    }


def main() -> None:
    out: Dict[str, dict] = {}
    diagrams_dir = os.path.join(HERE, "diagrams_15")
    for idx, spec in enumerate(SPECS, start=1):
        fname = f"diagram_{idx:02d}.png"
        target = os.path.join(diagrams_dir, fname)
        out[fname] = derive_pixel_boxes(spec, target)
        derived = out[fname]["image_size"]
        disk = out[fname]["on_disk_size"]
        match = "OK" if derived == disk else f"MISMATCH derived={derived} disk={disk}"
        print(f"[gt-boxes] {fname:<22} {len(out[fname]['boxes'])} boxes  {match}")

    out_path = os.path.join(HERE, "ground_truth_boxes.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[gt-boxes] Wrote {out_path}")


if __name__ == "__main__":
    main()
