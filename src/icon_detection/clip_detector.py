"""CLIP zero-shot icon detector.

Strategy: HSV-segment the image to find candidate icon-like coloured regions,
then for each crop run CLIP image-text similarity against a vocabulary of
icon-name prompts. Keep crops whose top-1 label has cosine similarity above
a threshold AND whose top-1 is NOT the "no icon" decoy class.

Uses `openai/clip-vit-base-patch32` via transformers (CPU). The model is
downloaded on first use (~600 MB) and cached. We share the HSV candidate
proposal across HSV and CLIP detectors so they're scored on the SAME
candidate set — the only difference is what label is assigned.

Vocabulary intentionally includes the two GT classes (`docker`, `MS SQL`)
plus 5 decoys (`aws`, `kubernetes`, `elasticsearch`, `azure`, `nothing`)
to make false-positive evaluation meaningful — without decoys CLIP could
not produce a wrong label.
"""
from __future__ import annotations

import os
import time
from typing import List

import cv2
import numpy as np
import torch
from PIL import Image

from src.icon_detection import hsv_detector

_MODEL_NAME = "openai/clip-vit-base-patch32"
_VOCAB = [
    "a docker container logo",
    "a Microsoft SQL Server database icon",
    "the AWS cloud logo",
    "the Kubernetes logo",
    "the elasticsearch logo",
    "the Microsoft Azure logo",
    "a small drawing or coloured shape, not a software logo",
]
# Map embedded prompts back to short labels for output
_VOCAB_LABELS = ["docker", "MS SQL", "AWS", "Kubernetes",
                 "elasticsearch", "Azure", "no_icon"]
_NO_ICON_INDEX = _VOCAB_LABELS.index("no_icon")
_SIM_THRESHOLD = 0.21   # cosine; CLIP zero-shot is well-known to score 0.2-0.35

_MODEL = None
_PROCESSOR = None
_TEXT_EMBEDS = None


def _embedding(out) -> torch.Tensor:
    """transformers < 5 returns the projected embedding as a tensor; 5.x wraps
    it in a BaseModelOutputWithPooling, as `pooler_output`."""
    return out if isinstance(out, torch.Tensor) else out.pooler_output


def _load():
    global _MODEL, _PROCESSOR, _TEXT_EMBEDS
    if _MODEL is not None:
        return
    from transformers import CLIPModel, CLIPProcessor   # noqa: E501
    _PROCESSOR = CLIPProcessor.from_pretrained(_MODEL_NAME)
    _MODEL = CLIPModel.from_pretrained(_MODEL_NAME).eval()
    with torch.no_grad():
        inputs = _PROCESSOR(text=_VOCAB, return_tensors="pt", padding=True)
        text_emb = _embedding(_MODEL.get_text_features(**inputs))
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        _TEXT_EMBEDS = text_emb


def detect(image_path: str) -> dict:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    start = time.perf_counter()
    _load()

    # Reuse the HSV detector's coloured-region proposals as candidate crops.
    proposals = hsv_detector.detect(image_path)["icons"]
    if not proposals:
        elapsed = time.perf_counter() - start
        return {"icons": [], "runtime_seconds": round(elapsed, 3)}

    crops_pil: List[Image.Image] = []
    for p in proposals:
        x, y, w, h = p["x"], p["y"], p["w"], p["h"]
        # Add a small margin to give CLIP some context
        pad = 4
        x0 = max(0, x - pad); y0 = max(0, y - pad)
        x1 = min(img.shape[1], x + w + pad); y1 = min(img.shape[0], y + h + pad)
        crop = img[y0:y1, x0:x1]
        crops_pil.append(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))

    icons: List[dict] = []
    with torch.no_grad():
        inputs = _PROCESSOR(images=crops_pil, return_tensors="pt")
        img_emb = _embedding(_MODEL.get_image_features(**inputs))
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        sims = (img_emb @ _TEXT_EMBEDS.T).numpy()   # [n_crops, vocab]

    for p, sim_row in zip(proposals, sims):
        # Argmax over labels EXCLUDING the "no_icon" decoy — i.e. find the
        # best-matching real label, then accept only if sim > threshold AND
        # that real label beats the no_icon class.
        real_sims = sim_row.copy()
        no_icon_sim = real_sims[_NO_ICON_INDEX]
        real_sims[_NO_ICON_INDEX] = -np.inf
        top = int(np.argmax(real_sims))
        top_sim = float(real_sims[top])
        if top_sim < _SIM_THRESHOLD or top_sim < no_icon_sim:
            continue
        icons.append({
            "x": p["x"], "y": p["y"],
            "w": p["w"], "h": p["h"],
            "label": _VOCAB_LABELS[top],
            "score": top_sim,
        })

    elapsed = time.perf_counter() - start
    return {"icons": icons, "runtime_seconds": round(elapsed, 3)}
