"""PaddleOCR text detector. Uses the English model; CPU-only.

paddleocr 2.x API: PaddleOCR(use_angle_cls=True, lang='en'). The .ocr()
return format is [[(bbox, (text, conf)), ...]] for the single-image case.
"""
from __future__ import annotations

import logging
import os
import time
from typing import List

import cv2
import numpy as np

# paddlepaddle 2.6's generated protobuf code only loads under protobuf's
# pure-Python implementation when protobuf 4+ is installed (requirements.txt
# pins 3.20.x, where this is already the case). Only takes effect if nothing
# has imported protobuf yet; harmless otherwise.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

# Silence paddle's verbose logger
logging.getLogger("ppocr").setLevel(logging.ERROR)

_OCR = None


def _get_ocr():
    global _OCR
    if _OCR is None:
        # torch must load before paddle. paddleocr imports albumentations,
        # which imports torch; by then paddle has loaded its older bundled
        # OpenMP runtime (libiomp5md.dll on Windows) and torch fails to load
        # ("WinError 127 ... shm.dll"), leaving every later torch-backed
        # detector (EasyOCR, CLIP, the arrow CNN) broken in this process too.
        try:
            import torch  # noqa: F401
        except ImportError:
            pass
        from paddleocr import PaddleOCR
        _OCR = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    return _OCR


def detect(image_path: str, min_conf: float = 0.10) -> dict:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    ocr = _get_ocr()
    start = time.perf_counter()
    # PaddleOCR accepts ndarray or path.
    result = ocr.ocr(img, cls=True)
    elapsed = time.perf_counter() - start

    texts: List[dict] = []
    if not result:
        return {"texts": texts, "runtime_seconds": round(elapsed, 3)}

    page = result[0] if isinstance(result[0], list) else result
    if page is None:
        return {"texts": texts, "runtime_seconds": round(elapsed, 3)}

    for entry in page:
        if entry is None:
            continue
        # entry = [bbox, (text, conf)]
        bbox, (text, conf) = entry
        if conf is None or conf < min_conf:
            continue
        pts = np.array(bbox, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        texts.append(
            {
                "text": text.strip(),
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "conf": float(conf),
            }
        )
    return {"texts": texts, "runtime_seconds": round(elapsed, 3)}
