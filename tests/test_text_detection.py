"""Tests for text detection backends."""
from __future__ import annotations

import os

import pytest


def test_easyocr_returns_schema_valid_dict(test_image: str):
    """EasyOCR detector returns a schema-valid dict with the expected keys."""
    from src.text_detection import easyocr_detector as det
    out = det.detect(test_image)
    assert set(out.keys()) >= {"texts", "runtime_seconds"}
    assert isinstance(out["texts"], list)
    assert out["runtime_seconds"] >= 0.0


def test_easyocr_text_elements_have_required_fields(test_image: str):
    """Every detected text element must have x/y/w/h/text/conf fields."""
    from src.text_detection import easyocr_detector as det
    out = det.detect(test_image)
    for t in out["texts"]:
        assert "text" in t
        assert all(k in t for k in ("x", "y", "w", "h"))
        assert "conf" in t
        assert isinstance(t["text"], str)


def test_easyocr_finds_some_text_on_benchmark(test_image: str):
    """diagram_01.png is a 3-tier web app with 'Browser', 'Web Server',
    'App Server', 'Database' — we expect at least 2 detected labels."""
    from src.text_detection import easyocr_detector as det
    out = det.detect(test_image)
    assert len(out["texts"]) >= 2, (
        f"Expected EasyOCR to find at least 2 text labels on diagram_01, "
        f"got {len(out['texts'])}"
    )


def test_paddleocr_returns_schema_valid_dict(test_image: str):
    """PaddleOCR detector returns a schema-valid dict with the expected keys."""
    from src.text_detection import paddle_detector as det
    out = det.detect(test_image)
    assert set(out.keys()) >= {"texts", "runtime_seconds"}
    for t in out["texts"]:
        assert all(k in t for k in ("text", "x", "y", "w", "h", "conf"))
        assert isinstance(t["text"], str)


def test_paddleocr_reads_benchmark_labels(test_image: str):
    """diagram_01.png's four labels should all be read exactly."""
    from src.text_detection import paddle_detector as det
    texts = {t["text"] for t in det.detect(test_image)["texts"]}
    assert {"Browser", "Web Server", "App Server", "Database"} <= texts, texts


def test_default_text_detector_is_paddleocr():
    from src.schemas import PipelineConfig
    assert PipelineConfig().text_detector == "paddleocr"


def test_default_pipeline_text_stage_runs(test_image: str):
    """The default text detector must actually load. A failing stage degrades
    to an empty list by design, so without this check a broken OCR install
    (e.g. a protobuf/paddlepaddle mismatch) would silently return no text."""
    from src import pipeline as pl
    from src.schemas import PipelineConfig
    result = pl.extract(test_image, PipelineConfig())
    assert "text" not in result.detectors.get("errors", {}), result.detectors["errors"]
    assert len(result.texts) >= 4


def test_default_pipeline_runs_in_fresh_process(test_image: str, repo_root: str):
    """Same check in a clean interpreter. Whether the OCR stack loads can
    depend on import order (on Windows torch must load before paddle), and
    the test process may already have imported torch, which would hide it."""
    import subprocess
    import sys
    code = (
        "from src import pipeline as pl; from src.schemas import PipelineConfig; "
        f"r = pl.extract(r'{test_image}', PipelineConfig()); "
        "assert not r.detectors['errors'], r.detectors['errors']; "
        "assert len(r.texts) >= 4 and len(r.relationships) >= 3, (len(r.texts), len(r.relationships))"
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=repo_root,
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
