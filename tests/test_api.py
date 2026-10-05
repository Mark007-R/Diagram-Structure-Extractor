"""Tests for the FastAPI service at `src/api.py`.

Uses Starlette's TestClient — no real network/uvicorn needed.
"""
from __future__ import annotations

import base64
import io
import os

import pytest


def _client():
    from fastapi.testclient import TestClient
    from src.api import app
    return TestClient(app)


def test_health_endpoint():
    c = _client()
    r = c.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "diagramine"


def test_extract_returns_schema_valid_json(test_image: str):
    """POST /extract must return a JSON body with structure.image, structure.boxes,
    structure.relationships keys — schema-valid by Pydantic construction."""
    c = _client()
    with open(test_image, "rb") as f:
        files = {"file": ("diagram_01.png", f.read(), "image/png")}
    r = c.post("/extract", files=files)
    assert r.status_code == 200, r.text
    body = r.json()
    assert "structure" in body
    assert "annotated_png_base64" in body
    assert "relationship_csv" in body
    structure = body["structure"]
    for k in ("image", "width", "height", "texts", "boxes", "regions",
              "arrows", "icons", "relationships", "runtime_seconds"):
        assert k in structure, f"Missing key in structure: {k}"


def test_extract_annotated_png_is_valid_image(test_image: str):
    """The base64-encoded annotated PNG must decode back to a valid image."""
    c = _client()
    with open(test_image, "rb") as f:
        files = {"file": ("diagram_01.png", f.read(), "image/png")}
    r = c.post("/extract", files=files)
    assert r.status_code == 200
    png_bytes = base64.b64decode(r.json()["annotated_png_base64"])
    # PNG magic header
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n", "Returned PNG header is invalid."


def test_extract_with_every_stage_set_to_none(test_image: str):
    """Selecting "none" skips a stage: it contributes nothing, records no error,
    and the response is still schema-valid."""
    c = _client()
    with open(test_image, "rb") as f:
        files = {"file": ("diagram_01.png", f.read(), "image/png")}
    r = c.post("/extract?text=none&box=none&arrow=none&icon=none", files=files)
    assert r.status_code == 200, r.text
    s = r.json()["structure"]
    for k in ("texts", "boxes", "regions", "arrows", "icons", "relationships"):
        assert s[k] == [], k
    assert s["detectors"]["errors"] == {}


def test_extract_relationships_all_detected_true(test_image: str):
    """Every relationship returned by the API must be marked detected=True —
    no hardcoded edges leak into the API surface."""
    c = _client()
    with open(test_image, "rb") as f:
        files = {"file": ("diagram_01.png", f.read(), "image/png")}
    r = c.post("/extract", files=files)
    body = r.json()
    for rel in body["structure"]["relationships"]:
        assert rel["detected"] is True, (
            f"API surface emitted a relationship with detected=False — "
            f"hardcoded edges should never leave the service: {rel}"
        )
