"""FastAPI service for DiagraMine.

    POST /extract   multipart image upload -> structured JSON + annotated PNG
                    (base64) + relationship CSV (text). Always returns
                    schema-valid JSON (the reliability guarantee).
    GET  /health    liveness probe.

Run:  uvicorn src.api:app --port 8000
"""
from __future__ import annotations

import base64
import csv
import io
import os
import tempfile

import cv2
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src import pipeline
from src.schemas import ExtractionResult, PipelineConfig

app = FastAPI(
    title="DiagraMine",
    description="Diagram understanding & structure extraction (data-driven, no hardcoded connections).",
    version="4.0",
)


class ExtractResponse(BaseModel):
    structure: ExtractionResult
    annotated_png_base64: str
    relationship_csv: str


def _csv_string(result: ExtractionResult) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["source", "target", "line_style", "direction", "relationship"])
    for r in result.relationships:
        w.writerow([r.source, r.target, r.line_style, r.direction, r.relationship])
    return buf.getvalue()


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "diagramine", "version": "4.0"}


@app.post("/extract", response_model=ExtractResponse)
async def extract(
    file: UploadFile = File(...),
    text: str = Query("easyocr"),
    box: str = Query("canny_contours"),
    arrow: str = Query("directed_lines"),
    icon: str = Query("template_matching"),
    outside_box_gate: bool = Query(False),
) -> JSONResponse:
    config = PipelineConfig(
        text_detector=text, box_detector=box, arrow_detector=arrow,
        icon_detector=icon, outside_box_gate=outside_box_gate,
    )
    suffix = os.path.splitext(file.filename or "upload.png")[1] or ".png"
    data = await file.read()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(data)
        tmp.close()
        result = pipeline.extract(tmp.name, config)
        annotated = pipeline.annotate(tmp.name, result)
        ok, png = cv2.imencode(".png", annotated)
        png_b64 = base64.b64encode(png.tobytes()).decode("ascii") if ok else ""
    finally:
        os.unlink(tmp.name)

    payload = ExtractResponse(
        structure=result,
        annotated_png_base64=png_b64,
        relationship_csv=_csv_string(result),
    )
    return JSONResponse(content=payload.model_dump())
