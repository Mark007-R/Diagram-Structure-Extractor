"""Typed Pydantic v2 schemas shared across the DiagraMine pipeline.

Every detection stage and the orchestrator validate their output against these
models, which is what guarantees DiagraMine's headline reliability metric:
`ExtractionResult.model_dump()` is always schema-valid machine-parseable JSON,
even when a stage detects nothing (it returns an empty list, not prose). This
is the property Day-6 measures against Claude Vision (~13% schema-valid).
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class TextElement(BaseModel):
    model_config = ConfigDict(extra="ignore")
    text: str
    x: int
    y: int
    w: int
    h: int
    conf: float = 0.0
    parent_box: Optional[int] = None


class BoxElement(BaseModel):
    model_config = ConfigDict(extra="ignore")
    x: int
    y: int
    w: int
    h: int
    area: int = 0
    is_region: bool = False
    label: str = ""
    entity_type: str = "component"


class ArrowElement(BaseModel):
    model_config = ConfigDict(extra="ignore")
    x1: int
    y1: int
    x2: int
    y2: int
    direction: str = "unknown"
    line_style: str = "solid"


class IconElement(BaseModel):
    model_config = ConfigDict(extra="ignore")
    x: int
    y: int
    w: int
    h: int
    label: str = "icon"
    score: float = 0.0
    parent_box: Optional[int] = None


class Relationship(BaseModel):
    model_config = ConfigDict(extra="ignore")
    source: str
    target: str
    line_style: str = "solid"
    direction: str = "unknown"
    relationship: str = "connects_to"
    # Always True now: every relationship is derived from a detected arrow.
    # The field is kept so downstream consumers that read the legacy schema
    # (which mixed detected + hardcoded edges) don't break.
    detected: bool = True


class PipelineConfig(BaseModel):
    """Selects which detector runs at each stage. Box, arrow and icon defaults
    are the leaderboard champions; text defaults to EasyOCR although PaddleOCR
    is the bake-off champion. "none" skips a stage, which then contributes an
    empty list."""
    model_config = ConfigDict(extra="forbid")
    text_detector: Literal["easyocr", "paddleocr", "tesseract", "none"] = "easyocr"
    box_detector: Literal["canny_contours", "hough", "yolo", "none"] = "canny_contours"
    arrow_detector: Literal["directed_lines", "hough_lines", "pixel_scan", "cnn", "none"] = "directed_lines"
    icon_detector: Literal["template_matching", "clip", "hsv", "none"] = "template_matching"
    # Day-3 finding: arrow segments lying on/inside a box bbox were inflating
    # recall via the snap-to-adjacency artifact. This gate rejects them. Off by
    # default: with directed_lines it lowers relationship F1 on both the
    # benchmark and the real-world diagram (border segments map both endpoints
    # to the same box and are discarded anyway).
    outside_box_gate: bool = False
    # Day-5 arrow-mapping fix: endpoint snap radius + ray/line intersection
    # fallback for short segments stopping in whitespace. On the synthetic
    # benchmark the ray path never triggers (boxes are dense, nearest-box always
    # resolves within rel_max_dist) so it is a no-op there; it is kept ON as a
    # proven-correct general improvement for real diagrams with sparser layouts.
    rel_max_dist: float = 160.0
    rel_ray_intersection: bool = True
    rel_max_proj: float = 400.0
    graph_layout: Literal["kamada_kawai", "spring"] = "kamada_kawai"


class StageRuntimes(BaseModel):
    model_config = ConfigDict(extra="ignore")
    text: float = 0.0
    box: float = 0.0
    arrow: float = 0.0
    icon: float = 0.0
    graph: float = 0.0


class ExtractionResult(BaseModel):
    """Canonical machine-parseable output of the whole pipeline."""
    model_config = ConfigDict(extra="ignore")
    image: str
    width: int
    height: int
    detectors: dict = Field(default_factory=dict)
    texts: List[TextElement] = Field(default_factory=list)
    boxes: List[BoxElement] = Field(default_factory=list)
    regions: List[BoxElement] = Field(default_factory=list)
    arrows: List[ArrowElement] = Field(default_factory=list)
    icons: List[IconElement] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)
    runtimes: StageRuntimes = Field(default_factory=StageRuntimes)
    runtime_seconds: float = 0.0

    def summary(self) -> dict:
        return {
            "text_labels": len(self.texts),
            "boxes": len(self.boxes),
            "regions": len(self.regions),
            "arrows": len(self.arrows),
            "icons": len(self.icons),
            "relationships": len(self.relationships),
        }
