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
    # Polyline from tail to head, including any elbow bends; (x1, y1) and
    # (x2, y2) are its first and last points.
    points: List[List[int]] = Field(default_factory=list)
    # An arrowhead was found at the last point. When False (and not
    # bidirectional) no direction was detected and the point order is arbitrary.
    has_head: bool = False
    # Arrowheads at both ends.
    bidirectional: bool = False


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
    # Double-headed connector: source -> target and target -> source.
    bidirectional: bool = False
    # Always True now: every relationship is derived from a detected arrow.
    # The field is kept so downstream consumers that read the legacy schema
    # (which mixed detected + hardcoded edges) don't break.
    detected: bool = True


class PipelineConfig(BaseModel):
    """Selects which detector runs at each stage. Defaults are the Phase-2
    leaderboard champions; "none" skips a stage, which then contributes an
    empty list."""
    model_config = ConfigDict(extra="forbid")
    text_detector: Literal["paddleocr", "easyocr", "tesseract", "none"] = "paddleocr"
    box_detector: Literal["canny_contours", "hough", "yolo", "none"] = "canny_contours"
    arrow_detector: Literal["directed_lines", "hough_lines", "pixel_scan", "cnn", "none"] = "directed_lines"
    icon_detector: Literal["template_matching", "clip", "hsv", "none"] = "template_matching"
    # Day-3 finding: arrow segments lying on/inside a box bbox were inflating
    # recall via the snap-to-adjacency artifact. This gate rejects them. Off by
    # default: with directed_lines it lowers relationship F1 on the synthetic
    # benchmark (0.993 -> 0.905) and both draw.io sets, and makes no
    # difference on the real diagram (border segments map both endpoints to
    # the same box and are discarded anyway).
    outside_box_gate: bool = False
    # Endpoint snap radius + ray/line intersection fallback for ends that stop
    # in whitespace. Connector ends are now extended to their arrowhead tips and
    # branches are traced back to their trunk, so real ends sit at their box;
    # a long radius mostly lets leftover border lines attach to a box. 60 px
    # replaced 160 when the real draw.io diagram was fixed (2026-10-05).
    rel_max_dist: float = 60.0
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
