"""
Generate 14 synthetic public-reference architecture diagrams for the
DiagraMine 15-diagram benchmark, plus emit ground_truth.json.

These diagrams are deliberately drawn in a style similar to the existing
test image (white background, rectangular boxes with text labels, mix of
solid and dashed arrows between boxes) so the existing pipeline has a fair
chance of detecting them. They mimic public reference patterns from
AWS Well-Architected, Kubernetes patterns, and microservices.io — they are
NOT derived from any proprietary diagram.

The 15th diagram is the pre-existing search_interview_test.png (the original
test diagram for the project), included so we can compare its score with
vs without `_known_connections()`.

Run:  python data/eval/generate_diagrams.py
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


# ---------------------------------------------------------------------------
# Diagram spec
# ---------------------------------------------------------------------------

Box = Tuple[float, float, float, float, str]   # x, y, w, h, label  (figure coords)
Arrow = Tuple[str, str, str]                   # src_label, tgt_label, "solid"|"dashed"
DiagramSpec = Dict[str, object]


def _box_center(box: Box) -> Tuple[float, float]:
    x, y, w, h, _ = box
    return x + w / 2, y + h / 2


def _exit_fraction(box: Box, dx: float, dy: float) -> float:
    """Fraction of the centre-to-centre vector (dx, dy) at which a ray from
    this box's centre crosses the box's border."""
    _, _, w, h, _ = box
    fx = (w / 2) / abs(dx) if dx else float("inf")
    fy = (h / 2) / abs(dy) if dy else float("inf")
    return min(fx, fy)


def _box_edge(src: Box, tgt: Box) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Compute arrow start/end on the box borders along the line between the
    two box centres, so connectors run through whitespace and never cross
    a label (arrows used to be aimed centre to centre with an 18 pt shrink,
    so the near-horizontal ones ran through the labels)."""
    sx, sy = _box_center(src)
    tx, ty = _box_center(tgt)
    dx, dy = tx - sx, ty - sy
    a = _exit_fraction(src, dx, dy)
    b = _exit_fraction(tgt, dx, dy)
    return (sx + a * dx, sy + a * dy), (tx - b * dx, ty - b * dy)


# Gap in points between a box border and the arrow's tail / head tip.
ARROW_SHRINK = 2


def build_figure(spec: DiagramSpec):
    """Draw a spec onto a new figure and return (fig, ax). Shared by
    render_diagram and derive_ground_truth_boxes so the answer key is always
    computed from exactly the figure that was saved."""
    boxes: List[Box] = spec["boxes"]
    arrows: List[Arrow] = spec["arrows"]
    canvas: Tuple[float, float] = spec.get("canvas", (12.0, 8.0))

    fig, ax = plt.subplots(figsize=canvas, dpi=120)
    ax.set_xlim(0, canvas[0])
    ax.set_ylim(0, canvas[1])
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    by_label: Dict[str, Box] = {}
    for box in boxes:
        x, y, w, h, label = box
        rect = Rectangle((x, y), w, h, linewidth=1.8,
                         edgecolor="black", facecolor="white")
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=11, fontweight="bold", color="black")
        by_label[label] = box

    for src_label, tgt_label, style in arrows:
        if src_label not in by_label or tgt_label not in by_label:
            continue
        (sx, sy), (tx, ty) = _box_edge(by_label[src_label], by_label[tgt_label])
        ls = "--" if style == "dashed" else "-"
        arrow = FancyArrowPatch(
            (sx, sy), (tx, ty),
            arrowstyle="-|>", mutation_scale=18,
            linewidth=1.6, linestyle=ls, color="black",
            shrinkA=ARROW_SHRINK, shrinkB=ARROW_SHRINK,
        )
        ax.add_patch(arrow)
    return fig, ax


def render_diagram(spec: DiagramSpec, out_path: str) -> None:
    fig, _ = build_figure(spec)
    fig.savefig(out_path, dpi=120, bbox_inches="tight",
                facecolor="white", pad_inches=0.3)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 14 synthetic public-reference architecture specs
# ---------------------------------------------------------------------------

def diagram_01_three_tier() -> DiagramSpec:
    return {
        "name": "3-tier web app",
        # 11 wide: at 10 the App Server box (x 8.0-10.4) ran off the canvas
        # and was saved with no right-hand border.
        "canvas": (11, 7),
        "boxes": [
            (1.0, 5.0, 2.4, 1.2, "Browser"),
            (4.5, 5.0, 2.4, 1.2, "Web Server"),
            (8.0, 5.0, 2.4, 1.2, "App Server"),
            (4.5, 2.0, 2.4, 1.2, "Database"),
        ],
        "arrows": [
            ("Browser", "Web Server", "dashed"),
            ("Web Server", "App Server", "dashed"),
            ("App Server", "Database", "dashed"),
        ],
    }


def diagram_02_microservices_gateway() -> DiagramSpec:
    return {
        "name": "Microservices with API gateway",
        "canvas": (12, 8),
        "boxes": [
            (4.7, 6.5, 2.4, 1.0, "Client App"),
            (4.7, 4.7, 2.4, 1.0, "API Gateway"),
            (0.8, 2.5, 2.4, 1.0, "User Svc"),
            (4.7, 2.5, 2.4, 1.0, "Order Svc"),
            (8.6, 2.5, 2.4, 1.0, "Payment Svc"),
            (4.7, 0.4, 2.4, 1.0, "Orders DB"),
        ],
        "arrows": [
            ("Client App", "API Gateway", "dashed"),
            ("API Gateway", "User Svc", "dashed"),
            ("API Gateway", "Order Svc", "dashed"),
            ("API Gateway", "Payment Svc", "dashed"),
            ("Order Svc", "Orders DB", "dashed"),
        ],
    }


def diagram_03_pubsub() -> DiagramSpec:
    return {
        "name": "Publish/subscribe messaging",
        "canvas": (12, 7),
        "boxes": [
            (0.8, 4.7, 2.4, 1.0, "Producer A"),
            (0.8, 2.5, 2.4, 1.0, "Producer B"),
            (4.8, 3.6, 2.4, 1.0, "Message Broker"),
            (8.8, 5.3, 2.4, 1.0, "Consumer A"),
            (8.8, 3.6, 2.4, 1.0, "Consumer B"),
            (8.8, 1.9, 2.4, 1.0, "Consumer C"),
        ],
        "arrows": [
            ("Producer A", "Message Broker", "dashed"),
            ("Producer B", "Message Broker", "dashed"),
            ("Message Broker", "Consumer A", "dashed"),
            ("Message Broker", "Consumer B", "dashed"),
            ("Message Broker", "Consumer C", "dashed"),
        ],
    }


def diagram_04_serverless_etl() -> DiagramSpec:
    return {
        "name": "AWS Lambda serverless ETL",
        "canvas": (13, 6),
        "boxes": [
            (0.6, 2.4, 2.4, 1.2, "S3 Source"),
            (3.8, 2.4, 2.4, 1.2, "Lambda Func"),
            (7.0, 2.4, 2.4, 1.2, "Glue Job"),
            (10.2, 2.4, 2.4, 1.2, "Redshift"),
        ],
        "arrows": [
            ("S3 Source", "Lambda Func", "dashed"),
            ("Lambda Func", "Glue Job", "dashed"),
            ("Glue Job", "Redshift", "dashed"),
        ],
    }


def diagram_05_kubernetes() -> DiagramSpec:
    return {
        "name": "Kubernetes deployment/service/pod",
        "canvas": (12, 7),
        "boxes": [
            (4.7, 5.5, 2.4, 1.0, "Ingress"),
            (4.7, 3.6, 2.4, 1.0, "Service"),
            (0.8, 1.5, 2.4, 1.0, "Pod A"),
            (4.7, 1.5, 2.4, 1.0, "Pod B"),
            (8.6, 1.5, 2.4, 1.0, "Pod C"),
        ],
        "arrows": [
            ("Ingress", "Service", "dashed"),
            ("Service", "Pod A", "dashed"),
            ("Service", "Pod B", "dashed"),
            ("Service", "Pod C", "dashed"),
        ],
    }


def diagram_06_master_replica() -> DiagramSpec:
    return {
        "name": "Database master/replica",
        "canvas": (11, 7),
        "boxes": [
            (4.0, 5.3, 2.4, 1.0, "App"),
            (4.0, 3.0, 2.4, 1.2, "Master DB"),
            (0.5, 0.7, 2.4, 1.2, "Replica 1"),
            (4.0, 0.7, 2.4, 1.2, "Replica 2"),
            (7.5, 0.7, 2.4, 1.2, "Replica 3"),
        ],
        "arrows": [
            ("App", "Master DB", "dashed"),
            ("Master DB", "Replica 1", "dashed"),
            ("Master DB", "Replica 2", "dashed"),
            ("Master DB", "Replica 3", "dashed"),
        ],
    }


def diagram_07_load_balancer() -> DiagramSpec:
    return {
        "name": "Load balancer + web servers",
        "canvas": (11, 6),
        "boxes": [
            (4.0, 4.3, 2.4, 1.0, "Load Balancer"),
            (0.5, 1.8, 2.4, 1.2, "Web 1"),
            (4.0, 1.8, 2.4, 1.2, "Web 2"),
            (7.5, 1.8, 2.4, 1.2, "Web 3"),
        ],
        "arrows": [
            ("Load Balancer", "Web 1", "dashed"),
            ("Load Balancer", "Web 2", "dashed"),
            ("Load Balancer", "Web 3", "dashed"),
        ],
    }


def diagram_08_cache_aside() -> DiagramSpec:
    return {
        "name": "Cache-aside pattern",
        "canvas": (11, 6),
        "boxes": [
            (4.0, 4.3, 2.4, 1.0, "App"),
            (0.5, 1.8, 2.4, 1.2, "Cache"),
            (7.5, 1.8, 2.4, 1.2, "Database"),
        ],
        "arrows": [
            ("App", "Cache", "dashed"),
            ("App", "Database", "dashed"),
            ("Cache", "Database", "dashed"),
        ],
    }


def diagram_09_etl_pipeline() -> DiagramSpec:
    return {
        "name": "ETL pipeline source/transform/warehouse",
        "canvas": (13, 5),
        "boxes": [
            (0.5, 2.0, 2.4, 1.2, "Source DB"),
            (3.7, 2.0, 2.4, 1.2, "Extract"),
            (6.9, 2.0, 2.4, 1.2, "Transform"),
            (10.1, 2.0, 2.4, 1.2, "Warehouse"),
        ],
        "arrows": [
            ("Source DB", "Extract", "dashed"),
            ("Extract", "Transform", "dashed"),
            ("Transform", "Warehouse", "dashed"),
        ],
    }


def diagram_10_cqrs() -> DiagramSpec:
    return {
        "name": "CQRS pattern",
        "canvas": (12, 7),
        "boxes": [
            (4.7, 5.5, 2.4, 1.0, "Command API"),
            (0.7, 3.4, 2.4, 1.0, "Write DB"),
            (4.7, 3.4, 2.4, 1.0, "Event Bus"),
            (8.7, 3.4, 2.4, 1.0, "Read DB"),
            (4.7, 1.2, 2.4, 1.0, "Query API"),
        ],
        "arrows": [
            ("Command API", "Write DB", "dashed"),
            ("Command API", "Event Bus", "dashed"),
            ("Event Bus", "Read DB", "dashed"),
            ("Query API", "Read DB", "dashed"),
        ],
    }


def diagram_11_saga() -> DiagramSpec:
    return {
        "name": "Saga distributed transaction",
        "canvas": (12, 7),
        "boxes": [
            (4.7, 5.5, 2.4, 1.0, "Orchestrator"),
            (0.7, 3.0, 2.4, 1.0, "Order Step"),
            (4.7, 3.0, 2.4, 1.0, "Payment Step"),
            (8.7, 3.0, 2.4, 1.0, "Shipping Step"),
            # Bottom right, not under Payment Step: in that column the
            # Orchestrator -> Saga Log arrow ran straight through Payment Step.
            (8.7, 0.7, 2.4, 1.0, "Saga Log"),
        ],
        "arrows": [
            ("Orchestrator", "Order Step", "dashed"),
            ("Orchestrator", "Payment Step", "dashed"),
            ("Orchestrator", "Shipping Step", "dashed"),
            ("Orchestrator", "Saga Log", "dashed"),
        ],
    }


def diagram_12_circuit_breaker() -> DiagramSpec:
    return {
        "name": "Circuit breaker pattern",
        "canvas": (12, 6),
        "boxes": [
            (0.7, 2.5, 2.4, 1.2, "Caller"),
            (4.3, 2.5, 2.4, 1.2, "Circuit Breaker"),
            (8.0, 2.5, 2.4, 1.2, "Remote Svc"),
            (4.3, 0.4, 2.4, 1.0, "Fallback"),
        ],
        "arrows": [
            ("Caller", "Circuit Breaker", "dashed"),
            ("Circuit Breaker", "Remote Svc", "dashed"),
            ("Circuit Breaker", "Fallback", "dashed"),
        ],
    }


def diagram_13_bff() -> DiagramSpec:
    return {
        "name": "Backend for Frontend",
        "canvas": (12, 7),
        "boxes": [
            (0.7, 5.3, 2.4, 1.0, "Web Client"),
            (8.6, 5.3, 2.4, 1.0, "Mobile Client"),
            (0.7, 3.2, 2.4, 1.0, "Web BFF"),
            (8.6, 3.2, 2.4, 1.0, "Mobile BFF"),
            (4.7, 1.2, 2.4, 1.0, "Core Svc"),
        ],
        "arrows": [
            ("Web Client", "Web BFF", "dashed"),
            ("Mobile Client", "Mobile BFF", "dashed"),
            ("Web BFF", "Core Svc", "dashed"),
            ("Mobile BFF", "Core Svc", "dashed"),
        ],
    }


def diagram_14_hexagonal() -> DiagramSpec:
    return {
        "name": "Hexagonal architecture",
        "canvas": (12, 7),
        "boxes": [
            (0.6, 5.0, 2.4, 1.0, "HTTP Adapter"),
            (0.6, 1.6, 2.4, 1.0, "CLI Adapter"),
            (4.7, 3.3, 2.4, 1.0, "Domain Core"),
            (8.8, 5.0, 2.4, 1.0, "DB Adapter"),
            (8.8, 1.6, 2.4, 1.0, "MQ Adapter"),
        ],
        "arrows": [
            ("HTTP Adapter", "Domain Core", "dashed"),
            ("CLI Adapter", "Domain Core", "dashed"),
            ("Domain Core", "DB Adapter", "dashed"),
            ("Domain Core", "MQ Adapter", "dashed"),
        ],
    }


SPECS: List[DiagramSpec] = [
    diagram_01_three_tier(),
    diagram_02_microservices_gateway(),
    diagram_03_pubsub(),
    diagram_04_serverless_etl(),
    diagram_05_kubernetes(),
    diagram_06_master_replica(),
    diagram_07_load_balancer(),
    diagram_08_cache_aside(),
    diagram_09_etl_pipeline(),
    diagram_10_cqrs(),
    diagram_11_saga(),
    diagram_12_circuit_breaker(),
    diagram_13_bff(),
    diagram_14_hexagonal(),
]


# ---------------------------------------------------------------------------
# Generation + ground-truth emit
# ---------------------------------------------------------------------------

def main() -> None:
    root = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(root, "diagrams_15")
    os.makedirs(out_dir, exist_ok=True)

    ground_truth: Dict[str, Dict[str, object]] = {}

    for idx, spec in enumerate(SPECS, start=1):
        fname = f"diagram_{idx:02d}.png"
        out_path = os.path.join(out_dir, fname)
        render_diagram(spec, out_path)
        print(f"[gen] {fname:<30} ({spec['name']})")

        ground_truth[fname] = {
            "name": spec["name"],
            "source": "synthetic-public-reference",
            "components": [b[4] for b in spec["boxes"]],
            "arrows": [
                {"source": s, "target": t, "style": st}
                for (s, t, st) in spec["arrows"]
            ],
            "icons": [],
        }

    # 15th diagram: the real draw.io export. Its connectors were re-audited on
    # 2026-10-05 (three independent readings of zoomed crops, unanimous): the
    # first key, written from an earlier extraction, listed a connector that is
    # not drawn (Server Website -> Plant An App - AWS), gave two the wrong end
    # (the long elbow ends at ELSER Model, not the Elasticsearch Serverless
    # container; the Database connector ends at the Back-End frame, not the
    # Elastic Language Client inside it), missed ELSER Model -> Indices, and
    # recorded double-headed connectors as one-way and dashed ones as solid.
    ground_truth["search_interview_test.png"] = {
        "name": "Plant An App AWS (original test diagram)",
        "source": "interview-test-original",
        "components": [
            "Server Website",
            "Plant An App - AWS",
            "Back-End",
            "Search UI",
            "Elastic Language Client",
            "Database",
            "Elastic Connector for MS SQL",
            "DB",
            "Elasticsearch Serverless",
            "Indices",
            "ELSER Model",
        ],
        "arrows": [
            {"source": "Search UI",                    "target": "Elastic Language Client",      "style": "dashed", "bidirectional": True},
            {"source": "Back-End",                     "target": "Database",                     "style": "dashed", "bidirectional": True},
            {"source": "Elastic Language Client",      "target": "ELSER Model",                  "style": "dashed", "bidirectional": True},
            {"source": "Database",                     "target": "Elastic Connector for MS SQL", "style": "dashed"},
            {"source": "Elastic Connector for MS SQL", "target": "Indices",                      "style": "dashed"},
            {"source": "Elastic Connector for MS SQL", "target": "DB",                           "style": "dashed"},
            {"source": "Indices",                      "target": "ELSER Model",                  "style": "dashed"},
            {"source": "ELSER Model",                  "target": "Indices",                      "style": "dashed"},
        ],
        "icons": ["docker", "MS SQL"],
    }

    gt_path = os.path.join(os.path.dirname(root), "eval", "ground_truth.json")
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2, ensure_ascii=False)
    print(f"\n[gt] Ground truth -> {gt_path}")
    print(f"[gt] {len(ground_truth)} diagrams in benchmark.")


if __name__ == "__main__":
    main()
