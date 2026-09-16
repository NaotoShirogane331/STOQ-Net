"""Publication tables, semantic chart specifications, renderers, and paper text."""

from __future__ import annotations

import csv
import io
import json
import math
import struct
import zlib
from pathlib import Path
from typing import Any

import numpy as np

from .constants import EVIDENCE_METADATA, RACE_PV, REPORT_LEADS
from .scenario_artifacts import (
    DISCLOSURE,
    atomic_bytes,
    canonical_bytes,
    register_stage_outputs,
    sha256_file,
    write_json,
)
from .scenario_manifest import require_stage

FIGURE_KINDS = {
    "performance-heatmap": "site_lead_heatmap",
    "seed-distribution": "seed_distribution",
    "coverage-width": "coverage_width_scatter",
    "significance-forest": "signed_dm_forest",
    "expert-specialization": "state_expert_specialization",
    "gate-utilization": "gate_utilization",
    "site-skill": "per_site_skill",
}

_FONT_5X7 = {
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01111", "10000", "10000", "10111", "10001", "10001", "01111"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("11111", "00100", "00100", "00100", "00100", "00100", "11111"),
    "J": ("00111", "00010", "00010", "00010", "10010", "10010", "01100"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "Q": ("01110", "10001", "10001", "10001", "10101", "10010", "01101"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "10101", "01010"),
    "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    "6": ("01110", "10000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00001", "01110"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    "_": ("00000", "00000", "00000", "00000", "00000", "00000", "11111"),
    ".": ("00000", "00000", "00000", "00000", "00000", "00110", "00110"),
    ":": ("00000", "00110", "00110", "00000", "00110", "00110", "00000"),
    ";": ("00000", "00110", "00110", "00000", "00110", "00100", "01000"),
    "=": ("00000", "11111", "00000", "11111", "00000", "00000", "00000"),
    "/": ("00001", "00010", "00100", "01000", "10000", "00000", "00000"),
    "(": ("00010", "00100", "01000", "01000", "01000", "00100", "00010"),
    ")": ("01000", "00100", "00010", "00010", "00010", "00100", "01000"),
    "%": ("11001", "11010", "00100", "01000", "10110", "00110", "00000"),
    "+": ("00000", "00100", "00100", "11111", "00100", "00100", "00000"),
}


def _rendered_text(spec: dict[str, Any]) -> list[str]:
    categories = ", ".join(spec.get("categories", [])[:8])
    return [
        spec["title"],
        f"X: {spec['labels']['x']}",
        f"Y: {spec['labels']['y']}",
        f"Categories: {categories}",
        spec["disclosure"],
    ]


def _draw_bitmap_text(
    pixels: np.ndarray, text: str, x: int, y: int, *, scale: int = 1
) -> None:
    cursor = x
    for character in text.upper():
        if character == " ":
            cursor += 4 * scale
            continue
        glyph = _FONT_5X7.get(character, ("11111", "10001", "00100", "00100", "00000", "00100", "00000"))
        for row, pattern in enumerate(glyph):
            for column, bit in enumerate(pattern):
                if bit == "1":
                    y1, y2 = y + row * scale, y + (row + 1) * scale
                    x1, x2 = cursor + column * scale, cursor + (column + 1) * scale
                    if y1 < pixels.shape[0] and x1 < pixels.shape[1]:
                        pixels[y1:min(y2, pixels.shape[0]), x1:min(x2, pixels.shape[1])] = 25
        cursor += 6 * scale


def _latex_value(value: Any) -> str:
    return str(value).translate(
        str.maketrans(
            {
                "\\": r"\textbackslash{}",
                "_": r"\_",
                "&": r"\&",
                "%": r"\%",
                "#": r"\#",
                "{": r"\{",
                "}": r"\}",
                "$": r"\$",
                "^": r"\textasciicircum{}",
                "~": r"\textasciitilde{}",
            }
        )
    )


def write_table(path: Path, rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    rows = rows or [{"note": DISCLOSURE}]
    fields = list(rows[0])
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    atomic_bytes(path, buffer.getvalue().encode("utf-8"))
    body = ["\\begin{tabular}{" + "l" * len(fields) + "}"]
    body.append(" & ".join(_latex_value(field) for field in fields) + r"\\")
    body.append(r"\hline")
    body.extend(
        " & ".join(
            _latex_value("see table note" if field == "note" else row.get(field, ""))
            for field in fields
        )
        + r"\\"
        for row in rows
    )
    body.extend((r"\end{tabular}", "% " + DISCLOSURE, ""))
    tex = path.with_suffix(".tex")
    atomic_bytes(tex, "\n".join(body).encode("utf-8"))
    return path, tex


def _scale(value: float, low: float, high: float, start: float, end: float) -> float:
    if not np.isfinite(value) or high <= low:
        return (start + end) / 2
    return start + (value - low) / (high - low) * (end - start)


def _geometry(spec: dict[str, Any]) -> list[dict[str, Any]]:
    coords = spec["coordinates"]
    kind = spec["kind"]
    geometry: list[dict[str, Any]] = []
    if kind in {"site_lead_heatmap", "state_expert_specialization"}:
        x_key, y_key = (
            ("lead", "site_id") if kind == "site_lead_heatmap" else ("expert", "state")
        )
        xs = list(dict.fromkeys(str(item[x_key]) for item in coords))
        ys = list(dict.fromkeys(str(item[y_key]) for item in coords))
        values = [float(item["value"]) for item in coords]
        low, high = min(values), max(values)
        for item in coords:
            x = xs.index(str(item[x_key]))
            y = ys.index(str(item[y_key]))
            intensity = int(round(_scale(float(item["value"]), low, high, 45, 225)))
            geometry.append(
                {
                    "type": "rect",
                    "x": 20 + x * 200 / max(1, len(xs)),
                    "y": 15 + y * 125 / max(1, len(ys)),
                    "w": 196 / max(1, len(xs)),
                    "h": 121 / max(1, len(ys)),
                    "color": [34, 80 + intensity // 3, intensity],
                }
            )
    elif kind in {"seed_distribution", "coverage_width_scatter"}:
        if kind == "seed_distribution":
            x_values = [float(item["seed"]) for item in coords]
            y_values = [float(item["value"]) for item in coords]
        else:
            x_values = [float(item["width"]) for item in coords]
            y_values = [float(item["coverage"]) for item in coords]
        for _item, x_value, y_value in zip(coords, x_values, y_values, strict=True):
            geometry.append(
                {
                    "type": "circle",
                    "x": _scale(x_value, min(x_values), max(x_values), 20, 220),
                    "y": _scale(y_value, min(y_values), max(y_values), 140, 15),
                    "r": 3 if kind == "seed_distribution" else 4,
                    "color": [31, 119, 180] if kind == "seed_distribution" else [214, 39, 40],
                }
            )
    elif kind == "signed_dm_forest":
        values = [float(item["statistic"]) for item in coords if np.isfinite(item["statistic"])]
        bound = max([abs(value) for value in values] or [1.0])
        zero = _scale(0, -bound, bound, 20, 220)
        geometry.append(
            {"type": "line", "x1": zero, "y1": 10, "x2": zero, "y2": 150, "color": [80, 80, 80]}
        )
        for index, item in enumerate(coords):
            value = float(item["statistic"])
            if not np.isfinite(value):
                continue
            y = 15 + index * 130 / max(1, len(coords) - 1)
            x = _scale(value, -bound, bound, 20, 220)
            geometry.extend(
                (
                    {"type": "line", "x1": zero, "y1": y, "x2": x, "y2": y, "color": [100, 100, 100]},
                    {"type": "circle", "x": x, "y": y, "r": 3, "color": [44, 160, 44] if value < 0 else [214, 39, 40]},
                )
            )
    elif kind in {"gate_utilization", "per_site_skill"}:
        values = [float(item["value"]) for item in coords]
        if kind == "gate_utilization":
            low, high, baseline = 0.0, max(values or [1.0]), 145.0
        else:
            bound = max([abs(value) for value in values] or [1.0])
            low, high = -bound, bound
            baseline = _scale(0, low, high, 145, 15)
            geometry.append(
                {"type": "line", "x1": 15, "y1": baseline, "x2": 225, "y2": baseline, "color": [80, 80, 80]}
            )
        width = 190 / max(1, len(values))
        for index, value in enumerate(values):
            y = _scale(value, low, high, 145, 15)
            geometry.append(
                {
                    "type": "rect",
                    "x": 20 + index * width,
                    "y": min(y, baseline),
                    "w": max(2, width - 2),
                    "h": max(1, abs(baseline - y)),
                    "color": [148, 103, 189] if kind == "gate_utilization" else ([44, 160, 44] if value >= 0 else [214, 39, 40]),
                }
            )
    return geometry


def _draw_line(pixels: np.ndarray, item: dict[str, Any]) -> None:
    x1, y1, x2, y2 = (int(round(item[key])) for key in ("x1", "y1", "x2", "y2"))
    steps = max(abs(x2 - x1), abs(y2 - y1), 1)
    xs = np.rint(np.linspace(x1, x2, steps + 1)).astype(int)
    ys = np.rint(np.linspace(y1, y2, steps + 1)).astype(int)
    valid = (0 <= xs) & (xs < pixels.shape[1]) & (0 <= ys) & (ys < pixels.shape[0])
    pixels[ys[valid], xs[valid]] = item["color"]


def _render_png(
    path: Path, geometry: list[dict[str, Any]], spec: dict[str, Any] | None = None
) -> None:
    width, height = 720, 440
    pixels = np.full((height, width, 3), 248, dtype=np.uint8)
    for item in geometry:
        item = dict(item)
        for key in ("x", "x1", "x2"):
            if key in item:
                item[key] = 100 + float(item[key]) * 2.1
        for key in ("y", "y1", "y2"):
            if key in item:
                item[key] = 60 + float(item[key]) * 1.5
        for key in ("w", "r"):
            if key in item:
                item[key] = float(item[key]) * 2.1
        if "h" in item:
            item["h"] = float(item["h"]) * 1.5
        if item["type"] == "rect":
            x1, y1 = max(0, int(item["x"])), max(0, int(item["y"]))
            x2 = min(width, int(math.ceil(item["x"] + item["w"])))
            y2 = min(height, int(math.ceil(item["y"] + item["h"])))
            pixels[y1:y2, x1:x2] = item["color"]
        elif item["type"] == "line":
            _draw_line(pixels, item)
        else:
            x1 = max(0, int(math.floor(item["x"] - item["r"])))
            x2 = min(width, int(math.ceil(item["x"] + item["r"] + 1)))
            y1 = max(0, int(math.floor(item["y"] - item["r"])))
            y2 = min(height, int(math.ceil(item["y"] + item["r"] + 1)))
            yy, xx = np.ogrid[y1:y2, x1:x2]
            mask = (xx - item["x"]) ** 2 + (yy - item["y"]) ** 2 <= item["r"] ** 2
            pixels[y1:y2, x1:x2][mask] = item["color"]
    text_items = _rendered_text(spec) if spec is not None else []
    if text_items:
        _draw_bitmap_text(pixels, text_items[0], 18, 14, scale=2)
        _draw_bitmap_text(pixels, text_items[1], 120, 315)
        _draw_bitmap_text(pixels, text_items[2], 18, 45)
        _draw_bitmap_text(pixels, text_items[3][:105], 18, 342)
        disclosure = text_items[4]
        _draw_bitmap_text(pixels, disclosure[:112], 18, 385)
        _draw_bitmap_text(pixels, disclosure[112:224], 18, 399)
    raw = b"".join(b"\x00" + pixels[row].tobytes() for row in range(height))

    def chunk(kind: bytes, value: bytes) -> bytes:
        return (
            struct.pack(">I", len(value))
            + kind
            + value
            + struct.pack(">I", zlib.crc32(kind + value) & 0xFFFFFFFF)
        )

    metadata = b"".join(
        chunk(
            b"tEXt",
            f"semantic_{index}\x00{text}".encode("latin-1", errors="replace"),
        )
        for index, text in enumerate(text_items)
    )
    content = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + metadata
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )
    atomic_bytes(path, content)


def _pdf_text(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _render_pdf(
    path: Path, geometry: list[dict[str, Any]], spec: dict[str, Any] | None = None
) -> None:
    commands = ["1 1 1 rg 0 0 720 440 re f"]
    for item in geometry:
        item = dict(item)
        for key in ("x", "x1", "x2"):
            if key in item:
                item[key] = 100 + float(item[key]) * 2.1
        for key in ("y", "y1", "y2"):
            if key in item:
                item[key] = 60 + float(item[key]) * 1.5
        for key in ("w", "r"):
            if key in item:
                item[key] = float(item[key]) * 2.1
        if "h" in item:
            item["h"] = float(item["h"]) * 1.5
        red, green, blue = [channel / 255 for channel in item["color"]]
        commands.append(f"{red:.6f} {green:.6f} {blue:.6f} rg")
        if item["type"] == "rect":
            commands.append(
                f"{item['x']:.3f} {440 - item['y'] - item['h']:.3f} {item['w']:.3f} {item['h']:.3f} re f"
            )
        elif item["type"] == "line":
            commands.append(
                f"{item['x1']:.3f} {440 - item['y1']:.3f} m {item['x2']:.3f} {440 - item['y2']:.3f} l S"
            )
        else:
            x, y, radius = item["x"], 440 - item["y"], item["r"]
            commands.append(f"{x - radius:.3f} {y - radius:.3f} {2 * radius:.3f} {2 * radius:.3f} re f")
    text_items = _rendered_text(spec) if spec is not None else []
    text_layout = ((18, 416, 14), (120, 115, 9), (18, 390, 9), (18, 86, 7), (18, 34, 5))
    commands.append("0 0 0 rg")
    for text, (x, y, size) in zip(text_items, text_layout, strict=True):
        commands.append(f"BT /F1 {size} Tf {x} {y} Td ({_pdf_text(text)}) Tj ET")
    stream = "\n".join(commands).encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 720 440] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for number, object_ in enumerate(objects, 1):
        offsets.append(len(output))
        output.extend(f"{number} 0 obj\n".encode())
        output.extend(object_)
        output.extend(b"\nendobj\n")
    xref = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    output.extend(b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets[1:]))
    output.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )
    atomic_bytes(path, bytes(output))


def _chart_specs(
    aggregate: dict[str, Any], evaluations: list[dict[str, Any]], source_hashes: dict[str, str]
) -> dict[str, dict[str, Any]]:
    selected = aggregate["reference_selection"]["selected_model"]
    selected_test = [
        row for row in evaluations if row["model"] == selected and row["split"] == "test"
    ]
    heatmap_groups: dict[tuple[str, int], list[float]] = {}
    for record in selected_test:
        for item in record["site_lead"]:
            heatmap_groups.setdefault((item["site_id"], int(item["lead"])), []).append(
                float(item["quantile_crps"])
            )
    heatmap = [
        {"site_id": site, "lead": lead, "value": float(np.mean(values))}
        for (site, lead), values in sorted(heatmap_groups.items())
    ]
    seed = [
        {"model": record["model"], "seed": record["seed"], "value": record["metrics"]["quantile_crps"]}
        for record in evaluations
        if record["model"] == RACE_PV.name and record["split"] == "test"
    ]
    coverage_groups: dict[str, list[dict[str, Any]]] = {}
    for record in evaluations:
        if record["model"] == RACE_PV.name and record["split"] == "test":
            for control, values in record["interval_metrics"].items():
                coverage_groups.setdefault(control, []).append(values)
    coverage = [
        {
            "coverage": float(np.mean([item["picp"] for item in values])),
            "width": float(np.mean([item["mean_width"] for item in values])),
            "control": control,
            "target": float(np.mean([item["coverage_target"] for item in values])),
            "model": RACE_PV.name,
            "split": "test",
        }
        for control, values in sorted(coverage_groups.items())
    ]
    forest = [
        {
            "model": item["model"],
            "statistic": item["statistic"],
            "adjusted_pvalue": item["bh_adjusted_pvalue"],
            "mean_loss_difference": item["mean_loss_difference"],
            "significant": item["bh_rejected"],
        }
        for item in aggregate["dm_bh"]
    ]
    expert_values: dict[tuple[str, int], list[float]] = {}
    gate_values: dict[int, list[float]] = {}
    for record in evaluations:
        if record["model"] != RACE_PV.name or record["split"] != "test":
            continue
        diagnostics = record.get("expert_diagnostics", {})
        for state, values in diagnostics.get("by_true_state", {}).items():
            for expert, value in enumerate(values.get("expert_nmae", [])):
                expert_values.setdefault((state, expert), []).append(float(value))
        for expert, value in enumerate(diagnostics.get("utilization", [])):
            gate_values.setdefault(expert, []).append(float(value))
    experts = [
        {"state": state, "expert": expert, "value": float(np.mean(values))}
        for (state, expert), values in sorted(expert_values.items())
    ]
    gates = [
        {"expert": expert, "value": float(np.mean(values)), "model": RACE_PV.name}
        for expert, values in sorted(gate_values.items())
    ]
    skills = [
        {
            "site_id": item["site_id"],
            "value": item["forecast_skill_nrmse"],
            "reference_model": item["reference_model"],
            "model": item["model"],
            "split": item["split"],
        }
        for item in aggregate["forecast_skill"]
        if item["model"] == RACE_PV.name and item["split"] == "test"
    ]
    coordinates = {
        "performance-heatmap": heatmap,
        "seed-distribution": seed,
        "coverage-width": coverage,
        "significance-forest": forest,
        "expert-specialization": experts,
        "gate-utilization": gates,
        "site-skill": skills,
    }
    labels = {
        "performance-heatmap": {"x": "lead", "y": "site", "value": "quantile CRPS"},
        "seed-distribution": {"x": "protocol seed", "y": "quantile CRPS"},
        "coverage-width": {"x": "mean interval width", "y": "empirical coverage"},
        "significance-forest": {"x": "signed DM statistic", "y": "comparison model"},
        "expert-specialization": {"x": "expert", "y": "oracle state", "value": "nMAE"},
        "gate-utilization": {"x": "expert", "y": "mean gate utilization"},
        "site-skill": {"x": "site", "y": "Forecast Skill (nRMSE)"},
    }
    specs = {}
    for name, values in coordinates.items():
        specs[name] = {
            **dict(EVIDENCE_METADATA),
            "disclosure": DISCLOSURE,
            "caption": f"{name.replace('-', ' ').title()} derived from persisted scenario artifacts; {DISCLOSURE}",
            "name": name,
            "kind": FIGURE_KINDS[name],
            "title": name.replace("-", " ").title(),
            "labels": labels[name],
            "coordinates": values,
            "series": sorted({str(item.get("model", item.get("control", "value"))) for item in values}),
            "categories": sorted({str(item.get("site_id", item.get("expert", item.get("model", "value")))) for item in values}),
            "values": [float(item.get("value", item.get("statistic", item.get("width", 0.0)))) for item in values],
            "intervals": [
                [float(item["bootstrap_ci_low"]), float(item["bootstrap_ci_high"])]
                for item in aggregate["rows"]
                if name == "seed-distribution" and item["model"] == RACE_PV.name and item["split"] == "test"
            ],
            "markers": [
                float(item.get("value", item.get("statistic", item.get("coverage", 0.0))))
                for item in values
            ],
            "source_hashes": source_hashes,
        }
    return specs


def export(root: Path) -> None:
    require_stage(root, "aggregate", ("aggregate.json", "selection.json", "evaluations.json"))
    aggregate = json.loads((root / "aggregate.json").read_text(encoding="utf-8"))
    evaluations = json.loads((root / "evaluations.json").read_text(encoding="utf-8"))["runs"]
    tables = root / "exports" / "tables"
    figures = root / "exports" / "figures"
    paper = root / "exports" / "paper"
    rows = aggregate["rows"]
    calibration_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for record in evaluations:
        for control, values in record["interval_metrics"].items():
            calibration_groups.setdefault(
                (record["model"], record["split"], control), []
            ).append(values)
    calibration_rows = [
        {
            "model": model,
            "split": split,
            "control": control,
            "picp": float(np.mean([item["picp"] for item in values])),
            "mean_width": float(np.mean([item["mean_width"] for item in values])),
            "coverage_target": float(np.mean([item["coverage_target"] for item in values])),
            "calibration_role": values[0]["calibration_role"],
            "n_seeds": len(values),
        }
        for (model, split, control), values in sorted(calibration_groups.items())
    ]
    site_groups: dict[tuple[str, str, str, int], list[dict[str, Any]]] = {}
    for record in evaluations:
        for item in record["site_lead"]:
            if int(item["lead"]) in REPORT_LEADS:
                site_groups.setdefault(
                    (record["model"], record["split"], item["site_id"], int(item["lead"])),
                    [],
                ).append(item)
    site_lead_rows = [
        {
            "model": model,
            "split": split,
            "site_id": site,
            "lead": lead,
            "mean_quantile_crps": float(np.mean([item["quantile_crps"] for item in values])),
            "mean_daylight_nrmse": float(
                np.mean([item["point"]["daylight"]["nrmse"] for item in values])
            ),
            "n_seeds": len(values),
        }
        for (model, split, site, lead), values in sorted(site_groups.items())
    ]
    table_rows = {
        "main-results": [
            row for row in rows if row["split"] == "test" and row["model"] not in {item["model"] for item in aggregate["sensitivities"]}
        ],
        "site-lead": site_lead_rows,
        "ablations": [row for row in rows if row["model"].startswith("a")],
        "calibration": calibration_rows,
        "significance": aggregate["dm_bh"],
        "transfer": aggregate["forecast_skill"],
    }
    outputs: list[Path] = []
    for name, values in table_rows.items():
        outputs.extend(
            write_table(
                tables / f"{name}.csv", [{**value, "note": DISCLOSURE} for value in values]
            )
        )
    source_hashes = {
        "aggregate.json": sha256_file(root / "aggregate.json"),
        "evaluations.json": sha256_file(root / "evaluations.json"),
        "selection.json": sha256_file(root / "selection.json"),
    }
    specs = _chart_specs(aggregate, evaluations, source_hashes)
    for name, spec in specs.items():
        geometry = _geometry(spec)
        geometry_digest = __import__("hashlib").sha256(canonical_bytes(geometry)).hexdigest()
        spec["renderer_contract"] = {
            "geometry_schema": "shared_primitives_v1",
            "png_geometry_sha256": geometry_digest,
            "pdf_geometry_sha256": geometry_digest,
            "primitive_count": len(geometry),
            "rendered_text_sha256": __import__("hashlib").sha256(
                canonical_bytes(_rendered_text(spec))
            ).hexdigest(),
        }
        spec_path = figures / f"{name}.json"
        png_path = figures / f"{name}.png"
        pdf_path = figures / f"{name}.pdf"
        write_json(spec_path, spec)
        _render_png(png_path, geometry, spec)
        _render_pdf(pdf_path, geometry, spec)
        outputs.extend((spec_path, png_path, pdf_path))

    best = min(
        (row for row in rows if row["split"] == "test"),
        key=lambda row: (row["mean_quantile_crps"], row["model"]),
    )
    race = next(row for row in rows if row["model"] == RACE_PV.name and row["split"] == "test")
    dm = aggregate["dm_bh"][0]
    skill = next(
        item
        for item in aggregate["forecast_skill"]
        if item["model"] == RACE_PV.name and item["split"] == "test"
    )
    selected_model = aggregate["reference_selection"]["selected_model"]
    heatmap_coordinate = specs["performance-heatmap"]["coordinates"][0]
    heatmap_row = next(
        row
        for row in site_lead_rows
        if row["model"] == selected_model
        and row["split"] == "test"
        and row["site_id"] == heatmap_coordinate["site_id"]
        and row["lead"] == heatmap_coordinate["lead"]
    )
    coverage_row = next(
        row
        for row in calibration_rows
        if row["model"] == RACE_PV.name
        and row["split"] == "test"
        and row["control"] == "state_lead"
    )
    claims = [
        (
            "selected_reference_site_lead_crps",
            "mean_quantile_crps",
            heatmap_row["mean_quantile_crps"],
            "site-lead",
            f"model={selected_model};split=test;site_id={heatmap_row['site_id']};lead={heatmap_row['lead']};field=mean_quantile_crps",
            "performance-heatmap",
            f"site_id={heatmap_coordinate['site_id']};lead={heatmap_coordinate['lead']};field=value",
            "value",
            [record["run_id"] for record in evaluations if record["model"] == selected_model and record["split"] == "test"],
        ),
        (
            "race_state_lead_coverage",
            "picp",
            coverage_row["picp"],
            "calibration",
            "model=race_pv;split=test;control=state_lead;field=picp",
            "coverage-width",
            "model=race_pv;split=test;control=state_lead;field=coverage",
            "coverage",
            [record["run_id"] for record in evaluations if record["model"] == RACE_PV.name and record["split"] == "test"],
        ),
        (
            "first_dm_statistic",
            "statistic",
            dm["statistic"],
            "significance",
            f"model={dm['model']};field=statistic",
            "significance-forest",
            f"model={dm['model']};field=statistic",
            "statistic",
            [record["run_id"] for record in evaluations if record["model"] in {dm["model"], dm["reference_model"]} and record["split"] == "test"],
        ),
        (
            "race_site_forecast_skill",
            "forecast_skill_nrmse",
            skill["forecast_skill_nrmse"],
            "transfer",
            f"model=race_pv;site_id={skill['site_id']};field=forecast_skill_nrmse",
            "site-skill",
            f"model=race_pv;split=test;site_id={skill['site_id']};field=value",
            "value",
            [record["run_id"] for record in evaluations if record["model"] in {RACE_PV.name, skill["reference_model"]} and record["split"] == "test"],
        ),
    ]
    claim_rows = []
    for (
        claim_id,
        field,
        value,
        table_name,
        coordinate,
        figure_name,
        figure_coordinate,
        figure_field,
        identities,
    ) in claims:
        csv_path = tables / f"{table_name}.csv"
        spec_path = figures / f"{figure_name}.json"
        png_path = figures / f"{figure_name}.png"
        pdf_path = figures / f"{figure_name}.pdf"
        claim_rows.append(
            {
                "claim_id": claim_id,
                "claim": f"material statistic {field}",
                "statistic_field": field,
                "value": str(float(value)),
                "artifact_path": str(csv_path.relative_to(root)),
                "artifact_sha256": sha256_file(csv_path),
                "csv_path": str(csv_path.relative_to(root)),
                "csv_sha256": sha256_file(csv_path),
                "csv_coordinate": coordinate,
                "figure_spec_path": str(spec_path.relative_to(root)),
                "figure_spec_sha256": sha256_file(spec_path),
                "figure_coordinate": figure_coordinate,
                "figure_statistic_field": figure_field,
                "png_path": str(png_path.relative_to(root)),
                "png_sha256": sha256_file(png_path),
                "pdf_path": str(pdf_path.relative_to(root)),
                "pdf_sha256": sha256_file(pdf_path),
                "source_hashes": json.dumps(source_hashes, sort_keys=True),
                "run_source_identities": ";".join(sorted(identities)),
                "coordinate": coordinate,
                "note": DISCLOSURE,
            }
        )
    outputs.extend(write_table(tables / "claim-evidence-map.csv", claim_rows))

    paper_values = {
        "best": best,
        "race": race,
        "dm": dm,
        "skill": skill,
    }
    results_path = paper / "results.md"
    atomic_bytes(
        results_path,
        (
            "# Results\n\n"
            f"From `aggregate.json`, the lowest held-out mean quantile CRPS is {best['mean_quantile_crps']:.6f} for `{best['model']}`. "
            f"RaCE-PV has selected-control coverage {race['selected_picp']:.6f} and mean width {race['selected_mean_width']:.6f}. "
            f"The first signed DM comparison is {dm['statistic']:.6f}; the per-site RaCE-PV Forecast Skill is {skill['forecast_skill_nrmse']:.6f}. "
            "Exact rows are in `exports/tables/main-results.csv`, `exports/tables/significance.csv`, and `exports/tables/transfer.csv`.\n\n"
            "All reported values are hypothetical and oracle-conditioned; they are not registered evidence.\n\n"
            f"{DISCLOSURE}\n"
        ).encode(),
    )
    captions_path = paper / "captions.md"
    atomic_bytes(
        captions_path,
        (
            "# Captions\n\n"
            + "\n".join(
                f"- `{name}` ({spec['kind']}): {len(spec['coordinates'])} persisted coordinates from {', '.join(spec['source_hashes'])}; {DISCLOSURE}"
                for name, spec in specs.items()
            )
            + f"\n\n{DISCLOSURE}\n"
        ).encode("utf-8"),
    )
    supplement_path = paper / "supplement.md"
    atomic_bytes(
        supplement_path,
        (
            "# Supplement\n\n"
            f"`aggregate.json` contains {len(rows)} model/split summaries and exactly {len(aggregate['sensitivities'])} validation sensitivity rows. "
            f"The selected model is `{aggregate['reference_selection']['selected_model']}` with {len(aggregate['dm_bh'])} DM/BH comparisons. "
            "See `exports/tables/ablations.csv` and `exports/tables/calibration.csv`.\n\n"
            f"{DISCLOSURE}\n"
        ).encode(),
    )
    chinese_path = paper / "explanation-zh.md"
    atomic_bytes(
        chinese_path,
        (
            "# 中文说明\n\n"
            f"`aggregate.json` 显示，测试集最低平均分位 CRPS 为 {best['mean_quantile_crps']:.6f}（`{best['model']}`）；"
            f"RaCE-PV 的所选校准覆盖率为 {race['selected_picp']:.6f}。具体证据坐标见 `exports/tables/claim-evidence-map.csv`，"
            "表格见 `exports/tables/`，图见 `exports/figures/`。这些结果是假设性、oracle 条件化的，不可作为注册证据。\n\n"
            f"{DISCLOSURE}\n"
        ).encode(),
    )
    replacement_path = paper / "replacement-guide.md"
    atomic_bytes(
        replacement_path,
        (
            "# Replacement guide\n\nReplace values by following each exact CSV coordinate in "
            "`exports/tables/claim-evidence-map.csv`; regenerate the linked PNG/PDF from its JSON chart spec and replace the references in Results and Supplement.\n\n"
            f"{DISCLOSURE}\n"
        ).encode(),
    )
    outputs.extend(
        (results_path, captions_path, supplement_path, chinese_path, replacement_path)
    )
    _ = paper_values
    register_stage_outputs(
        root,
        "export",
        outputs,
        source_paths=("aggregate.json", "evaluations.json", "selection.json"),
    )


__all__ = ["FIGURE_KINDS", "export", "write_table"]
