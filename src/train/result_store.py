"""Append-only, idempotent CSV persistence for baseline evaluation rows."""

from __future__ import annotations

import csv
import numbers
from datetime import datetime, timezone
from uuid import uuid4
from pathlib import Path


FIELDS = [
    "run_id", "timestamp_utc",
    "dataset", "model", "n_classes", "mode", "protocol", "fold", "accuracy", "f1_macro",
    "sensitivity", "specificity", "precision", "auc",
    "published_reference_accuracy", "delta_vs_published", "within_band",
    "baseline_status", "notes",
]


def persist_baseline_result(row: dict, report_path: Path = Path("results/baseline_accuracy_report.csv")) -> None:
    """Append a distinct run; retrying an identical run_id is idempotent."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    accuracy = row.get("accuracy", "")
    macro_f1 = row.get("macro_f1", row.get("f1_macro", ""))
    reference = row.get("published_reference_accuracy", "")
    accuracy_pct = float(accuracy) * 100 if isinstance(accuracy, numbers.Real) and abs(float(accuracy)) <= 1 else accuracy
    f1_pct = float(macro_f1) * 100 if isinstance(macro_f1, numbers.Real) and abs(float(macro_f1)) <= 1 else macro_f1
    reference_pct = float(reference) * 100 if isinstance(reference, numbers.Real) and abs(float(reference)) <= 1 else reference
    delta_pp = (float(accuracy_pct) - float(reference_pct)
                if isinstance(accuracy_pct, numbers.Real) and isinstance(reference_pct, numbers.Real) else "")
    within_band = ("yes" if abs(delta_pp) <= 3 else "no") if delta_pp != "" else row.get("within_band", "not_assessed")
    output = {
        "run_id": row.get("run_id") or str(uuid4()),
        "timestamp_utc": row.get("timestamp_utc") or datetime.now(timezone.utc).isoformat(),
        "dataset": row.get("dataset", "unknown"), "model": row.get("model", "unknown"),
        "n_classes": int(row.get("n_classes", 4)), "mode": row.get("mode", "global"),
        "protocol": row.get("protocol", ""),
        "fold": row.get("fold", "unknown"), "accuracy": accuracy_pct, "f1_macro": f1_pct,
        "sensitivity": row.get("sensitivity", ""), "specificity": row.get("specificity", ""),
        "precision": row.get("precision", ""), "auc": row.get("auc", ""),
        "published_reference_accuracy": reference_pct, "delta_vs_published": delta_pp,
        "within_band": within_band, "baseline_status": row.get("baseline_status", "measured"),
        "notes": row.get("notes", ""),
    }
    rows: list[dict] = []
    columns = list(FIELDS)
    if report_path.is_file():
        with report_path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            columns = list(dict.fromkeys((reader.fieldnames or []) + FIELDS))
            rows = list(reader)
    key = str(output["run_id"])
    for index, existing in enumerate(rows):
        existing_key = existing.get("run_id", "")
        if existing_key == key:
            rows[index] = {**existing, **{name: str(value) for name, value in output.items()}}
            break
    else:
        rows.append({name: str(value) for name, value in output.items()})
    with report_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
