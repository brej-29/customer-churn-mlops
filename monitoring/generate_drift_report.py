from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Tuple

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from training.preprocess import FEATURE_COLUMNS, generate_synthetic_churn_data

DEFAULT_REFERENCE_DATA_PATH = os.getenv(
    "REFERENCE_DATA_PATH", "data/churn_reference.csv"
)
DEFAULT_LOG_DB_PATH = os.getenv("LOG_DB_PATH", "logs/predictions.db")
DEFAULT_DRIFT_WINDOW = int(os.getenv("DRIFT_WINDOW", "500"))
DEFAULT_REPORT_OUTPUT_DIR = os.getenv("REPORT_OUTPUT_DIR", "reports")


def load_reference_data(path: str = DEFAULT_REFERENCE_DATA_PATH) -> pd.DataFrame:
    """Load the reference dataset used during training.

    If the file does not exist yet, generate it from the same synthetic data
    pipeline used by training and save it to disk for future runs.
    """
    ref_path = Path(path)
    if ref_path.is_file():
        return pd.read_csv(ref_path)

    ref_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_churn_data(n_samples=1000, random_state=42)
    df.to_csv(ref_path, index=False)
    return df


def load_current_data(
    db_path: str = DEFAULT_LOG_DB_PATH, window: int = DEFAULT_DRIFT_WINDOW
) -> pd.DataFrame:
    """Load the most recent prediction inputs from the logging store.

    If the logging database or table is missing, fall back to an empty frame
    with the correct schema.
    """
    db_file = Path(db_path)
    if not db_file.is_file():
        # No logs yet; return an empty frame with the right columns.
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    conn = sqlite3.connect(db_file)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(
            """
            SELECT request_payload
            FROM prediction_logs
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(window),),
        )
        rows = cursor.fetchall()
    except sqlite3.Error:
        conn.close()
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    conn.close()

    if not rows:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    payloads = []
    for row in rows:
        try:
            payload = json.loads(row["request_payload"])
        except Exception:  # noqa: BLE001
            continue
        payloads.append(payload)

    if not payloads:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    df = pd.DataFrame(payloads)
    # Keep only known feature columns in a stable order.
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    for col in missing:
        df[col] = 0
    return df[FEATURE_COLUMNS]


def _save_html_and_json(result: object, html_path: Path, json_path: Path) -> None:
    """Persist an Evidently evaluation result to HTML and JSON with fallbacks.

    Newer Evidently versions expose save_html/save_json on the result object
    returned by Report.run(). Older or variant builds may expose these methods
    on the Report instance itself, or only provide json()/dict() helpers.
    """
    # HTML
    if hasattr(result, "save_html"):
        result.save_html(str(html_path))
    else:
        # Minimal fallback: embed JSON in a simple HTML shell if possible.
        try:
            if hasattr(result, "json"):
                html_path.write_text(
                    "<html><body><pre>"
                    + json.dumps(json.loads(result.json()), indent=2)
                    + "</pre></body></html>",
                    encoding="utf-8",
                )
            elif hasattr(result, "dict"):
                html_path.write_text(
                    "<html><body><pre>"
                    + json.dumps(result.dict(), indent=2)
                    + "</pre></body></html>",
                    encoding="utf-8",
                )
            else:
                html_path.write_text(
                    "<html><body><p>No HTML available.</p></body></html>",
                    encoding="utf-8",
                )
        except Exception:  # noqa: BLE001
            html_path.write_text(
                "<html><body><p>Failed to generate HTML.</p></body></html>",
                encoding="utf-8",
            )

    # JSON
    if hasattr(result, "save_json"):
        result.save_json(str(json_path))
    else:
        try:
            if hasattr(result, "json"):
                json_path.write_text(result.json(), encoding="utf-8")
            elif hasattr(result, "dict"):
                json_path.write_text(
                    json.dumps(result.dict(), indent=2),
                    encoding="utf-8",
                )
            else:
                json_path.write_text("{}", encoding="utf-8")
        except Exception:  # noqa: BLE001
            json_path.write_text("{}", encoding="utf-8")


def generate_drift_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    output_dir: str = DEFAULT_REPORT_OUTPUT_DIR,
) -> Tuple[str, str]:
    """Generate an Evidently drift report and save HTML and JSON artifacts.

    Uses Evidently v0.7+ style imports:
    - from evidently import Report
    - from evidently.presets import DataDriftPreset
    """
    report = Report(metrics=[DataDriftPreset()])

    # Evidently v0.7+ returns an evaluation result object from run().
    snapshot = report.run(
        reference_data=reference_data.reset_index(drop=True),
        current_data=current_data.reset_index(drop=True),
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    html_path = out_dir / "drift_report.html"
    json_path = out_dir / "drift_report.json"

    # Prefer the snapshot if available; fall back to the Report instance.
    target = snapshot if snapshot is not None else report
    _save_html_and_json(target, html_path=html_path, json_path=json_path)

    return str(html_path), str(json_path)


def main() -> None:
    """Entry point for `python -m monitoring.generate_drift_report`."""
    reference_df = load_reference_data()
    current_df = load_current_data()

    # Use only feature columns for drift analysis.
    reference_features = reference_df[FEATURE_COLUMNS]
    current_features = current_df[FEATURE_COLUMNS]

    html_path, json_path = generate_drift_report(
        reference_data=reference_features,
        current_data=current_features,
        output_dir=DEFAULT_REPORT_OUTPUT_DIR,
    )
    print(f"Drift report saved to: {html_path}")
    print(f"Drift JSON snapshot saved to: {json_path}")


if __name__ == "__main__":
    main()