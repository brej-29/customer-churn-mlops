import json
from pathlib import Path

from monitoring.generate_drift_report import generate_drift_report
from training.preprocess import FEATURE_COLUMNS, generate_synthetic_churn_data


def test_generate_drift_report_small_sample(tmp_path):
    """Drift report should be generated successfully on a small sample.

    This is a CI-safe smoke test that validates both HTML and JSON outputs.
    """
    reference_df = generate_synthetic_churn_data(n_samples=50, random_state=1)[
        FEATURE_COLUMNS
    ]
    current_df = generate_synthetic_churn_data(n_samples=50, random_state=2)[
        FEATURE_COLUMNS
    ]

    html_path, json_path = generate_drift_report(
        reference_data=reference_df,
        current_data=current_df,
        output_dir=str(tmp_path),
    )

    html_file = Path(html_path)
    json_file = Path(json_path)

    assert html_file.is_file()
    assert json_file.is_file()

    # JSON should be valid and contain some top-level fields.
    content = json_file.read_text(encoding="utf-8")
    data = json.loads(content)
    assert isinstance(data, dict)
    assert data  # not empty
    # Evidently reports typically contain a \"metrics\" section in JSON output.
    assert "metrics" in data