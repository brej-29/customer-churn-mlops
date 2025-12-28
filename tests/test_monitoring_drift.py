from pathlib import Path

from monitoring.generate_drift_report import generate_drift_report
from training.preprocess import FEATURE_COLUMNS, generate_synthetic_churn_data


def test_generate_drift_report_small_sample(tmp_path):
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

    assert Path(html_path).is_file()
    assert Path(json_path).is_file()