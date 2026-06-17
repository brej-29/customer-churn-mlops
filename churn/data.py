from pathlib import Path

import pandas as pd

from churn.config import settings

TELCO_EXPECTED_SHAPE = (7043, 21)


def load_telco_raw(csv_path: Path | None = None) -> pd.DataFrame:
    """Load the raw Telco Customer Churn CSV and return it as a DataFrame.

    Raises FileNotFoundError with an actionable message if the file is absent.
    """
    path = Path(csv_path) if csv_path is not None else settings.telco_csv_path
    if not path.exists():
        raise FileNotFoundError(
            f"Telco CSV not found at '{path}'. "
            "Download it from https://www.kaggle.com/datasets/blastchar/telco-customer-churn "
            f"and place it at '{path}' (relative to the project root)."
        )
    return pd.read_csv(path)
