from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from churn.config import settings

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

TELCO_EXPECTED_SHAPE = (7043, 21)
TELCO_CLEAN_SHAPE = (7043, 20)

TARGET = "Churn"

NUMERIC_FEATURES: list[str] = ["tenure", "MonthlyCharges", "TotalCharges"]

# All feature columns after dropping customerID and the target (16 columns).
# Note: SeniorCitizen is already integer-encoded (0/1) in the raw CSV, unlike
# the other binary features which arrive as "Yes"/"No" strings. It is kept here
# as a categorical so the downstream ColumnTransformer handles it uniformly with
# the other binary indicators rather than treating it as a continuous numeric.
CATEGORICAL_FEATURES: list[str] = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

ALL_FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ---------------------------------------------------------------------------
# Raw loading
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------


def clean_telco(df: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic, row-wise cleaning to the raw Telco DataFrame.

    Steps (no learned statistics — safe to run on the full dataset pre-split):
    1. Coerce TotalCharges to numeric; verify that every NaN corresponds to
       tenure == 0, then fill with 0.0.  These are new customers whose total
       charges are structurally zero, not missing-at-random — imputation by
       mean/median would be wrong and belongs to a different class of missingness.
    2. Drop customerID (identifier; no predictive value; leakage/noise risk).
    3. Map Churn "Yes" -> 1 / "No" -> 0 and validate.
    4. Enforce dtypes: numeric features as float64, target as int, categoricals
       remain object/string for the downstream ColumnTransformer.
    5. Assert no nulls remain.
    """
    df = df.copy()

    # --- TotalCharges ---
    numeric_tc = pd.to_numeric(df["TotalCharges"], errors="coerce")
    blank_mask = numeric_tc.isna()
    bad_rows = df.loc[blank_mask & (df["tenure"] != 0)]
    if not bad_rows.empty:
        raise ValueError(
            f"Found {len(bad_rows)} row(s) where TotalCharges is blank but tenure != 0. "
            "The assumption that all blanks are tenure-0 new customers is violated. "
            f"Row indices: {bad_rows.index.tolist()}"
        )
    # Fill tenure-0 blanks with 0.0 (structurally correct, not imputation).
    numeric_tc = numeric_tc.fillna(0.0)
    df["TotalCharges"] = numeric_tc

    # --- Drop identifier ---
    df = df.drop(columns=["customerID"])

    # --- Target encoding ---
    unexpected = set(df[TARGET].unique()) - {"Yes", "No"}
    if unexpected:
        raise ValueError(
            f"Unexpected values in {TARGET} column: {unexpected}. Expected only 'Yes' and 'No'."
        )
    df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0}).astype(int)

    # --- Dtype enforcement ---
    for col in NUMERIC_FEATURES:
        df[col] = df[col].astype("float64")
    # Categoricals stay as object (string); SeniorCitizen becomes string too so
    # the ColumnTransformer can treat it uniformly with the other Yes/No columns.
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str)

    # --- Final invariant ---
    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if not cols_with_nulls.empty:
        raise AssertionError(
            f"Cleaning left null values in columns: {cols_with_nulls.to_dict()}"
        )

    return df


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------


def load_clean_telco(
    csv_path: Path | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """Load raw CSV, clean it, and optionally persist to data/processed/.

    Parameters
    ----------
    csv_path:
        Override for the raw CSV path (defaults to settings.telco_csv_path).
    save:
        If True (default), write the cleaned frame to
        settings.data_processed_dir / 'telco_clean.csv' for inspection.
    """
    df = load_telco_raw(csv_path=csv_path)
    df = clean_telco(df)
    if save:
        out = settings.data_processed_dir / "telco_clean.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
    return df


# ---------------------------------------------------------------------------
# Canonical split — single source of truth for every downstream step
# ---------------------------------------------------------------------------


def get_splits(
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return the canonical stratified train/test split for the Telco dataset.

    Every downstream step (feature engineering, model training, evaluation)
    must obtain its split from this function so that train/test indices are
    identical across all steps.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    df = load_clean_telco(save=False)
    X = df[ALL_FEATURES]
    y = df[TARGET]
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=settings.random_seed,
    )
