"""Feature engineering and preprocessing pipeline for the Telco churn dataset."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from churn.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES

# ---------------------------------------------------------------------------
# Feature-engineering constants
# ---------------------------------------------------------------------------

TENURE_BINS: list[int] = [0, 12, 24, 48, 60, 72]
TENURE_LABELS: list[str] = ["0-12", "12-24", "24-48", "48-60", "60-72"]

# Service columns for num_services: each "Yes" adds 1.
# "No phone service" and "No internet service" are NOT "Yes" — they count as 0.
SERVICE_COLS: list[str] = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

ENGINEERED_FEATURES: list[str] = [
    "tenure_bucket",
    "num_services",
    "avg_monthly_spend",
    "spend_growth_ratio",
]

# Columns routed through the numeric branch of the ColumnTransformer:
# the three original numerics plus the three numeric engineered features.
CT_NUMERIC: list[str] = NUMERIC_FEATURES + [
    "num_services",
    "avg_monthly_spend",
    "spend_growth_ratio",
]

# Columns routed through the categorical branch: original 16 plus tenure_bucket.
CT_CATEGORICAL: list[str] = CATEGORICAL_FEATURES + ["tenure_bucket"]


# ---------------------------------------------------------------------------
# Custom transformer
# ---------------------------------------------------------------------------


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """Stateless transformer that appends four domain-driven features.

    Engineered features and churn rationale
    ----------------------------------------
    tenure_bucket
        Bins raw tenure (months) into five ordered string labels:
        "0-12", "12-24", "24-48", "48-60", "60-72" (right-closed, leftmost
        bin is closed on both sides via include_lowest=True so tenure=0 maps
        to "0-12"). Churn risk is highly non-linear in tenure: very new
        customers and very long-tenured ones behave fundamentally differently.
        A categorical bucket lets tree models split cleanly on segment
        boundaries and gives the linear baseline piecewise-constant encoding.

    num_services
        Count of active services: number of "Yes" values across [PhoneService,
        MultipleLines, OnlineSecurity, OnlineBackup, DeviceProtection,
        TechSupport, StreamingTV, StreamingMovies], PLUS 1 if InternetService
        is not "No". "No phone service" and "No internet service" values are
        not "Yes" and count as 0. Customers with more services are more
        entrenched and face higher switching costs — a well-documented
        retention signal.

    avg_monthly_spend
        TotalCharges / tenure for tenure > 0; MonthlyCharges for tenure == 0
        (brand-new customers whose TotalCharges is structurally zero, not
        missing). The historical average spend can diverge from the current
        MonthlyCharges when price changes or service changes occurred.

    spend_growth_ratio
        MonthlyCharges / avg_monthly_spend. A ratio > 1.0 means the current
        bill exceeds the historical average, signalling a price increase — a
        known churn driver. Set to 1.0 when avg_monthly_spend == 0 to guard
        against division by zero (covers the edge case of tenure == 0 with
        MonthlyCharges == 0).

    Note on tree models vs. domain features
    ----------------------------------------
    Tree-based models can capture tenure non-linearity and service counts
    implicitly through splits. These features are included to (a) provide
    explicit domain signal that helps the linear baseline and (b) make
    the signal interpretable upfront in SHAP/importance analyses. The real
    contribution of each feature will be assessed in the next step.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "ChurnFeatureEngineer":
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add engineered features; does NOT mutate the input DataFrame."""
        X = X.copy()

        # tenure_bucket — ordered string label per tenure segment.
        # include_lowest=True makes the first interval [0, 12] (not (0, 12])
        # so that tenure == 0 lands in "0-12" rather than becoming NaN.
        X["tenure_bucket"] = pd.cut(
            X["tenure"],
            bins=TENURE_BINS,
            labels=TENURE_LABELS,
            include_lowest=True,
            right=True,
        ).astype(str)

        # num_services — count active add-ons + internet connectivity
        yes_count = (X[SERVICE_COLS] == "Yes").sum(axis=1)
        internet_active = (X["InternetService"] != "No").astype(int)
        X["num_services"] = (yes_count + internet_active).astype(int)

        # avg_monthly_spend — historical per-month rate vs current charge
        X["avg_monthly_spend"] = np.where(
            X["tenure"].astype(float) > 0,
            X["TotalCharges"].astype(float) / X["tenure"].astype(float),
            X["MonthlyCharges"].astype(float),
        )

        # spend_growth_ratio — current bill relative to historical average
        X["spend_growth_ratio"] = np.where(
            X["avg_monthly_spend"] != 0.0,
            X["MonthlyCharges"].astype(float) / X["avg_monthly_spend"],
            1.0,
        )

        return X

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Return feature names: original input columns plus the four engineered names."""
        check_is_fitted(self, "feature_names_in_")
        if input_features is None:
            input_features = self.feature_names_in_
        return np.array(list(input_features) + ENGINEERED_FEATURES, dtype=object)


# ---------------------------------------------------------------------------
# Preprocessing pipeline
# ---------------------------------------------------------------------------


def build_preprocessor() -> Pipeline:
    """Return an unfitted sklearn Pipeline:

        Pipeline([
            ("fe", ChurnFeatureEngineer()),
            ("ct", ColumnTransformer([
                ("num", SimpleImputer(median) → StandardScaler, 6 cols),
                ("cat", OneHotEncoder(handle_unknown="ignore"), 17 cols),
            ])),
        ])

    Numeric branch (6 columns = 3 original + 3 engineered):
        SimpleImputer(strategy="median") → StandardScaler().
        The imputer is a serving-time safety net even though training data has
        no nulls. StandardScaler is needed for the linear baseline and
        harmless for trees.

    Categorical branch (17 columns = 16 original + tenure_bucket):
        OneHotEncoder(handle_unknown="ignore", sparse_output=False).
        handle_unknown="ignore" is required so categories unseen during
        training produce zero vectors at inference rather than raising.
    """
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, CT_NUMERIC),
            ("cat", categorical_transformer, CT_CATEGORICAL),
        ],
        remainder="drop",
    )

    return Pipeline([
        ("fe", ChurnFeatureEngineer()),
        ("ct", column_transformer),
    ])
