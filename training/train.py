import os

import mlflow
import mlflow.xgboost
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from xgboost import XGBClassifier

from training.preprocess import FEATURE_COLUMNS, get_train_test_data

EXPERIMENT_NAME = "customer-churn"
MODEL_NAME = "CustomerChurnModel"
DEFAULT_TRACKING_URI = "file:./mlruns"
DEFAULT_SERVING_URI = f"models:/{MODEL_NAME}/Production"


def get_tracking_uri() -> str:
    return os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)


def train() -> None:
    X_train, X_test, y_train, y_test = get_train_test_data()

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    mlflow.set_tracking_uri(get_tracking_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        mlflow.log_params(
            {
                "n_estimators": model.n_estimators,
                "max_depth": model.max_depth,
                "learning_rate": model.learning_rate,
                "subsample": model.subsample,
                "colsample_bytree": model.colsample_bytree,
                "features": ",".join(FEATURE_COLUMNS),
                "experiment_name": EXPERIMENT_NAME,
                "model_name": MODEL_NAME,
            }
        )

        model.fit(X_train, y_train)

        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        loss = log_loss(y_test, y_pred_proba)

        mlflow.log_metrics(
            {
                "accuracy": float(accuracy),
                "roc_auc": float(roc_auc),
                "log_loss": float(loss),
            }
        )

        # Log model with schema information so MLflow stores input/output schema.
        input_example = X_train.iloc[:5]
        signature = infer_signature(X_train, model.predict_proba(X_train)[:, 1])

        model_info = mlflow.xgboost.log_model(
            xgb_model=model.get_booster(),
            artifact_path="model",
            signature=signature,
            input_example=input_example,
        )

        print(f"Model logged to MLflow with run URI: {model_info.model_uri}")

        registered_model_uri = None
        try:
            registered_model = mlflow.register_model(
                model_uri=model_info.model_uri,
                name=MODEL_NAME,
            )
            registered_model_uri = f"models:/{registered_model.name}/{registered_model.version}"
            print(
                f"Registered model '{registered_model.name}' as version {registered_model.version} "
                "in the MLflow Model Registry."
            )
            print(f"Model URI for this version: {registered_model_uri}")
        except MlflowException as exc:
            print(
                "Model registration skipped or failed. "
                "The MLflow Model Registry may not be available for the current tracking URI."
            )
            print(f"Error details: {exc}")

        serving_uri = os.getenv("CHURN_MODEL_URI", DEFAULT_SERVING_URI)
        print(f"Suggested serving URI (CHURN_MODEL_URI) for the churn model: {serving_uri}")
        print(
            "If you promote the registered model to a stage (for example, 'Production'), "
            f"you can use a stage-based URI such as 'models:/{MODEL_NAME}/Production'."
        )

        print(
            f"Test accuracy: {accuracy:.3f}, ROC AUC: {roc_auc:.3f}, log loss: {loss:.3f}"
        )


if __name__ == "__main__":
    train()