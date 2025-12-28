from pathlib import Path

import mlflow
import mlflow.xgboost
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from xgboost import XGBClassifier

from training.preprocess import FEATURE_COLUMNS, get_train_test_data

EXPERIMENT_NAME = "churn_xgboost"


def train():
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

        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        model_path = models_dir / "xgboost_churn_model.json"

        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
        )

        model.save_model(model_path)
        mlflow.log_artifact(str(model_path), artifact_path="model_files")

        print(f"Model saved to {model_path}")
        print(f"Test accuracy: {accuracy:.3f}, ROC AUC: {roc_auc:.3f}, log loss: {loss:.3f}")


if __name__ == "__main__":
    train()