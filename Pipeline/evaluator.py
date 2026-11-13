from sklearn.metrics import (
    r2_score, mean_squared_error,
    accuracy_score, f1_score, roc_auc_score
)
import numpy as np


def evaluate_models(trained_models, X_test, y_test, problem_type):
    """
    Evaluate all trained models and return their performance metrics.
    """

    results = []

    for name, info in trained_models.items():
        pipeline = info["pipeline"]
        train_time = info["train_time"]

        # Prediction
        y_pred = pipeline.predict(X_test)

        metrics = {
            "model": name,
            "train_time": train_time,
            "tuned": info.get("tuned", False),
            "best_params": str(info.get("best_params")) if info.get("best_params") else None,
        }

        # ---------------- Regression Metrics ----------------
        if problem_type == "regression":
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            metrics.update({
                "RMSE": rmse,
                "R2": r2
            })

        # ---------------- Classification Metrics ----------------
        else:
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")

            metrics.update({
                "Accuracy": acc,
                "F1": f1
            })

            # ROC-AUC (only for binary classification)
            try:
                if len(np.unique(y_test)) == 2:
                    y_proba = pipeline.predict_proba(X_test)[:, 1]
                    auc = roc_auc_score(y_test, y_proba)
                    metrics["ROC_AUC"] = auc
            except:
                metrics["ROC_AUC"] = None

        results.append(metrics)

    return results
