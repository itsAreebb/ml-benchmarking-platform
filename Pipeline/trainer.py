# Pipeline/trainer.py

import time
import numpy as np
from scipy import sparse

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from .models import REGRESSION_MODELS, CLASSIFICATION_MODELS
from .preprocessing import build_preprocessor

# XGBoost (optional)
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# ------------------ Helpers ------------------

def make_dense(X):
    """Convert sparse matrices to dense arrays (some models require dense)."""
    if sparse.issparse(X):
        return X.toarray()
    return X


# ------------------ Trainer ------------------

def train_models(X_train, y_train, problem_type):
    """
    Trains all models for the given problem type.
    Assumes:
    - y is already numeric and encoded globally in dataset loader.
    - Labels are 0..K-1 for classification.
    """

    preprocessor = build_preprocessor(X_train)
    trained_models = {}

    if problem_type == "regression":
        models = REGRESSION_MODELS
    else:
        models = CLASSIFICATION_MODELS

    # For classification: number of classes
    n_classes = None
    if problem_type == "classification":
        n_classes = len(np.unique(y_train))

    for name, base_model in models.items():
        print(f"Training {name}...")

        model = base_model

        # ==========================
        # XGBoost Special Handling
        # ==========================
        if "XGBoost" in name:
            if not XGBOOST_AVAILABLE:
                print("❌ XGBoost not installed. Skipping.")
                continue

            if problem_type == "regression":
                model = XGBRegressor(
                    
                    objective="reg:squarederror",
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=4,
                    verbosity=0
                )

            else:  # classification
                if n_classes < 2:
                    print(f"Skipping {name}: only one class present.")
                    continue

                if n_classes == 2:
                    model = XGBClassifier(
                       
                        objective="binary:logistic",
                        eval_metric="logloss",
                        n_estimators=200,
                        learning_rate=0.05,
                        max_depth=4,
                        verbosity=0
                    )
                else:
                    model = XGBClassifier(
                        
                        objective="multi:softprob",
                        num_class=n_classes,
                        eval_metric="mlogloss",
                        n_estimators=200,
                        learning_rate=0.05,
                        max_depth=4,
                        verbosity=0
                    )

        # ==========================
        # Build Pipeline
        # ==========================
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("to_dense", FunctionTransformer(make_dense, accept_sparse=True)),
            ("model", model)
        ])

        # ==========================
        # Train Safely
        # ==========================
        try:
            start_time = time.time()
            pipeline.fit(X_train, y_train)
            train_time = time.time() - start_time

            trained_models[name] = {
                "pipeline": pipeline,
                "train_time": train_time
            }

        except Exception as e:
            print(f"❌ Failed training {name}: {e}")

    return trained_models
