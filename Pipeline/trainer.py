# Pipeline/trainer.py

import time
import numpy as np
from scipy import sparse

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import RandomizedSearchCV, ParameterGrid

from .models import (
    REGRESSION_MODELS,
    CLASSIFICATION_MODELS,
    REGRESSION_PARAM_GRIDS,
    CLASSIFICATION_PARAM_GRIDS,
)
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

def tune_pipeline(pipeline, param_grid, X_train, y_train, problem_type,
                  n_iter=10, cv=3, random_state=42):
    """
    Hyperparameter tuning with RandomizedSearchCV.
    Samples up to `n_iter` settings from `param_grid`, scores each one with
    `cv`-fold cross-validation on the training data only (the test set is
    never touched), then refits the best setting on the full training set.

    Scoring matches the ranking metric:
    - classification -> weighted F1
    - regression     -> RMSE
    """
    scoring = "f1_weighted" if problem_type == "classification" else "neg_root_mean_squared_error"
    n_candidates = len(ParameterGrid(param_grid))

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=min(n_iter, n_candidates),
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        random_state=random_state,
        error_score="raise",
    )
    search.fit(X_train, y_train)

    best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    return search.best_estimator_, best_params, search.best_score_


def train_models(X_train, y_train, problem_type, tune=True, n_iter=10, cv=3):
    """
    Trains all models for the given problem type.
    Assumes:
    - Labels are already encoded as 0..K-1 for classification.

    If `tune` is True, each model with a search space in Pipeline/models.py is
    tuned with RandomizedSearchCV (`n_iter` candidates, `cv` folds). If tuning
    fails for a model (e.g. too few samples in a class for the folds), it falls
    back to the default settings so the run never crashes.
    """

    preprocessor = build_preprocessor(X_train)
    trained_models = {}

    if problem_type == "regression":
        models = REGRESSION_MODELS
        param_grids = REGRESSION_PARAM_GRIDS
    else:
        models = CLASSIFICATION_MODELS
        param_grids = CLASSIFICATION_PARAM_GRIDS

    # For classification: number of classes
    n_classes = None
    if problem_type == "classification":
        n_classes = len(np.unique(y_train))

    for name, base_model in models.items():
        print(f"Training {name}...")

        model = clone(base_model)

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
            ("preprocessor", clone(preprocessor)),
            ("to_dense", FunctionTransformer(make_dense, accept_sparse=True)),
            ("model", model)
        ])

        # ==========================
        # Train Safely
        # ==========================
        param_grid = param_grids.get(name) if tune else None
        best_params, cv_score, tuned = None, None, False

        try:
            start_time = time.time()

            if param_grid:
                try:
                    pipeline, best_params, cv_score = tune_pipeline(
                        pipeline, param_grid, X_train, y_train, problem_type,
                        n_iter=n_iter, cv=cv
                    )
                    tuned = True
                except Exception as e:
                    print(f"⚠️ Tuning failed for {name} ({e}). Using default settings.")
                    pipeline.fit(X_train, y_train)
            else:
                pipeline.fit(X_train, y_train)

            train_time = time.time() - start_time

            trained_models[name] = {
                "pipeline": pipeline,
                "train_time": train_time,
                "tuned": tuned,
                "best_params": best_params,
                "cv_score": cv_score,
            }

        except Exception as e:
            print(f"❌ Failed training {name}: {e}")

    return trained_models
