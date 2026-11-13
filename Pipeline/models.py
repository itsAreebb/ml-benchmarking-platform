from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
    AdaBoostRegressor,
    AdaBoostClassifier
)

try:
    from xgboost import XGBRegressor, XGBClassifier
    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False


# ---------------- REGRESSION MODELS ----------------
REGRESSION_MODELS = {
    "LinearRegression": LinearRegression(),
    "DecisionTreeRegressor": DecisionTreeRegressor(random_state=42),
    "RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=42),
    "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
    "AdaBoostRegressor": AdaBoostRegressor(random_state=42)
}
if XGBOOST_AVAILABLE:
    REGRESSION_MODELS["XGBoostRegressor"] = XGBRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=4, verbosity=0
    )

# ---------------- CLASSIFICATION MODELS ----------------
CLASSIFICATION_MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "NaiveBayes": GaussianNB(),
    "RandomForestClassifier": RandomForestClassifier(n_estimators=100, random_state=42),
    "GradientBoostingClassifier": GradientBoostingClassifier(random_state=42),
    "AdaBoostClassifier": AdaBoostClassifier(random_state=42)
}

if XGBOOST_AVAILABLE:
    CLASSIFICATION_MODELS["XGBoostClassifier"] = XGBClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4, eval_metric="logloss", verbosity=0
    )

# ---------------- HYPERPARAMETER SEARCH SPACES ----------------
# Keys use the "model__" prefix because each model sits inside a
# preprocessing Pipeline under the step name "model".
# Models without an entry here are trained with their default settings.

CLASSIFICATION_PARAM_GRIDS = {
    "LogisticRegression": {
        "model__C": [0.01, 0.1, 1, 10, 100],
    },
    "DecisionTreeClassifier": {
        "model__max_depth": [None, 3, 5, 10, 20],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
    },
    "KNN": {
        "model__n_neighbors": [3, 5, 7, 11, 15],
        "model__weights": ["uniform", "distance"],
    },
    "NaiveBayes": {
        "model__var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6],
    },
    "RandomForestClassifier": {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2"],
    },
    "GradientBoostingClassifier": {
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.03, 0.1, 0.2],
        "model__max_depth": [2, 3, 4],
        "model__subsample": [0.8, 1.0],
    },
    "AdaBoostClassifier": {
        "model__n_estimators": [50, 100, 200],
        "model__learning_rate": [0.1, 0.5, 1.0],
    },
    "XGBoostClassifier": {
        "model__n_estimators": [100, 200, 300],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__max_depth": [3, 4, 6],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0],
    },
}

REGRESSION_PARAM_GRIDS = {
    "DecisionTreeRegressor": {
        "model__max_depth": [None, 3, 5, 10, 20],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
    },
    "RandomForestRegressor": {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": [1.0, "sqrt"],
    },
    "GradientBoostingRegressor": {
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.03, 0.1, 0.2],
        "model__max_depth": [2, 3, 4],
        "model__subsample": [0.8, 1.0],
    },
    "AdaBoostRegressor": {
        "model__n_estimators": [50, 100, 200],
        "model__learning_rate": [0.1, 0.5, 1.0],
    },
    "XGBoostRegressor": {
        "model__n_estimators": [100, 200, 300],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__max_depth": [3, 4, 6],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0],
    },
}
