from sklearn.datasets import fetch_openml

from sklearn.datasets import (
    load_iris,
    load_wine,
    load_breast_cancer,
    load_diabetes,
    fetch_california_housing,
    load_linnerud
)

# Dataset Registry
DATASETS = [
    # ---------- Classification ----------
    {
        "name": "iris",
        "source": "sklearn",
        "task": "classification",
        "loader": load_iris,
        "target": "target"
    },
    {
        "name": "wine",
        "source": "sklearn",
        "task": "classification",
        "loader": load_wine,
        "target": "target"
    },
    {
        "name": "breast_cancer",
        "source": "sklearn",
        "task": "classification",
        "loader": load_breast_cancer,
        "target": "target"
    },

    # ---------- Regression ----------
    {
        "name": "diabetes",
        "source": "sklearn",
        "task": "regression",
        "loader": load_diabetes,
        "target": "target"
    },
    {
        "name": "california_housing",
        "source": "sklearn",
        "task": "regression",
        "loader": fetch_california_housing,
        "target": "target"
    },
    {
        "name": "linnerud",
        "source": "sklearn",
        "task": "regression",
        "loader": load_linnerud,
        "target": "target"
    },


# ----------- OpenML Classification Datasets (12) -----------
{
    "name": "adult",
    "source": "openml",
    "task": "classification",
    "loader": fetch_openml,
    "target": "class"
},
{
    "name": "titanic",
    "source": "openml",
    "task": "classification",
    "loader": fetch_openml,
    "target": "survived"
},
{
    "name": "heart-disease",
    "source": "openml",
    "task": "classification",
    "loader": fetch_openml,
    "target": "class"
},
{
    "name": "bank-marketing",
    "source": "openml",
    "task": "classification",
    "loader": fetch_openml,
    "target": "y"
},
{
    "name": "credit-g",
    "source": "openml",
    "task": "classification",
    "loader": fetch_openml,
    "target": "class"
},
{
    "name": "mushroom",
    "source": "openml",
    "task": "classification",
    "loader": fetch_openml,
    "target": "class"
},
{
    "name": "spambase",
    "source": "openml",
    "task": "classification",
    "loader": fetch_openml,
    "target": "class"
},
{
    "name": "car",
    "source": "openml",
    "task": "classification",
    "loader": fetch_openml,
    "target": "class"
},
{
    "name": "electricity",
    "source": "openml",
    "task": "classification",
    "loader": fetch_openml,
    "target": "class"
},
{
    "name": "churn",
    "source": "openml",
    "task": "classification",
    "loader": fetch_openml,
    "target": "Exited"
},
{
    "name": "phishing",
    "source": "openml",
    "task": "classification",
    "loader": fetch_openml,
    "target": "Result"
},
{
    "name": "diabetes",
    "source": "openml",
    "task": "classification",
    "loader": fetch_openml,
    "target": "class"
},

# ----------- OpenML Regression Datasets (12) -----------
{
    "name": "house_prices",
    "source": "openml",
    "task": "regression",
    "loader": fetch_openml,
    "target": "SalePrice"
},
{
    "name": "energy_efficiency",
    "source": "openml",
    "task": "regression",
    "loader": fetch_openml,
    "target": "Y1"
},
{
    "name": "concrete",
    "source": "openml",
    "task": "regression",
    "loader": fetch_openml,
    "target": "strength"
},
{
    "name": "wine-quality-red",
    "source": "openml",
    "task": "regression",
    "loader": fetch_openml,
    "target": "quality"
},
{
    "name": "bike-sharing-demand",
    "source": "openml",
    "task": "regression",
    "loader": fetch_openml,
    "target": "count"
},
{
    "name": "cpu_small",
    "source": "openml",
    "task": "regression",
    "loader": fetch_openml,
    "target": "perf"
},
{
    "name": "airfoil_self_noise",
    "source": "openml",
    "task": "regression",
    "loader": fetch_openml,
    "target": "Scaled_sound_pressure_level"
},
{
    "name": "yacht_hydrodynamics",
    "source": "openml",
    "task": "regression",
    "loader": fetch_openml,
    "target": "Residuary_resistance"
},
{
    "name": "automobile",
    "source": "openml",
    "task": "regression",
    "loader": fetch_openml,
    "target": "price"
},
{
    "name": "student_performance",
    "source": "openml",
    "task": "regression",
    "loader": fetch_openml,
    "target": "G3"
},
{
    "name": "forest_fires",
    "source": "openml",
    "task": "regression",
    "loader": fetch_openml,
    "target": "area"
},
{
    "name": "real_estate",
    "source": "openml",
    "task": "regression",
    "loader": fetch_openml,
    "target": "Y house price of unit area"
},
]




def get_all_datasets():
    """
    Returns all dataset configurations.
    """
    return DATASETS


