"""
Train the used-car price model and save everything the web app needs.

    python train.py

Produces:
    models/car_price_pipeline.joblib  - preprocessing + model in ONE object
    models/model_metadata.json        - metrics, error band, dropdown options,
                                        feature importance, library versions

Why this replaces the notebook's final model
--------------------------------------------
1. `original_price` is NOT the new-car showroom price. In this dataset the
   sale price is always 64-100% of it (median 94%), so it is the listing's
   pre-discount asking price. Training on it leaks the answer, and in the
   app users type the showroom price instead, which the model never saw.
2. `car_rating` ("great" / "fair" / "overpriced") is the listing site's
   verdict on the price itself, so it also leaks the target.
3. `times_viewed`, `is_hot`, `reserved`, `assured_buy`, `car_availability`,
   `source` and the ad date are listing-platform fields a buyer or seller
   can't know for their own car, so they can't be model inputs.
4. Age is now measured at the time of the listing (ad year - manufacturing
   year) instead of 2026 - manufacturing year, so "age" means the same
   thing in training and in the app.
5. Encoding lives inside a scikit-learn Pipeline, so the app no longer has
   to rebuild 76 one-hot columns by hand (where mismatches silently became 0).
"""

import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
RAW_DATA = ROOT / "data" / "Used_Car_Price_Prediction.csv"
MODEL_DIR = ROOT / "models"
RANDOM_STATE = 42

CATEGORICAL = ["make", "model", "variant", "fuel_type", "transmission", "body_type", "city"]
NUMERIC = ["car_age", "kms_run", "total_owners"]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "sale_price"
MISSING = "unknown"


# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------

def load_and_clean(path=RAW_DATA):
    df = pd.read_csv(path)
    n0 = len(df)

    df = df.drop_duplicates()
    # A handful of rows have prices of 0 or a few rupees - data errors.
    df = df[df[TARGET] >= 30_000]

    for col in CATEGORICAL:
        # Missing -> "unknown" so every value is a plain string. The app
        # sends "unknown" for fields the user leaves blank.
        df[col] = df[col].fillna(MISSING).astype(str).str.strip().str.lower()

    ad_year = pd.to_datetime(df["ad_created_on"], errors="coerce").dt.year
    df["car_age"] = (ad_year.fillna(ad_year.median()) - df["yr_mfr"]).clip(lower=0)

    print(f"Rows: {n0} raw -> {len(df)} after cleaning")
    return df.reset_index(drop=True)


# ------------------------------------------------------------------
# Candidate models (all predict log(price), converted back automatically)
# ------------------------------------------------------------------

def log_target(regressor):
    return TransformedTargetRegressor(regressor=regressor, func=np.log1p, inverse_func=np.expm1)


def make_hist_gb():
    # Ordinal-encode categories and let the gradient boosting model treat
    # them as native categoricals. Unknown values at prediction time become
    # "missing", which the model handles, instead of crashing.
    prep = ColumnTransformer(
        [
            ("cat", OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=np.nan,
                encoded_missing_value=np.nan, min_frequency=3, max_categories=250,
            ), CATEGORICAL),
            ("num", "passthrough", NUMERIC),
        ],
        verbose_feature_names_out=False,
    )
    model = HistGradientBoostingRegressor(
        categorical_features=list(range(len(CATEGORICAL))),
        learning_rate=0.05, max_iter=800, max_leaf_nodes=31,
        min_samples_leaf=10, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=40,
        random_state=RANDOM_STATE,
    )
    return log_target(Pipeline([("prep", prep), ("model", model)]))


def make_random_forest():
    prep = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=3), CATEGORICAL),
            ("num", "passthrough", NUMERIC),
        ]
    )
    model = RandomForestRegressor(
        n_estimators=300, min_samples_leaf=2, max_features=0.5,
        n_jobs=-1, random_state=RANDOM_STATE,
    )
    return log_target(Pipeline([("prep", prep), ("model", model)]))


def make_ridge():
    prep = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=3), CATEGORICAL),
            ("num", StandardScaler(), NUMERIC),
        ]
    )
    return log_target(Pipeline([("prep", prep), ("model", Ridge(alpha=1.0))]))


CANDIDATES = {
    "Ridge regression (baseline)": make_ridge,
    "Random forest": make_random_forest,
    "Histogram gradient boosting": make_hist_gb,
}


def metrics(y_true, y_pred):
    return {
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
        "mape_pct": round(float(np.mean(np.abs(y_pred - y_true) / y_true) * 100), 2),
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    df = load_and_clean()
    X, y = df[FEATURES], df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # 1. Compare candidates with 5-fold CV on the training set only.
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    comparison = []
    for name, factory in CANDIDATES.items():
        scores = cross_validate(
            factory(), X_train, y_train, cv=cv, n_jobs=1,
            scoring={"r2": "r2", "mae": "neg_mean_absolute_error"},
        )
        row = {
            "model": name,
            "cv_r2": round(float(scores["test_r2"].mean()), 4),
            "cv_mae": round(float(-scores["test_mae"].mean()), 2),
        }
        comparison.append(row)
        print(f"{name:32s}  CV R2 {row['cv_r2']:.4f}   CV MAE Rs {row['cv_mae']:,.0f}")

    best = min(comparison, key=lambda r: r["cv_mae"])
    print(f"\nBest by CV MAE: {best['model']}")

    # 2. Fit the winner on the full training set, evaluate ONCE on the test set.
    pipeline = CANDIDATES[best["model"]]()
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    test_metrics = metrics(y_test.to_numpy(), y_pred)
    print("Held-out test:", test_metrics)

    # Empirical error band: 80% of test cars sold within this ratio of the
    # prediction. More honest than a fixed +/- MAE for every price level.
    ratio = y_test.to_numpy() / y_pred
    band = {
        "low_ratio": round(float(np.percentile(ratio, 10)), 4),
        "high_ratio": round(float(np.percentile(ratio, 90)), 4),
        "coverage_pct": 80,
    }
    print("80% band (actual / predicted):", band)

    # 3. Permutation importance on the test set, on the ORIGINAL inputs
    #    (so the chart says "Model", not "model_freq" or "make_bmw").
    perm = permutation_importance(
        pipeline, X_test, y_test, n_repeats=5, random_state=RANDOM_STATE,
        scoring="neg_mean_absolute_error",
    )
    imp = np.clip(perm.importances_mean, 0, None)
    imp = imp / imp.sum() * 100 if imp.sum() else imp
    importance = sorted(
        ({"feature": f, "importance": round(float(v), 2)} for f, v in zip(FEATURES, imp)),
        key=lambda d: d["importance"], reverse=True,
    )

    # 4. Refit on ALL data for deployment.
    final = CANDIDATES[best["model"]]()
    final.fit(X, y)

    # 5. Dropdown options for the web form (only values the model has seen).
    make_models = {
        make: sorted(set(g["model"]) - {MISSING})
        for make, g in df.groupby("make")
    }
    variants = {
        f"{mk}|{md}": sorted(set(g["variant"]) - {MISSING})
        for (mk, md), g in df.groupby(["make", "model"])
    }
    options = {
        "make_models": make_models,
        "variants": variants,
        "fuel_type": sorted(set(df["fuel_type"]) - {MISSING}),
        "transmission": sorted(set(df["transmission"]) - {MISSING}),
        "body_type": sorted(set(df["body_type"]) - {MISSING}),
        "city": sorted(set(df["city"]) - {MISSING}),
        "max_car_age": int(df["car_age"].max()),
        "missing_token": MISSING,
    }

    ad_dates = pd.to_datetime(df["ad_created_on"], errors="coerce")
    metadata = {
        "model_name": best["model"],
        "features": FEATURES,
        "n_rows": int(len(df)),
        "data_period": f"{ad_dates.min():%b %Y} - {ad_dates.max():%b %Y}",
        "cv_comparison": comparison,
        "test_metrics": test_metrics,
        "error_band": band,
        "feature_importance": importance,
        "options": options,
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(final, MODEL_DIR / "car_price_pipeline.joblib", compress=3)
    (MODEL_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"\nSaved {MODEL_DIR / 'car_price_pipeline.joblib'}")
    print(f"Saved {MODEL_DIR / 'model_metadata.json'}")


if __name__ == "__main__":
    main()
