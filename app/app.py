import os
import traceback
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

MODEL_PATH = os.path.join(
    PROJECT_ROOT, "models", "used_car_price_xgboost.pkl"
)

FEATURE_COLUMNS_PATH = os.path.join(
    PROJECT_ROOT, "models", "feature_columns.pkl"
)

DATA_PATH = os.path.join(
    PROJECT_ROOT, "data", "Used_Car_Price_Prediction_Cleaned.csv"
)


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading XGBoost model...")

model = joblib.load(MODEL_PATH)
feature_columns = list(joblib.load(FEATURE_COLUMNS_PATH))

print("Model loaded successfully!")
print("Expected features:", len(feature_columns))

# Model's own reported error from your notebook - used to draw a
# confidence / uncertainty band around the predicted price.
# Update this if you retrain and get a different MAE.
MODEL_MAE = 40996.51


# ============================================================
# LOAD TRAINING DATA
# ============================================================

df_train = pd.read_csv(DATA_PATH)

print("Training data loaded:", df_train.shape)

# Which column in the raw CSV holds the actual sale price - needed for
# the "similar cars" lookup. Adjust this list if your CSV uses a
# different name for the target variable.
TARGET_COLUMN = None
for candidate in ["sale_price", "price", "selling_price", "original_price"]:
    if candidate in df_train.columns:
        TARGET_COLUMN = candidate
        break


# ============================================================
# CATEGORICAL FEATURE MAPPING (built once at startup)
# ------------------------------------------------------------
# The old approach guessed how pd.get_dummies() named each one-hot
# column (value, value.title(), value.upper(), value_with_underscores)
# and used whichever guess happened to exist in feature_columns. If the
# real training data used a casing/spacing convention that didn't match
# any of those guesses, the feature silently stayed at 0 - no error,
# just a categorical input the model never actually saw.
#
# This was the main cause of predictions barely reacting to make,
# fuel type, body type, transmission, city, rating, etc.: those inputs
# were mostly landing on nothing, leaving the model to lean almost
# entirely on the numeric features (especially original_price).
#
# Fix: look at the ACTUAL raw values that existed in the training CSV
# for each categorical column, and match them directly against
# feature_columns. This guarantees a correct match regardless of the
# casing/spacing convention used when the model was trained.
# ============================================================

CATEGORICAL_RAW_COLUMNS = [
    "fuel_type", "city", "body_type", "transmission",
    "source", "make", "car_availability", "car_rating",
]


def build_category_maps():
    maps = {}

    for col in CATEGORICAL_RAW_COLUMNS:
        if col not in df_train.columns:
            continue

        prefix = col + "_"
        # every one-hot column in the trained model that belongs to this field
        candidate_columns = [c for c in feature_columns if c.startswith(prefix)]

        value_map = {}
        for raw in df_train[col].dropna().astype(str).unique():
            raw_clean = raw.strip()
            candidate = f"{prefix}{raw_clean}"
            if candidate in candidate_columns:
                value_map[raw_clean.lower()] = candidate

        maps[col] = value_map

    return maps


CATEGORY_MAPS = build_category_maps()


# ============================================================
# FEATURE IMPORTANCE (computed once at startup)
# ============================================================

def humanize_feature_name(name):
    """Turns a raw trained feature name into something readable for
    the chart, e.g. 'fuel_type_diesel' -> 'Fuel Type: Diesel',
    'model_freq' -> 'Model (popularity)'."""

    if name.endswith("_freq"):
        base = name[:-5].replace("_", " ")
        return f"{base.title()} (popularity)"

    prefixes = [
        "fuel_type", "city", "body_type", "transmission", "source",
        "make", "car_availability", "car_rating", "fitness_certificate",
    ]
    for prefix in prefixes:
        if name.startswith(prefix + "_"):
            value = name[len(prefix) + 1:]
            return f"{prefix.replace('_', ' ').title()}: {value.title()}"

    return name.replace("_", " ").title()


def build_feature_importance():
    try:
        importances = model.feature_importances_
    except AttributeError:
        return []

    pairs = list(zip(feature_columns, importances))
    pairs.sort(key=lambda p: p[1], reverse=True)

    top = pairs[:8]
    total = sum(v for _, v in pairs) or 1.0

    return [
        {
            "feature": humanize_feature_name(name),
            "importance": round(float(value) / float(total) * 100, 2),
        }
        for name, value in top
    ]


FEATURE_IMPORTANCE = build_feature_importance()


# ============================================================
# FREQUENCY ENCODING
# IMPORTANT:
# Notebook used normalize=True.
# The trained feature name is "<col>_freq" (e.g. "model_freq"),
# NOT the bare column name.
# ============================================================

frequency_columns = [
    "car_name",
    "variant",
    "registered_city",
    "registered_state",
    "rto",
    "model"
]

frequency_maps = {}

for col in frequency_columns:

    if col == "car_name":
        # car_name is derived (make + " " + model), not a raw column —
        # rebuild it the same way we will at request time.
        if "make" in df_train.columns and "model" in df_train.columns:
            car_name_series = (
                df_train["make"].astype(str).str.strip().str.lower()
                + " "
                + df_train["model"].astype(str).str.strip().str.lower()
            )
            frequency_maps["car_name"] = (
                car_name_series.value_counts(normalize=True).to_dict()
            )
        continue

    if col in df_train.columns:

        frequency_maps[col] = (
            df_train[col]
            .astype(str)
            .str.strip()
            .str.lower()
            .value_counts(normalize=True)
            .to_dict()
        )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_text(value):

    if value is None:
        return ""

    return str(value).strip().lower()


def to_float(value):

    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def to_bool(value):

    value = clean_text(value)

    return 1 if value in [
        "yes",
        "true",
        "1",
        "on"
    ] else 0


def get_request_data():
    """
    Works no matter which way the frontend sends data:
    - JSON body (fetch with Content-Type: application/json)
    - classic form-encoded POST (multipart/form-data or
      application/x-www-form-urlencoded)
    This was the main bug: the previous version only ever read
    request.form, which is always empty for a JSON POST.
    """

    if request.is_json:
        return request.get_json(silent=True) or {}

    if request.form:
        return request.form

    # last resort: some clients send JSON without setting the header
    return request.get_json(silent=True) or {}


def set_categorical_feature(row, prefix, value):
    """
    Looks up the exact trained one-hot column for this raw value using
    CATEGORY_MAPS (built from the real training data), falling back to
    the old guessing behavior only if nothing was found (e.g. a brand
    new value the model never saw during training - in that case there
    is genuinely no matching column, and leaving it at 0 is correct).
    """

    value_clean = clean_text(value)
    if not value_clean:
        return

    exact_map = CATEGORY_MAPS.get(prefix, {})
    exact_match = exact_map.get(value_clean)
    if exact_match:
        row[exact_match] = 1
        return

    # Fallback: old guessing logic, kept only as a last resort for
    # values that truly weren't in the training data at all.
    possible_names = [
        f"{prefix}_{value_clean}",
        f"{prefix}_{value_clean.replace(' ', '_')}",
        f"{prefix}_{value_clean.title()}",
        f"{prefix}_{value_clean.upper()}",
    ]
    for name in possible_names:
        if name in feature_columns:
            row[name] = 1
            return

    # Nothing matched at all - log it so it's visible instead of silent.
    print(f"[warn] no trained column found for {prefix}='{value_clean}'")


def build_feature_row(inputs):
    """
    Builds the exact-column, exact-order row the model expects, from a
    dict of parsed inputs. Shared by /predict and the price-trend
    calculation so both use identical logic - avoids duplicating (and
    accidentally desyncing) the encoding rules in two places.
    """

    row = pd.Series(0.0, index=feature_columns, dtype="float64")

    frequency_values = {
        "car_name": f"{inputs['make']} {inputs['model_name']}".strip(),
        "variant": inputs["variant"],
        "registered_city": inputs["registered_city"],
        "registered_state": inputs["registered_state"],
        "rto": inputs["rto"],
        "model": inputs["model_name"],
    }
    for col, value in frequency_values.items():
        feature_name = f"{col}_freq"
        if feature_name in feature_columns:
            freq_map = frequency_maps.get(col, {})
            row[feature_name] = freq_map.get(clean_text(value), 0)

    categorical_values = {
        "fuel_type": inputs["fuel_type"],
        "city": inputs["city"],
        "body_type": inputs["body_type"],
        "transmission": inputs["transmission"],
        "source": inputs.get("source", ""),
        "make": inputs["make"],
        "car_availability": inputs.get("car_availability", ""),
        "car_rating": inputs["car_rating"],
    }
    for col, value in categorical_values.items():
        set_categorical_feature(row, col, value)

    # fitness_certificate is boolean-flavoured in the training data
    # (its two raw values are True/False), so get_dummies names its
    # column "fitness_certificate_True" / "fitness_certificate_False".
    # Use to_bool() here too so checkbox values like "on" are handled
    # the same way as every other boolean-ish field.
    fitness_bool_str = "True" if to_bool(inputs.get("fitness_certificate", "")) else "False"
    fitness_column = f"fitness_certificate_{fitness_bool_str}"
    if fitness_column in feature_columns:
        row[fitness_column] = 1

    numeric_values = {
        "yr_mfr": inputs["manufacturing_year"],
        "kms_run": inputs["kms_driven"],
        "total_owners": inputs["total_owners"],
        "original_price": inputs["original_price"],
        "is_hot": inputs.get("is_hot", 0),
        "reserved": inputs.get("reserved", 0),
        "warranty_avail": inputs.get("warranty_avail", 0),
        "assured_buy": inputs.get("assured_buy", 0),
        "times_viewed": inputs.get("times_viewed", 0),
    }
    for feature_name, value in numeric_values.items():
        if feature_name in feature_columns:
            row[feature_name] = value

    # NOTE: your notebook hardcoded 2026 as "current year" when this
    # feature was engineered — using datetime.now().year is fine while
    # it's still 2026, but if you retrain later or run this next year,
    # switch back to a fixed year to stay consistent with training.
    if "car_age" in feature_columns and inputs["manufacturing_year"]:
        current_year = datetime.now().year
        row["car_age"] = max(current_year - int(inputs["manufacturing_year"]), 0)

    return row


def predict_price(row):
    X_input = pd.DataFrame([row], columns=feature_columns)
    price = float(model.predict(X_input)[0])
    return max(price, 0), X_input


# ============================================================
# DETERMINISTIC DEPRECIATION CEILING
# ------------------------------------------------------------
# The trained model has no concept of "must be worth less than
# what was paid" - it just pattern-matches from historical listings,
# so it can (correctly, statistically) land within its own margin
# of error on either side of the original price. To guarantee the
# business rule "a used car is always valued below its original
# price, and more so with age / km / accidents", this layer computes
# an independent, rule-based ceiling and takes whichever of the
# model's prediction or this ceiling is LOWER. It never overrides the
# model upward - only ever caps it downward.
# ============================================================

EXPECTED_KM_PER_YEAR = 12000       # "normal" usage assumption
MAX_TOTAL_DEPRECIATION = 0.75      # never value a car below 25% of original
ACCIDENT_PENALTY = 0.15            # flat extra cut for accident history


def compute_depreciation_breakdown(age_years, kms_driven, had_accident):
    """Returns (total_fraction, breakdown_dict). total_fraction is how
    much value is cut from the original price - e.g. 0.32 = 32% down."""

    age_years = max(age_years, 0)

    # Year-wise: cars typically lose the most value in year 1, then a
    # smaller, fairly steady percentage each year after that.
    if age_years <= 0:
        age_pct = 0.05
    else:
        age_pct = 0.18 + 0.10 * (age_years - 1)
    age_pct = min(age_pct, 0.65)

    # Km-wise: extra depreciation only for driving MORE than the
    # "expected" km for the car's age (12,000 km/year assumption).
    # Every 10,000 km of excess usage knocks off another 1.5%, capped.
    expected_km = EXPECTED_KM_PER_YEAR * max(age_years, 1)
    excess_km = max(0.0, (kms_driven or 0) - expected_km)
    km_pct = min(0.20, (excess_km / 10000) * 0.015)

    # Accident history: flat penalty on top, since damage/repair
    # history affects resale value regardless of age or mileage.
    accident_pct = ACCIDENT_PENALTY if had_accident else 0.0

    total_pct = min(MAX_TOTAL_DEPRECIATION, age_pct + km_pct + accident_pct)

    return total_pct, {
        "age_pct": round(age_pct * 100, 1),
        "km_pct": round(km_pct * 100, 1),
        "accident_pct": round(accident_pct * 100, 1),
        "total_pct": round(total_pct * 100, 1),
    }


def apply_depreciation_ceiling(raw_predicted_price, original_price, age_years, kms_driven, had_accident):
    """Caps the model's raw prediction so it can never exceed the
    rule-based depreciation ceiling. Returns (final_price, breakdown)."""

    total_pct, breakdown = compute_depreciation_breakdown(age_years, kms_driven, had_accident)

    if original_price and original_price > 0:
        ceiling_price = original_price * (1 - total_pct)
        final_price = min(raw_predicted_price, ceiling_price)
    else:
        # No original price to cap against - still apply the accident
        # penalty directly to the model's own number, since a real
        # accident should reduce value regardless of a reference price.
        final_price = raw_predicted_price * (1 - (ACCIDENT_PENALTY if had_accident else 0.0))

    return max(final_price, 0), breakdown


def find_similar_cars(inputs, limit=5):
    """Looks up real listings from the training CSV that are close to
    the entered car - same make (if enough exist), similar year and km."""

    if df_train.empty or TARGET_COLUMN is None:
        return []

    candidates = df_train.copy()

    make = inputs["make"]
    if make and "make" in candidates.columns:
        same_make = candidates[candidates["make"].astype(str).str.strip().str.lower() == make]
        if len(same_make) >= 3:
            candidates = same_make

    mfr_year = inputs.get("manufacturing_year")
    if "yr_mfr" in candidates.columns and mfr_year is not None:
        candidates = candidates.assign(
            _year_diff=(candidates["yr_mfr"] - mfr_year).abs()
        )
    else:
        candidates = candidates.assign(_year_diff=0)

    kms = inputs.get("kms_driven")
    if "kms_run" in candidates.columns and kms is not None:
        candidates = candidates.assign(
            _km_diff=(candidates["kms_run"] - kms).abs()
        )
    else:
        candidates = candidates.assign(_km_diff=0)

    candidates["_score"] = candidates["_year_diff"] * 50000 + candidates["_km_diff"] / 10
    candidates = candidates.sort_values("_score").head(limit)

    results = []
    for _, r in candidates.iterrows():
        results.append({
            "make": str(r.get("make", "")).title(),
            "model": str(r.get("model", "")).title(),
            "year": int(r["yr_mfr"]) if "yr_mfr" in r and pd.notna(r["yr_mfr"]) else None,
            "kms": int(r["kms_run"]) if "kms_run" in r and pd.notna(r["kms_run"]) else None,
            "price": round(float(r[TARGET_COLUMN]), 2) if pd.notna(r[TARGET_COLUMN]) else None,
        })
    return results


def build_price_trend(inputs, span=8):
    """Holds everything fixed except manufacturing year, and predicts
    the price at each year in the last `span` years - shows the
    depreciation curve for this exact car. Each point goes through the
    same depreciation ceiling as the main prediction, so the curve is
    a real downward depreciation slope rather than raw (and sometimes
    noisy) model output."""

    current_year = datetime.now().year
    original_price = inputs.get("original_price", 0)
    had_accident = bool(inputs.get("had_accident", 0))

    trend = []
    for yr in range(current_year - span + 1, current_year + 1):
        trial_inputs = dict(inputs)
        trial_inputs["manufacturing_year"] = yr
        row = build_feature_row(trial_inputs)
        raw_price, _ = predict_price(row)

        trial_age = max(current_year - yr, 0)
        final_price, _ = apply_depreciation_ceiling(
            raw_price, original_price, trial_age, inputs.get("kms_driven", 0), had_accident
        )
        trend.append({"year": yr, "price": round(final_price, 2)})
    return trend


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def index():

    return render_template("index.html")


# ============================================================
# PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        form = get_request_data()

        # ----------------------------------------------------
        # READ USER INPUT
        # Field names below match what the current frontend
        # actually sends. .get(a, form.get(b)) pattern lets an
        # older/newer form use either name safely.
        # ----------------------------------------------------

        inputs = {
            "make": clean_text(form.get("make")),
            "model_name": clean_text(form.get("model")),
            "variant": clean_text(form.get("variant")),
            "manufacturing_year": to_float(form.get("yr_mfr", form.get("manufacturing_year"))),
            "kms_driven": to_float(form.get("kms_run", form.get("kms_driven"))),
            "fuel_type": clean_text(form.get("fuel_type")),
            "transmission": clean_text(form.get("transmission")),
            "body_type": clean_text(form.get("body_type")),
            "total_owners": to_float(form.get("total_owners")),
            "city": clean_text(form.get("city")),
            "registered_city": clean_text(form.get("registered_city")),
            "registered_state": clean_text(form.get("registered_state")),
            "rto": clean_text(form.get("rto")),
            "car_rating": clean_text(form.get("car_rating")),
            "original_price": to_float(form.get("original_price")),
            "car_availability": clean_text(form.get("car_availability")),
            "source": clean_text(form.get("source")),
            "fitness_certificate": clean_text(form.get("fitness_certificate")),
            "assured_buy": to_bool(form.get("assured_buy")),
            "is_hot": to_bool(form.get("is_hot")),
            "reserved": to_bool(form.get("reserved")),
            "warranty_avail": to_bool(form.get("warranty_avail", form.get("warranty_available"))),
            "times_viewed": to_float(form.get("times_viewed")),
            "had_accident": to_bool(form.get("had_accident")),
        }

        print("\nParsed input:", {
            "make": inputs["make"], "model": inputs["model_name"],
            "yr_mfr": inputs["manufacturing_year"], "kms_run": inputs["kms_driven"],
            "fuel_type": inputs["fuel_type"], "transmission": inputs["transmission"],
            "body_type": inputs["body_type"], "total_owners": inputs["total_owners"],
            "city": inputs["city"], "original_price": inputs["original_price"],
            "car_rating": inputs["car_rating"],
        })

        # ----------------------------------------------------
        # BUILD FEATURE ROW + PREDICT
        # ----------------------------------------------------

        row = build_feature_row(inputs)
        raw_predicted_price, X_input = predict_price(row)

        # Sanity check while you're debugging: how many features
        # actually ended up non-zero? If this number stays tiny no
        # matter what you type, something upstream still isn't
        # reaching this row.
        nonzero = int((X_input.iloc[0] != 0).sum())
        print("Prediction input created")
        print("Feature count:", X_input.shape[1], "| non-zero features:", nonzero)

        # ----------------------------------------------------
        # DEPRECIATION CEILING (age + km + accident)
        # Guarantees the final number is always at or below what the
        # rule-based depreciation curve says the car should be worth -
        # never above the model's own raw number, only ever capped down.
        # ----------------------------------------------------

        current_year = datetime.now().year
        age_years = max(current_year - int(inputs["manufacturing_year"]), 0) if inputs["manufacturing_year"] else 0

        predicted_price, depreciation_breakdown = apply_depreciation_ceiling(
            raw_predicted_price,
            inputs["original_price"],
            age_years,
            inputs["kms_driven"],
            bool(inputs["had_accident"]),
        )

        # ----------------------------------------------------
        # FAIR PRICE RANGE
        # ----------------------------------------------------

        fair_low = predicted_price * 0.90

        fair_high = predicted_price * 1.10

        # ----------------------------------------------------
        # VALUATION STATUS
        # ----------------------------------------------------

        original_price = inputs["original_price"]

        if original_price and original_price < predicted_price * 0.90:

            valuation_status = "UNDERPRICED"
            recommendation = "BUY"

        elif original_price and original_price > predicted_price * 1.10:

            valuation_status = "OVERPRICED"
            recommendation = "AVOID"

        else:

            valuation_status = "FAIR"
            recommendation = "NEGOTIATE"

        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        response = {

            "success": True,

            "predicted_price": round(predicted_price, 2),

            "fair_price": round(predicted_price, 2),
            "fair_price_low": round(fair_low, 2),
            "fair_price_high": round(fair_high, 2),

            # Confidence / uncertainty band, based on the model's own MAE.
            "confidence_low": round(max(predicted_price - MODEL_MAE, 0), 2),
            "confidence_high": round(predicted_price + MODEL_MAE, 2),
            "model_mae": MODEL_MAE,

            "depreciation": round(
                (1 - (predicted_price / original_price)) * 100, 1
            ) if original_price else 0,

            "depreciation_breakdown": depreciation_breakdown,

            "status": valuation_status,
            "valuation_status": valuation_status,

            "recommendation": recommendation,

            "feature_importance": FEATURE_IMPORTANCE,
            "similar_cars": find_similar_cars(inputs),
            "price_trend": build_price_trend(inputs),

            "summary": {
                "make": inputs["make"],
                "model": inputs["model_name"],
                "variant": inputs["variant"],
                "manufacturing_year": inputs["manufacturing_year"],
                "kms_driven": inputs["kms_driven"],
                "total_owners": inputs["total_owners"],
                "city": inputs["city"],
                "original_price": original_price,
                "had_accident": bool(inputs["had_accident"]),
            }
        }

        return jsonify(response)

    except Exception:

        print("\nPrediction error:")
        traceback.print_exc()

        return jsonify({
            "success": False,
            "error": "Prediction failed. Please check the input values."
        }), 400


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("======================================")
    print("Smart Used Car Valuation System")
    print("======================================")

    app.run(debug=True, port=5000)
