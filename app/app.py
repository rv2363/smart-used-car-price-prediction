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


# ============================================================
# LOAD TRAINING DATA
# ============================================================

df_train = pd.read_csv(DATA_PATH)

print("Training data loaded:", df_train.shape)


# ============================================================
# FREQUENCY ENCODING
# IMPORTANT:
# Notebook used normalize=True.
# The trained feature name is "<col>_freq" (e.g. "model_freq"),
# NOT the bare column name — this is what was wrong before.
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

    value = clean_text(value)

    if not value:
        return

    possible_names = [

        f"{prefix}_{value}",

        f"{prefix}_{value.replace(' ', '_')}",

        f"{prefix}_{value.title()}",

        f"{prefix}_{value.upper()}",

    ]

    for name in possible_names:

        if name in feature_columns:

            row[name] = 1

            return


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

        make = clean_text(form.get("make"))

        model_name = clean_text(form.get("model"))

        variant = clean_text(form.get("variant"))

        manufacturing_year = to_float(
            form.get("yr_mfr", form.get("manufacturing_year"))
        )

        kms_driven = to_float(
            form.get("kms_run", form.get("kms_driven"))
        )

        fuel_type = clean_text(form.get("fuel_type"))

        transmission = clean_text(form.get("transmission"))

        body_type = clean_text(form.get("body_type"))

        total_owners = to_float(form.get("total_owners"))

        city = clean_text(form.get("city"))

        registered_city = clean_text(form.get("registered_city"))

        registered_state = clean_text(form.get("registered_state"))

        rto = clean_text(form.get("rto"))

        car_rating = clean_text(form.get("car_rating"))

        original_price = to_float(form.get("original_price"))

        car_availability = clean_text(form.get("car_availability"))

        source = clean_text(form.get("source"))

        fitness_certificate = clean_text(form.get("fitness_certificate"))

        assured_buy = to_bool(form.get("assured_buy"))

        is_hot = to_bool(form.get("is_hot"))

        reserved = to_bool(form.get("reserved"))

        warranty_avail = to_bool(
            form.get("warranty_avail", form.get("warranty_available"))
        )

        times_viewed = to_float(form.get("times_viewed"))

        print("\nParsed input:", {
            "make": make, "model": model_name, "yr_mfr": manufacturing_year,
            "kms_run": kms_driven, "fuel_type": fuel_type,
            "transmission": transmission, "body_type": body_type,
            "total_owners": total_owners, "city": city,
            "original_price": original_price, "car_rating": car_rating,
        })

        # ----------------------------------------------------
        # CREATE EMPTY ROW
        # ----------------------------------------------------

        row = pd.Series(
            0.0,
            index=feature_columns,
            dtype="float64"
        )

        # ----------------------------------------------------
        # FREQUENCY ENCODING
        # Trained feature name is "<col>_freq", not the bare name.
        # ----------------------------------------------------

        frequency_values = {

            "car_name": f"{make} {model_name}".strip(),

            "variant": variant,

            "registered_city": registered_city,

            "registered_state": registered_state,

            "rto": rto,

            "model": model_name
        }

        for col, value in frequency_values.items():

            feature_name = f"{col}_freq"

            if feature_name in feature_columns:

                freq_map = frequency_maps.get(col, {})

                row[feature_name] = freq_map.get(clean_text(value), 0)

        # ----------------------------------------------------
        # ONE-HOT ENCODING
        # ----------------------------------------------------

        categorical_values = {

            "fuel_type": fuel_type,

            "city": city,

            "body_type": body_type,

            "transmission": transmission,

            "source": source,

            "make": make,

            "car_availability": car_availability,

            "car_rating": car_rating,

        }

        for col, value in categorical_values.items():

            set_categorical_feature(row, col, value)

        # fitness_certificate is boolean-flavoured in the training data
        # (its two raw values are True/False), so get_dummies names its
        # column "fitness_certificate_True" / "fitness_certificate_False".
        fitness_bool_str = "True" if fitness_certificate in ("yes", "true", "1") else "False"
        fitness_column = f"fitness_certificate_{fitness_bool_str}"
        if fitness_column in feature_columns:
            row[fitness_column] = 1

        # ----------------------------------------------------
        # NUMERIC FEATURES (trained column name -> value)
        # ----------------------------------------------------

        numeric_values = {

            "yr_mfr": manufacturing_year,

            "kms_run": kms_driven,

            "total_owners": total_owners,

            "original_price": original_price,

            "is_hot": is_hot,

            "reserved": reserved,

            "warranty_avail": warranty_avail,

            "assured_buy": assured_buy,

            "times_viewed": times_viewed
        }

        for feature_name, value in numeric_values.items():

            if feature_name in feature_columns:

                row[feature_name] = value

        # ----------------------------------------------------
        # CAR AGE
        # NOTE: your notebook hardcoded 2026 as "current year" when
        # this feature was engineered — using datetime.now().year is
        # fine while it's still 2026, but if you retrain later or run
        # this next year, switch back to a fixed year to stay
        # consistent with how the model was actually trained.
        # ----------------------------------------------------

        if "car_age" in feature_columns and manufacturing_year:

            current_year = datetime.now().year

            row["car_age"] = max(
                current_year - int(manufacturing_year),
                0
            )

        # ----------------------------------------------------
        # FINAL DATAFRAME
        # ----------------------------------------------------

        X_input = pd.DataFrame(
            [row],
            columns=feature_columns
        )

        # Sanity check while you're debugging: how many features
        # actually ended up non-zero? If this number stays tiny no
        # matter what you type, something upstream still isn't
        # reaching this row.
        nonzero = int((X_input.iloc[0] != 0).sum())
        print("Prediction input created")
        print("Feature count:", X_input.shape[1], "| non-zero features:", nonzero)

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        predicted_price = float(model.predict(X_input)[0])

        predicted_price = max(predicted_price, 0)

        # ----------------------------------------------------
        # FAIR PRICE RANGE
        # ----------------------------------------------------

        fair_low = predicted_price * 0.90

        fair_high = predicted_price * 1.10

        # ----------------------------------------------------
        # VALUATION STATUS
        # ----------------------------------------------------

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

            "depreciation": round(
                (1 - (predicted_price / original_price)) * 100, 1
            ) if original_price else 0,

            "status": valuation_status,
            "valuation_status": valuation_status,

            "recommendation": recommendation,

            "summary": {
                "make": make,
                "model": model_name,
                "variant": variant,
                "manufacturing_year": manufacturing_year,
                "kms_driven": kms_driven,
                "total_owners": total_owners,
                "city": city,
                "original_price": original_price
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