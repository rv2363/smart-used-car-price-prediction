"""
Smart Used Car Valuation - Flask app.

All preprocessing lives inside models/car_price_pipeline.joblib (built by
train.py), so this file only has to turn the form into a one-row DataFrame
with the same raw columns the model was trained on.
"""

import json
import os
import sys
import traceback
from datetime import datetime

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
sys.path.insert(0, PROJECT_ROOT)

from train import CATEGORICAL, FEATURES, MISSING, load_and_clean  # noqa: E402

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "car_price_pipeline.joblib")
META_PATH = os.path.join(PROJECT_ROOT, "models", "model_metadata.json")

app = Flask(__name__)

# ------------------------------------------------------------------
# Load model, metadata and reference data once at startup
# ------------------------------------------------------------------

pipeline = joblib.load(MODEL_PATH)
with open(META_PATH) as f:
    META = json.load(f)

OPTIONS = META["options"]
MAX_CAR_AGE = OPTIONS["max_car_age"]
BAND_LOW = META["error_band"]["low_ratio"]
BAND_HIGH = META["error_band"]["high_ratio"]
MODEL_MAE = META["test_metrics"]["mae"]

# Cleaned listings, used for the "similar cars" table.
df_ref = load_and_clean()
df_ref["yr_mfr"] = df_ref["yr_mfr"].astype(int)

print(f"Loaded {META['model_name']} | test R2 {META['test_metrics']['r2']} "
      f"| MAE Rs {MODEL_MAE:,.0f} | {len(df_ref)} reference listings")

# ------------------------------------------------------------------
# Business rules (clearly separate from the ML model)
# ------------------------------------------------------------------

# The dataset has no accident information, so the model can't learn this.
# A flat, disclosed rule-of-thumb is applied instead.
ACCIDENT_ADJUSTMENT = 0.15
SCRAP_AGE_YEARS = 15
MIN_YEAR = 1990

FEATURE_LABELS = {
    "make": "Brand", "model": "Model", "variant": "Variant",
    "fuel_type": "Fuel type", "transmission": "Transmission",
    "body_type": "Body type", "city": "City", "car_age": "Car age",
    "kms_run": "Kilometers driven", "total_owners": "Number of owners",
}
FEATURE_IMPORTANCE = [
    {"feature": FEATURE_LABELS.get(d["feature"], d["feature"]), "importance": d["importance"]}
    for d in META["feature_importance"]
]

GENERAL_ADVICE = [
    "Compare a few similar live listings - this is one estimate, not the only data point.",
    "If the asking price is above the estimated range, use the gap as your negotiating starting point.",
    "Always check service and accident history and get an independent mechanic's inspection.",
    "A price far below the range can signal hidden problems - inspect it even more carefully.",
]


# ------------------------------------------------------------------
# Input handling
# ------------------------------------------------------------------

def text(value):
    return str(value).strip().lower() if value not in (None, "") else ""


def number(value):
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def truthy(value):
    return text(value) in {"yes", "true", "1", "on"}


def get_request_data():
    if request.is_json:
        return request.get_json(silent=True) or {}
    return request.form or request.get_json(silent=True) or {}


def parse_inputs(form):
    return {
        "make": text(form.get("make")),
        "model": text(form.get("model")),
        "variant": text(form.get("variant")),
        "fuel_type": text(form.get("fuel_type")),
        "transmission": text(form.get("transmission")),
        "body_type": text(form.get("body_type")),
        "city": text(form.get("city")),
        "yr_mfr": number(form.get("yr_mfr", form.get("manufacturing_year"))),
        "kms_run": number(form.get("kms_run", form.get("kms_driven"))),
        "total_owners": number(form.get("total_owners")) or 1,
        # New-car (ex-showroom) price: display only, NOT a model feature.
        "new_price": number(form.get("new_price", form.get("original_price"))),
        "asking_price": number(form.get("asking_price")),
        "had_accident": truthy(form.get("had_accident")),
    }


def validate(inp):
    errors = []
    this_year = datetime.now().year
    for field, label in [("make", "Make"), ("model", "Model"), ("fuel_type", "Fuel type"),
                         ("transmission", "Transmission")]:
        if not inp[field]:
            errors.append(f"{label} is required.")
    if inp["yr_mfr"] is None or not (MIN_YEAR <= inp["yr_mfr"] <= this_year):
        errors.append(f"Manufacturing year must be between {MIN_YEAR} and {this_year}.")
    if inp["kms_run"] is None or not (0 <= inp["kms_run"] <= 1_000_000):
        errors.append("Kilometers driven must be between 0 and 10,00,000.")
    if not (1 <= inp["total_owners"] <= 10):
        errors.append("Total owners must be between 1 and 10.")
    for field, label in [("new_price", "New car price"), ("asking_price", "Asking price")]:
        if inp[field] is not None and inp[field] <= 0:
            errors.append(f"{label} must be a positive number.")
    return errors


def reliability_notes(inp):
    """Tell the user when their input is outside what the model has seen."""
    notes = []
    if inp["make"] not in OPTIONS["make_models"]:
        notes.append(f"'{inp['make'].title()}' isn't in the training data, so this estimate is less reliable.")
    elif inp["model"] not in OPTIONS["make_models"][inp["make"]]:
        notes.append(f"The model '{inp['model'].title()}' isn't in the training data, so the estimate "
                     "relies on brand-level patterns and is less reliable.")
    if inp["city"] and inp["city"] not in OPTIONS["city"]:
        notes.append("Your city isn't in the training data; prices are based on the average across all cities.")
    age = datetime.now().year - inp["yr_mfr"]
    if age > MAX_CAR_AGE:
        notes.append(f"The training data only covers cars up to {MAX_CAR_AGE} years old.")
    notes.append(f"Prices are learned from listings dated {META['data_period']}; "
                 "today's market may be somewhat higher.")
    return notes


# ------------------------------------------------------------------
# Prediction helpers
# ------------------------------------------------------------------

def to_frame(inp, age):
    row = {col: (inp[col] or MISSING) for col in CATEGORICAL}
    row.update({
        "car_age": min(max(age, 0), MAX_CAR_AGE),
        "kms_run": inp["kms_run"],
        "total_owners": inp["total_owners"],
    })
    return pd.DataFrame([row], columns=FEATURES)


def estimate(inp, age):
    price = float(pipeline.predict(to_frame(inp, age))[0])
    if inp["had_accident"]:
        price *= 1 - ACCIDENT_ADJUSTMENT
    return max(price, 0.0)


def price_trend(inp, span=8):
    this_year = datetime.now().year
    years = range(this_year - span + 1, this_year + 1)
    frame = pd.concat([to_frame(inp, this_year - y) for y in years], ignore_index=True)
    prices = pipeline.predict(frame)
    if inp["had_accident"]:
        prices = prices * (1 - ACCIDENT_ADJUSTMENT)
    return [{"year": y, "price": round(float(p), -2)} for y, p in zip(years, prices)]


def similar_cars(inp, age, limit=5):
    """Real listings closest to this car. Matched on age AT THE TIME OF SALE
    (not manufacturing year), because the data is from 2019-2021."""
    pool = df_ref[(df_ref["make"] == inp["make"]) & (df_ref["model"] == inp["model"])]
    if len(pool) < 3:
        pool = df_ref[df_ref["make"] == inp["make"]]
    if pool.empty:
        return []
    score = (pool["car_age"] - age).abs() * 20_000 + (pool["kms_run"] - inp["kms_run"]).abs()
    return [
        {
            "make": r.make.title(), "model": r.model.title(), "variant": r.variant.upper(),
            "year": f"{int(r.car_age)} yrs", "kms": int(r.kms_run), "city": r.city.title(),
            "price": float(r.sale_price),
        }
        for r in pool.loc[score.nsmallest(limit).index].itertuples()
    ]


def verdict(asking, low, high, age):
    if age >= SCRAP_AGE_YEARS:
        return "SCRAP", "SCRAP"
    if asking is None:
        return "ESTIMATE", None
    if asking < low:
        return "UNDERPRICED", "BUY"
    if asking > high:
        return "OVERPRICED", "AVOID"
    return "FAIR", "NEGOTIATE"


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html", options=OPTIONS, meta=META,
                           this_year=datetime.now().year, min_year=MIN_YEAR)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": META["model_name"]})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        inp = parse_inputs(get_request_data())
        errors = validate(inp)
        if errors:
            return jsonify({"success": False, "error": " ".join(errors), "errors": errors}), 400

        this_year = datetime.now().year
        age = int(this_year - inp["yr_mfr"])

        price = estimate(inp, age)
        low, high = price * BAND_LOW, price * BAND_HIGH
        status, recommendation = verdict(inp["asking_price"], low, high, age)

        scrap_message = None
        if age >= SCRAP_AGE_YEARS:
            scrap_message = (
                f"This vehicle is {age} years old. Many cities restrict older vehicles, and resale value "
                "at this age depends mostly on condition. Check your state's RTO rules and consider an "
                "authorised scrapping facility (RVSF), which can issue a certificate with benefits on "
                "your next purchase."
            )

        notes = reliability_notes(inp)
        if inp["had_accident"]:
            notes.insert(0, f"Reduced by {int(ACCIDENT_ADJUSTMENT * 100)}% for accident history "
                            "(a rule of thumb; the dataset has no accident information).")

        new_price = inp["new_price"]
        return jsonify({
            "success": True,
            "predicted_price": round(price, -2),
            "fair_price": round(price, -2),
            "fair_price_low": round(low, -2),
            "fair_price_high": round(high, -2),
            "band_coverage_pct": META["error_band"]["coverage_pct"],
            "model_mae": MODEL_MAE,
            "depreciation": round((1 - price / new_price) * 100, 1) if new_price else None,
            "status": status,
            "valuation_status": status,
            "recommendation": recommendation,
            "asking_vs_estimate": round(inp["asking_price"] / price, 3) if inp["asking_price"] else None,
            "scrap_recommended": scrap_message is not None,
            "scrap_message": scrap_message,
            "notes": notes,
            "advice": GENERAL_ADVICE,
            "feature_importance": FEATURE_IMPORTANCE,
            "similar_cars": similar_cars(inp, age),
            "price_trend": price_trend(inp),
            "summary": {
                "make": inp["make"], "model": inp["model"], "variant": inp["variant"],
                "manufacturing_year": int(inp["yr_mfr"]), "kms_driven": inp["kms_run"],
                "total_owners": inp["total_owners"], "city": inp["city"],
                "new_price": new_price, "asking_price": inp["asking_price"],
                "had_accident": inp["had_accident"],
            },
        })

    except Exception:
        traceback.print_exc()
        return jsonify({"success": False, "error": "Something went wrong on our side. Please try again."}), 500


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
