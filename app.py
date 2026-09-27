"""
Smart Car Valuator - Flask app.

Routes
    GET  /          the website
    GET  /health    health check for Render
    POST /predict   valuation JSON (used by the valuation form and Compare Cars)
    POST /report    the same valuation as a downloadable PDF

The ML model (models/car_price_pipeline.joblib, built by train.py) contains
all preprocessing, so this file only turns the form into a one-row
DataFrame. Everything that is NOT the model - the verdict, recommendation,
health check score and future projection - is plain, documented logic in
this file so it can be explained and tested on its own.
"""

import json
import os
import sys
from datetime import datetime

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file

# ------------------------------------------------------------------
# Paths (work whether app.py is in the repo root or in app/)
# ------------------------------------------------------------------

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join("models", "car_price_pipeline.joblib")
META_FILE = os.path.join("models", "model_metadata.json")


def find_project_root():
    candidates = [APP_DIR, os.path.dirname(APP_DIR), os.getcwd()]
    for folder in candidates:
        if os.path.isfile(os.path.join(folder, MODEL_FILE)):
            return folder
    raise FileNotFoundError(
        f"Could not find {MODEL_FILE}. Looked in: " + ", ".join(candidates)
        + ". Make sure the models/ folder is committed to the repository."
    )


def find_dir(root, name, must_contain):
    for folder in [os.path.join(APP_DIR, name), os.path.join(root, "app", name), os.path.join(root, name)]:
        if os.path.isfile(os.path.join(folder, must_contain)):
            return folder
    raise FileNotFoundError(f"Could not find {name}/{must_contain} next to app.py.")


PROJECT_ROOT = find_project_root()
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from report import build_report_pdf  # noqa: E402
from train import CATEGORICAL, FEATURES, MISSING, load_and_clean  # noqa: E402

app = Flask(
    __name__,
    template_folder=find_dir(PROJECT_ROOT, "templates", "index.html"),
    static_folder=find_dir(PROJECT_ROOT, "static", "style.css"),
)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024  # form posts are tiny

# ------------------------------------------------------------------
# Load model, metadata and reference data ONCE at startup
# ------------------------------------------------------------------

pipeline = joblib.load(os.path.join(PROJECT_ROOT, MODEL_FILE))
with open(os.path.join(PROJECT_ROOT, META_FILE)) as f:
    META = json.load(f)

OPTIONS = META["options"]
MAX_CAR_AGE = OPTIONS["max_car_age"]
BAND_LOW = META["error_band"]["low_ratio"]      # e.g. 0.86
BAND_HIGH = META["error_band"]["high_ratio"]    # e.g. 1.15

df_ref = load_and_clean()                         # used only for "similar cars"
df_ref["yr_mfr"] = df_ref["yr_mfr"].astype(int)

print(f"Project root: {PROJECT_ROOT}")
print(f"Loaded {META['model_name']} | test R2 {META['test_metrics']['r2']} "
      f"| MAE Rs {META['test_metrics']['mae']:,.0f} | {len(df_ref)} reference listings")

# ------------------------------------------------------------------
# Constants and plain-logic rules (NOT part of the ML model)
# ------------------------------------------------------------------

MIN_YEAR = 1990
SCRAP_AGE_YEARS = 15
FUTURE_YEARS = [1, 2, 3, 5]
DEFAULT_KM_PER_YEAR = 10_000

FEATURE_LABELS = {
    "make": "Brand", "model": "Model", "variant": "Variant",
    "fuel_type": "Fuel type", "transmission": "Transmission",
    "body_type": "Body type", "city": "City", "car_age": "Vehicle age",
    "kms_run": "Kilometers driven", "total_owners": "Number of owners",
}
FEATURE_IMPORTANCE = [
    {"feature": FEATURE_LABELS.get(d["feature"], d["feature"]), "importance": d["importance"]}
    for d in META["feature_importance"]
]

# Quick Vehicle Health Check: a user-filled checklist, scored out of 100.
# It never changes the ML valuation.
HEALTH_CHECK = [
    {"key": "engine", "label": "Engine condition", "options": [
        ("smooth", "Starts easily, runs smoothly", 20), ("minor", "Minor noises or leaks", 10),
        ("major", "Major issues / warning lights", 0)]},
    {"key": "exterior", "label": "Exterior condition", "options": [
        ("good", "Clean, no visible damage", 10), ("minor", "Minor scratches or dents", 6),
        ("poor", "Rust, major dents or repaint", 0)]},
    {"key": "interior", "label": "Interior condition", "options": [
        ("good", "Clean, everything works", 10), ("fair", "Some wear or small faults", 6),
        ("poor", "Heavy wear or broken parts", 0)]},
    {"key": "tyres", "label": "Tyre condition", "options": [
        ("good", "Good tread, even wear", 10), ("fair", "Usable, replace soon", 5),
        ("poor", "Worn out or uneven", 0)]},
    {"key": "accident", "label": "Accident history", "options": [
        ("none", "No accidents", 15), ("minor", "Minor, repaired", 8),
        ("major", "Major / structural", 0)]},
    {"key": "service", "label": "Service history", "options": [
        ("full", "Full records available", 15), ("partial", "Partial records", 8),
        ("none", "No records", 0)]},
    {"key": "insurance", "label": "Insurance validity", "options": [
        ("comprehensive", "Valid comprehensive", 10), ("third_party", "Valid third-party only", 6),
        ("expired", "Expired", 0)]},
    {"key": "documents", "label": "RC and documents", "options": [
        ("complete", "RC and papers complete", 10), ("pending", "Transfer or NOC pending", 4),
        ("missing", "Missing documents", 0)]},
]


def score_health(answers):
    """Return None if nothing answered, else score/100 for the answered items."""
    earned = possible = answered = 0
    for item in HEALTH_CHECK:
        choice = (answers or {}).get(item["key"])
        points = {k: p for k, _, p in item["options"]}
        if choice in points:
            answered += 1
            earned += points[choice]
            possible += max(points.values())
    if not answered:
        return None
    score = round(earned / possible * 100)
    band = "Excellent" if score >= 80 else "Good" if score >= 60 else "Needs attention"
    return {"score": score, "band": band, "answered": answered, "total": len(HEALTH_CHECK)}


# ------------------------------------------------------------------
# Input handling and validation
# ------------------------------------------------------------------

class ValidationError(Exception):
    def __init__(self, errors):
        super().__init__(" ".join(errors))
        self.errors = errors


def text(value):
    return str(value).strip().lower() if value not in (None, "") else ""


def parse_number(form, key, label, errors, required=False):
    raw = form.get(key)
    if raw in (None, ""):
        if required:
            errors.append(f"{label} is required.")
        return None
    try:
        value = float(str(raw).replace(",", "").strip())
    except ValueError:
        errors.append(f"{label} must be a number.")
        return None
    if value != value or value in (float("inf"), float("-inf")):
        errors.append(f"{label} must be a number.")
        return None
    return value


def parse_and_validate(form):
    errors = []
    this_year = datetime.now().year
    inp = {k: text(form.get(k)) for k in CATEGORICAL}

    for key, label in [("make", "Brand"), ("model", "Model"), ("fuel_type", "Fuel type"),
                       ("transmission", "Transmission")]:
        if not inp[key]:
            errors.append(f"{label} is required.")

    # Categorical values must be ones the model knows (dropdowns guarantee
    # this; the check protects the API from hand-written requests).
    if inp["make"] and inp["make"] not in OPTIONS["make_models"]:
        errors.append(f"Brand '{inp['make']}' is not supported.")
    for key, label in [("fuel_type", "Fuel type"), ("transmission", "Transmission"),
                       ("body_type", "Body type"), ("city", "City")]:
        if inp[key] and inp[key] not in OPTIONS[key]:
            if key == "city":
                inp[key] = ""          # unknown city -> model uses the all-city average
            else:
                errors.append(f"{label} '{inp[key]}' is not supported.")

    yr = parse_number(form, "yr_mfr", "Manufacturing year", errors, required=True)
    if yr is not None and not (MIN_YEAR <= yr <= this_year and yr == int(yr)):
        errors.append(f"Manufacturing year must be a whole year between {MIN_YEAR} and {this_year}.")
    kms = parse_number(form, "kms_run", "Kilometers driven", errors, required=True)
    if kms is not None and not (0 <= kms <= 1_000_000):
        errors.append("Kilometers driven must be between 0 and 10,00,000.")
    owners = parse_number(form, "total_owners", "Total owners", errors) or 1
    if not (1 <= owners <= 10):
        errors.append("Total owners must be between 1 and 10.")

    prices = {}
    for key, label in [("asking_price", "Asking price"), ("new_price", "Original price")]:
        value = parse_number(form, key, label, errors)
        if value is not None and not (10_000 <= value <= 50_000_000):
            errors.append(f"{label} must be between ₹10,000 and ₹5,00,00,000.")
        prices[key] = value

    if errors:
        raise ValidationError(errors)

    inp.update({"yr_mfr": int(yr), "kms_run": kms, "total_owners": int(owners), **prices})
    return inp


def reliability_notes(inp, age):
    notes = []
    if inp["model"] not in OPTIONS["make_models"][inp["make"]]:
        notes.append(f"The model '{inp['model'].title()}' isn't in the training data, so this "
                     "estimate relies on brand-level patterns and is less reliable.")
    if not inp["city"]:
        notes.append("City not in the data - the estimate uses the average across all cities.")
    if age > MAX_CAR_AGE:
        notes.append(f"The training data only covers cars up to {MAX_CAR_AGE} years old.")
    notes.append(f"Learned from listings dated {META['data_period']}; today's prices may be somewhat higher.")
    return notes


# ------------------------------------------------------------------
# Model calls
# ------------------------------------------------------------------

def rows_for(inp, scenarios):
    """scenarios: list of (car_age, kms_run) -> DataFrame with the model's raw columns."""
    base = {col: (inp[col] or MISSING) for col in CATEGORICAL}
    rows = [{**base, "car_age": min(max(a, 0), MAX_CAR_AGE), "kms_run": k,
             "total_owners": inp["total_owners"]} for a, k in scenarios]
    return pd.DataFrame(rows, columns=FEATURES)


def run_valuation(inp):
    this_year = datetime.now().year
    age = this_year - inp["yr_mfr"]
    km_per_year = min(max(inp["kms_run"] / age, 5_000), 25_000) if age >= 1 else DEFAULT_KM_PER_YEAR

    trend_years = list(range(this_year - 7, this_year + 1))
    scenarios = (
        [(age, inp["kms_run"])]                                                        # the car today
        + [(this_year - y, inp["kms_run"]) for y in trend_years]                       # by mfg year
        + [(age + n, inp["kms_run"] + n * km_per_year) for n in FUTURE_YEARS]          # future
    )
    preds = pipeline.predict(rows_for(inp, scenarios))   # ONE model call per request
    price = float(preds[0])
    trend = [{"year": y, "price": round(float(p), -2)} for y, p in zip(trend_years, preds[1:9])]
    future = [
        {"years": n, "age": age + n, "kms": round(inp["kms_run"] + n * km_per_year, -3),
         "price": round(float(p), -2), "change_pct": round((float(p) / price - 1) * 100, 1)}
        for n, p in zip(FUTURE_YEARS, preds[9:])
    ]
    return age, price, trend, future, km_per_year


def price_difference(reference, price):
    """Signed difference of the market value vs. a reference price."""
    if not reference:
        return None
    diff = price - reference
    pct = diff / reference * 100
    return {
        "amount": round(diff, -2),
        "pct": round(pct, 1),
        "direction": "above" if diff > 0 else "below" if diff < 0 else "equal",
    }


def verdict(inp, price, low, high, age):
    """Status + recommendation. Status needs an asking price to compare with."""
    asking = inp["asking_price"]
    if age >= SCRAP_AGE_YEARS:
        rec = ("VERIFY VEHICLE CONDITION",
               f"At {age} years old, value depends mostly on condition and local RTO rules. "
               "Get a thorough inspection before paying anything.")
    elif asking is None:
        rec = ("COMPARE BEFORE BUYING",
               "Add the seller's asking price to see whether it is fair. Meanwhile, use the "
               "estimated range as your reference when comparing listings.")
    elif asking < low * 0.85:
        rec = ("VERIFY VEHICLE CONDITION",
               "The asking price is far below the estimated range. Deals this cheap can hide "
               "accident damage or paperwork problems - inspect carefully.")
    elif asking < low:
        rec = ("GOOD VALUE",
               "The asking price is below the estimated market range. Confirm the car's condition "
               "and documents, then it looks like a good deal.")
    elif asking <= price:
        rec = ("GOOD VALUE",
               "The asking price is within the estimated range and at or below the estimate.")
    elif asking <= high:
        rec = ("NEGOTIATE",
               "The asking price is within the range but above the estimate, so there is room "
               f"to negotiate towards about ₹{price:,.0f}.")
    else:
        rec = ("NEGOTIATE",
               "The asking price is above the estimated market value. Compare similar vehicles "
               "and negotiate, or keep looking.")

    if asking is None:
        status = None
    elif asking < low:
        status = "UNDERPRICED"
    elif asking > high:
        status = "OVERPRICED"
    else:
        status = "FAIR VALUE"
    return status, {"code": rec[0], "reason": rec[1]}


def similar_cars(inp, age, limit=5):
    pool = df_ref[(df_ref["make"] == inp["make"]) & (df_ref["model"] == inp["model"])]
    if len(pool) < 3:
        pool = df_ref[df_ref["make"] == inp["make"]]
    if pool.empty:
        return []
    score = (pool["car_age"] - age).abs() * 20_000 + (pool["kms_run"] - inp["kms_run"]).abs()
    return [
        {"name": f"{r.make.title()} {r.model.title()}", "variant": r.variant.upper(),
         "age": int(r.car_age), "kms": int(r.kms_run), "city": r.city.title(),
         "price": float(r.sale_price)}
        for r in pool.loc[score.nsmallest(limit).index].itertuples()
    ]


def build_result(inp):
    age, price, trend, future, km_per_year = run_valuation(inp)
    low, high = price * BAND_LOW, price * BAND_HIGH
    status, recommendation = verdict(inp, price, low, high, age)

    scrap_message = None
    if age >= SCRAP_AGE_YEARS:
        scrap_message = (f"This vehicle is {age} years old. Many cities restrict older vehicles. "
                         "Check your state's RTO rules; an authorised scrapping facility (RVSF) "
                         "can issue a certificate with benefits on your next purchase.")

    return {
        "success": True,
        "predicted_price": round(price, -2),
        "range_low": round(low, -2),
        "range_high": round(high, -2),
        "range_coverage_pct": META["error_band"]["coverage_pct"],
        "status": status,
        "recommendation": recommendation,
        "vs_asking": price_difference(inp["asking_price"], price),
        "vs_new": price_difference(inp["new_price"], price),
        "vehicle_age": age,
        "km_per_year": round(km_per_year, -2),
        "scrap_message": scrap_message,
        "notes": reliability_notes(inp, age),
        "feature_importance": FEATURE_IMPORTANCE,
        "similar_cars": similar_cars(inp, age),
        "price_trend": trend,
        "future_values": future,
        "car": {
            "make": inp["make"].title(), "model": inp["model"].title(),
            "variant": inp["variant"].upper() if inp["variant"] else "",
            "yr_mfr": inp["yr_mfr"], "kms_run": inp["kms_run"], "fuel_type": inp["fuel_type"].title(),
            "transmission": inp["transmission"].title(), "body_type": inp["body_type"].title(),
            "city": inp["city"].title() if inp["city"] else "Other",
            "total_owners": inp["total_owners"],
            "asking_price": inp["asking_price"], "new_price": inp["new_price"],
        },
    }


def request_data():
    if request.is_json:
        return request.get_json(silent=True) or {}
    return request.form.to_dict()


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@app.route("/")
def index():
    return render_template(
        "index.html", options=OPTIONS, meta=META, health_check=HEALTH_CHECK,
        this_year=datetime.now().year, min_year=MIN_YEAR,
        n_features=len(META["features"]),
    )


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": META["model_name"]})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        return jsonify(build_result(parse_and_validate(request_data())))
    except ValidationError as e:
        return jsonify({"success": False, "error": str(e), "errors": e.errors}), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"success": False, "error": "The valuation could not be calculated. Please try again."}), 500


@app.route("/report", methods=["POST"])
def report():
    """Recalculate the valuation server-side (never trust client numbers) and return a PDF."""
    try:
        data = request_data()
        inp = parse_and_validate(data)
        result = build_result(inp)
        health = score_health(data.get("health") if isinstance(data.get("health"), dict) else {})
        pdf = build_report_pdf(result, health, META, HEALTH_CHECK, data.get("health") or {})
        name = f"valuation-{inp['make']}-{inp['model']}-{datetime.now():%Y%m%d}.pdf".replace(" ", "-")
        return send_file(pdf, mimetype="application/pdf", as_attachment=True, download_name=name)
    except ValidationError as e:
        return jsonify({"success": False, "error": str(e), "errors": e.errors}), 400
    except Exception:
        app.logger.exception("Report failed")
        return jsonify({"success": False, "error": "The report could not be created. Please try again."}), 500


@app.errorhandler(404)
def not_found(_):
    if request.path.startswith(("/predict", "/report")) or request.is_json:
        return jsonify({"success": False, "error": "Not found."}), 404
    return render_template("index.html", options=OPTIONS, meta=META, health_check=HEALTH_CHECK,
                           this_year=datetime.now().year, min_year=MIN_YEAR,
                           n_features=len(META["features"])), 404


@app.errorhandler(413)
def too_large(_):
    return jsonify({"success": False, "error": "Request too large."}), 413


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
