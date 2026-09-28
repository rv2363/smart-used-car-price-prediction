"""Server-side PDF valuation report (ReportLab).

Every number comes from the /predict result (model output, dataset listings,
user input or plain arithmetic). Sections are wrapped in KeepTogether so a
heading is never separated from its content.
"""

from datetime import datetime
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

VIOLET = colors.HexColor("#5B2EE0")
INK = colors.HexColor("#1E1433")
MUTED = colors.HexColor("#6B6485")
LINE = colors.HexColor("#E4E0F0")
TINT = colors.HexColor("#F3EFFF")
STATUS_COLORS = {"UNDERPRICED": "#127A4B", "FAIR VALUE": "#5B2EE0", "OVERPRICED": "#C0352B"}

PAGE_W = A4[0] - 32 * mm          # usable width
HALF = (PAGE_W - 6 * mm) / 2      # two-column width

DISCLAIMER = (
    "This valuation is an estimated market value generated from historical data and machine "
    "learning. It is not a guaranteed selling or buying price. Actual value may vary based on "
    "vehicle condition, location, market demand, service history, accident history and other factors."
)


def inr(value):
    """Indian digit grouping: 1234567 -> Rs. 12,34,567 (built-in PDF fonts have no rupee glyph)."""
    if value is None:
        return "-"
    s = str(int(round(abs(value))))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        s = ",".join(groups) + "," + tail
    return ("-" if value < 0 else "") + "Rs. " + s


S = {
    "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=17, leading=20, textColor=VIOLET),
    "sub": ParagraphStyle("sub", fontName="Helvetica", fontSize=8.5, leading=11, textColor=MUTED),
    "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=10.5, leading=13, textColor=INK,
                         spaceBefore=6, spaceAfter=2),
    "body": ParagraphStyle("body", fontName="Helvetica", fontSize=9, leading=12, textColor=INK),
    "small": ParagraphStyle("small", fontName="Helvetica", fontSize=7.5, leading=9.5, textColor=MUTED),
    "label": ParagraphStyle("label", fontName="Helvetica", fontSize=7.5, leading=9.5, textColor=MUTED),
    "big": ParagraphStyle("big", fontName="Helvetica-Bold", fontSize=20, leading=23, textColor=INK),
    "cell": ParagraphStyle("cell", fontName="Helvetica-Bold", fontSize=8.5, leading=10.5, textColor=INK),
}


def kv(rows, width):
    """Compact two-column label/value table."""
    t = Table([[Paragraph(k, S["label"]), Paragraph(str(v), S["cell"])] for k, v in rows],
              colWidths=(width * 0.40, width * 0.60), hAlign="LEFT")
    t.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 1.8), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.8),
        ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def kv_pairs(rows, width):
    """Label/value pairs laid out two per line, for short lists like the key factors."""
    rows = rows + [["", ""]] * (len(rows) % 2)
    half = len(rows) // 2
    data = [[Paragraph(a[0], S["label"]), Paragraph(a[1], S["cell"]),
             Paragraph(b[0], S["label"]), Paragraph(b[1], S["cell"])]
            for a, b in zip(rows[:half], rows[half:])]
    w = width / 2
    t = Table(data, colWidths=(w * 0.64, w * 0.36, w * 0.64, w * 0.36), hAlign="LEFT")
    t.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (1, -1), 0.4, LINE), ("LINEBELOW", (2, 0), (3, -1), 0.4, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 1.8), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.8),
        ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (1, 0), (1, -1), 8),
    ]))
    return t


def section(title, *flowables):
    return KeepTogether([Paragraph(title, S["h2"]), *flowables])


def column(title, *flowables):
    """A section used inside a side-by-side row (the row itself never splits)."""
    return [Paragraph(title, S["h2"]), *flowables]


def side_by_side(left, right):
    t = Table([[left, right]], colWidths=(HALF + 6 * mm, HALF), hAlign="LEFT")
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (0, 0), 6 * mm),
        ("RIGHTPADDING", (1, 0), (1, 0), 0), ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    return t


def diff_line(diff, name):
    """Estimated price difference = (predicted - reference) / reference x 100."""
    if not diff:
        return "Add an asking price to compare" if name == "asking price" else "Not provided"
    if diff["direction"] == "equal":
        return f"Same as {name}"
    sign = "+" if diff["pct"] > 0 else "-"
    return f"{sign}{abs(diff['pct']):.1f}% {diff['direction']} {name}<br/>({sign}{inr(abs(diff['amount']))})"


def build_report_pdf(result, health, meta, health_items, health_answers):
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm,
                            topMargin=11 * mm, bottomMargin=10 * mm,
                            title="Smart Car Valuator - Valuation Report")
    car = result["car"]
    story = []

    # ---- Header
    story += [
        Paragraph("Smart Car Valuator", S["title"]),
        Paragraph(f"Historical data-based valuation report &nbsp;|&nbsp; Generated {datetime.now():%d %b %Y, %I:%M %p}",
                  S["sub"]),
        Spacer(1, 6),
    ]

    # ---- Headline card: value + range | status + recommendation
    status = result["status"]
    status_html = (f"<font color='{STATUS_COLORS[status]}'><b>{status}</b></font>" if status
                   else "<b>Add an asking price for a verdict</b>")
    left = [
        Paragraph("Estimated market value (historical data-based estimate)", S["label"]),
        Paragraph(inr(result["predicted_price"]), S["big"]),
        Paragraph(f"Estimated market range: {inr(result['range_low'])} - {inr(result['range_high'])}", S["body"]),
        Paragraph("This is an approximate valuation range around the model prediction, "
                  "not a statistical confidence interval.", S["small"]),
    ]
    right = [
        Paragraph("Valuation status", S["label"]), Paragraph(status_html, S["body"]),
        Paragraph(result["status_explanation"], S["small"]), Spacer(1, 3),
        Paragraph("Recommendation", S["label"]),
        Paragraph(f"<b>{result['recommendation']['code']}</b>", S["body"]),
        Paragraph(result["recommendation"]["reason"].replace("₹", "Rs. "), S["small"]),
    ]
    card = Table([[left, right]], colWidths=(PAGE_W * 0.56, PAGE_W * 0.44), hAlign="LEFT")
    card.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), TINT), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(card)

    # ---- Vehicle details | Price analysis
    vehicle = column("Vehicle details", kv([
        ["Car", f"{car['make']} {car['model']} {car['variant']}".strip()],
        ["Manufacturing year", f"{car['yr_mfr']} ({result['vehicle_age']} years old)"],
        ["Kilometers driven", f"{int(car['kms_run']):,} km"],
        ["Fuel / transmission", f"{car['fuel_type']} / {car['transmission']}"],
        ["Body type", car["body_type"] or "Not specified"],
        ["Owners / city", f"{car['total_owners']} / {car['city']}"],
    ], HALF))
    price = column("Price analysis", kv([
        ["Estimated market value", inr(result["predicted_price"])],
        ["Estimated market range", f"{inr(result['range_low'])} - {inr(result['range_high'])}"],
        ["Seller's asking price", inr(car["asking_price"]) if car["asking_price"] else "Not provided"],
        ["Price difference vs asking", diff_line(result["vs_asking"], "asking price")],
        ["Original (new) price", inr(car["new_price"]) if car["new_price"] else "Not provided"],
        ["Price difference vs original", diff_line(result["vs_new"], "original price")],
    ], HALF))
    story.append(side_by_side(vehicle, price))

    # ---- Market position (compact, after price analysis)
    mp = result["market_position"]
    if mp["available"]:
        cells = [
            [Paragraph("Estimated market value", S["label"]), Paragraph("Historical comparable range", S["label"]),
             Paragraph("Comparable listings", S["label"])],
            [Paragraph(inr(mp["prediction"]), S["cell"]),
             Paragraph(f"{inr(mp['low'])} - {inr(mp['high'])}", S["cell"]),
             Paragraph(str(mp["count"]), S["cell"])],
        ]
        mpt = Table(cells, colWidths=(PAGE_W / 3,) * 3, hAlign="LEFT")
        mpt.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8F7FC")),
            ("LEFTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(section("Market position", mpt))
    else:
        story.append(section("Market position", Paragraph(mp["message"], S["body"])))

    # ---- Similar historical listings
    if result["similar_listings"]:
        rows = [["Car", "Age when sold", "KM driven", "City", "Listed price"]] + [
            [f"{l['name']} {l['variant']}", f"{l['age']} yrs", f"{l['kms']:,} km", l["city"], inr(l["price"])]
            for l in result["similar_listings"]
        ]
        t = Table(rows, colWidths=(PAGE_W * 0.36, PAGE_W * 0.14, PAGE_W * 0.16, PAGE_W * 0.16, PAGE_W * 0.18),
                  hAlign="LEFT")
        t.setStyle(TableStyle([
            ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 7.5), ("FONT", (0, 1), (-1, -1), "Helvetica", 8),
            ("TEXTCOLOR", (0, 0), (-1, 0), MUTED), ("ALIGN", (-1, 0), (-1, -1), "RIGHT"),
            ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE),
            ("TOPPADDING", (0, 0), (-1, -1), 1.8), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.8),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ]))
        story.append(section(
            "Similar historical listings",
            Paragraph("Similar historical listings from the project dataset.", S["small"]),
            Spacer(1, 2), t, Spacer(1, 2),
            Paragraph(f"These listings are from the historical dataset ({result['data_period']}) and may not "
                      "represent current market prices.", S["small"]),
            Spacer(1, 4),
        ))

    # ---- Key factors | Future projection
    factors = column(
        "Key factors considered by the model",
        Paragraph("These factors describe the model's overall behavior and are not the exact contribution "
                  "of each factor to this individual prediction.", S["small"]),
        Spacer(1, 2),
        kv_pairs([[f["feature"], f"{f['importance']:.1f}%"] for f in result["feature_importance"]], HALF),
        Spacer(1, 2),
        Paragraph("Source: permutation importance on held-out test data (share of total).", S["small"]),
    )
    future = column(
        "Illustrative future value projection",
        Paragraph("This is an illustrative projection based on the current valuation model and assumed future "
                  "usage. It is not a separately trained resale-price forecasting model and is not a guaranteed "
                  "future resale value.", S["small"]),
        Spacer(1, 2),
        kv([[f"In {f['years']} year{'s' if f['years'] > 1 else ''} (~{int(f['kms']):,} km)",
             f"{inr(f['price'])} ({f['change_pct']:+.1f}%)"] for f in result["future_values"]], HALF),
        Spacer(1, 2),
        Paragraph(f"Assumed usage: about {int(result['km_per_year']):,} km a year.", S["small"]),
    )
    # With a health check the report can't fit on one page, so split it into two balanced
    # pages: valuation (page 1) and analysis (page 2), instead of a nearly empty page 2.
    if health:
        story.append(PageBreak())
    story.append(side_by_side(factors, future))

    # ---- Health check (optional)
    if health:
        rows = []
        for item in health_items:
            choice = health_answers.get(item["key"])
            rows.append([item["label"], next((lbl for k, lbl, _ in item["options"] if k == choice), "Not answered")])
        half = (len(rows) + 1) // 2
        story.append(section(
            f"Quick vehicle health check: {health['score']}/100 ({health['band']})",
            side_by_side(kv(rows[:half], HALF), kv(rows[half:], HALF)),
            Spacer(1, 2),
            Paragraph("User-provided checklist, not an inspection by this tool. "
                      "Always verify the vehicle physically before purchase.", S["small"]),
        ))

    # ---- Model information + notes
    notes = ([result["scrap_message"]] if result["scrap_message"] else []) + result["notes"]
    tm = meta["test_metrics"]
    labels = ["Algorithm", "Training data", "Model inputs", "Test R-squared", "Mean absolute error"]
    values = [meta["model_name"], f"{meta['n_rows']:,} listings<br/>{meta['data_period']}",
              f"{len(meta['features'])} vehicle details", f"{tm['r2']:.3f}",
              f"{inr(tm['mae'])}<br/>({tm['mape_pct']:.1f}% on average)"]
    strip = Table([[Paragraph(l, S["label"]) for l in labels], [Paragraph(v, S["cell"]) for v in values]],
                  colWidths=(PAGE_W * 0.24, PAGE_W * 0.24, PAGE_W * 0.16, PAGE_W * 0.14, PAGE_W * 0.22),
                  hAlign="LEFT")
    strip.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8F7FC")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6), ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(section(
        "Model information", strip, Spacer(1, 2),
        Paragraph("Model performance is based on the project's held-out test dataset. "
                  + " ".join(notes), S["small"]),
    ))

    # ---- Disclaimer (always last)
    story += [Spacer(1, 6), Paragraph("<b>Disclaimer.</b> " + DISCLAIMER, S["small"])]

    doc.build(story)
    buf.seek(0)
    return buf
