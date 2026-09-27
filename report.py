"""Server-side PDF valuation report (ReportLab)."""

from datetime import datetime
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

VIOLET = colors.HexColor("#5B2EE0")
INK = colors.HexColor("#1E1433")
MUTED = colors.HexColor("#6B6485")
LINE = colors.HexColor("#E4E0F0")
TINT = colors.HexColor("#F3EFFF")
STATUS_COLORS = {"UNDERPRICED": "#127A4B", "FAIR VALUE": "#5B2EE0", "OVERPRICED": "#C0352B"}

DISCLAIMER = (
    "This valuation is an estimated market value generated from historical data and machine "
    "learning. It is not a guaranteed selling or buying price. Actual value may vary based on "
    "vehicle condition, location, market demand, service history, accident history and other factors."
)


def inr(value):
    """Indian digit grouping: 1234567 -> Rs. 12,34,567 (built-in PDF fonts have no rupee glyph)."""
    if value is None:
        return "-"
    n = int(round(abs(value)))
    s = str(n)
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


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=base["Title"], fontName="Helvetica-Bold", fontSize=20,
                                textColor=VIOLET, alignment=TA_LEFT, spaceAfter=2),
        "sub": ParagraphStyle("s", parent=base["Normal"], fontSize=9, textColor=MUTED),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=12,
                             textColor=INK, spaceBefore=10, spaceAfter=5),
        "body": ParagraphStyle("b", parent=base["Normal"], fontSize=9.5, leading=13.5, textColor=INK),
        "small": ParagraphStyle("sm", parent=base["Normal"], fontSize=8, leading=11, textColor=MUTED),
        "big": ParagraphStyle("big", parent=base["Normal"], fontName="Helvetica-Bold", fontSize=22,
                              leading=26, textColor=INK),
    }


def _kv_table(rows, col_widths=(55 * mm, 115 * mm)):
    t = Table(rows, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("FONT", (0, 0), (0, -1), "Helvetica", 9),
        ("FONT", (1, 0), (1, -1), "Helvetica-Bold", 9),
        ("TEXTCOLOR", (0, 0), (0, -1), MUTED),
        ("TEXTCOLOR", (1, 0), (1, -1), INK),
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, LINE),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def build_report_pdf(result, health, meta, health_items, health_answers):
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm,
                            title="Smart Car Valuator - Valuation Report")
    st = _styles()
    car = result["car"]
    story = []

    story += [
        Paragraph("Smart Car Valuator", st["title"]),
        Paragraph(f"AI-powered used car valuation report &nbsp;|&nbsp; Generated {datetime.now():%d %b %Y, %I:%M %p}",
                  st["sub"]),
        Spacer(1, 8),
    ]

    # Headline value
    headline = Table([[
        [Paragraph("Estimated market value", st["small"]), Paragraph(inr(result["predicted_price"]), st["big"]),
         Paragraph(f"Estimated market range: {inr(result['range_low'])} - {inr(result['range_high'])}", st["body"])],
        [Paragraph("Valuation status", st["small"]),
         Paragraph(f"<font color='{STATUS_COLORS.get(result['status'], '#6B6485')}'><b>"
                   f"{result['status'] or 'No asking price given'}</b></font>", st["body"]),
         Spacer(1, 4),
         Paragraph("Recommendation", st["small"]),
         Paragraph(f"<b>{result['recommendation']['code']}</b>", st["body"])],
    ]], colWidths=(105 * mm, 65 * mm))
    headline.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), TINT),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 9), ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    story += [headline, Spacer(1, 4), Paragraph(result["recommendation"]["reason"], st["body"])]

    # Vehicle details
    story.append(Paragraph("Vehicle details", st["h2"]))
    story.append(_kv_table([
        ["Car", f"{car['make']} {car['model']} {car['variant']}".strip()],
        ["Manufacturing year", f"{car['yr_mfr']} ({result['vehicle_age']} years old)"],
        ["Kilometers driven", f"{int(car['kms_run']):,} km"],
        ["Fuel / transmission", f"{car['fuel_type']} / {car['transmission']}"],
        ["Body type", car["body_type"] or "Not specified"],
        ["Owners", str(car["total_owners"])],
        ["City", car["city"]],
    ]))

    # Price analysis
    def diff_line(diff, name):
        if not diff:
            return "-"
        if diff["direction"] == "equal":
            return f"Same as {name}"
        sign = "+" if diff["amount"] > 0 else "-"
        return f"{sign}{inr(abs(diff['amount']))} ({abs(diff['pct']):.1f}% {diff['direction']} {name})"

    story.append(Paragraph("Price analysis", st["h2"]))
    story.append(_kv_table([
        ["Estimated market value", inr(result["predicted_price"])],
        ["Estimated market range", f"{inr(result['range_low'])} - {inr(result['range_high'])}"],
        ["Seller's asking price", inr(car["asking_price"]) if car["asking_price"] else "Not provided"],
        ["Market value vs asking price", diff_line(result["vs_asking"], "asking price")],
        ["Original (new) price", inr(car["new_price"]) if car["new_price"] else "Not provided"],
        ["Market value vs original price", diff_line(result["vs_new"], "original price")],
    ]))

    # Key factors
    story.append(Paragraph("What the model weighs most", st["h2"]))
    story.append(Paragraph("Share of the model's accuracy that depends on each input, measured on unseen "
                           "test data (permutation importance). It describes the model overall, not this car alone.",
                           st["small"]))
    story.append(Spacer(1, 3))
    story.append(_kv_table([[f["feature"], f"{f['importance']:.1f}%"] for f in result["feature_importance"][:7]]))

    # Future values
    story.append(Paragraph("Illustrative future value projection", st["h2"]))
    story.append(_kv_table(
        [[f"In {f['years']} year{'s' if f['years'] > 1 else ''} (~{int(f['kms']):,} km)",
          f"{inr(f['price'])} ({f['change_pct']:+.1f}%)"] for f in result["future_values"]]
    ))
    story.append(Paragraph("Assumes the car keeps being driven about "
                           f"{int(result['km_per_year']):,} km a year, at the same market price level. "
                           "This is an illustrative projection, not a guaranteed resale value.", st["small"]))

    # Health check
    if health:
        story.append(Paragraph(f"Quick vehicle health check: {health['score']}/100 ({health['band']})", st["h2"]))
        rows = []
        for item in health_items:
            choice = health_answers.get(item["key"])
            label = next((lbl for k, lbl, _ in item["options"] if k == choice), "Not answered")
            rows.append([item["label"], label])
        story.append(_kv_table(rows))
        story.append(Paragraph("User-provided checklist, not an inspection by this tool. "
                               "Always verify the vehicle physically before purchase.", st["small"]))

    if result["notes"] or result["scrap_message"]:
        story.append(Paragraph("Notes", st["h2"]))
        for n in ([result["scrap_message"]] if result["scrap_message"] else []) + result["notes"]:
            story.append(Paragraph("- " + n, st["body"]))

    # Model information
    tm = meta["test_metrics"]
    story.append(Paragraph("Model information", st["h2"]))
    story.append(_kv_table([
        ["Algorithm", meta["model_name"] + " (scikit-learn)"],
        ["Training data", f"{meta['n_rows']:,} used-car listings, {meta['data_period']}"],
        ["Model inputs", f"{len(meta['features'])} vehicle details"],
        ["Test R-squared", f"{tm['r2']:.3f}"],
        ["Mean absolute error", f"{inr(tm['mae'])} ({tm['mape_pct']:.1f}% on average)"],
    ]))
    story.append(Paragraph("Model performance is based on the project's held-out test dataset.", st["small"]))

    story += [Spacer(1, 10), Paragraph("<b>Disclaimer.</b> " + DISCLAIMER, st["small"])]

    doc.build(story)
    buf.seek(0)
    return buf
