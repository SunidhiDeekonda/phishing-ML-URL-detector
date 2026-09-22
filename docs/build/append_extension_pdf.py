"""Append the frozen two-page research-extension summary to the base report PDF.

Run the base PDF generator first. This script refuses to append a duplicate.
"""

from __future__ import annotations

import json
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "docs/AI_Based_Phishing_Detection_System_KLH_Bachupally.pdf"
TEMP = ROOT / "docs/build/research_extension_appendix.pdf"


def main() -> None:
    metrics = json.loads((ROOT / "results/robustness_final_metrics.json").read_text())
    original_clean = metrics["clean_test"]["original_selected_95_5"]
    robust_clean = metrics["clean_test"]["robust_candidate_95_5"]
    original_adv = metrics["adversarial_test"]["original_selected_95_5"]
    robust_adv = metrics["adversarial_test"]["robust_candidate_95_5"]
    current = PdfReader(str(OUTPUT))
    if any("Extensions Beyond the Reference Study" in (page.extract_text() or "") for page in current.pages[-3:]):
        raise SystemExit("Extension appendix is already present; regenerate the base PDF before appending again.")

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ExtensionTitle", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=18, leading=22, textColor=colors.HexColor("#17213B"), spaceAfter=10))
    styles.add(ParagraphStyle(name="ExtensionHead", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=12, leading=15, textColor=colors.HexColor("#2557B8"), spaceBefore=6, spaceAfter=4))
    styles.add(ParagraphStyle(name="ExtensionBody", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.2, leading=13, textColor=colors.HexColor("#17213B"), spaceAfter=6))

    def paragraph(text: str, style: str = "ExtensionBody") -> Paragraph:
        return Paragraph(text, styles[style])

    def footer(canvas, _doc) -> None:
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#CBD5E5"))
        canvas.line(18 * mm, 14 * mm, 192 * mm, 14 * mm)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#56627A"))
        canvas.drawString(18 * mm, 9 * mm, "Research extension appendix - deterministic seed 42")
        canvas.restoreState()

    document = SimpleDocTemplate(str(TEMP), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=18 * mm, bottomMargin=18 * mm)
    rows = [["Held-out suite / model", "Accuracy", "Precision", "Recall", "ROC-AUC", "FN"]]
    for label, result in (("Clean / original", original_clean), ("Clean / robust candidate", robust_clean), ("Adversarial / original", original_adv), ("Adversarial / robust candidate", robust_adv)):
        rows.append([label, f"{result['accuracy']:.3%}", f"{result['precision']:.3%}", f"{result['recall']:.3%}", f"{result['roc_auc']:.3%}", str(result["false_negatives"])])
    table = Table(rows, colWidths=[48 * mm, 24 * mm, 24 * mm, 22 * mm, 24 * mm, 13 * mm], repeatRows=1)
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2557B8")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 7.4), ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E5")), ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F5F8FE")), ("ALIGN", (1, 1), (-1, -1), "CENTER"), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    story = [paragraph("Extensions Beyond the Reference Study", "ExtensionTitle"), paragraph("<b>Research question.</b> Can training-only, label-preserving URL mutations reduce false negatives under evolved phishing strings without damaging clean held-out performance? All mutations are inert text; no URL is fetched."), paragraph("Adversarial robustness methodology", "ExtensionHead"), paragraph("Eight deterministic transformations use seed 42. The experiment adds 1,751 training-derived phishing mutations, selects by mean clean/adversarial validation ROC-AUC with a 0.002 clean-AUC floor, and opens the 3,998-row adversarial test only after selection is frozen."), Spacer(1, 3 * mm), table, Spacer(1, 4 * mm), paragraph(f"The robust candidate increased adversarial recall by {robust_adv['recall'] - original_adv['recall']:+.3%} and reduced false negatives from {original_adv['false_negatives']} to {robust_adv['false_negatives']}. Clean accuracy was {robust_clean['accuracy']:.3%}. These results are specific to the declared synthetic mutations, not a universal security guarantee."), Image(str(ROOT / "results/plots/robustness_comparison.png"), width=145 * mm, height=80 * mm), PageBreak(), paragraph("Safe Context and Controlled Adaptation", "ExtensionTitle"), paragraph("Optional context analysis", "ExtensionHead"), paragraph("POST /predict-context extracts local signals from caller-supplied HTML and email text. The server never retrieves the URL, and unvalidated context evidence is never mixed into the validated ensemble probability."), paragraph("Human-verified continuous learning", "ExtensionHead"), paragraph("Feedback is quarantined, deduplicated, blocked from test contamination, and requires human approval. Retraining and validation-gated model promotion are explicit offline actions to resist poisoning."), paragraph("Drift and governance", "ExtensionHead"), paragraph("PSI monitors all 36 feature distributions. The registry records version, training date, data count, clean and robustness validation evidence, and status."), paragraph("What is new", "ExtensionHead"), paragraph("The reference already reported 100% precision. This contribution is adversarial benchmarking and training, separate clean-versus-adversarial evaluation, safe context evidence, and controlled adaptation rather than a higher saturated precision claim."), paragraph("Limitations", "ExtensionHead"), paragraph("Synthetic mutations do not cover every attacker; the subset is fixed; context is heuristic; and Vercel feedback storage is ephemeral without an authenticated external datastore.")]
    document.build(story, onFirstPage=footer, onLaterPages=footer)

    writer = PdfWriter()
    writer.clone_document_from_reader(current)
    appendix = PdfReader(str(TEMP))
    for page in appendix.pages:
        writer.add_page(page)
    writer.add_metadata({"/Title": "AI-Based Phishing Detection System Using Ensemble Learning", "/Author": "D Sunidhi; P Manvitha; G Likith", "/Subject": "Academic project report with robustness and adaptability extension"})
    staged = OUTPUT.with_suffix(".staged.pdf")
    with staged.open("wb") as handle:
        writer.write(handle)
    staged.replace(OUTPUT)
    TEMP.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
