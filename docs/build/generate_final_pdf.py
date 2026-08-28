from __future__ import annotations

import json
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "docs" / "AI_Based_Phishing_Detection_System_KLH_Bachupally.pdf"
PLOTS = ROOT / "results" / "plots"
ASSETS = ROOT / "docs" / "assets"

PAGE_W, PAGE_H = A4
LEFT = 18 * mm
RIGHT = 18 * mm
TOP = 18 * mm
BOTTOM = 17 * mm
CONTENT_W = PAGE_W - LEFT - RIGHT
CONTENT_H = PAGE_H - TOP - BOTTOM

INK = colors.HexColor("#17213B")
BLUE = colors.HexColor("#2557B8")
LIGHT_BLUE = colors.HexColor("#EAF0FC")
PALE_BLUE = colors.HexColor("#F5F8FE")
MID = colors.HexColor("#56627A")
LINE = colors.HexColor("#CBD5E5")
GREEN = colors.HexColor("#1B7A52")
RED = colors.HexColor("#A53B45")


def register_fonts() -> tuple[str, str, str]:
    candidates = [
        (
            Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
            Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
            Path("/System/Library/Fonts/Supplemental/Arial Italic.ttf"),
        ),
        (
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf"),
        ),
    ]
    for regular, bold, italic in candidates:
        if regular.exists() and bold.exists() and italic.exists():
            pdfmetrics.registerFont(TTFont("ReportRegular", str(regular)))
            pdfmetrics.registerFont(TTFont("ReportBold", str(bold)))
            pdfmetrics.registerFont(TTFont("ReportItalic", str(italic)))
            return "ReportRegular", "ReportBold", "ReportItalic"
    return "Helvetica", "Helvetica-Bold", "Helvetica-Oblique"


FONT, FONT_BOLD, FONT_ITALIC = register_fonts()


def build_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            fontName=FONT_BOLD,
            fontSize=25,
            leading=31,
            textColor=INK,
            alignment=TA_CENTER,
            spaceAfter=9,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportSubtitle",
            fontName=FONT,
            fontSize=12.5,
            leading=17,
            textColor=MID,
            alignment=TA_CENTER,
            spaceAfter=7,
        )
    )
    styles.add(
        ParagraphStyle(
            name="H1",
            fontName=FONT_BOLD,
            fontSize=17,
            leading=21,
            textColor=INK,
            spaceBefore=0,
            spaceAfter=8,
            keepWithNext=True,
        )
    )
    styles.add(
        ParagraphStyle(
            name="H2",
            fontName=FONT_BOLD,
            fontSize=11.5,
            leading=14,
            textColor=BLUE,
            spaceBefore=7,
            spaceAfter=4,
            keepWithNext=True,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Body",
            fontName=FONT,
            fontSize=9.2,
            leading=12.4,
            textColor=INK,
            alignment=TA_LEFT,
            spaceAfter=5.5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyTight",
            parent=styles["Body"],
            fontSize=8.4,
            leading=10.7,
            spaceAfter=3,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Small",
            parent=styles["Body"],
            fontSize=7.7,
            leading=9.8,
            textColor=MID,
            spaceAfter=3,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Caption",
            fontName=FONT_ITALIC,
            fontSize=8,
            leading=10.2,
            textColor=MID,
            alignment=TA_CENTER,
            spaceBefore=3,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TOCEntry1",
            fontName=FONT,
            fontSize=9.5,
            leading=13,
            textColor=INK,
            leftIndent=0,
            firstLineIndent=0,
            spaceBefore=2,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TOCEntry2",
            fontName=FONT,
            fontSize=8.2,
            leading=10.5,
            textColor=MID,
            leftIndent=12,
            firstLineIndent=0,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportCode",
            fontName="Courier",
            fontSize=8.2,
            leading=11,
            textColor=INK,
            leftIndent=8,
            spaceAfter=2,
        )
    )
    return styles


STYLES = build_styles()


class AcademicDocTemplate(BaseDocTemplate):
    def __init__(self, filename: str):
        super().__init__(
            filename,
            pagesize=A4,
            leftMargin=LEFT,
            rightMargin=RIGHT,
            topMargin=TOP,
            bottomMargin=BOTTOM,
            title="AI-Based Phishing Detection System Using Ensemble Learning",
            author="D Sunidhi; P Manvitha; G Likith",
            subject="Academic project and experiment report",
        )
        frame = Frame(LEFT, BOTTOM, CONTENT_W, CONTENT_H, id="normal")
        self.addPageTemplates([PageTemplate(id="academic", frames=[frame], onPage=self._decorate)])
        self._bookmark_id = 0

    def beforeDocument(self):
        self._bookmark_id = 0

    def _decorate(self, canvas, doc):
        canvas.saveState()
        if doc.page > 1:
            canvas.setStrokeColor(LINE)
            canvas.setLineWidth(0.45)
            canvas.line(LEFT, PAGE_H - 12 * mm, PAGE_W - RIGHT, PAGE_H - 12 * mm)
            canvas.setFont(FONT, 7.2)
            canvas.setFillColor(MID)
            canvas.drawString(LEFT, PAGE_H - 9.3 * mm, "AI-Based Phishing Detection System Using Ensemble Learning")
            canvas.drawRightString(PAGE_W - RIGHT, 9 * mm, f"Page {doc.page}")
        canvas.restoreState()

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and flowable.style.name in ("H1", "H2"):
            level = 0 if flowable.style.name == "H1" else 1
            text = flowable.getPlainText()
            key = f"section-{self._bookmark_id}"
            self._bookmark_id += 1
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, level=level, closed=False)
            if level == 0:
                self.notify("TOCEntry", (level, text, self.page, key))


def para(text: str, style: str = "Body") -> Paragraph:
    return Paragraph(text, STYLES[style])


def heading(text: str, level: int = 1) -> Paragraph:
    return Paragraph(text, STYLES["H1" if level == 1 else "H2"])


def bullet(text: str) -> Paragraph:
    style = ParagraphStyle(
        name=f"Bullet-{abs(hash(text))}",
        parent=STYLES["BodyTight"],
        leftIndent=12,
        firstLineIndent=-8,
        bulletIndent=2,
        spaceAfter=2.5,
    )
    return Paragraph(text, style, bulletText="-")


def fit_image(path: Path, max_width: float, max_height: float) -> Image:
    with PILImage.open(path) as image:
        width, height = image.size
    scale = min(max_width / width, max_height / height)
    return Image(str(path), width=width * scale, height=height * scale)


def styled_table(data, col_widths, font_size=7.5, header=True, alignments=None):
    normalized = []
    for row in data:
        normalized.append(
            [cell if hasattr(cell, "wrap") else para(str(cell), "Small") for cell in row]
        )
    table = Table(normalized, colWidths=col_widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("GRID", (0, 0), (-1, -1), 0.35, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("FONTNAME", (0, 0), (-1, -1), FONT),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1), [colors.white, PALE_BLUE]),
    ]
    if header:
        commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), BLUE),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
            ]
        )
    if alignments:
        for col, alignment in alignments.items():
            commands.append(("ALIGN", (col, 1 if header else 0), (col, -1), alignment))
    table.setStyle(TableStyle(commands))
    return table


def info_box(title: str, body: str, color=LIGHT_BLUE):
    inner = [
        [para(f"<b>{title}</b>", "BodyTight")],
        [para(body, "BodyTight")],
    ]
    table = Table(inner, colWidths=[CONTENT_W - 10], hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), color),
                ("BOX", (0, 0), (-1, -1), 0.6, LINE),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def pct(value: float) -> str:
    return f"{value * 100:.3f}%"


def load_data():
    with (ROOT / "results" / "final_test_metrics.json").open() as handle:
        final = json.load(handle)
    with (ROOT / "results" / "lightgbm_validation_metrics.json").open() as handle:
        lightgbm = json.load(handle)
    with (ROOT / "results" / "charcnn_training_history.json").open() as handle:
        cnn = json.load(handle)
    with (ROOT / "results" / "ensemble_config.json").open() as handle:
        ensemble = json.load(handle)
    return final, lightgbm, cnn, ensemble


def add_title_page(story):
    story.extend(
        [
            Spacer(1, 24 * mm),
            para("KLH Bachupally", "ReportSubtitle"),
            Spacer(1, 7 * mm),
            para("AI-Based Phishing Detection System<br/>Using Ensemble Learning", "ReportTitle"),
            para("Academic Project and Experiment Report", "ReportSubtitle"),
            Spacer(1, 16 * mm),
            styled_table(
                [
                    ["Department", "DEPARTMENT OF CSIT"],
                    ["Academic Year", "2026 - 2027"],
                    ["Project Guide", "Dr. K Venkateshwara Rao"],
                ],
                [42 * mm, 92 * mm],
                font_size=10,
                header=False,
            ),
            Spacer(1, 11 * mm),
            para("<b>Project Team</b>", "ReportSubtitle"),
            styled_table(
                [
                    ["2320090017", "D Sunidhi"],
                    ["2320090060", "P Manvitha"],
                    ["2320090069", "G Likith"],
                ],
                [42 * mm, 92 * mm],
                font_size=10,
                header=False,
            ),
            Spacer(1, 18 * mm),
            para("Character-Level CNN and LightGBM Ensemble for Safe URL-String Analysis", "ReportSubtitle"),
            PageBreak(),
        ]
    )


def add_contents(story):
    story.append(heading("Contents"))
    toc = TableOfContents()
    toc.levelStyles = [STYLES["TOCEntry1"], STYLES["TOCEntry2"]]
    story.append(toc)
    story.append(Spacer(1, 8 * mm))
    story.append(
        info_box(
            "Report scope",
            "This document reports an independent limited-compute reproduction using 20,000 URLs. Reference-study measurements are labelled separately and are never presented as outcomes of the local experiment.",
        )
    )
    story.append(PageBreak())


def add_overview(story, final):
    story.append(heading("1. Abstract"))
    story.append(
        para(
            "Phishing URLs imitate legitimate services to steal credentials or other sensitive information. Blacklists and fixed rules remain useful for known threats, but newly created links may evade them. This project implements a URL-only phishing detector that combines a Character-Level Convolutional Neural Network (Char-CNN), which learns sequential patterns directly from character strings, with Light Gradient Boosting Machine (LightGBM), which uses 36 engineered lexical, structural, and statistical URL features. The work is an independent limited-compute reproduction inspired by the PhishX reference study. A balanced deterministic subset of 20,000 URLs - 10,000 legitimate and 10,000 phishing - was constructed from frozen public snapshots. Root-domain-separated train, validation, and test partitions prevented domain leakage, and all model decisions were frozen before the held-out test set was evaluated. The validation-selected 95/5 CNN-LightGBM ensemble achieved 99.525% accuracy, 100% precision, 99.048% recall, and 99.931% ROC-AUC on 3,998 held-out URLs. A local FastAPI application applies the same saved preprocessing and models in real time while treating every submitted URL only as text; it never visits the destination. The completed artifacts, saved models, predictions, tests, and audit records demonstrate that the hybrid method can be reproduced on modest local hardware without overstating its scope or equivalence to the full reference study."
        )
    )
    story.append(heading("2. Introduction and Problem Statement"))
    story.append(
        para(
            "A phishing URL can use misleading domains, subdomains, paths, punctuation, encoding, or brand-like tokens to appear trustworthy. Reputation lists may not yet contain a new attack, while fixed rules can be evaded through small lexical changes. Machine learning offers a complementary first-stage defense by learning patterns that generalize beyond exact blacklist matches. URL-only analysis is particularly useful because it can produce a rapid risk estimate without downloading potentially hostile content."
        )
    )
    story.append(
        para(
            "Feature-based models explicitly represent known warning signals but may miss complex character sequences. Character-based deep learning can learn such sequences directly but does not make engineered indicators explicit. This project investigates whether combining both probability estimates provides a robust, reproducible detector under limited local compute."
        )
    )
    story.append(heading("3. Objectives"))
    for item in [
        "Create a balanced, traceable 20,000-URL dataset and analyse URLs safely as text.",
        "Implement 36 engineered features and released-compatible 200-character tokenization.",
        "Train and validate LightGBM and Char-CNN models independently.",
        "Prevent root-domain leakage and preserve a genuinely held-out test set.",
        "Select ensemble weights using validation ROC-AUC before test evaluation.",
        "Build a local web interface using the saved preprocessing and models.",
    ]:
        story.append(bullet(item))
    story.append(PageBreak())


def add_related_work(story):
    story.append(heading("4. Related Work and Reference Study"))
    story.append(
        para(
            "The principal reference is the PhishX study by Dubey, Tripathi, Srivastava, and Singh. Its hybrid design combines a character-level neural network with engineered URL features and LightGBM. URLNet by Le et al. motivates direct learned representations of malicious URLs, while work by Garera et al., Blum et al., and Yang et al. illustrates the value of lexical, structural, and multidimensional phishing indicators."
        )
    )
    story.append(heading("Reference architecture", 2))
    architecture = [
        [para("Raw URL", "BodyTight")],
        [para("Character branch: URL -> Char-CNN -> probability", "BodyTight")],
        [para("Feature branch: URL -> engineered features -> LightGBM -> probability", "BodyTight")],
        [para("Weighted probability combination -> prediction", "BodyTight")],
    ]
    arch_table = Table(architecture, colWidths=[CONTENT_W - 12], hAlign="LEFT")
    arch_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PALE_BLUE),
                ("BOX", (0, 0), (-1, -1), 0.7, BLUE),
                ("INNERGRID", (0, 0), (-1, -1), 0.35, LINE),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(arch_table)
    story.append(Spacer(1, 5 * mm))
    story.append(heading("Reference-study results", 2))
    story.append(
        styled_table(
            [
                ["Dataset", "Accuracy", "Precision", "Recall", "ROC-AUC"],
                ["Approximately 99,361 URLs", "99.819%", "100%", "99.635%", "99.947%"],
            ],
            [50 * mm, 25 * mm, 25 * mm, 25 * mm, 25 * mm],
            font_size=8.3,
            alignments={1: "CENTER", 2: "CENTER", 3: "CENTER", 4: "CENTER"},
        )
    )
    story.append(Spacer(1, 5 * mm))
    story.append(
        info_box(
            "Research-integrity boundary",
            "The values above are REFERENCE STUDY RESULTS. The local reproduction used a separate deterministic 20,000-URL subset, stricter domain-separated splits, its own model runs, and independently stored metrics.",
        )
    )
    story.append(PageBreak())


def add_dataset(story):
    story.append(heading("5. Dataset, Preprocessing, and Leakage Prevention"))
    story.append(
        para(
            "Frozen CSV snapshots were obtained from the released PhishX repository. The phishing snapshot was based on PhishTank and contained 49,363 unique cleaned URLs after eight duplicates were removed. The legitimate snapshot was based on Tranco and contained 50,000 unique URLs. These available snapshot totals are distinct from the reference paper's approximately reported 99,361-URL corpus."
        )
    )
    story.append(
        styled_table(
            [
                ["Quantity", "Legitimate", "Phishing", "Total"],
                ["Available unique frozen URLs", "50,000", "49,363", "99,363"],
                ["Used in this experiment", "10,000", "10,000", "20,000"],
            ],
            [58 * mm, 31 * mm, 31 * mm, 31 * mm],
            font_size=8.2,
            alignments={1: "CENTER", 2: "CENTER", 3: "CENTER"},
        )
    )
    story.append(heading("Preprocessing", 2))
    story.append(
        para(
            "Nulls and duplicates were audited, each class was sampled deterministically with seed 42, and labels were defined as 0 for legitimate and 1 for phishing. Character input was lowercased and encoded to length 200 using PAD 0, 49 explicit characters at indices 1-49, and UNK 50. Long URLs retain the rightmost 200 characters; shorter strings are right-padded with zeros."
        )
    )
    story.append(heading("Domain-leakage correction", 2))
    story.append(
        para(
            "The first stratified split placed URLs sharing a registered root domain in different partitions. That could allow domain-specific patterns learned during training to reappear in testing and inflate performance. The problem was detected and corrected before either model was trained."
        )
    )
    story.append(
        styled_table(
            [
                ["Split pair", "Initial overlapping domains", "Final overlapping domains"],
                ["Train - validation", "153", "0"],
                ["Train - test", "232", "0"],
                ["Validation - test", "108", "0"],
            ],
            [65 * mm, 43 * mm, 43 * mm],
            font_size=8.2,
            alignments={1: "CENTER", 2: "CENTER"},
        )
    )
    story.append(Spacer(1, 4 * mm))
    story.append(
        styled_table(
            [
                ["Final partition", "Rows", "Legitimate", "Phishing"],
                ["Train", "14,002", "6,997", "7,005"],
                ["Validation", "2,000", "1,000", "1,000"],
                ["Held-out test", "3,998", "2,003", "1,995"],
            ],
            [55 * mm, 32 * mm, 32 * mm, 32 * mm],
            font_size=8.2,
            alignments={1: "CENTER", 2: "CENTER", 3: "CENTER"},
        )
    )
    story.append(Spacer(1, 4 * mm))
    story.append(
        info_box(
            "Safety design",
            "Dataset and submitted URLs were never visited. Feature extraction performed no DNS lookup, external HTTP request, webpage download, or JavaScript execution; each URL was treated only as text.",
        )
    )
    story.append(PageBreak())


def add_features(story):
    story.append(heading("6. Feature Engineering"))
    story.append(
        para(
            "The LightGBM branch receives exactly 36 numeric features implemented in <font name='Courier'>src/features.py</font>. The design combines compact interpretable indicators with the CNN's automatically learned character patterns."
        )
    )
    data = [
        ["Category", "Implemented examples", "Purpose"],
        ["Lexical", "URL, host and path length; digit and letter counts; digit-letter ratio; special characters; hyphens", "Measure character composition and obfuscation."],
        ["Structural", "Dots, path segments, query parameters, HTTPS, port, fragment, at-sign, IP host", "Describe URL hierarchy and unusual structure."],
        ["Statistical/domain", "Hostname entropy, path entropy, vowel fraction, suspicious TLD", "Represent randomness, linguistic form, and domain indicators."],
        ["Length buckets", "0-20, 21-40, 41-60, 61-80, 81-100, 101+", "Expose broad URL-length ranges."],
        ["Suspicious tokens", "login, signin, secure, webscr, bank, verify, update, account, confirm", "Mark common credential and account lures."],
    ]
    story.append(styled_table(data, [29 * mm, 78 * mm, 44 * mm], font_size=7.4))
    story.append(Spacer(1, 5 * mm))
    story.append(heading("Interpretation boundaries", 2))
    story.append(
        para(
            "No single feature is treated as proof of phishing. HTTPS, for example, is not automatically legitimate, and a long URL is not automatically malicious. LightGBM combines thresholds and interactions across the feature set. Feature importance describes tree usage rather than causal security evidence."
        )
    )
    story.append(heading("Character representation", 2))
    story.append(
        styled_table(
            [
                ["Property", "Value"],
                ["Explicit alphabet", "49 letters, digits, and URL symbols"],
                ["Effective indices", "PAD 0; explicit 1-49; UNK 50"],
                ["Sequence length", "200"],
                ["Truncation", "Keep rightmost 200 characters"],
                ["Padding", "Zeros on the right"],
            ],
            [56 * mm, 95 * mm],
            font_size=8,
        )
    )
    story.append(PageBreak())


def add_models(story, lightgbm, cnn):
    story.append(heading("7. Model Architectures and Training"))
    story.append(heading("Character-Level CNN", 2))
    cnn_steps = [
        ["1", "URL character encoding", "200 indices"],
        ["2", "Embedding", "16 dimensions"],
        ["3", "Parallel Conv1D", "kernels 3, 5, 7; 128 filters each"],
        ["4", "Activation and pooling", "ReLU; adaptive max pooling"],
        ["5", "Concatenation", "384 dimensions"],
        ["6", "Dense and dropout", "384 -> 64; ReLU; dropout 0.3"],
        ["7", "Output", "64 -> 1 binary logit"],
    ]
    story.append(styled_table([["Stage", "Layer", "Configuration"]] + cnn_steps, [17 * mm, 56 * mm, 78 * mm], font_size=7.5))
    story.append(
        para(
            "The network has 56,625 trainable parameters. It used Adam with learning rate 0.001, batch size 64, and Apple MPS acceleration. Early stopping patience was two epochs; seven epochs completed, the best checkpoint was epoch 5, and training took approximately 16.9 seconds."
        )
    )
    story.append(heading("LightGBM", 2))
    story.append(
        para(
            "LightGBM builds a sequence of decision trees over the 36 engineered features. It efficiently models nonlinear thresholds and feature interactions on CPU. Sigmoid calibration was applied after fitting to improve probability interpretation; calibration was not expected to maximize fixed-threshold accuracy."
        )
    )
    story.append(
        styled_table(
            [
                ["Objective", "Max estimators", "Learning rate", "Leaves", "Best iteration", "Training time"],
                ["Binary", "1,000", "0.05", "64", str(lightgbm["best_iteration"]), f'{lightgbm["training_duration_seconds"]:.2f} s'],
            ],
            [28 * mm, 27 * mm, 25 * mm, 20 * mm, 26 * mm, 25 * mm],
            font_size=7.7,
            alignments={1: "CENTER", 2: "CENTER", 3: "CENTER", 4: "CENTER", 5: "CENTER"},
        )
    )
    story.append(Spacer(1, 4 * mm))
    story.append(
        info_box(
            "Validation control",
            "LightGBM early stopping used validation ROC-AUC with patience 50. CNN checkpoint selection used validation performance with patience 2. Neither model used held-out test outcomes for fitting or model selection.",
        )
    )
    story.append(PageBreak())


def add_ensemble_protocol(story, ensemble):
    story.append(heading("8. Ensemble Methodology and Experimental Protocol"))
    story.append(heading("Probability combination", 2))
    story.append(
        info_box(
            "Ensemble equation",
            "P(ensemble) = w(CNN) x P(CNN) + w(LightGBM) x P(LightGBM)",
        )
    )
    story.append(Spacer(1, 4 * mm))
    story.append(
        para(
            "The reference weighting was 0.60 CNN and 0.40 LightGBM. The local validation search evaluated CNN weights from 0.00 to 1.00 in steps of 0.05 and used validation ROC-AUC as the selection metric. The selected configuration was 0.95 CNN and 0.05 LightGBM, with validation ROC-AUC 99.7066%. The gain was small and is not presented as a major improvement."
        )
    )
    story.append(
        styled_table(
            [
                ["Configuration", "CNN weight", "LightGBM weight", "Selection basis"],
                ["Reference ensemble", "0.60", "0.40", "Reference methodology"],
                ["Selected ensemble", "0.95", "0.05", "Highest validation ROC-AUC"],
            ],
            [50 * mm, 28 * mm, 35 * mm, 38 * mm],
            font_size=8,
            alignments={1: "CENTER", 2: "CENTER"},
        )
    )
    story.append(heading("Train, validation, and test roles", 2))
    protocol = [
        ["Partition", "Permitted role"],
        ["Train", "Fit LightGBM trees and Char-CNN parameters."],
        ["Validation", "Early stopping, LightGBM probability calibration, and ensemble weight selection."],
        ["Held-out test", "One final evaluation after preprocessing, models, calibration, weights, and threshold were frozen."],
    ]
    story.append(styled_table(protocol, [38 * mm, 113 * mm], font_size=8.1))
    story.append(Spacer(1, 5 * mm))
    story.append(
        info_box(
            "Held-out evaluation rule",
            "The classification threshold was fixed at 0.50. No threshold, weight, architecture, or training decision was changed after test results were observed. Selecting a better-looking test configuration afterward would constitute test-set tuning.",
        )
    )
    story.append(PageBreak())


def add_results(story, final):
    story.append(heading("9. Final Held-Out Test Results"))
    rows = []
    labels = [
        ("Calibrated LightGBM", "lightgbm"),
        ("Char-CNN", "charcnn"),
        ("Reference ensemble 60/40", "reference_ensemble"),
        ("Selected ensemble 95/5", "selected_ensemble"),
    ]
    for label, key in labels:
        m = final[key]
        rows.append(
            [
                label,
                pct(m["accuracy"]),
                pct(m["precision"]),
                pct(m["recall"]),
                pct(m["f1"]),
                pct(m["roc_auc"]),
                pct(m["pr_auc"]),
                str(m["fp"]),
                str(m["fn"]),
            ]
        )
    story.append(
        styled_table(
            [["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "FP", "FN"]] + rows,
            [36 * mm, 16.5 * mm, 17 * mm, 16.5 * mm, 16 * mm, 18 * mm, 18 * mm, 7 * mm, 7 * mm],
            font_size=6.8,
            alignments={1: "CENTER", 2: "CENTER", 3: "CENTER", 4: "CENTER", 5: "CENTER", 6: "CENTER", 7: "CENTER", 8: "CENTER"},
        )
    )
    story.append(Spacer(1, 5 * mm))
    story.append(heading("Selected ensemble confusion matrix", 2))
    selected = final["selected_ensemble"]
    story.append(
        styled_table(
            [
                ["True negatives", "False positives", "False negatives", "True positives"],
                [str(selected["tn"]), str(selected["fp"]), str(selected["fn"]), str(selected["tp"])],
            ],
            [37.75 * mm] * 4,
            font_size=8.4,
            alignments={0: "CENTER", 1: "CENTER", 2: "CENTER", 3: "CENTER"},
        )
    )
    story.append(heading("Interpretation", 2))
    story.append(
        para(
            "Calibrated LightGBM produced the highest test accuracy (99.600%) and recall (99.449%). Char-CNN produced the highest test ROC-AUC (99.945%). The validation-selected ensemble achieved zero false positives and ROC-AUC 99.931%, but its fixed-threshold decisions matched the CNN on this test set. The 95/5 ensemble is retained because it was selected before test inspection, not because it dominated every final metric."
        )
    )
    story.append(
        para(
            "Zero false positives means no legitimate held-out URL was incorrectly blocked. Nineteen phishing URLs were missed, illustrating the security tradeoff between user disruption and attack recall. The threshold was not changed after observing these errors."
        )
    )
    story.append(PageBreak())


def add_visual_page_one(story):
    story.append(heading("10. Visual Experimental Evidence"))
    model_img = fit_image(PLOTS / "model_comparison.png", CONTENT_W, 64 * mm)
    story.append(model_img)
    story.append(para("<b>Figure 1.</b> Comparison of component and ensemble performance on the final held-out test set.", "Caption"))
    story.append(
        para(
            "Differences are small but metric-dependent: LightGBM leads test accuracy, while Char-CNN leads ROC-AUC. The chart supports multi-metric reporting rather than a single winner claim.",
            "BodyTight",
        )
    )
    confusion_img = fit_image(PLOTS / "confusion_matrix.png", CONTENT_W, 80 * mm)
    story.append(confusion_img)
    story.append(para("<b>Figure 2.</b> Confusion matrix of the validation-selected 95/5 ensemble on 3,998 held-out URLs.", "Caption"))
    story.append(
        para(
            "At threshold 0.50, the ensemble produced 2,003 true negatives, zero false positives, nineteen false negatives, and 1,976 true positives.",
            "BodyTight",
        )
    )
    story.append(PageBreak())


def add_visual_page_two(story):
    story.append(heading("11. Threshold-Independent Evaluation"))
    roc = fit_image(PLOTS / "roc_curve.png", 74 * mm, 62 * mm)
    pr = fit_image(PLOTS / "precision_recall_curve.png", 74 * mm, 62 * mm)
    visuals = Table([[roc, pr]], colWidths=[76 * mm, 76 * mm], hAlign="CENTER")
    visuals.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    story.append(visuals)
    captions = Table(
        [[
            para("<b>Figure 3.</b> Receiver operating characteristic curves on held-out data.", "Caption"),
            para("<b>Figure 4.</b> Precision-recall curves on held-out data.", "Caption"),
        ]],
        colWidths=[76 * mm, 76 * mm],
    )
    story.append(captions)
    story.append(Spacer(1, 4 * mm))
    story.append(heading("Interpretation", 2))
    story.append(
        para(
            "The ROC curves summarize false-positive and true-positive tradeoffs across thresholds, while precision-recall curves focus on phishing detection quality as the decision threshold changes. All configurations rank the two classes strongly, with final ROC-AUC values between 99.851% and 99.945%."
        )
    )
    story.append(
        para(
            "Threshold-independent curves do not replace the fixed-threshold confusion matrix. A deployment must still decide the relative cost of false alerts and missed attacks using validation or operational data, not the held-out test set."
        )
    )
    story.append(
        info_box(
            "Metric context",
            "ROC-AUC evaluates ranking over thresholds. PR-AUC emphasizes precision and recall. Accuracy and confusion counts describe the chosen 0.50 operating point. Reporting all of them prevents a narrow interpretation of performance.",
        )
    )
    story.append(PageBreak())


def add_feature_analysis(story, lightgbm):
    story.append(heading("12. Feature and Probability Analysis"))
    fi = fit_image(PLOTS / "feature_importance.png", 74 * mm, 55 * mm)
    pd = fit_image(PLOTS / "probability_distribution.png", 74 * mm, 55 * mm)
    visuals = Table([[fi, pd]], colWidths=[76 * mm, 76 * mm], hAlign="CENTER")
    visuals.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    story.append(visuals)
    story.append(
        Table(
            [[
                para("<b>Figure 5.</b> Leading engineered features by LightGBM importance.", "Caption"),
                para("<b>Figure 6.</b> Held-out phishing-probability distributions.", "Caption"),
            ]],
            colWidths=[76 * mm, 76 * mm],
        )
    )
    story.append(heading("Leading observed features", 2))
    top = lightgbm["top_10_features"]
    half = 5
    feature_rows = [["Rank", "Feature", "Importance", "Rank", "Feature", "Importance"]]
    for idx in range(half):
        left = top[idx]
        right = top[idx + half]
        feature_rows.append(
            [
                str(idx + 1), left["feature"], str(left["importance"]),
                str(idx + half + 1), right["feature"], str(right["importance"]),
            ]
        )
    story.append(styled_table(feature_rows, [12 * mm, 43 * mm, 20 * mm, 12 * mm, 43 * mm, 21 * mm], font_size=7.4, alignments={0: "CENTER", 2: "CENTER", 3: "CENTER", 5: "CENTER"}))
    story.append(Spacer(1, 4 * mm))
    story.append(
        para(
            "Hostname entropy, vowel fraction, and host length were the strongest recorded LightGBM contributors, followed by URL length and character-count features. These values indicate tree-split usage, not causality. The probability distribution shows strong overall class separation while retaining an ambiguous region where threshold-dependent errors occur."
        )
    )
    story.append(PageBreak())


def add_application_legitimate(story):
    story.append(heading("13. Working Local Application"))
    story.append(
        para(
            "The demonstration application uses FastAPI, HTML, CSS, and vanilla JavaScript. <font name='Courier'>GET /health</font> reports readiness, and <font name='Courier'>POST /predict</font> accepts a URL string. The server applies the same 36-feature extractor, 200-character tokenizer, calibrated LightGBM model, Char-CNN checkpoint, and frozen ensemble weights used in evaluation. It is a local application at <font name='Courier'>http://127.0.0.1:8000</font>, not a public deployment."
        )
    )
    img = fit_image(ASSETS / "frontend_legitimate.png", CONTENT_W, 145 * mm)
    story.append(img)
    story.append(
        para(
            "<b>Figure 7.</b> Local application correctly classifying <font name='Courier'>https://www.google.com</font> as legitimate and displaying component probabilities and URL signals.",
            "Caption",
        )
    )
    story.append(
        para(
            "The selected ensemble reports approximately 0.6% phishing probability for this example. One example is illustrative only and is not used as a separate performance benchmark.",
            "BodyTight",
        )
    )
    story.append(PageBreak())


def add_application_phishing(story):
    story.append(heading("14. Synthetic Suspicious-String Demonstration"))
    img = fit_image(ASSETS / "frontend_phishing.png", CONTENT_W, 155 * mm)
    story.append(img)
    story.append(
        para(
            "<b>Figure 8.</b> Local application classifying the synthetic suspicious-looking string <font name='Courier'>http://secure-account-login-example.xyz/verify</font> as phishing. The URL was analysed only as text and was not visited.",
            "Caption",
        )
    )
    story.append(
        info_box(
            "Application safety",
            "Inference performs no DNS lookup, external HTTP request, webpage download, script execution, or browser navigation to the submitted address. The screenshot records a local string-classification result, not a visit to a phishing website.",
            color=colors.HexColor("#FDEEEF"),
        )
    )
    story.append(Spacer(1, 4 * mm))
    story.append(
        para(
            "The component models and both ensemble configurations assigned approximately 99.9-100.0% phishing probability to this synthetic string. This is a useful demonstration of the interface and signal presentation, but it must not be generalized beyond the held-out quantitative evaluation."
        )
    )
    story.append(PageBreak())


def add_comparison_and_evidence(story):
    story.append(heading("15. Reference Comparison and Implementation Evidence"))
    story.append(
        styled_table(
            [
                ["Measure", "Reference study", "Local reproduction"],
                ["Dataset", "Approximately 99,361 URLs", "20,000 URLs"],
                ["Accuracy", "99.819%", "99.525%"],
                ["Precision", "100%", "100%"],
                ["Recall", "99.635%", "99.048%"],
                ["ROC-AUC", "99.947%", "99.931%"],
            ],
            [48 * mm, 51.5 * mm, 51.5 * mm],
            font_size=8.1,
            alignments={1: "CENTER", 2: "CENTER"},
        )
    )
    story.append(
        para(
            "The local selected-ensemble accuracy is approximately 0.294 percentage points lower. Plausible factors include the smaller deterministic subset, strict domain separation, different sampling, local implementation details, and limited training and tuning. The results are close, but the studies are not directly equivalent and the local system is not claimed to be better."
        )
    )
    story.append(heading("Implementation Evidence", 2))
    evidence = [
        ["Evidence", "Repository artifact"],
        ["Dataset provenance", "DATA_PROVENANCE.md"],
        ["Experiment history", "RUN_LOG.md"],
        ["Feature engineering and tokenizer", "src/features.py; src/char_tokenizer.py"],
        ["LightGBM and CNN implementation", "src/train_lightgbm.py; src/train_charcnn.py"],
        ["Ensemble selection", "src/select_ensemble.py; results/ensemble_config.json"],
        ["Final held-out metrics", "results/final_test_metrics.json"],
        ["Saved models", "models/lightgbm_calibrated.pkl; models/char_cnn.pt"],
        ["Application inference", "app/inference.py"],
        ["Automated tests", "24 passed; 0 failed"],
    ]
    story.append(styled_table(evidence, [52 * mm, 99 * mm], font_size=7.7))
    story.append(Spacer(1, 4 * mm))
    story.append(
        info_box(
            "Auditability",
            "The repository retains source implementations, saved indices, trained artifacts, validation history, ensemble search results, final predictions, plots, tests, and chronological run notes. Reproducibility is visible without embedding raw logs or source-code pages in this report.",
        )
    )
    story.append(PageBreak())


def add_limits_conclusion(story):
    story.append(heading("16. Limitations, Future Work, and Conclusion"))
    story.append(heading("Limitations", 2))
    limitations = [
        "The experiment used a balanced 20,000-URL subset rather than the complete available snapshots.",
        "Detection is URL-only and does not inspect webpage content, certificates, redirects, or visual similarity.",
        "No live reputation, WHOIS, domain-age, or hosting-history signals were used.",
        "The historical snapshots do not establish temporal generalization to future campaigns.",
        "Adversarial robustness was not deeply tested, and balanced-test performance may differ under operational class imbalance.",
        "The selected ensemble did not dominate every component metric.",
    ]
    for item in limitations:
        story.append(bullet(item))
    story.append(heading("Future work", 2))
    story.append(
        para(
            "Future evaluation could use larger and newer frozen datasets, time-based splits, adversarial URL transformations, validation-driven security thresholds, and safe optional domain-reputation or webpage-content services. SHAP explanations could improve LightGBM interpretability, and a browser extension could provide a practical client. None of these extensions is claimed as completed work."
        )
    )
    story.append(heading("Conclusion", 2))
    story.append(
        para(
            "The project reproduced the core hybrid CNN and LightGBM methodology on limited local hardware using 20,000 URLs. It implemented 36 engineered features, exact character metadata, zero-overlap domain-separated evaluation, independently trained components, validation-only ensemble selection, and a working local application. The final selected ensemble achieved 99.525% accuracy, 100% precision, 99.048% recall, and 99.931% ROC-AUC on 3,998 held-out URLs. These results demonstrate a strong reproducible academic prototype without claiming equivalence to or superiority over the reference study."
        )
    )
    story.append(heading("Reproducibility", 2))
    story.append(para("Run the completed project from its repository root:", "BodyTight"))
    for command in [
        "source .venv/bin/activate",
        "pytest",
        "uvicorn app.main:app --reload",
    ]:
        story.append(para(command, "ReportCode"))
    story.append(para("Local application: <font name='Courier'>http://127.0.0.1:8000</font>", "BodyTight"))
    story.append(PageBreak())


def add_references(story):
    story.append(heading("17. References"))
    references = [
        "R. Dubey, A. M. Tripathi, A. Srivastava, and S. Singh, \"Phishing Detection System: An Ensemble Approach Using Character-Level CNN and Feature Engineering,\" arXiv:2512.16717, 2025. Reference implementation: https://github.com/dubeyrudra-1808/PhishX.",
        "PhishTank, PhishTank Valid Phishing URLs. https://phishtank.org/.",
        "Tranco, Tranco Top 1M List. https://tranco-list.eu/.",
        "H. Le, Q. Pham, D. Sahoo, and S. C. H. Hoi, \"URLNet: Learning a URL representation with deep learning for malicious URL detection,\" arXiv:1802.03162, 2018.",
        "S. Garera, N. Provos, M. Chew, and A. D. Rubin, \"A framework for detection and measurement of phishing attacks,\" Proceedings of the 2007 ACM Workshop on Large Scale Attack Defence, pp. 1-8, 2007.",
        "A. Blum, B. Wardman, T. Solorio, and G. Warner, \"Lexical feature-based phishing URL detection using online learning,\" Proceedings of the 3rd ACM Workshop on Artificial Intelligence and Security, pp. 54-60, 2010.",
        "P. Yang, G. Zhao, and P. Zeng, \"Phishing website detection based on multidimensional features driven by deep learning,\" IEEE Access, vol. 7, pp. 15196-15209, 2019.",
    ]
    for index, reference in enumerate(references, start=1):
        style = ParagraphStyle(
            name=f"Reference{index}",
            parent=STYLES["Body"],
            leftIndent=12,
            firstLineIndent=-12,
            fontSize=8.5,
            leading=11.5,
            spaceAfter=7,
        )
        story.append(Paragraph(f"{index}. {reference}", style))
    story.append(Spacer(1, 8 * mm))
    story.append(
        info_box(
            "Reproduction statement",
            "This report documents an independent educational reproduction. Reference metrics, local metrics, public data provenance, and implementation evidence are presented as separate, auditable quantities.",
        )
    )


def main():
    final, lightgbm, cnn, ensemble = load_data()
    required = [
        PLOTS / "confusion_matrix.png",
        PLOTS / "roc_curve.png",
        PLOTS / "precision_recall_curve.png",
        PLOTS / "model_comparison.png",
        PLOTS / "feature_importance.png",
        PLOTS / "probability_distribution.png",
        ASSETS / "frontend_legitimate.png",
        ASSETS / "frontend_phishing.png",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required report images: {missing}")

    story = []
    add_title_page(story)
    add_contents(story)
    add_overview(story, final)
    add_related_work(story)
    add_dataset(story)
    add_features(story)
    add_models(story, lightgbm, cnn)
    add_ensemble_protocol(story, ensemble)
    add_results(story, final)
    add_visual_page_one(story)
    add_visual_page_two(story)
    add_feature_analysis(story, lightgbm)
    add_application_legitimate(story)
    add_application_phishing(story)
    add_comparison_and_evidence(story)
    add_limits_conclusion(story)
    add_references(story)

    document = AcademicDocTemplate(str(OUTPUT))
    document.multiBuild(story)
    print(OUTPUT)


if __name__ == "__main__":
    main()
