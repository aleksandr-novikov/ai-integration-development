"""Render the two-page homework from STRATEGY.md; requires ReportLab."""

import argparse
import re
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer


def inline(text):
    text = escape(text)
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
    return text.replace("`", "")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font", type=Path, required=True)
    parser.add_argument("--bold-font", type=Path, required=True)
    args = parser.parse_args()
    pdfmetrics.registerFont(TTFont("Body", str(args.font)))
    pdfmetrics.registerFont(TTFont("BodyBold", str(args.bold_font)))
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="BodyBold")
    root = Path(__file__).resolve().parent
    body = ParagraphStyle("Body", fontName="Body", fontSize=11, leading=14.5,
                          spaceAfter=8, textColor=colors.HexColor("#243447"))
    heading = ParagraphStyle("Heading", parent=body, fontName="BodyBold", fontSize=12,
                             leading=15, spaceBefore=7, spaceAfter=8,
                             textColor=colors.HexColor("#12395A"))
    title = ParagraphStyle("Title", parent=heading, fontSize=18, leading=21, spaceAfter=10)
    story = []
    for block in (root / "STRATEGY.md").read_text().strip().split("\n\n"):
        if block.startswith("## 3."):
            story.append(PageBreak())
        style = title if block.startswith("# ") else heading if block.startswith("## ") else body
        text = re.sub(r"^#{1,2} ", "", block)
        story.append(Paragraph(inline(text), style))
    story.append(Spacer(1, 4))

    def footer(canvas, doc):
        canvas.setFont("Body", 8)
        canvas.setFillColor(colors.HexColor("#607080"))
        canvas.drawString(40, 25, "Интеграция ИИ в разработку | В13 | стратегия проекта")
        canvas.drawRightString(A4[0] - 40, 25, str(doc.page))

    SimpleDocTemplate(str(root / "STRATEGY.pdf"), pagesize=A4,
                      rightMargin=40, leftMargin=40, topMargin=32, bottomMargin=42,
                      title="DB Monitoring: дрейф и переобучение", author="Команда DB Monitoring").build(
                          story, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    main()
