"""Render the two-page homework from STRATEGY.md; requires ReportLab.

Supported Markdown subset: `#`/`##` headings, paragraphs, `-`/`1.` lists, pipe
tables (column widths from a preceding `<!-- widths: ... -->` comment, in
percent), fenced code blocks (the text diagram) and `<!-- pagebreak -->`.
"""

import argparse
import re
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

INK = colors.HexColor("#243447")
ACCENT = colors.HexColor("#12395A")
RULE = colors.HexColor("#B8C4D0")
SHADE = colors.HexColor("#EEF2F6")
MARGIN = 34


def inline(text, style):
    """Links keep only their text; code is set in a slightly smaller mono face."""
    text = escape(text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
    size = round(style.fontSize * 0.88, 2)
    return re.sub(r"`([^`]*)`", rf'<font face="Mono" size="{size}">\1</font>', text)


def blocks(lines):
    """Yield (kind, payload) for the Markdown subset above."""
    i, widths = 0, None
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
        elif line.startswith("```"):
            end = lines.index("```", i + 1)
            yield "code", "\n".join(lines[i + 1:end])
            i = end + 1
        elif line.startswith("<!--"):
            match = re.match(r"<!-- widths: ([\d ]+) -->", line)
            if match:
                widths = [int(value) for value in match.group(1).split()]
            elif "pagebreak" in line:
                yield "pagebreak", None
            i += 1
        elif line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [cell.strip() for cell in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", cell) for cell in cells):
                    rows.append(cells)
                i += 1
            yield "table", (rows, widths)
            widths = None
        elif re.match(r"(- |\d+\. )", line):
            items = []
            while i < len(lines) and lines[i].strip():
                if re.match(r"(- |\d+\. )", lines[i]):
                    items.append(lines[i])
                else:
                    items[-1] += " " + lines[i].strip()
                i += 1
            yield "list", items
        else:
            text = []
            while i < len(lines) and lines[i].strip() and not lines[i].startswith(("```", "|")):
                text.append(lines[i].strip())
                i += 1
            joined = " ".join(text)
            kind = "title" if joined.startswith("# ") else "heading" if joined.startswith("## ") else "text"
            yield kind, re.sub(r"^#{1,2} ", "", joined)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font", type=Path, required=True)
    parser.add_argument("--bold-font", type=Path, required=True)
    parser.add_argument("--mono-font", type=Path, required=True)
    args = parser.parse_args()
    rl_config.invariant = 1  # Fixed timestamps and IDs: rebuilding gives identical bytes.
    pdfmetrics.registerFont(TTFont("Body", str(args.font)))
    pdfmetrics.registerFont(TTFont("BodyBold", str(args.bold_font)))
    pdfmetrics.registerFont(TTFont("Mono", str(args.mono_font)))
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="BodyBold",
                                  italic="Body", boldItalic="BodyBold")
    root = Path(__file__).resolve().parent
    body = ParagraphStyle("Body", fontName="Body", fontSize=8.6, leading=10.7,
                          spaceAfter=3.5, textColor=INK)
    bullet = ParagraphStyle("Bullet", parent=body, leftIndent=10, bulletIndent=1, spaceAfter=1.5)
    heading = ParagraphStyle("Heading", parent=body, fontName="BodyBold", fontSize=10,
                             leading=12, spaceBefore=4, spaceAfter=3, textColor=ACCENT)
    title = ParagraphStyle("Title", parent=heading, fontSize=14, leading=17, spaceBefore=0)
    cell = ParagraphStyle("Cell", parent=body, fontSize=7.3, leading=8.6, spaceAfter=0)
    head_cell = ParagraphStyle("HeadCell", parent=cell, fontName="BodyBold", textColor=ACCENT)
    code = ParagraphStyle("Code", fontName="Mono", fontSize=6.9, leading=8.1, textColor=INK)
    width = A4[0] - 2 * MARGIN

    story = []
    lines = (root / "STRATEGY.md").read_text(encoding="utf-8").splitlines()
    for kind, payload in blocks(lines):
        if kind == "pagebreak":
            story.append(PageBreak())
        elif kind in ("title", "heading", "text"):
            style = {"title": title, "heading": heading}.get(kind, body)
            story.append(Paragraph(inline(payload, style), style))
        elif kind == "list":
            for item in payload:
                marker, text = re.match(r"(- |\d+\. )(.*)", item).groups()
                label = "•" if marker == "- " else marker.strip()
                story.append(Paragraph(inline(text, bullet), bullet, bulletText=label))
        elif kind == "code":
            frame = Table([[Preformatted(payload, code)]], colWidths=[width])
            frame.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), SHADE),
                ("BOX", (0, 0), (-1, -1), 0.5, RULE),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story += [frame, Spacer(1, 4)]
        elif kind == "table":
            rows, percents = payload
            percents = percents or [100 / len(rows[0])] * len(rows[0])
            data = [[Paragraph(inline(value, cell), head_cell if r == 0 else cell) for value in row]
                    for r, row in enumerate(rows)]
            table = Table(data, colWidths=[width * p / sum(percents) for p in percents],
                          repeatRows=1)
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), SHADE),
                ("GRID", (0, 0), (-1, -1), 0.4, RULE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ]))
            story += [table, Spacer(1, 4)]

    def footer(canvas, doc):
        canvas.setFont("Body", 7)
        canvas.setFillColor(colors.HexColor("#607080"))
        canvas.drawString(MARGIN, 18, "Интеграция ИИ в разработку | В10 | стратегия деплоя DB Monitoring")
        canvas.drawRightString(A4[0] - MARGIN, 18, str(doc.page))

    SimpleDocTemplate(str(root / "STRATEGY.pdf"), pagesize=A4,
                      rightMargin=MARGIN, leftMargin=MARGIN, topMargin=28, bottomMargin=32,
                      title="DB Monitoring: стратегия деплоя и отката",
                      author="Команда DB Monitoring").build(
                          story, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    main()
