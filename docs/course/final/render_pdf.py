"""Render the final report REPORT.md to REPORT.pdf; requires ReportLab.

Follows docs/course/deployment/render_pdf.py. Supported Markdown subset: `#`/`##`
headings, paragraphs, `-`/`1.` lists, pipe tables (column widths from a preceding
`<!-- widths: ... -->` comment, in percent), fenced code blocks and
`<!-- pagebreak -->`. A paragraph starting with «Таблица» is set as a caption.
Relative links become absolute GitHub links to the same file on master, so the
PDF stays usable outside the repository.
"""

import argparse
import posixpath
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

REPO = "https://github.com/aleksandr-novikov/ai-integration-development"
SOURCE_DIR = "docs/course/final"
INK = colors.HexColor("#243447")
ACCENT = colors.HexColor("#12395A")
LINK = colors.HexColor("#1F5C99")
RULE = colors.HexColor("#B8C4D0")
SHADE = colors.HexColor("#EEF2F6")
MARGIN = 48


def href(target):
    """Absolute URL for a Markdown link target relative to this directory."""
    if re.match(r"https?://", target):
        return target
    path, _, anchor = target.partition("#")
    resolved = posixpath.normpath(posixpath.join(SOURCE_DIR, path))
    return f"{REPO}/blob/master/{resolved}" + (f"#{anchor}" if anchor else "")


def inline(text, style):
    """Bold, code in a slightly smaller mono face, links clickable."""
    text = escape(text)
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
    size = round(style.fontSize * 0.88, 2)
    text = re.sub(r"`([^`]*)`", rf'<font face="Mono" size="{size}">\1</font>', text)
    return re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        lambda m: f'<link href="{escape(href(m.group(2)))}" color="{LINK.hexval()}">'
                  f"{m.group(1)}</link>",
        text,
    )


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
            if joined.startswith("# "):
                kind = "title"
            elif joined.startswith("## "):
                kind = "heading"
            elif joined.startswith("Таблица "):
                kind = "caption"
            else:
                kind = "text"
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
    body = ParagraphStyle("Body", fontName="Body", fontSize=9.6, leading=12.6,
                          spaceAfter=4.5, textColor=INK)
    bullet = ParagraphStyle("Bullet", parent=body, leftIndent=12, bulletIndent=2, spaceAfter=2.5)
    heading = ParagraphStyle("Heading", parent=body, fontName="BodyBold", fontSize=12,
                             leading=15, spaceBefore=8, spaceAfter=4, textColor=ACCENT,
                             keepWithNext=1)
    title = ParagraphStyle("Title", parent=heading, fontSize=15.5, leading=19, spaceBefore=0)
    caption = ParagraphStyle("Caption", parent=body, fontSize=8.4, leading=10.6,
                             textColor=colors.HexColor("#4A5B6C"), spaceAfter=6)
    cell = ParagraphStyle("Cell", parent=body, fontSize=8.4, leading=10.2, spaceAfter=0)
    head_cell = ParagraphStyle("HeadCell", parent=cell, fontName="BodyBold", textColor=ACCENT)
    code = ParagraphStyle("Code", fontName="Mono", fontSize=7.4, leading=9.2, textColor=INK)
    width = A4[0] - 2 * MARGIN

    story = []
    lines = (root / "REPORT.md").read_text(encoding="utf-8").splitlines()
    for kind, payload in blocks(lines):
        if kind == "pagebreak":
            story.append(PageBreak())
        elif kind in ("title", "heading", "caption", "text"):
            style = {"title": title, "heading": heading, "caption": caption}.get(kind, body)
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
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))
            story += [frame, Spacer(1, 6)]
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
                ("LEFTPADDING", (0, 0), (-1, -1), 3.5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3.5),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            story += [table, Spacer(1, 5)]

    def footer(canvas, doc):
        canvas.setFont("Body", 7.5)
        canvas.setFillColor(colors.HexColor("#607080"))
        canvas.drawString(MARGIN, 22, "Интеграция ИИ в разработку | итоговый отчёт | DB Monitoring")
        canvas.drawRightString(A4[0] - MARGIN, 22, str(doc.page))

    SimpleDocTemplate(str(root / "REPORT.pdf"), pagesize=A4,
                      rightMargin=MARGIN, leftMargin=MARGIN, topMargin=42, bottomMargin=42,
                      title="DB Monitoring: итоговый отчёт",
                      author="Команда DB Monitoring").build(
                          story, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    main()
