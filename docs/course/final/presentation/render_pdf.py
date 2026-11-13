"""Render PRESENTATION.pptx to PRESENTATION.pdf; requires python-pptx and Chromium.

LibreOffice is not available in the course environment, so the PDF is drawn from the
.pptx itself: every shape of every slide is read back with python-pptx and placed into
an HTML page at the same position (rectangles, rounded rectangles, ovals, straight
connectors with arrowheads, freeform polylines, cropped pictures and text frames with
their insets, anchors, alignment, line spacing and bullets). Headless Chromium prints
that page. Only the shape kinds produced by build_presentation.py are supported; any
other shape stops the render instead of being skipped.

Arial and Courier New are replaced by the metric-compatible Liberation Sans and
Liberation Mono (SIL OFL) from --font-dir, so line breaks match PowerPoint closely.
Text that does not fit its frame is reported, and the render fails.
"""

import argparse
import base64
import html
import json
import re
import subprocess
import tempfile
from pathlib import Path

from pptx import Presentation
from pptx.enum.dml import MSO_FILL
from pptx.enum.shapes import MSO_SHAPE, MSO_SHAPE_TYPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn

HERE = Path(__file__).resolve().parent
EMU = 914400
LINE_HEIGHT = 1.149  # Arial / Liberation Sans ascent + descent + line gap per em
FONTS = {
    "Arial": ("LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf"),
    "Courier New": ("LiberationMono-Regular.ttf", "LiberationMono-Bold.ttf"),
}


def inches(emu):
    return emu / EMU


def color_of(fill):
    if fill.type == MSO_FILL.SOLID:
        return "#" + str(fill.fore_color.rgb)
    if fill.type in (None, MSO_FILL.BACKGROUND):
        return None
    raise ValueError(f"unsupported fill {fill.type}")


def line_style(shape):
    ln = shape._element.spPr.find(qn("a:ln"))
    if ln is None or ln.find(qn("a:noFill")) is not None:
        return None
    color = "#" + str(shape.line.color.rgb)
    width = shape.line.width.pt if shape.line.width else 0.75
    dash = ln.find(qn("a:prstDash")) is not None
    return color, width, dash


def box_css(shape):
    return (
        f"left:{inches(shape.left):.4f}in;top:{inches(shape.top):.4f}in;"
        f"width:{inches(shape.width):.4f}in;height:{inches(shape.height):.4f}in;"
    )


def render_text(shape, name):
    frame = shape.text_frame
    body = frame._txBody.find(qn("a:bodyPr"))
    insets = [
        inches(int(body.get(key, default)))
        for key, default in (("tIns", 45720), ("rIns", 91440), ("bIns", 45720), ("lIns", 91440))
    ]
    anchor = {
        MSO_ANCHOR.TOP: "flex-start",
        MSO_ANCHOR.MIDDLE: "center",
        MSO_ANCHOR.BOTTOM: "flex-end",
        None: "flex-start",
    }[frame.vertical_anchor]
    paragraphs = []
    for paragraph in frame.paragraphs:
        align = {
            PP_ALIGN.LEFT: "left",
            PP_ALIGN.CENTER: "center",
            PP_ALIGN.RIGHT: "right",
            None: "left",
        }[paragraph.alignment]
        spacing = paragraph.line_spacing or 1.0
        after = paragraph.space_after.pt if paragraph.space_after is not None else 0
        ppr = paragraph._p.pPr
        style = f"text-align:{align};margin:0 0 {after}pt 0;"
        bullet = ""
        if ppr is not None and ppr.find(qn("a:buChar")) is not None:
            indent = inches(int(ppr.get("marL", 0)))
            style += f"padding-left:{indent:.3f}in;position:relative;"
            color = ppr.find(qn("a:buClr")).find(qn("a:srgbClr")).get("val")
            face = ppr.find(qn("a:buFont")).get("typeface")
            char = html.escape(ppr.find(qn("a:buChar")).get("char"))
            bullet = (
                f'<span style="position:absolute;left:0;color:#{color};'
                f"font-family:'{face}'\">{char}</span>"
            )
        sizes, spans = [], []
        for run in paragraph.runs:
            font = run.font
            sizes.append(font.size.pt)
            css = (
                f"font-family:'{font.name}';font-size:{font.size.pt}pt;"
                f"color:#{font.color.rgb};font-weight:{700 if font.bold else 400};"
                f"font-style:{'italic' if font.italic else 'normal'};"
            )
            spans.append(f'<span style="{css}">{html.escape(run.text)}</span>')
        size = max(sizes) if sizes else 12
        face = paragraph.runs[0].font.name if paragraph.runs else "Arial"
        style += (
            f"font-family:'{face}';font-size:{size}pt;"
            f"line-height:{size * LINE_HEIGHT * spacing:.2f}pt;"
        )
        paragraphs.append(f'<p style="{style}">{bullet}{"".join(spans)}</p>')
    padding = ";".join(
        f"padding-{side}:{value:.4f}in"
        for side, value in zip(("top", "right", "bottom", "left"), insets, strict=True)
    )
    return (
        f'<div class="tf" data-name="{html.escape(name)}" style="{box_css(shape)}'
        f'{padding};justify-content:{anchor}"><div class="in">{"".join(paragraphs)}'
        f"</div></div>"
    )


def render_autoshape(shape):
    kind = shape.auto_shape_type
    fill = color_of(shape.fill)
    border = line_style(shape)
    css = box_css(shape) + (f"background:{fill};" if fill else "")
    if border:
        color, width, dash = border
        css += f"border:{width}pt {'dashed' if dash else 'solid'} {color};"
    if kind == MSO_SHAPE.ROUNDED_RECTANGLE:
        radius = shape.adjustments[0] * min(inches(shape.width), inches(shape.height))
        css += f"border-radius:{radius:.4f}in;"
    elif kind == MSO_SHAPE.OVAL:
        css += "border-radius:50%;"
    elif kind != MSO_SHAPE.RECTANGLE:
        raise ValueError(f"unsupported autoshape {kind} in {shape.name}")
    return f'<div class="sp" style="{css}"></div>'


def svg(path_d, color, width, dash=False, arrow=False):
    """One stroked path on a slide-sized overlay; coordinates and widths in inches."""
    stroke = width / 72
    dash_css = f' stroke-dasharray="{stroke * 4:.4f} {stroke * 3:.4f}"' if dash else ""
    marker = ""
    if arrow:  # PowerPoint "triangle, med": about three line widths, tip at the end point
        size = max(6.0, 3.5 * width) / 72
        marker = (
            f'<defs><marker id="a" viewBox="0 0 10 10" refX="10" refY="5" '
            f'markerWidth="{size:.4f}" markerHeight="{size:.4f}" '
            f'markerUnits="userSpaceOnUse" orient="auto">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{color}"/></marker></defs>'
        )
    end = ' marker-end="url(#a)"' if arrow else ""
    return (
        f'<svg class="ov" viewBox="0 0 13.333 7.5">{marker}'
        f'<path d="{path_d}" fill="none" stroke="{color}" stroke-width="{stroke:.4f}" '
        f'stroke-linejoin="round" stroke-linecap="{"butt" if arrow else "round"}"'
        f"{dash_css}{end}/></svg>"
    )


def render_connector(shape):
    color, width, dash = line_style(shape)
    ln = shape._element.spPr.find(qn("a:ln"))
    arrow = ln.find(qn("a:tailEnd")) is not None
    x1, y1 = inches(shape.begin_x), inches(shape.begin_y)
    x2, y2 = inches(shape.end_x), inches(shape.end_y)
    return svg(f"M{x1:.4f},{y1:.4f} L{x2:.4f},{y2:.4f}", color, width, dash, arrow)


def render_freeform(shape):
    color, width, dash = line_style(shape)
    path = shape._element.spPr.find(qn("a:custGeom")).find(qn("a:pathLst")).find(qn("a:path"))
    pw, ph = int(path.get("w")), int(path.get("h"))
    commands = []
    for node in path:
        tag = node.tag.split("}")[1]
        if tag == "close":
            commands.append("Z")
            continue
        point = node.find(qn("a:pt"))
        x = (
            inches(shape.left + int(point.get("x")) * shape.width / pw)
            if pw
            else inches(shape.left)
        )
        y = inches(shape.top + int(point.get("y")) * shape.height / ph) if ph else inches(shape.top)
        commands.append({"moveTo": "M", "lnTo": "L"}[tag] + f"{x:.4f},{y:.4f}")
    return svg(" ".join(commands), color, width, dash)


def render_picture(shape):
    blob = base64.b64encode(shape.image.blob).decode()
    left, right = shape.crop_left, shape.crop_right
    top, bottom = shape.crop_top, shape.crop_bottom
    width = inches(shape.width) / (1 - left - right)
    height = inches(shape.height) / (1 - top - bottom)
    return (
        f'<div class="sp" style="{box_css(shape)}overflow:hidden">'
        f'<img src="data:{shape.image.content_type};base64,{blob}" style="position:absolute;'
        f"left:{-left * width:.4f}in;top:{-top * height:.4f}in;width:{width:.4f}in;"
        f'height:{height:.4f}in"></div>'
    )


def render_slide(slide, number):
    background = color_of(slide.background.fill) or "#FFFFFF"
    parts = []
    for shape in slide.shapes:
        name = f"slide {number}: {shape.name}"
        kind = shape.shape_type
        if kind == MSO_SHAPE_TYPE.AUTO_SHAPE:
            parts.append(render_autoshape(shape))
        elif kind == MSO_SHAPE_TYPE.FREEFORM:
            parts.append(render_freeform(shape))
        elif kind == MSO_SHAPE_TYPE.LINE:
            parts.append(render_connector(shape))
        elif kind == MSO_SHAPE_TYPE.PICTURE:
            parts.append(render_picture(shape))
        elif kind != MSO_SHAPE_TYPE.TEXT_BOX:
            raise ValueError(f"unsupported shape {kind} ({name})")
        if kind != MSO_SHAPE_TYPE.PICTURE and shape.has_text_frame and shape.text_frame.text:
            parts.append(render_text(shape, name))
    return f'<section style="background:{background}">{"".join(parts)}</section>'


CHECK = """
<script>
const issues = [];
for (const el of document.querySelectorAll('.tf')) {
  const cs = getComputedStyle(el);
  const room = el.clientHeight - parseFloat(cs.paddingTop) - parseFloat(cs.paddingBottom);
  const inner = el.firstElementChild;
  if (inner.offsetHeight > room + 1 || inner.scrollWidth > inner.clientWidth + 1)
    issues.push(el.dataset.name + ' (' + inner.offsetHeight + ' > ' + Math.round(room) + ')');
}
document.getElementById('report').textContent = JSON.stringify(issues);
</script>"""


def build_html(pptx, font_dir):
    prs = Presentation(pptx)
    width, height = inches(prs.slide_width), inches(prs.slide_height)
    faces = []
    for family, (regular, bold) in FONTS.items():
        for weight, file in ((400, regular), (700, bold)):
            faces.append(
                f"@font-face{{font-family:'{family}';font-weight:{weight};"
                f"src:url('{(font_dir / file).resolve().as_uri()}')}}"
            )
    slides = "".join(render_slide(slide, index + 1) for index, slide in enumerate(prs.slides))
    return (
        f"""<!doctype html><html lang="ru"><head><meta charset="utf-8"><style>
{"".join(faces)}
@page {{ size: {width}in {height}in; margin: 0 }}
html, body {{ margin: 0; padding: 0 }}
section {{ position: relative; width: {width}in; height: {height}in; overflow: hidden;
  page-break-after: always; break-after: page }}
.sp, .tf, .ov {{ position: absolute; box-sizing: border-box }}
.ov {{ left: 0; top: 0; width: {width}in; height: {height}in; overflow: visible }}
.tf {{ display: flex; flex-direction: column; overflow: visible; font-kerning: normal }}
.tf p {{ white-space: pre-wrap; overflow-wrap: normal }}
#report {{ display: none }}
</style></head><body>{slides}<pre id="report"></pre>{CHECK}</body></html>""",
        len(prs.slides),
    )


def chromium(binary, *args):
    flags = [
        "--headless",
        "--no-sandbox",
        "--disable-gpu",
        "--disable-dev-shm-usage",
        "--no-zygote",
        "--single-process",
        "--allow-file-access-from-files",
    ]
    return subprocess.run(
        [binary, *flags, *args], check=True, capture_output=True, text=True, timeout=180
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chromium", required=True, help="headless Chromium executable")
    parser.add_argument(
        "--font-dir",
        type=Path,
        required=True,
        help="directory with Liberation Sans / Mono TTF files",
    )
    parser.add_argument("--pptx", type=Path, default=HERE / "PRESENTATION.pptx")
    parser.add_argument("--output", type=Path, default=HERE / "PRESENTATION.pdf")
    parser.add_argument("--html", type=Path, help="also keep the intermediate HTML here")
    parser.add_argument(
        "--draft",
        action="store_true",
        help="print the PDF even if some text does not fit (for review)",
    )
    args = parser.parse_args()
    page, count = build_html(args.pptx, args.font_dir)
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(args.html or Path(tmp) / "slides.html")
        source.write_text(page, encoding="utf-8")
        dom = chromium(
            args.chromium, "--virtual-time-budget=5000", "--dump-dom", source.resolve().as_uri()
        ).stdout
        issues = json.loads(
            html.unescape(re.search(r'<pre id="report">(.*?)</pre>', dom, re.S).group(1))
        )
        if issues and args.draft:
            print("text does not fit its frame:\n  " + "\n  ".join(issues))
        elif issues:
            raise SystemExit("text does not fit its frame:\n  " + "\n  ".join(issues))
        chromium(
            args.chromium,
            "--no-pdf-header-footer",
            "--virtual-time-budget=5000",
            f"--print-to-pdf={args.output.resolve()}",
            source.resolve().as_uri(),
        )
    print(
        f"{args.output}: {count} slides from {args.pptx.name}; "
        + ("DRAFT, text overflows" if issues else "all text frames fit")
    )


if __name__ == "__main__":
    main()
