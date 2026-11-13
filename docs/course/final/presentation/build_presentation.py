"""Build PRESENTATION.pptx for the final defense (#20); requires python-pptx.

Numbers come from the saved artifacts, not from the code: the five validation runs
from experiments/comparison.csv, the single test evaluation from experiments/summary.json
and the alert scenario from monitoring/model/demo.json. Each value shown on a slide is
also looked up in final/REPORT.md, so a slide cannot silently diverge from the report.
Text is set in Arial; slide size is 13.333 x 7.5 in (16:9). Speaker notes carry the
timing. The PDF is rendered from the finished .pptx by render_pdf.py.
"""

import argparse
import csv
import json
import re
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Pt

HERE = Path(__file__).resolve().parent
COURSE = HERE.parents[1]
REPORT = COURSE / "final" / "REPORT.md"
SWAGGER = COURSE / "inference" / "evidence" / "swagger-predictAnomaly.png"
TOTAL = 11

SANS = "Arial"
MONO = "Courier New"
BG = "0B1324"
CARD = "131E35"
CARD2 = "1A2844"
EDGE = "2B3D60"
TEXT = "EAF0FA"
MUTED = "9AACC9"
TEAL = "2DD4BF"
TEAL_DARK = "123A44"
BLUE = "60A5FA"
BLUE_DARK = "16304F"
AMBER = "FBBF24"
AMBER_DARK = "3A3018"
CORAL = "F87171"
CORAL_DARK = "3D1E2A"
GRID = "33456A"

NBSP = " "


def inch(value):
    return Emu(round(value * 914400))


def rgb(value):
    return RGBColor.from_string(value)


def pct(value, digits=2):
    """0.13267 -> '13,27 %' with a non-breaking space."""
    return f"{value * 100:.{digits}f}".replace(".", ",") + NBSP + "%"


def num(value):
    """8061 -> '8 061' with a non-breaking space."""
    return f"{value:,}".replace(",", NBSP)


def dec(value, digits):
    return f"{value:.{digits}f}".replace(".", ",")


# --------------------------------------------------------------------------- data


def load_data():
    report = REPORT.read_text(encoding="utf-8").replace(NBSP, " ")
    with (COURSE / "experiments" / "comparison.csv").open(encoding="utf-8") as stream:
        runs = list(csv.DictReader(stream))
    summary = json.loads((COURSE / "experiments" / "summary.json").read_text(encoding="utf-8"))
    alert = json.loads((COURSE / "monitoring" / "model" / "demo.json").read_text(encoding="utf-8"))
    test = summary["test"]
    assert test["name"] == summary["selection"]["name"] == "baseline"
    assert summary["selection"]["selected_using"] == "validation_only"
    metrics = test["metrics"]
    for value in (
        pct(metrics["false_positive_rate"]),
        pct(metrics["point_precision"]),
        pct(metrics["point_recall"]),
        dec(metrics["point_f1"], 4),
        f"{metrics['false_positives']}/1 809",
        *(pct(float(run["false_positive_rate"])) for run in runs),
        *(dec(float(run["point_f1"]), 4) for run in runs),
    ):
        assert value.replace(NBSP, " ") in report, f"{value!r} is not in REPORT.md"
    assert metrics["events_detected"] == metrics["events_total"] == 9
    assert alert["passed"] is True
    events = {item["event"]: item["tick"] for item in alert["events"]}
    assert events == {"firing": 33, "resolved": 54}, events
    for fact in ("firing` на тике 33", "`resolved` на тике 54", "19 проверок `true`",
                 "8 061 наблюдение", "train 4 029 / validation 2 016 / test 2 016",
                 "20 проверок пройдено", "14 проверок провалено", "190 из 200",
                 "FPR 1,66 % (52/3 133)", "13 тестовых случаев"):
        assert fact in report, f"{fact!r} is not in REPORT.md"
    return runs, summary, alert


# ------------------------------------------------------------------------ helpers


def shape_box(slide, x, y, w, h, fill=None, line=None, width=1.0, radius=None, dash=False,
              kind=None, name=None):
    if kind is None:
        kind = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    shape = slide.shapes.add_shape(kind, inch(x), inch(y), inch(w), inch(h))
    if radius and kind == MSO_SHAPE.ROUNDED_RECTANGLE:
        shape.adjustments[0] = min(0.5, radius / min(w, h))
    if fill:
        shape.fill.solid()
        shape.fill.fore_color.rgb = rgb(fill)
    else:
        shape.fill.background()
    if line:
        shape.line.color.rgb = rgb(line)
        shape.line.width = Pt(width)
        if dash:
            shape.line.dash_style = 4  # MSO_LINE.DASH
    else:
        shape.line.fill.background()
    shape.shadow.inherit = False
    if name:
        shape.name = name
    _no_text(shape)
    return shape


def _no_text(shape):
    frame = shape.text_frame
    frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0


def _runs(paragraph, runs, size, color, bold, font):
    for item in runs:
        text, opts = (item, {}) if isinstance(item, str) else item
        run = paragraph.add_run()
        run.text = text
        run.font.name = opts.get("font", font)
        run.font.size = Pt(opts.get("size", size))
        run.font.bold = opts.get("bold", bold)
        run.font.italic = opts.get("italic", False)
        run.font.color.rgb = rgb(opts.get("color", color))
        if opts.get("font", font) == MONO:
            run.font._rPr.set("lang", "en-US")


def fill_text(frame, paras, size=14, color=TEXT, bold=False, align="l", anchor="t",
              font=SANS, spacing=1.0, after=0.0, inset=(0, 0, 0, 0)):
    """paras: str | list of (str | list of runs | dict(runs=..., align=..., ...))."""
    frame.word_wrap = True
    frame.auto_size = MSO_AUTO_SIZE.NONE
    frame.margin_left, frame.margin_top, frame.margin_right, frame.margin_bottom = (
        inch(v) for v in inset)
    frame.vertical_anchor = {"t": MSO_ANCHOR.TOP, "m": MSO_ANCHOR.MIDDLE,
                             "b": MSO_ANCHOR.BOTTOM}[anchor]
    if isinstance(paras, str):
        paras = [paras]
    for index, para in enumerate(paras):
        opts = para if isinstance(para, dict) else {"runs": para}
        runs = opts["runs"]
        if isinstance(runs, str):
            runs = [runs]
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER,
                               "r": PP_ALIGN.RIGHT}[opts.get("align", align)]
        paragraph.line_spacing = opts.get("spacing", spacing)
        gap = opts.get("after", after)
        if gap and index < len(paras) - 1:
            paragraph.space_after = Pt(gap)
        if opts.get("bullet"):
            ppr = paragraph._p.get_or_add_pPr()
            ppr.set("marL", str(inch(0.22)))
            ppr.set("indent", str(-inch(0.22)))
            bullet_color = ppr.makeelement(qn("a:buClr"), {})
            bullet_color.append(bullet_color.makeelement(
                qn("a:srgbClr"), {"val": opts.get("bullet_color", TEAL)}))
            ppr.append(bullet_color)
            ppr.append(ppr.makeelement(qn("a:buFont"), {"typeface": SANS}))
            ppr.append(ppr.makeelement(qn("a:buChar"), {"char": "•"}))
        _runs(paragraph, runs, opts.get("size", size), opts.get("color", color),
              opts.get("bold", bold), opts.get("font", font))


def text(slide, x, y, w, h, paras, name=None, **opts):
    box = slide.shapes.add_textbox(inch(x), inch(y), inch(w), inch(h))
    fill_text(box.text_frame, paras, **opts)
    if name:
        box.name = name
    return box


def card(slide, x, y, w, h, paras=None, fill=CARD, line=None, radius=0.12, pad=0.2,
         name=None, **opts):
    shape = shape_box(slide, x, y, w, h, fill=fill, line=line, radius=radius, name=name,
                      dash=opts.pop("dash", False))
    if paras is not None:
        fill_text(shape.text_frame, paras, inset=(pad, pad, pad, pad), **opts)
    return shape


def line(slide, x1, y1, x2, y2, color=MUTED, width=1.5, arrow=False, dash=False, name=None):
    connector = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, inch(x1), inch(y1),
                                           inch(x2), inch(y2))
    connector.line.color.rgb = rgb(color)
    connector.line.width = Pt(width)
    if dash:
        connector.line.dash_style = 4
    if arrow:
        ln = connector.line._get_or_add_ln()
        ln.append(ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"}))
    if name:
        connector.name = name
    return connector


def polyline(slide, points, color, width=2.0, name=None):
    builder = slide.shapes.build_freeform(inch(points[0][0]), inch(points[0][1]))
    builder.add_line_segments([(inch(x), inch(y)) for x, y in points[1:]], close=False)
    shape = builder.convert_to_shape()
    shape.fill.background()
    shape.line.color.rgb = rgb(color)
    shape.line.width = Pt(width)
    shape.shadow.inherit = False
    if name:
        shape.name = name
    return shape


def dot(slide, x, y, d, color, name=None):
    return shape_box(slide, x - d / 2, y - d / 2, d, d, fill=color, kind=MSO_SHAPE.OVAL,
                     name=name)


def badge(slide, x, y, d, kind, color):
    """Status icon in a circle: check, cross, warn or a short label (number)."""
    shape_box(slide, x, y, d, d, fill=color, kind=MSO_SHAPE.OVAL, name=f"Icon {kind}")
    s = d / 10
    if kind == "check":
        polyline(slide, [(x + 2.6 * s, y + 5.2 * s), (x + 4.3 * s, y + 6.9 * s),
                         (x + 7.4 * s, y + 3.4 * s)], BG, width=max(1.5, d * 9))
    elif kind == "cross":
        line(slide, x + 3.2 * s, y + 3.2 * s, x + 6.8 * s, y + 6.8 * s, BG, width=max(1.5, d * 9))
        line(slide, x + 6.8 * s, y + 3.2 * s, x + 3.2 * s, y + 6.8 * s, BG, width=max(1.5, d * 9))
    else:
        label = "!" if kind == "warn" else kind
        text(slide, x, y, d, d, [[(label, {"bold": True})]], size=max(10, round(d * 30)),
             color=BG, align="c", anchor="m")


def chip(slide, x, y, label, color, w=None, dashed=False, size=10):
    w = w or 0.16 + len(label) * size * 0.0082
    shape = shape_box(slide, x, y, w, 0.26, fill=None if dashed else color,
                      line=color if dashed else None, radius=0.13, dash=dashed,
                      name=f"Chip {label}")
    fill_text(shape.text_frame, [[(label, {"bold": True})]], size=size,
              color=color if dashed else BG, align="c", anchor="m")
    return w


def implemented(slide, x, y):
    chip(slide, x, y, "реализовано", TEAL, w=1.12)


def designed(slide, x, y):
    chip(slide, x, y, "проект", MUTED, w=0.78, dashed=True)


def frame(prs, number, kicker, title, notes):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = rgb(BG)
    if kicker:
        text(slide, 0.6, 0.38, 9.0, 0.3, [[(f"{number:02d}   ", {"color": MUTED}),
                                          (kicker, {})]],
             size=12, color=TEAL, bold=True, name="Kicker")
    if title:
        text(slide, 0.6, 0.66, 12.13, 0.66, title, size=30, bold=True, name="Title",
             spacing=0.95)
    text(slide, 0.6, 7.05, 8.0, 0.25, "DB Monitoring · итоговый проект курса «Интеграция ИИ"
         " в разработку»", size=10, color=MUTED, name="Footer")
    text(slide, 11.73, 7.05, 1.0, 0.25, f"{number} / {TOTAL}", size=10, color=MUTED,
         align="r", name="Slide number")
    slide.notes_slide.notes_text_frame.text = notes
    return slide


def arrow_right(slide, x, y, w, color=MUTED):
    line(slide, x, y, x + w, y, color, width=1.75, arrow=True)


def score_chart(slide, x, y, w, h, timeline, lo, hi, *, axes=True, line_color=TEAL,
                width=2.0):
    """Score per tick from demo.json; returns a function mapping (tick, score) to inches."""
    first, last = timeline[0]["tick"], timeline[-1]["tick"]

    def at(tick, score):
        return (x + (tick - first) / (last - first) * w, y + (hi - score) / (hi - lo) * h)

    segment = []
    for row in timeline:
        if row["score"] is None:
            if len(segment) > 1:
                polyline(slide, segment, line_color, width, name="Score line")
            segment = []
            continue
        segment.append(at(row["tick"], row["score"]))
    if len(segment) > 1:
        polyline(slide, segment, line_color, width, name="Score line")
    for row in timeline:
        if row["is_anomaly"]:
            dot(slide, *at(row["tick"], row["score"]), 0.11 if axes else 0.07, CORAL,
                name=f"Anomaly tick {row['tick']}")
    return at


# ------------------------------------------------------------------------- slides


def slide_title(prs, alert):
    slide = frame(prs, 1, None, None,
                  "≈10 с · Раиль. Проект DB Monitoring: ищем аномалии в метриках качества "
                  "данных. Покажем, что построено, какие результаты получены и где "
                  "ограничения. Команда из четырёх человек, зоны ответственности на слайде.")
    text(slide, 0.6, 0.9, 7.6, 0.3, "ИТОГОВЫЙ ПРОЕКТ · «ИНТЕГРАЦИЯ ИИ В РАЗРАБОТКУ»",
         size=12, color=TEAL, bold=True)
    text(slide, 0.6, 1.3, 7.6, 1.0, "DB Monitoring", size=54, bold=True)
    text(slide, 0.6, 2.4, 7.75, 0.5, "Обнаружение аномалий в метриках качества данных",
         size=22, color=TEXT)
    text(slide, 0.6, 3.15, 7.4, 0.95,
         "ML-компонент (Isolation Forest) в системе мониторинга БД: датасет, эксперименты"
         " MLflow, тесты, сервис инференса, контейнеры и мониторинг", size=15, color=MUTED)
    # Real score trace as the visual motif.
    card(slide, 8.55, 0.9, 4.18, 3.3, fill=CARD, name="Score card")
    text(slide, 8.8, 1.08, 3.7, 0.3, "score demo-модели, реальный прогон", size=11,
         color=MUTED)
    lo, hi = -0.12, 0.28
    at = score_chart(slide, 8.85, 1.55, 3.6, 1.95, alert["timeline"], lo, hi, axes=False)
    x0, y0 = at(1, 0)
    x1, _ = at(61, 0)
    line(slide, x0, y0, x1, y0, GRID, width=1, dash=True)
    text(slide, 8.8, 3.68, 3.7, 0.42, [[("score < 0", {"color": CORAL, "bold": True}),
                                        " → аномалия → алерт"]], size=12, color=TEXT)
    team = [("Александр", "aleksandr-novikov", "Архитектура, MLflow, версии моделей, интеграция"),
            ("Раиль", "rail-ss", "Постановка задачи, roadmap, отчёт и презентация"),
            ("Виталий", "vtm9", "Тесты, CI, Docker, выпуск и эксплуатация"),
            ("Оксана", "ksdergach", "Данные, валидация, API и коллекторы")]
    text(slide, 0.6, 4.62, 6.0, 0.3, "КОМАНДА И ЗОНЫ ОТВЕТСТВЕННОСТИ", size=12, color=MUTED,
         bold=True)
    w = (12.13 - 3 * 0.3) / 4
    for index, (name, handle, zone) in enumerate(team):
        x = 0.6 + index * (w + 0.3)
        card(slide, x, 5.0, w, 1.6, [
            {"runs": [(name, {"bold": True, "size": 17})], "after": 2},
            {"runs": [(handle, {"color": TEAL, "font": MONO, "size": 11})], "after": 6},
            {"runs": [zone], "size": 13, "color": MUTED}], name=f"Team {name}")


def slide_problem(prs):
    slide = frame(prs, 2, "ПРОБЛЕМА", "БД работает, а данные уже испорчены",
                  "≈25 с · Раиль. Отказ БД виден сразу, а порча данных — нет: таблица "
                  "доступна, запросы проходят, но растёт доля NULL или ломается объём. "
                  "Это доходит до отчётности. Пользователь — дата-инженер или аналитик; "
                  "цели качества мы зафиксировали заранее, к ним вернёмся на слайде 6.")
    kinds = [
        ("Всплеск NULL", "резко растёт доля пропусков",
         [(0, .70), (.25, .68), (.5, .71), (.62, .69), (.7, .15), (.8, .2), (1, .66)]),
        ("Скачок объёма", "нетипично меняется число строк",
         [(0, .72), (.3, .70), (.55, .68), (.56, .2), (.8, .22), (1, .2)]),
        ("Смена режима", "ломается привычная динамика",
         [(0, .7), (.12, .5), (.24, .7), (.36, .5), (.48, .7), (.6, .45), (.8, .42), (1, .40)]),
    ]
    w = 2.3
    for index, (title, desc, points) in enumerate(kinds):
        x = 0.6 + index * (w + 0.3)
        card(slide, x, 1.7, w, 2.75, name=f"Incident {title}")
        px, py, pw, ph = x + 0.25, 1.95, w - 0.5, 1.05
        line(slide, px, py + ph, px + pw, py + ph, GRID, width=1)
        polyline(slide, [(px + a * pw, py + b * ph) for a, b in points],
                 CORAL if index != 2 else AMBER, width=2.5)
        text(slide, x + 0.25, 3.2, w - 0.35, 0.35, title, size=16, bold=True)
        text(slide, x + 0.25, 3.6, w - 0.4, 0.75, desc, size=13, color=MUTED)
    text(slide, 0.6, 4.55, 7.5, 0.3, "Графики схематичные, не данные проекта", size=10,
         color=MUTED)
    card(slide, 0.6, 5.05, 7.5, 1.55, fill=CARD2, name="Gap statement")
    badge(slide, 0.85, 5.33, 0.42, "warn", AMBER)
    text(slide, 1.5, 5.25, 6.4, 1.2, [
        {"runs": [("Мониторинг доступности этого не видит", {"bold": True})], "after": 4},
        {"runs": ["БД отвечает, запросы проходят — ошибочные данные доходят до отчётов. "
                  "Цель: сократить время обнаружения таких инцидентов."],
         "size": 14, "color": MUTED}], size=16)
    card(slide, 8.45, 1.7, 4.28, 1.45, [
        {"runs": [("ПОЛЬЗОВАТЕЛЬ", {"bold": True, "color": TEAL, "size": 12})], "after": 4},
        {"runs": ["Дата-инженер или аналитик, отвечающий за качество данных. "
                  "Система подсвечивает, решает человек."], "size": 14}], name="User")
    card(slide, 8.45, 3.35, 4.28, 3.25, name="Targets")
    text(slide, 8.65, 3.52, 3.9, 0.55, "ЦЕЛИ, ЗАФИКСИРОВАННЫЕ ДО ЭКСПЕРИМЕНТОВ", size=12,
         color=TEAL, bold=True)
    targets = [("≥" + NBSP + "80" + NBSP + "%", "размеченных инцидентов обнаружено"),
               ("≤" + NBSP + "10" + NBSP + "%", "ложных сигналов вне инцидентов (FPR)"),
               ("≤" + NBSP + "1", "цикла (15 мин) — медианная задержка")]
    for index, (value, label) in enumerate(targets):
        y = 4.15 + index * 0.78
        text(slide, 8.65, y, 1.45, 0.6, value, size=24, bold=True, color=TEXT, anchor="m")
        text(slide, 10.15, y, 2.45, 0.6, label, size=13, color=MUTED, anchor="m")


def slide_system(prs):
    slide = frame(prs, 3, "ЧТО ДЕЛАЕТ СИСТЕМА", "От метрик таблицы к решению специалиста",
                  "≈20 с · Оксана. Система по расписанию собирает метрики таблиц, строит "
                  "четыре признака и отдаёт их Isolation Forest. Результат — оценка и флаг "
                  "с версией модели: на дашборде, в API, в уведомлении. Решение принимает "
                  "специалист.")
    steps = [
        ("Источник", ["PostgreSQL, MySQL, ClickHouse, Iceberg"], BLUE),
        ("Сбор метрик", ["по расписанию, по умолчанию раз в 15" + NBSP + "мин"], BLUE),
        ("4 признака", ["row_count", "null_rate", "d_row_count", "d_null_rate"], BLUE),
        ("Isolation Forest", ["score, «норма / аномалия», версия модели"], TEAL),
        ("Результат", ["дашборд, API, Telegram, гипотеза причины"], TEAL),
    ]
    w, gap, y, h = 2.11, 0.395, 1.75, 2.0
    for index, (title, desc, color) in enumerate(steps):
        x = 0.6 + index * (w + gap)
        card(slide, x, y, w, h, name=f"Step {title}")
        badge(slide, x + 0.2, y + 0.22, 0.4, str(index + 1), color)
        text(slide, x + 0.2, y + 0.75, w - 0.35, 0.4, title, size=16, bold=True)
        font = MONO if index == 2 else SANS
        size = 11 if index == 2 else 12
        text(slide, x + 0.2, y + 1.15, w - 0.3, 0.8,
             [[(part, {"font": font})] for part in desc], size=size, color=MUTED)
        if index < len(steps) - 1:
            arrow_right(slide, x + w + 0.06, y + h / 2, gap - 0.12)
    card(slide, 0.6, 4.1, 5.9, 1.65, [
        {"runs": [("ВХОД", {"bold": True, "color": BLUE, "size": 12})], "after": 4},
        {"runs": ["Метрики каждой таблицы: число строк, доля NULL, распределения, схема. "
                  "Модель видит объём, долю NULL и их изменения между соседними "
                  "наблюдениями."], "size": 14}], name="Input")
    card(slide, 6.83, 4.1, 5.9, 1.65, [
        {"runs": [("ВЫХОД", {"bold": True, "color": TEAL, "size": 12})], "after": 4},
        {"runs": ["Для наблюдения: объект, время, признаки, «норма / аномалия», score и "
                  "версия модели. Не исправляет данные и не решает за человека."],
         "size": 14}], name="Output")
    card(slide, 0.6, 5.95, 12.13, 0.72, fill=TEAL_DARK, name="Principle")
    badge(slide, 0.8, 6.11, 0.4, "check", TEAL)
    text(slide, 1.4, 5.95, 11.1, 0.72, [[
        ("Сигнал модели — повод для проверки, а не доказательство ошибки. ", {"bold": True}),
        ("Решение остаётся за специалистом.", {"color": MUTED})]], size=15, anchor="m")


def slide_architecture(prs):
    slide = frame(prs, 4, "АРХИТЕКТУРА", "Flask-монолит и отдельный сервис инференса",
                  "≈30 с · Александр. Flask с планировщиком собирает метрики в хранилище и "
                  "показывает дашборд. Инференс вынесен в FastAPI: он проверяет токен, "
                  "источник и неизменяемую версию модели из реестра и не обучает модель сам "
                  "(ADR-001). Пунктир внизу — то, что спроектировано, но не развёрнуто.")
    cols = [0.6, 4.86, 9.12]
    w = 3.61

    def node(col, y, h, title, lines, name, built=True):
        x = cols[col]
        card(slide, x, y, w, h, line=EDGE if built else None, name=name)
        text(slide, x + 0.2, y + 0.14, w - 1.5, 0.35, title, size=15, bold=True)
        if built:
            implemented(slide, x + w - 1.3, y + 0.17)
        text(slide, x + 0.2, y + 0.52, w - 0.35, h - 0.6, lines, size=12, color=MUTED)

    node(0, 1.55, 1.0, "Источники данных", "PostgreSQL, MySQL, ClickHouse, Iceberg", "Sources")
    node(1, 1.55, 1.0, "Сборщики", "APScheduler: каждые 5–1440 мин, обычно 15", "Collectors")
    node(2, 1.55, 1.0, "Хранилище метрик", "TimescaleDB / PostgreSQL / SQLite", "Storage")
    node(2, 3.0, 1.25, "Flask", "Дашборд, REST API, Telegram, гипотеза причины от LLM",
         "Flask")
    node(1, 3.0, 1.25, "FastAPI-инференс",
         [[("/healthz  /v1/model  /v1/predict", {"font": MONO, "size": 11})],
          "токен + полный источник + версия"], "FastAPI")
    node(0, 3.0, 1.25, "Реестр моделей",
         "skops + манифест; явный выбор версии и допуск качества", "Registry")
    node(0, 4.7, 1.15, "Ночное обучение", "кандидаты по истории из хранилища, без активации",
         "Training")
    node(1, 4.7, 1.15, "Наблюдаемость",
         "/healthz, /metrics, JSON-логи; 2 алерта → журнал", "Observability")
    node(2, 4.7, 1.15, "Специалист", "видит оценку и версию модели, принимает решение",
         "User", built=False)
    arrow_right(slide, cols[0] + w + 0.07, 2.05, 0.5)
    arrow_right(slide, cols[1] + w + 0.07, 2.05, 0.5)
    line(slide, cols[2] + w / 2, 2.6, cols[2] + w / 2, 2.95, MUTED, 1.75, arrow=True)
    line(slide, cols[2] - 0.07, 3.45, cols[1] + w + 0.07, 3.45, BLUE, 1.75, arrow=True)
    line(slide, cols[1] + w + 0.07, 3.85, cols[2] - 0.07, 3.85, TEAL, 1.75, arrow=True)
    text(slide, cols[1] + w - 0.05, 3.1, 0.75, 0.3, "признаки", size=10, color=BLUE,
         align="c")
    text(slide, cols[1] + w - 0.05, 3.9, 0.75, 0.3, "score", size=10, color=TEAL, align="c")
    arrow_right(slide, cols[0] + w + 0.07, 3.62, 0.5)
    line(slide, cols[0] + w / 2, 4.65, cols[0] + w / 2, 4.3, MUTED, 1.75, arrow=True)
    line(slide, cols[2] + w / 2, 4.3, cols[2] + w / 2, 4.65, MUTED, 1.75, arrow=True)
    card(slide, 0.6, 6.08, 12.13, 0.78, fill=None, line=MUTED, dash=True, name="Designed")
    designed(slide, 0.82, 6.34)
    text(slide, 1.78, 6.08, 10.8, 0.78,
         "Prometheus-сервер · Grafana · доставка алертов вовне · переключатель трафика · "
         "shadow / canary · автоматическое переобучение", size=14, color=MUTED, anchor="m")


def slide_data(prs, runs):
    slide = frame(prs, 5, "ДАННЫЕ И ЭКСПЕРИМЕНТЫ",
                  "Синтетический датасет и пять экспериментов MLflow",
                  "≈25 с · Оксана. Синтетическая история трёх таблиц с размеченными "
                  "инцидентами, разбиение строго по времени; train — только норма. Перед "
                  "обучением 20 проверок данных. Протокол зафиксирован в Git до запусков: "
                  "5 конфигураций, выбор по validation, test — один раз. contamination 0,05 "
                  "исключена: FPR 24,5 %.")
    text(slide, 0.6, 1.5, 6.2, 0.4, [[("Seed 42 · ", {"color": TEAL, "bold": True}),
                                      ("3 таблицы · шаг 15 мин · " + num(8061)
                                       + " наблюдений", {"bold": True})]], size=16)
    total = 4029 + 2016 + 2016
    x, y, width, h = 0.6, 2.05, 6.2, 0.95
    parts = [("train", 4029, "только норма", BLUE_DARK, BLUE),
             ("validation", 2016, "9 инцидентов", TEAL_DARK, TEAL),
             ("test", 2016, "9 инцидентов", CORAL_DARK, CORAL)]
    for label, count, desc, fill, color in parts:
        w = width * count / total
        card(slide, x, y, w - 0.06, h, [
            {"runs": [(label, {"bold": True, "color": color})], "after": 2},
            {"runs": [num(count)], "size": 13, "color": TEXT, "bold": True},
            {"runs": [desc], "size": 12, "color": MUTED}],
            fill=fill, radius=0.08, pad=0.14, size=14, name=f"Split {label}")
        x += w
    line(slide, 0.6, 3.17, 6.8, 3.17, MUTED, 1.25, arrow=True)
    text(slide, 0.6, 3.22, 6.2, 0.3, "время →  scaler обучается только на train",
         size=11, color=MUTED)
    card(slide, 0.6, 3.75, 6.2, 1.35, name="Validation")
    badge(slide, 0.82, 4.22, 0.4, "check", TEAL)
    text(slide, 1.4, 3.85, 5.25, 1.15, anchor="m", paras=[
        {"runs": [("Валидация данных: 20 из 20 проверок", {"bold": True})], "after": 3},
        {"runs": ["На испорченной копии — 14 провалов, обучение не стартует"],
         "size": 14, "color": MUTED}], size=16)
    card(slide, 0.6, 5.35, 6.2, 1.35, name="Protocol")
    badge(slide, 0.82, 5.82, 0.4, "check", TEAL)
    text(slide, 1.4, 5.45, 5.25, 1.15, anchor="m", paras=[
        {"runs": [("Протокол в Git до запусков", {"bold": True})], "after": 3},
        {"runs": ["Один параметр за раз; выбор — max F1 среди выполнивших цели на "
                  "validation; test — один раз, отдельным run"],
         "size": 14, "color": MUTED}], size=16)
    # FPR on validation for the five runs.
    cx, cw = 7.2, 5.53
    card(slide, cx, 1.5, cw, 5.2, name="Validation runs")
    text(slide, cx + 0.25, 1.65, cw - 0.5, 0.35, "5 runs на validation: FPR и F1", size=15,
         bold=True)
    text(slide, cx + 0.25, 1.98, cw - 0.5, 0.3, "Все пять нашли 9 / 9 инцидентов", size=12,
         color=MUTED)
    bar_x, bar_w, scale = cx + 1.95, 1.75, 0.25
    top, step = 2.7, 0.62
    for index, run in enumerate(runs):
        fpr, f1 = float(run["false_positive_rate"]), float(run["point_f1"])
        ry = top + index * step
        chosen, excluded = run["name"] == "baseline", fpr > 0.10
        color = TEAL if chosen else CORAL if excluded else BLUE
        text(slide, cx + 0.25, ry, 1.65, 0.42, [[(run["name"], {"font": MONO})]], size=11,
             color=TEXT if chosen else MUTED, anchor="m", bold=chosen)
        shape_box(slide, bar_x, ry + 0.08, max(0.04, bar_w * fpr / scale), 0.26, fill=color,
                  name=f"FPR {run['name']}")
        text(slide, cx + cw - 1.75, ry, 0.95, 0.42, pct(fpr), size=12, color=color,
             anchor="m", align="r", bold=True)
        text(slide, cx + cw - 0.85, ry, 0.65, 0.42, dec(f1, 3), size=12, color=MUTED,
             anchor="m", align="r")
    limit = bar_x + bar_w * 0.10 / scale
    line(slide, limit, 2.62, limit, top + 5 * step - 0.1, AMBER, 1.25, dash=True)
    text(slide, limit - 0.5, 2.36, 1.0, 0.25, "10" + NBSP + "%", size=10, color=AMBER,
         align="c")
    text(slide, cx + cw - 0.85, 2.36, 0.65, 0.25, "F1", size=10, color=MUTED, align="r")
    text(slide, cx + cw - 1.75, 2.36, 0.95, 0.25, "FPR", size=10, color=MUTED, align="r")
    text(slide, cx + 0.25, 5.95, cw - 0.5, 0.65, [
        [("baseline", {"color": TEAL, "bold": True}),
         " — выбрана (max F1 среди допустимых)"],
        [("contamination_050", {"color": CORAL, "bold": True}), " — исключена по FPR"]],
        size=12, color=MUTED, after=2)


def slide_results(prs, summary):
    m = summary["test"]["metrics"]
    validation = next(r for r in summary["validation"] if r["name"] == "baseline")["metrics"]
    slide = frame(prs, 6, "РЕЗУЛЬТАТ ОСНОВНОЙ МОДЕЛИ · TEST",
                  "9 из 9 инцидентов найдено, но FPR выше цели",
                  "≈40 с · Александр. Ключевой слайд. На независимом test baseline нашла "
                  "все 9 инцидентов, медианная задержка 0 циклов. Но 13,27 % ложных "
                  "сигналов при цели 10 % — модель не прошла допуск. 9/9 — полнота по "
                  "событиям; по отдельным наблюдениям recall 44 %. Порог не ослабляли и "
                  "contamination по test не подбирали. Demo-модель стенда сюда не входит.")
    stats = [
        (f"{m['events_detected']} / {m['events_total']}", "инцидентов обнаружено",
         "цель ≥" + NBSP + "80" + NBSP + "% · выполнена", TEAL, "check"),
        (pct(m["false_positive_rate"]),
         f"ложных сигналов: {m['false_positives']} из {num(1809)} нормальных",
         "цель ≤" + NBSP + "10" + NBSP + "% · не выполнена", CORAL, "cross"),
        (dec(m["abrupt_median_delay_ticks"], 0) + " циклов",
         "медианная задержка резких инцидентов", "цель ≤" + NBSP + "1 цикла · выполнена",
         TEAL, "check"),
    ]
    w = (12.13 - 2 * 0.3) / 3
    for index, (value, label, target, color, icon) in enumerate(stats):
        x = 0.6 + index * (w + 0.3)
        card(slide, x, 1.55, w, 2.3, line=color if icon == "cross" else None,
             fill=CORAL_DARK if icon == "cross" else CARD, name=f"Stat {label}")
        text(slide, x + 0.25, 1.68, w - 0.5, 0.95, value, size=44, bold=True, color=color,
             anchor="m")
        text(slide, x + 0.25, 2.62, w - 0.5, 0.6, label, size=14, color=TEXT)
        badge(slide, x + 0.25, 3.3, 0.34, icon, color)
        text(slide, x + 0.7, 3.27, w - 0.9, 0.4, target, size=13, color=color, bold=True,
             anchor="m")
    # Validation vs test FPR.
    card(slide, 0.6, 4.1, 5.9, 2.6, name="FPR chart")
    text(slide, 0.85, 4.22, 5.4, 0.35, "FPR baseline: validation → test", size=15,
         bold=True)
    base_y, chart_h, scale = 6.3, 1.4, 0.15
    for index, (label, value, color) in enumerate(
            [("validation", validation["false_positive_rate"], BLUE),
             ("test", m["false_positive_rate"], CORAL)]):
        bx = 1.35 + index * 1.75
        bh = chart_h * value / scale
        shape_box(slide, bx, base_y - bh, 1.0, bh, fill=color, name=f"FPR bar {label}")
        text(slide, bx - 0.3, base_y - bh - 0.35, 1.6, 0.32, pct(value), size=14, bold=True,
             color=TEXT, align="c")
        text(slide, bx - 0.3, base_y + 0.03, 1.6, 0.3, label, size=12, color=MUTED,
             align="c")
    limit_y = base_y - chart_h * 0.10 / scale
    line(slide, 1.05, limit_y, 4.75, limit_y, AMBER, 1.5, dash=True)
    text(slide, 4.82, limit_y - 0.17, 1.6, 0.34, "цель 10" + NBSP + "%", size=12,
         color=AMBER, bold=True, anchor="m")
    line(slide, 1.05, base_y, 4.75, base_y, GRID, 1)
    # Event recall vs point recall and the verdict.
    card(slide, 6.83, 4.1, 5.9, 1.35, [
        {"runs": [("Две полноты — не путать", {"bold": True})], "after": 4},
        {"runs": [("по событиям: ", {"color": MUTED}), ("9 / 9 = 100" + NBSP + "%",
                                                        {"bold": True})], "size": 14,
         "after": 1},
        {"runs": [("по наблюдениям: ", {"color": MUTED}),
                  ("recall " + pct(m["point_recall"]) + " · precision "
                   + pct(m["point_precision"]), {"bold": True})], "size": 14}],
        size=15, pad=0.18, name="Recall kinds")
    card(slide, 6.83, 5.6, 5.9, 1.1, fill=CORAL_DARK, line=CORAL, name="Verdict")
    badge(slide, 7.03, 5.83, 0.4, "cross", CORAL)
    text(slide, 7.6, 5.66, 5.0, 1.0, [
        {"runs": [("Модель не допущена реестром", {"bold": True})], "after": 2},
        {"runs": ["Порог не ослабляли, параметры по test не подбирали"], "size": 13,
         "color": TEXT}], size=15, anchor="m")


def slide_testing(prs):
    slide = frame(prs, 7, "ТЕСТИРОВАНИЕ И НАЙДЕННЫЕ ПРОБЛЕМЫ",
                  "Проверяем код, данные и поведение модели",
                  "≈30 с · Виталий. Тест-план построен по рискам: от unit до Docker-"
                  "интеграции с пятью хранилищами данных и E2E. Отдельная команда проверки "
                  "качества модели для baseline ожидаемо возвращает FAIL. Проблемы нашли "
                  "реальные: архивная модель вне области обучения на стенде, вероятностные "
                  "ложные тревоги demo-модели (не исправлены — это ограничение), нестабильная "
                  "проверка Compose (исправлена).")
    levels = [
        ("Unit и контракт", "признаки, OpenAPI, изоляция проектов · Python 3.11–3.13"),
        ("Интеграция модели", "настоящие IsolationForest и scaler → API"),
        ("Поведение модели", "13 случаев: NULL, объём, режим, шум, пропуски"),
        ("Docker-интеграция", "5 хранилищ данных и Compose-сценарий в CI"),
        ("E2E", "Chromium → Flask, push в master"),
    ]
    for index, (title, desc) in enumerate(levels):
        y = 1.55 + index * 0.78
        card(slide, 0.6, y, 5.6, 0.66, fill=CARD, name=f"Level {title}")
        text(slide, 0.8, y, 2.1, 0.66, title, size=14, bold=True, anchor="m")
        text(slide, 2.85, y, 3.25, 0.66, desc, size=12, color=MUTED, anchor="m")
    card(slide, 0.6, 5.5, 5.6, 1.2, fill=AMBER_DARK, name="Quality gate")
    text(slide, 0.8, 5.58, 5.25, 1.05, [
        {"runs": [("check_model_quality → код 1 (FAIL)", {"bold": True, "font": MONO,
                                                          "size": 13})], "after": 3},
        {"runs": ["для baseline — ожидаемо: зелёный pytest ≠ допуск модели. CI PR #40: "
                  "3 × test, integration, ruff — SUCCESS"], "size": 13, "color": TEXT}],
        size=14, anchor="m")
    text(slide, 6.55, 1.5, 6.2, 0.35, "НАЙДЕННЫЕ ПРОБЛЕМЫ → РЕАКЦИЯ", size=12, color=TEAL,
         bold=True)
    issues = [
        ("Архивная модель #9 на стенде: score −0,077 у всех точек — данные вне области"
         " обучения; генератор поднимал NULL в «норме»",
         "Генератор исправлен, для стенда — отдельная demo-модель", "исправлено", TEAL),
        ("Demo-модель: сценарий проходит в 190 из 200 — ≈1 ложная тревога на 20 показов;"
         " CI упал на check-outcome",
         "Зафиксировано как ограничение; порог и seed не подбирали", "ограничение", AMBER),
        ("Проверка Compose теряла тик: окно с микросекундами, метки — с секундами",
         "Исправлено, добавлены регрессионные тесты", "исправлено", TEAL),
    ]
    for index, (problem, reaction, status, color) in enumerate(issues):
        y = 1.95 + index * 1.6
        card(slide, 6.55, y, 6.18, 1.45, name=f"Issue {index + 1}")
        text(slide, 6.75, y + 0.13, 4.45, 0.8, problem, size=12, color=TEXT)
        chip(slide, 11.38, y + 0.15, status, color, w=1.17)
        text(slide, 6.75, y + 0.95, 5.8, 0.4, [[("→ ", {"color": color, "bold": True}),
                                                (reaction, {"bold": True})]],
             size=13, color=TEXT)


def slide_operations(prs):
    slide = frame(prs, 8, "ЗАПУСК, ДЕПЛОЙ И МОНИТОРИНГ",
                  "Стенд и алерты проверены, релиз — только проект",
                  "≈25 с · Виталий. Docker Compose поднимает источник, хранилище, Flask и "
                  "FastAPI; порты только на localhost. CI проходит маршрут из чистого "
                  "checkout. Есть технические метрики и два алерта, они пишутся в журнал. "
                  "Blue-green, shadow/canary и откат описаны, но не испытаны; ручной "
                  "независимый прогон стенда не выполнен.")
    card(slide, 0.6, 1.5, 6.3, 2.35, name="Compose")
    text(slide, 0.82, 1.62, 4.2, 0.4, "Docker Compose-стенд", size=16, bold=True)
    implemented(slide, 5.55, 1.67)
    services = [("PostgreSQL", "источник"), ("TimescaleDB", "метрики"),
                ("Flask", ":15001"), ("FastAPI", ":18000")]
    for index, (name, role) in enumerate(services):
        x = 0.82 + index * 1.47
        card(slide, x, 2.12, 1.35, 0.82, [
            {"runs": [(name, {"bold": True})], "after": 1},
            {"runs": [role], "size": 11, "color": MUTED}], fill=CARD2, size=12, pad=0.1,
            align="c", anchor="m", name=f"Service {name}")
    text(slide, 0.82, 3.02, 5.9, 0.8, [
        {"runs": ["Порты только на 127.0.0.1, .env без внешних ключей"], "after": 3},
        {"runs": [("CI из чистого checkout: ", {"color": TEXT, "bold": True}),
                  "сборка --no-cache, readiness, норма, инцидент, перезапуск БД"]}],
        size=12, color=MUTED)
    card(slide, 0.6, 4.05, 6.3, 2.65, name="Swagger")
    text(slide, 0.82, 4.15, 5.9, 0.35, [[("FastAPI-инференс: ", {"bold": True}),
                                         ("POST /v1/predict → 200", {"font": MONO,
                                                                     "size": 12})]],
         size=14)
    left, top, right, bottom = 104, 1206, 770, 1358  # response body, pixels of the PNG
    width = 5.86
    height = width * (bottom - top) / (right - left)
    picture = slide.shapes.add_picture(str(SWAGGER), inch(0.82), inch(4.6), inch(width),
                                       inch(height))
    source_w, source_h = 1384, 4329
    picture.crop_left, picture.crop_right = left / source_w, 1 - right / source_w
    picture.crop_top, picture.crop_bottom = top / source_h, 1 - bottom / source_h
    picture.name = "Swagger response (real screenshot)"
    text(slide, 0.82, 4.6 + height + 0.1, 5.86, 0.45,
         "Фрагмент реального снимка Swagger UI от 25.10.2026 (inference/evidence): score, "
         "is_anomaly, версия модели, demo, quality_admitted", size=10, color=MUTED)
    card(slide, 7.2, 1.5, 5.53, 2.6, name="Monitoring")
    text(slide, 7.42, 1.62, 3.8, 0.4, "Мониторинг и алерты", size=16, bold=True)
    implemented(slide, 11.38, 1.67)
    text(slide, 7.42, 2.05, 5.1, 0.6, [[("/healthz · /metrics", {"font": MONO, "size": 12}),
                                        " · JSON-логи с request_id · p95 латентности"]],
         size=12, color=MUTED)
    alerts = [("HttpErrorRatioHigh", "5xx > 10" + NBSP + "% за 5 мин"),
              ("ModelAnomalyRateHigh", "≥ 3 аномалий из 20 последних")]
    for index, (name, rule) in enumerate(alerts):
        y = 2.68 + index * 0.68
        card(slide, 7.42, y, 5.1, 0.58, fill=CARD2, radius=0.08, name=f"Alert {name}")
        text(slide, 7.6, y, 2.6, 0.58, [[(name, {"font": MONO, "bold": True})]], size=11,
             anchor="m")
        text(slide, 10.2, y, 2.25, 0.58, rule, size=11, color=MUTED, anchor="m")
    card(slide, 7.2, 4.3, 5.53, 2.4, fill=None, line=MUTED, dash=True, name="Designed only")
    text(slide, 7.42, 4.42, 3.8, 0.4, "Спроектировано, не испытано", size=16, bold=True)
    designed(slide, 11.73, 4.47)
    gaps = ["Blue-green приложения, shadow → canary модели",
            "Откат по runbook; release.yml не запускался",
            "Prometheus, Grafana, доставка алертов вовне",
            "Дрейф KS / PSI описан, не автоматизирован",
            "Ручной прогон стенда другим участником не выполнен"]
    text(slide, 7.42, 4.9, 5.1, 1.75,
         [{"runs": [gap], "bullet": True, "bullet_color": MUTED} for gap in gaps],
         size=13, color=MUTED, after=3)


def slide_ai(prs):
    slide = frame(prs, 9, "ИИ-АССИСТЕНТЫ В РАЗРАБОТКЕ",
                  "Ассистент предлагает, человек проверяет",
                  "≈20 с · Раиль. Codex и Claude Code писали код, тесты и документы. "
                  "Результат принимаем только с воспроизводимой проверкой и ревью другого "
                  "участника. Четыре примера, где ревью нашло и исправило недостатки. "
                  "Промпты не сохранялись, журнал ведётся до 08.11. LLM внутри продукта "
                  "только формулирует гипотезу причины.")
    card(slide, 0.6, 1.5, 12.13, 0.78, fill=TEAL_DARK, name="Rule")
    badge(slide, 0.82, 1.69, 0.4, "check", TEAL)
    text(slide, 1.42, 1.5, 11.1, 0.78, [[
        ("Правило: ", {"bold": True, "color": TEAL}),
        ("результат ассистента принят, только если есть команда, тест или артефакт "
         "и ревью другого участника", {"bold": True})]], size=15, anchor="m")
    examples = [
        ("#10 · Качество модели", "Поведенческие тесты, аудит архива",
         "Ревью Раиля: исправлена граница инцидента, закреплён SHA-256 архива"),
        ("#3 · Архитектура и ADR", "Схема, ADR Flask / FastAPI",
         "Ревью Раиля: уточнены источник активной модели и откат"),
        ("#18 · Аудит LLM-объяснений", "Анализ пути запроса, тесты",
         "Другой ассистент нашёл 6 замечаний — исправлены; APPROVED Оксаны"),
        ("#17 · Стратегия дрейфа", "Codex: стратегия и синтетический сценарий",
         "Ревью Раиля: проверка значений до KS / PSI и рост row_count; доработал "
         "Claude Code"),
    ]
    w, h = (12.13 - 0.3) / 2, 1.55
    for index, (title, made, review) in enumerate(examples):
        x = 0.6 + (index % 2) * (w + 0.3)
        y = 2.5 + (index // 2) * (h + 0.22)
        card(slide, x, y, w, h, name=f"AI {title}")
        text(slide, x + 0.22, y + 0.14, w - 0.44, 0.35, title, size=15, bold=True)
        text(slide, x + 0.22, y + 0.52, w - 0.44, 0.35, [[("ИИ: ", {"color": BLUE,
                                                                    "bold": True}), made]],
             size=13, color=MUTED)
        text(slide, x + 0.22, y + 0.88, w - 0.44, 0.6, [[("Человек: ", {"color": TEAL,
                                                                        "bold": True}),
                                                         review]], size=13, color=TEXT)
    card(slide, 0.6, 6.06, 5.9, 0.64, [[("Codex, Claude Code. ", {"bold": True}),
                                        ("Промпты не сохранялись, журнал до 08.11",
                                         {"color": MUTED})]],
         size=12, pad=0.15, anchor="m", name="Tools")
    card(slide, 6.83, 6.06, 5.9, 0.64, [[("LLM в продукте: ", {"bold": True}),
                                         ("только гипотеза причины; confidence не калиброван",
                                          {"color": MUTED})]],
         size=12, pad=0.15, anchor="m", name="LLM in product")


def slide_demo(prs, alert):
    slide = frame(prs, 10, "ДЕМОНСТРАЦИЯ",
                  "Норма → отклонение → алерт → восстановление",
                  "≈45 с · Виталий. Показываем сценарий demo-модели стенда: 30 нормальных "
                  "сборов, затем пачка аварийных строк — score уходит ниже нуля, на 33-м "
                  "сборе срабатывает ModelAnomalyRateHigh, после восстановления на 54-м "
                  "алерт снимается. Это demo-модель (demo=true), а не основная. Живой "
                  "показ — по README презентации; если стенд недоступен, этот график "
                  "построен из сохранённого прогона.")
    cx, cy, cw, ch = 0.6, 1.5, 8.35, 4.45
    card(slide, cx, cy, cw, ch, name="Demo chart")
    timeline = alert["timeline"]
    px, py, pw, ph = cx + 0.75, cy + 0.6, cw - 1.05, ch - 1.25
    lo, hi = -0.12, 0.32
    first, last = timeline[0]["tick"], timeline[-1]["tick"]

    def tx(tick):
        return px + (tick - first) / (last - first) * pw

    phases = [("норма", 1, 30.5, CARD), ("отклонение", 30.5, 36.5, CORAL_DARK),
              ("восстановление", 36.5, 61, CARD)]
    for label, start, end, fill in phases:
        if fill != CARD:
            shape_box(slide, tx(start), py, tx(end) - tx(start), ph, fill=fill,
                      name=f"Phase {label}")
    text(slide, tx(1), cy + 0.18, tx(30.5) - tx(1), 0.3, "норма: 30 сборов", size=11,
         color=MUTED, align="c")
    text(slide, tx(28), cy + 0.18, tx(39) - tx(28), 0.3, "отклонение", size=11, color=CORAL,
         align="c", bold=True)
    text(slide, tx(39), cy + 0.18, tx(61) - tx(39), 0.3, "восстановление", size=11,
         color=MUTED, align="c")
    for value in (-0.1, 0.0, 0.1, 0.2):
        gy = py + (hi - value) / (hi - lo) * ph
        line(slide, px, gy, px + pw, gy, AMBER if value == 0 else GRID,
             1.25 if value == 0 else 0.75, dash=value == 0)
        text(slide, cx + 0.1, gy - 0.14, 0.58, 0.28, dec(value, 1).replace("-", "−"),
             size=10, color=AMBER if value == 0 else MUTED, align="r")
    zero_y = py + hi / (hi - lo) * ph
    text(slide, tx(41), zero_y + 0.04, tx(61) - tx(41), 0.3, "score < 0 → аномалия",
         size=11, color=AMBER, align="r")
    for event, tick, color in (("firing", 33, CORAL), ("resolved", 54, TEAL)):
        line(slide, tx(tick), py, tx(tick), py + ph, color, 1.25)
        dot(slide, tx(tick), py, 0.16, color, name=f"Event {event}")
        text(slide, tx(tick) - 1.62, py + 0.04, 1.5, 0.3, f"{event} · сбор {tick}",
             size=11, color=color, bold=True, align="r")
    score_chart(slide, px, py, pw, ph, timeline, lo, hi)
    text(slide, tx(10) - 0.75, py + (hi - 0.08) / (hi - lo) * ph, 1.5, 0.45,
         ["сбор 10: нет ответа", "инференса"], size=10, color=MUTED, align="c")
    for tick in (1, 10, 20, 30, 40, 50, 61):
        text(slide, tx(tick) - 0.3, py + ph + 0.04, 0.6, 0.25, str(tick), size=10,
             color=MUTED, align="c")
    text(slide, px, py + ph + 0.27, pw, 0.25, "номер сбора метрик (раз в 5 мин в сценарии)",
         size=10, color=MUTED, align="c")
    text(slide, 0.6, 6.05, 8.35, 0.62,
         "Реальный прогон scripts.check_model_quality_alert (10.11.2026), demo-модель "
         "стенда: demo=true, quality_admitted=false; 19 из 19 проверок. "
         "Данные: docs/course/monitoring/model/demo.json", size=10, color=MUTED)
    steps = [("Норма", "score 0,06…0,24 — тревог нет", BLUE),
             ("Отклонение", "пачка строк: 40" + NBSP + "% NULL ip_address", CORAL),
             ("Анализ", "FastAPI: score < 0 → is_anomaly", TEAL),
             ("Алерт", "≥ 3 из 20 → firing; затем resolved", AMBER)]
    for index, (title, desc, color) in enumerate(steps):
        y = 1.5 + index * 0.92
        card(slide, 9.25, y, 3.48, 0.8, name=f"Demo step {title}")
        badge(slide, 9.42, y + 0.2, 0.4, str(index + 1), color)
        text(slide, 9.97, y + 0.08, 2.65, 0.3, title, size=14, bold=True)
        text(slide, 9.97, y + 0.38, 2.7, 0.4, desc, size=11, color=MUTED)
    card(slide, 9.25, 5.25, 3.48, 1.42, anchor="m", paras=[
        {"runs": [("ПЛАН ПОКАЗА", {"bold": True, "color": TEAL, "size": 11})], "after": 3},
        {"runs": [("A ", {"bold": True, "color": TEAL}), "стенд Compose, заранее"],
         "after": 1},
        {"runs": [("B ", {"bold": True, "color": TEAL}), "этот сценарий без Docker"],
         "after": 1},
        {"runs": [("C ", {"bold": True, "color": TEAL}), "сохранённые результаты"]}],
        size=12, pad=0.15, name="Demo plans")


def slide_conclusions(prs):
    slide = frame(prs, 11, "ОГРАНИЧЕНИЯ И ВЫВОДЫ",
                  "Построен учебный ИИ-контур; модель пока не готова",
                  "≈25 с · Раиль. Построен воспроизводимый контур: данные, проверки, "
                  "эксперименты, тесты, сервис, стенд, алерты. Главный результат по "
                  "качеству — отрицательный, и мы его честно зафиксировали. Дальше: "
                  "признаки, устойчивые к росту таблиц, новый закрытый test, реальные "
                  "данные, доставка алертов и испытание отката. Готовы к вопросам.")
    built = ["Датасет с разметкой и временным разбиением",
             "Валидация данных: 20 проверок",
             "5 экспериментов MLflow + одна оценка на test",
             "Тесты модели и CI на каждый PR",
             "FastAPI-инференс и реестр версий",
             "Compose-стенд, проверенный в CI",
             "Технический алерт и алерт модели"]
    card(slide, 0.6, 1.5, 5.9, 4.15, name="Built")
    text(slide, 0.85, 1.65, 5.4, 0.35, "ЧТО ПОСТРОЕНО", size=12, color=TEAL, bold=True)
    for index, item in enumerate(built):
        y = 2.1 + index * 0.49
        badge(slide, 0.85, y + 0.05, 0.3, "check", TEAL)
        text(slide, 1.3, y, 5.05, 0.4, item, size=14, anchor="m")
    limits = [("Синтетические данные", "перенос на реальные БД не доказан", AMBER, "warn"),
              ("FPR 13,27" + NBSP + "% > 10" + NBSP + "%", "основная модель не допущена",
               CORAL, "cross"),
              ("Demo-модель без допуска", "≈1 ложная тревога на 20 показов", AMBER, "warn"),
              ("Релиз не испытан", "blue-green, shadow / canary, откат", AMBER, "warn"),
              ("Не всё автоматизировано", "алерты в журнал, дрейф вручную", AMBER, "warn")]
    card(slide, 6.83, 1.5, 5.9, 4.15, name="Limits")
    text(slide, 7.08, 1.65, 5.4, 0.35, "ОГРАНИЧЕНИЯ", size=12, color=CORAL, bold=True)
    for index, (title, desc, color, icon) in enumerate(limits):
        y = 2.1 + index * 0.69
        badge(slide, 7.08, y + 0.08, 0.34, icon, color)
        text(slide, 7.6, y, 4.95, 0.62, [{"runs": [(title, {"bold": True})], "after": 0},
                                         {"runs": [desc], "size": 12, "color": MUTED}],
             size=14)
    card(slide, 0.6, 5.85, 12.13, 0.85, fill=TEAL_DARK, name="Next")
    text(slide, 0.85, 5.85, 11.7, 0.85, [[
        ("Дальше: ", {"bold": True, "color": TEAL}),
        ("признаки, устойчивые к росту таблиц, и новый закрытый test · реальная история "
         "инцидентов · доставка алертов · испытание отката", {})]], size=14, anchor="m")


def build(output):
    runs, summary, alert = load_data()
    prs = Presentation()
    prs.slide_width, prs.slide_height = inch(13.333), inch(7.5)
    props = prs.core_properties
    props.title = "DB Monitoring — защита итогового проекта"
    props.author = "Команда DB Monitoring"
    props.subject = "Обнаружение аномалий в метриках качества данных"
    props.language = "ru-RU"
    slide_title(prs, alert)
    slide_problem(prs)
    slide_system(prs)
    slide_architecture(prs)
    slide_data(prs, runs)
    slide_results(prs, summary)
    slide_testing(prs)
    slide_operations(prs)
    slide_ai(prs)
    slide_demo(prs, alert)
    slide_conclusions(prs)
    assert len(prs.slides) == TOTAL
    for slide in prs.slides:
        for shape in slide.shapes:
            if shape.has_text_frame and re.search(r"TODO|lorem|xxx", shape.text_frame.text,
                                                  re.I):
                raise AssertionError(shape.text_frame.text)
    prs.save(output)
    print(f"{output}: {len(prs.slides)} slides")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=HERE / "PRESENTATION.pptx")
    build(parser.parse_args().output)


if __name__ == "__main__":
    main()
