"""Pure synthetic metric profiles shared by dashboard seeds and course datasets."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import datetime

# Базовый тренд + шум (используется как дефолт TableProfile).
ROW_COUNT_START_FRACTION = 0.70
NOISE_AMPLITUDE = 0.005

# Недельная сезонность — кормит Prophet в forecast.py (дефолт).
WEEKLY_AMPLITUDE = 0.08

# Ступенчатая регрессия null_rate (change-point + страница истории).
REGRESSION_THRESHOLD = 0.10
REGRESSION_DAYS = 7
BASELINE_NULL_RATE = 0.02

# Бэкфил-ступенька на row_count: даёт PELT отдельный change-point на росте.
BACKFILL_PROGRESS = 0.30
BACKFILL_FRACTION = 0.06

# Точечные аномалии для IsolationForest. (progress, multiplier) — на каждую
# точку приходится ровно один тик, ближайший к указанному прогрессу.
ANOMALY_POINTS: tuple[tuple[float, float], ...] = (
    (0.55, 1.18),
    (0.78, 0.80),
    (0.88, 1.15),
)

# Транзиентный выброс null_rate для одного тика. История
# (`/dashboard/history`) считает «Выбросы NULL» как Δnull_rate ≥ 5 п.п. между
# соседними тиками по серии table-level avg null_rate — поэтому пик надо
# дать сразу всем колонкам таблицы, иначе среднее размывается ниже порога.
# Имитирует короткий инцидент качества данных: один час null_rate подскочил
# на 18 п.п. сразу по всем колонкам, потом откатился к норме.
NULL_SPIKE_PROGRESS = 0.20  # ~ 11 дней назад в 14-дневном окне
NULL_SPIKE_DELTA = 0.18     # +18 п.п. к каждой колонке на один тик


@dataclass(frozen=True)
class TableProfile:
    """Параметры синтетической траектории row_count под конкретную таблицу.

    Все четыре монитор-таблицы (users/events/products/orders) идут по
    единому паттерну append-only — монотонный рост с тремя кумулятивными
    ступеньками в последних 7 днях окна. Различаются только величины
    ступенек (10–30 тыс. строк) и start_fraction, чтобы графики не
    выглядели дублями.
    """

    start_fraction: float = ROW_COUNT_START_FRACTION
    weekly_amplitude: float = WEEKLY_AMPLITUDE
    backfill_fraction: float = BACKFILL_FRACTION
    anomalies: tuple[tuple[float, float], ...] = ANOMALY_POINTS
    noise_amplitude: float = NOISE_AMPLITUDE
    # Накопительные ступеньки-«всплески» для append-only таблиц: пара
    # (progress, abs_rows_added). На указанной фазе значение прыгает вверх
    # ровно на abs_rows_added и остаётся повышенным до конца окна — никаких
    # возвратов. Сумма ступенек «съедается» из ramp'а, чтобы последняя
    # точка осталась ≈ current. Если сумма превосходит 80% current, шаги
    # пропорционально масштабируются, чтобы тренд не схлопнулся в ноль.
    growth_steps: tuple[tuple[float, int], ...] = ()


DEFAULT_PROFILE = TableProfile()

# Все четыре таблицы — append-only: монотонный рост + ровно три кумулятивные
# ступеньки-«всплеска» в последних 7 днях 14-дневного окна. Каждая ступенька
# добавляет 10–30 тыс. строк (точное значение масштабируется вниз, если оно
# превысит 80% current — см. _step_scale) и остаётся повышенной до конца
# окна — никаких возвратов вниз. Никакой сезонности, шума и
# мультипликативных аномалий: ровный наклонный тренд, три резких положительных
# Δrow_count в правой половине окна — это ровно три точки, которые
# IsolationForest помечает как аномалии (contamination=0.01 при 336 тиках ≈ 3).
#
# Прогрессы ступенек разнесены по таблицам, чтобы графики users/events/
# products/orders не выглядели зеркальными друг другу — спайки приходятся
# на разные дни.
# Соседние ступеньки разнесены минимум на 72 часа — это окно дедупликации
# в `ml/changepoint.py` (DEDUPE_WINDOW_HOURS): однонаправленные сдвиги
# ближе чем 72ч схлопываются в один changepoint, и тогда дашборд показал
# бы только одну метку «Сдвиг» вместо трёх. С 14-дневным окном это
# означает, что прогрессы соседних ступенек должны различаться не меньше
# чем на 72/(14·24) ≈ 0.215.
PROFILES: dict[str, TableProfile] = {
    "users": TableProfile(
        start_fraction=0.65,
        weekly_amplitude=0.0,
        backfill_fraction=0.0,
        anomalies=(),
        noise_amplitude=0.0,
        growth_steps=((0.50, 12_000), (0.74, 18_000), (0.97, 25_000)),
    ),
    "events": TableProfile(
        start_fraction=0.55,
        weekly_amplitude=0.0,
        backfill_fraction=0.0,
        anomalies=(),
        noise_amplitude=0.0,
        growth_steps=((0.52, 13_000), (0.75, 19_000), (0.98, 26_000)),
    ),
    "products": TableProfile(
        start_fraction=0.55,
        weekly_amplitude=0.0,
        backfill_fraction=0.0,
        anomalies=(),
        noise_amplitude=0.0,
        growth_steps=((0.51, 10_000), (0.74, 14_000), (0.97, 20_000)),
    ),
    "orders": TableProfile(
        start_fraction=0.55,
        weekly_amplitude=0.0,
        backfill_fraction=0.0,
        anomalies=(),
        noise_amplitude=0.0,
        growth_steps=((0.53, 11_000), (0.76, 16_000), (0.98, 22_000)),
    ),
}


def _profile_for(table_name: str) -> TableProfile:
    return PROFILES.get(table_name, DEFAULT_PROFILE)

# ─────── трансформации ряда row_count ───────


def _seasonality_factor(ts: datetime, amplitude: float = WEEKLY_AMPLITUDE) -> float:
    """Множитель недельной сезонности: пик в середине недели, провал на выходных.
    При amplitude=0 функция возвращает строго 1.0 (для append-only таблиц)."""
    if amplitude == 0:
        return 1.0
    dow = ts.weekday()  # Mon=0..Sun=6
    return 1.0 + amplitude * math.cos((dow - 2) * 2 * math.pi / 7)


def _backfill_offset(
    progress: float, current: int, fraction: float = BACKFILL_FRACTION,
) -> int:
    """До ступеньки вычитаем константу, после — ничего не делаем.
    Якорит последнюю точку на `current`, делая ступеньку «накоплением».
    При fraction=0 эффект отключён."""
    if fraction <= 0 or progress >= BACKFILL_PROGRESS:
        return 0
    return -round(current * fraction)


def _anomaly_multiplier(
    progress: float,
    n_ticks: int,
    anomalies: tuple[tuple[float, float], ...] = ANOMALY_POINTS,
) -> float:
    """Возвращает множитель аномалии для тика. 1.0 для нормальных и при
    пустом списке anomalies."""
    if n_ticks <= 1 or not anomalies:
        return 1.0
    half_step = 0.5 / (n_ticks - 1)
    for p, mult in anomalies:
        if abs(progress - p) < half_step:
            return mult
    return 1.0


_STEP_BUDGET_FRACTION = 0.80  # ступеньки могут «съесть» максимум 80% current


def _step_scale(current: int, steps: tuple) -> float:
    """Если сумма ступенек > 80% current, масштабируем все вниз пропорционально.
    Иначе scale=1.0 (используются абсолютные значения как есть)."""
    if not steps or current <= 0:
        return 1.0
    total = sum(c for _, c in steps)
    budget = current * _STEP_BUDGET_FRACTION
    return budget / total if total > budget else 1.0


def _step_contribution(progress: float, steps: tuple, scale: float) -> int:
    if not steps:
        return 0
    return round(scale * sum(c for p, c in steps if progress >= p))


def _row_count_at(
    progress: float,
    current: int,
    ts: datetime,
    rng: random.Random,
    profile: TableProfile = DEFAULT_PROFILE,
    anomaly_mult: float = 1.0,
) -> int:
    scale = _step_scale(current, profile.growth_steps)
    total_steps = round(scale * sum(c for _, c in profile.growth_steps))
    # Ramp заканчивается в (current - total_steps), чтобы вместе со ступеньками
    # дать ≈ current на progress=1.
    ramp_target = max(0, current - total_steps)
    base = ramp_target * (
        profile.start_fraction + (1.0 - profile.start_fraction) * progress
    )
    base += _step_contribution(progress, profile.growth_steps, scale)
    base += _backfill_offset(progress, current, profile.backfill_fraction)
    base *= _seasonality_factor(ts, profile.weekly_amplitude)
    base *= anomaly_mult
    if profile.noise_amplitude > 0:
        base *= 1 + rng.uniform(-profile.noise_amplitude, profile.noise_amplitude)
    return max(0, round(base))


def _null_rate_at(
    progress: float,
    current_rate: float,
    regression_progress_start: float,
) -> float:
    """Резкий шаг: BASELINE до regression_progress_start, иначе current_rate.
    PELT ловит шаг лучше, чем рампу — score не размазывается по соседним тикам."""
    if current_rate <= REGRESSION_THRESHOLD:
        return current_rate
    return BASELINE_NULL_RATE if progress < regression_progress_start else current_rate
