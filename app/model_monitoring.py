"""Мониторинг входов и работы опубликованной модели Isolation Forest (#16).

Всё считается для одного ряда: полный источник (project, connection, schema,
table) + ``model_version`` + ``feature_schema_version`` + режим ``demo``.
Наблюдения берутся из ``app.source_storage``, признаки и их проверка — из
``ml.features``; пропуски нулями не заполняются, наблюдения разных источников
не смешиваются.

- ``input_summary`` — пропуски, физические границы, распределения, свежесть
  и число валидных векторов четырёх признаков;
- ``score_summary`` / ``version_info`` / ``inference_summary`` — score и доля
  аномалий, версия и возраст модели, ошибки и время вычисления инференса;
- ``AnomalyRateAlert`` — одно правило: доля аномальных предсказаний за окно
  против зафиксированного baseline той же версии.

Доля аномалий — не FPR: правило показывает изменение поведения детектора,
а не ухудшение precision/recall. Сигнал — запись ``app.alerts`` (WARNING при
срабатывании, INFO при восстановлении); доставка во внешний канал не
выполняется.

Наблюдатель против хранилища и ``/metrics`` работающего приложения::

    python -m app.model_monitoring --source source.json --model-version if-... \\
        --baseline baseline.json --registry models/registry \\
        --metrics-url http://127.0.0.1:5001/metrics
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from itertools import pairwise

from ml.features import FEATURE_NAMES, build_vector, validate_vector

logger = logging.getLogger("app.alerts")

# Ниже этого числа валидных векторов / результатов вывод недостоверен; равно
# окну алерта, чтобы входы и правило опирались на одинаковый объём данных.
MIN_OBSERVATIONS = 20
# Физические границы ml.features.validate_vector, проверяемые перед инференсом.
BOUNDS = {
    "row_count": "row_count >= 0",
    "null_rate": "0 <= null_rate <= 1",
    "d_row_count": "row_count - d_row_count >= 0 (previous row_count)",
    "d_null_rate": "-1 <= d_null_rate <= 1, 0 <= null_rate - d_null_rate <= 1",
}
STATE_INACTIVE = "inactive"
STATE_FIRING = "firing"
WILSON_Z = 1.96


def _instant(ts: str) -> datetime:
    stamp = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=UTC)


def _finite(value) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _distribution(values: list[float]) -> dict | None:
    if not values:
        return None
    import numpy as np

    data = np.asarray(values, dtype=float)
    p05, p50, p95 = np.percentile(data, [5, 50, 95])
    return {
        name: float(f"{value:.6g}")
        for name, value in {
            "min": data.min(), "p05": p05, "median": p50, "p95": p95, "max": data.max(),
            "mean": data.mean(), "std": data.std(),
        }.items()
    }


def input_summary(row_counts, null_rates, *, now: datetime, max_age: timedelta) -> dict:
    """Показатели четырёх входов по истории одного источника.

    Векторы строятся так же, как в ``ml.features.build_rows`` (inner join
    по времени, разность с предыдущим согласованным наблюдением), но каждая
    пара проверяется отдельно: невалидная пара учитывается, а не обрывает
    расчёт и не заменяется нулём. Распределения — только по валидным векторам.
    """
    rc = {row["ts"]: row["value"] for row in row_counts}
    nr = {row["ts"]: row["value"] for row in null_rates}
    instants = sorted(rc.keys() | nr.keys())
    aligned = sorted(rc.keys() & nr.keys())
    pairs = list(pairwise(aligned))
    present = {
        "row_count": sum(_finite(rc.get(ts)) for ts in instants),
        "null_rate": sum(_finite(nr.get(ts)) for ts in instants),
    }
    computable = sum(
        all(_finite(value) for value in (rc[a], nr[a], rc[b], nr[b])) for a, b in pairs
    )
    present["d_row_count"] = present["d_null_rate"] = computable
    expected = {"row_count": len(instants), "null_rate": len(instants)}
    expected["d_row_count"] = expected["d_null_rate"] = max(len(instants) - 1, 0)

    valid, invalid = [], {"not_a_finite_number": 0, "out_of_bounds": 0}
    for before, current in pairs:
        try:
            vector = build_vector(rc[current], nr[current], rc[before], nr[before])
        except ValueError:
            invalid["not_a_finite_number"] += 1
            continue
        if not all(math.isfinite(value) for value in vector):
            invalid["not_a_finite_number"] += 1
            continue
        try:
            validate_vector(dict(zip(FEATURE_NAMES, vector, strict=True)))
        except ValueError:
            invalid["out_of_bounds"] += 1
            continue
        valid.append((current, vector))

    latest = valid[-1][0] if valid else None
    age = (now - _instant(latest)).total_seconds() if latest else None
    fresh = age is not None and age <= max_age.total_seconds()
    reasons = []
    if len(valid) < MIN_OBSERVATIONS:
        reasons.append("no_valid_vectors" if not valid else "insufficient_valid_vectors")
    if any(invalid.values()):
        reasons.append("invalid_values")
    if valid and not fresh:
        reasons.append("stale")
    return {
        "observation_instants": len(instants),
        "aligned_instants": len(aligned),
        "vectors": len(pairs),
        "valid_vectors": len(valid),
        "invalid_vectors": invalid,
        "min_valid_vectors": MIN_OBSERVATIONS,
        "features": {
            name: {
                "missing": expected[name] - present[name],
                "bounds": BOUNDS[name],
                "distribution": _distribution([vector[i] for _, vector in valid]),
            }
            for i, name in enumerate(FEATURE_NAMES)
        },
        "freshness": {
            "latest_valid_observation": latest,
            "age_seconds": round(age, 1) if age is not None else None,
            "max_age_seconds": max_age.total_seconds(),
            "fresh": fresh,
        },
        "conclusive": not reasons,
        "reasons": reasons,
    }


def score_summary(scores, *, demo: bool) -> dict:
    """Score и доля аномалий одного ряда; результаты другого режима исключаются."""
    rows = [row for row in scores if row["demo"] is demo]
    anomalies = sum(row["is_anomaly"] for row in rows)
    return {
        "results": len(rows),
        "excluded_other_mode": len(scores) - len(rows),
        "anomalies": anomalies,
        "anomaly_rate": round(anomalies / len(rows), 6) if rows else None,
        "score": _distribution([row["score"] for row in rows]),
        "first_observed_at": rows[0]["ts"] if rows else None,
        "last_observed_at": rows[-1]["ts"] if rows else None,
    }


def version_info(manifest: dict, *, now: datetime) -> dict:
    """Версия, возраст и допуск из проверенного манифеста реестра."""
    from ml.model_registry import admitted

    training = manifest["training"]
    trained_at = training.get("trained_at")
    evaluation = manifest.get("evaluation")
    return {
        "available": True,
        "model_version": manifest["model_version"],
        "feature_schema_version": manifest["feature_schema_version"],
        "origin": (manifest.get("origin") or {}).get("kind"),
        "trained_at": trained_at,
        # Исторический экспорт #9 не содержит времени обучения: возраст неизвестен.
        "time_status": training.get("time_status", "recorded"),
        "age_days": (
            round((now - _instant(trained_at)).total_seconds() / 86400, 3) if trained_at else None
        ),
        "quality_admitted": admitted(training, evaluation),
        "labelled_evaluation": (
            {
                key: evaluation.get(key)
                for key in ("split", "data_sha256", "run_id", "metrics")
            }
            if isinstance(evaluation, dict)
            else None
        ),
    }


def inference_summary(metrics_text: str, labels: dict) -> dict:
    """Вызовы, ошибки и время вычисления ряда из текста Prometheus ``/metrics``."""
    from prometheus_client.parser import text_string_to_metric_families

    outcomes: dict[str, float] = {}
    count = total = 0.0
    buckets = []
    for family in text_string_to_metric_families(metrics_text):
        for sample in family.samples:
            if any(sample.labels.get(key) != value for key, value in labels.items()):
                continue
            if sample.name == "inference_predictions_total":
                outcome = sample.labels["outcome"]
                outcomes[outcome] = outcomes.get(outcome, 0) + sample.value
            elif sample.name == "inference_predict_duration_seconds_count":
                count = sample.value
            elif sample.name == "inference_predict_duration_seconds_sum":
                total = sample.value
            elif sample.name == "inference_predict_duration_seconds_bucket":
                buckets.append((float(sample.labels["le"]), sample.value))
    calls = int(sum(outcomes.values()))
    errors = {key: int(value) for key, value in sorted(outcomes.items()) if key != "ok"}
    p95 = next((le for le, cumulative in sorted(buckets) if count and cumulative >= 0.95 * count),
               None)
    return {
        "calls": calls,
        "ok": int(outcomes.get("ok", 0)),
        "errors": errors,
        "error_rate": round(sum(errors.values()) / calls, 6) if calls else None,
        "predict_seconds": {
            "count": int(count),
            "mean": float(f"{total / count:.6g}") if count else None,
            "p95_bucket_le": p95,
        },
    }


# ── Алерт качества ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AnomalyRateRule:
    """Окно — последние ``window`` результатов ряда; решение только по полному окну.

    Порог — наименьшее число аномалий k, при котором P(X >= k) <= ``alpha``
    для X ~ Binomial(window, p), где p — верхняя 95% граница Уилсона доли
    аномалий baseline. Граница учитывает конечный объём baseline; baseline
    меньше ``min_baseline`` не различает долю порядка contamination 0,5–1%
    (при 0 аномалиях из 500 верхняя граница уже 0,76%).
    """

    name: str = "ModelAnomalyRateHigh"
    window: int = MIN_OBSERVATIONS
    alpha: float = 0.01
    min_baseline: int = 500


def wilson_upper(successes: int, total: int, z: float = WILSON_Z) -> float:
    p = successes / total
    centre = p + z * z / (2 * total)
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return (centre + margin) / (1 + z * z / total)


def binomial_tail(k: int, n: int, p: float) -> float:
    """P(X >= k) для X ~ Binomial(n, p)."""
    return max(0.0, 1.0 - sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k)))


def threshold_count(rule: AnomalyRateRule, baseline: dict) -> int | None:
    """Порог числа аномалий в окне; None — правило при таком baseline не срабатывает."""
    p = wilson_upper(baseline["anomalies"], baseline["observations"])
    return next(
        (k for k in range(1, rule.window + 1) if binomial_tail(k, rule.window, p) <= rule.alpha),
        None,
    )


def baseline_problem(baseline, *, source, model_version, feature_schema_version, demo,
                     rule: AnomalyRateRule) -> str | None:
    """Причина, по которой baseline нельзя применить к ряду, или None."""
    if baseline is None:
        return "baseline_missing"
    try:
        observations, anomalies = baseline["observations"], baseline["anomalies"]
        identity = (baseline["source"], baseline["model_version"],
                    baseline["feature_schema_version"], baseline["demo"])
    except (KeyError, TypeError):
        return "baseline_invalid"
    if (type(observations) is not int or type(anomalies) is not int
            or not 0 <= anomalies <= observations):
        return "baseline_invalid"
    if identity != (source, model_version, feature_schema_version, demo):
        return "baseline_mismatch"
    if observations < rule.min_baseline:
        return "baseline_insufficient"
    if threshold_count(rule, baseline) is None:
        return "baseline_rate_too_high"
    return None


@dataclass(frozen=True)
class Evaluation:
    status: str  # evaluated | insufficient_observations | baseline_*
    observed_at: str | None
    observations: int
    anomalies: int | None
    anomaly_rate: float | None
    baseline_rate: float | None
    threshold_count: int | None
    threshold_rate: float | None
    condition: bool | None  # None — вывод невозможен, это не «норма»
    state: str
    event: str | None  # "firing" / "resolved" на переходе, иначе None


class AnomalyRateAlert:
    """Состояние правила для одного ряда: источник + версия + схема + режим."""

    def __init__(self, baseline, *, source, model_version, feature_schema_version, demo,
                 rule: AnomalyRateRule | None = None) -> None:
        self.rule = rule or AnomalyRateRule()
        self.series = {"source": dict(source), "model_version": model_version,
                       "feature_schema_version": feature_schema_version, "demo": demo}
        self.problem = baseline_problem(baseline, **self.series, rule=self.rule)
        self.baseline_rate = self.threshold = None
        if self.problem is None:
            self.baseline_rate = round(baseline["anomalies"] / baseline["observations"], 6)
            self.threshold = threshold_count(self.rule, baseline)
        self.state = STATE_INACTIVE

    def observe(self, scores) -> Evaluation:
        """Оценить последнее окно; ``scores`` — результаты ряда по возрастанию времени."""
        rows = [
            row for row in scores
            if row["demo"] is self.series["demo"]
            and row["source"] == self.series["source"]
            and row["model_version"] == self.series["model_version"]
            and row["feature_schema_version"] == self.series["feature_schema_version"]
        ]
        window = rows[-self.rule.window:]
        observed_at = window[-1]["ts"] if window else None
        threshold_rate = (
            round(self.threshold / self.rule.window, 6) if self.threshold is not None else None
        )
        if self.problem is not None or len(window) < self.rule.window:
            # Без применимого baseline или полного окна решение не принимается,
            # состояние сохраняется: firing не снимается без доказательства.
            return Evaluation(self.problem or "insufficient_observations", observed_at,
                              len(window), None, None, self.baseline_rate, self.threshold,
                              threshold_rate, None, self.state, None)
        anomalies = sum(row["is_anomaly"] for row in window)
        rate = round(anomalies / len(window), 6)
        condition = anomalies >= self.threshold
        event = None
        if condition and self.state == STATE_INACTIVE:
            self.state, event = STATE_FIRING, "firing"
        elif not condition and self.state == STATE_FIRING:
            self.state, event = STATE_INACTIVE, "resolved"
        if event:
            self._log(event, observed_at, anomalies, len(window), rate)
        return Evaluation("evaluated", observed_at, len(window), anomalies, rate,
                          self.baseline_rate, self.threshold, threshold_rate, condition,
                          self.state, event)

    def _log(self, event, observed_at, anomalies, observations, rate) -> None:
        logger.log(
            logging.WARNING if event == "firing" else logging.INFO,
            "alert %s %s: anomalies %d/%d = %.1f%% (threshold >= %d, baseline %.2f%%) "
            "for %s model %s",
            self.rule.name, event, anomalies, observations, rate * 100, self.threshold,
            self.baseline_rate * 100, self.series["source"], self.series["model_version"],
            extra={
                "event": f"alert_{event}",
                "alert": self.rule.name,
                **self.series,
                "observed_at": observed_at,
                "anomalies": anomalies,
                "observations": observations,
                "anomaly_rate": rate,
                "threshold_count": self.threshold,
                "baseline_rate": self.baseline_rate,
            },
        )


def build_baseline(flags, *, source, model_version, feature_schema_version, demo,
                   origin: dict) -> dict:
    """Baseline из предсказаний этой версии на нормальных размеченных данных."""
    flags = [bool(flag) for flag in flags]
    return {
        "kind": "anomaly_rate_baseline",
        "source": dict(source),
        "model_version": model_version,
        "feature_schema_version": feature_schema_version,
        "demo": demo,
        "observations": len(flags),
        "anomalies": sum(flags),
        "origin": origin,
    }


# ── Отчёт по хранилищу ─────────────────────────────────────────────────────


def freshness_limit(source) -> timedelta:
    """Тот же предел, что у событий адаптера: две периодичности сбора, минимум 30 минут."""
    from app.inference_adapter import FRESH_COLLECTIONS, MIN_FRESHNESS
    from app.metrics_storage import get_connection

    connection = get_connection(source["project_id"], source["connection_id"]) or {}
    interval = timedelta(minutes=connection.get("interval_minutes") or 0)
    return max(MIN_FRESHNESS, FRESH_COLLECTIONS * interval)


def report(source, model_version, feature_schema_version, *, demo, alert: AnomalyRateAlert,
           registry=None, metrics_text=None, now=None) -> dict:
    """Входы, работа модели и состояние правила для одного ряда."""
    from app import source_storage
    from ml.model_registry import RegistryError, inspect_version

    now = now or datetime.now(UTC)
    row_counts = source_storage.metrics(source, "row_count", quarantine=True)
    null_rates = source_storage.metrics(source, "null_rate", quarantine=True)
    scores = source_storage.scores(source, model_version, feature_schema_version)
    if registry is None:
        version = {"model_version": model_version, "available": False,
                   "reason": "registry_not_given"}
    else:
        try:
            version = version_info(
                inspect_version(registry, model_version, source=source), now=now
            )
        except RegistryError:
            version = {"model_version": model_version, "available": False,
                       "reason": "manifest_unavailable"}
    labels = {**source, "model_version": model_version,
              "feature_schema_version": feature_schema_version, "demo": str(demo).lower()}
    return {
        "generated_at": now.isoformat(timespec="seconds"),
        "source": source,
        "model_version": model_version,
        "feature_schema_version": feature_schema_version,
        "demo": demo,
        "inputs": input_summary(row_counts, null_rates, now=now,
                                max_age=freshness_limit(source)),
        "model": {
            "version": version,
            "scores": score_summary(scores, demo=demo),
            "inference": (
                inference_summary(metrics_text, labels) if metrics_text is not None
                else {"available": False, "reason": "metrics_not_given"}
            ),
        },
        "alert": {"rule": asdict(alert.rule), **asdict(alert.observe(scores))},
    }


def main(argv: list[str] | None = None) -> None:
    from app.logging_setup import configure_logging

    parser = argparse.ArgumentParser(description="Report inputs, model behaviour and the alert.")
    parser.add_argument("--source", required=True, help="JSON file with the full source")
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--feature-schema-version", default="1.0.0")
    parser.add_argument("--demo", action="store_true", help="series of a demo-mode binding")
    parser.add_argument("--baseline", help="JSON baseline of this source and version")
    parser.add_argument("--registry", help="registry root for version and age")
    parser.add_argument("--metrics-url", help="Flask /metrics with inference counters")
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument("--iterations", type=int, default=1, help="0 = run until interrupted")
    args = parser.parse_args(argv)

    configure_logging(os.environ.get("LOG_FORMAT", "text"), os.environ.get("LOG_LEVEL", "INFO"))
    with open(args.source, encoding="utf-8") as stream:
        source = json.load(stream)
    baseline = None
    if args.baseline:
        with open(args.baseline, encoding="utf-8") as stream:
            baseline = json.load(stream)
    alert = AnomalyRateAlert(baseline, source=source, model_version=args.model_version,
                             feature_schema_version=args.feature_schema_version, demo=args.demo)
    done = 0
    while not args.iterations or done < args.iterations:
        if done:
            time.sleep(args.interval)
        done += 1
        metrics_text = None
        if args.metrics_url:
            try:
                with urllib.request.urlopen(args.metrics_url, timeout=5) as response:
                    metrics_text = response.read().decode("utf-8")
            except OSError as exc:
                logger.warning("metrics scrape failed: %s", exc, extra={"event": "scrape_failed"})
        result = report(source, args.model_version, args.feature_schema_version, demo=args.demo,
                        alert=alert, registry=args.registry, metrics_text=metrics_text)
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
