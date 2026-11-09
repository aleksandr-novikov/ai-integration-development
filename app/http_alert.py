"""Технический алерт «доля HTTP 5xx» (#15).

Одно правило поверх существующего ``http_requests_total`` из ``/metrics``:

- окно 5 минут, разность счётчиков между самым старым снимком окна и
  последним — так же, как ``increase()`` в Prometheus, но без экстраполяции;
- решение принимается только при ≥ 10 запросах в окне: Docker HEALTHCHECK
  опрашивает ``/healthz`` каждые 30 с, поэтому 10 запросов за 5 минут есть
  даже без пользовательского трафика;
- срабатывание при доле 5xx > 10 %: при минимальном объёме нужны две ошибки
  из десяти, одиночный сбой алерт не поднимает;
- ``/metrics`` в расчёт не входит: запросы сборщика не являются трафиком
  сервиса и разбавляли бы долю ошибок.

Сигнал — запись ``app.alerts`` (WARNING при срабатывании, INFO при
восстановлении) и поле ``event`` результата. Доставка (Telegram, почта и
т. п.) здесь не выполняется: модуль только вычисляет правило.

Запуск против работающего сервиса::

    python -m app.http_alert --url http://127.0.0.1:5001/metrics
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
import urllib.request
from collections import deque
from dataclasses import asdict, dataclass, field

from prometheus_client.parser import text_string_to_metric_families

logger = logging.getLogger("app.alerts")

STATE_INACTIVE = "inactive"
STATE_FIRING = "firing"


@dataclass(frozen=True)
class HttpErrorRule:
    name: str = "HttpErrorRatioHigh"
    threshold: float = 0.10
    window_seconds: float = 300.0
    min_requests: int = 10
    excluded_endpoints: frozenset[str] = field(default_factory=lambda: frozenset({"metrics"}))


@dataclass(frozen=True)
class Snapshot:
    ts: float
    requests: float
    errors: float


@dataclass(frozen=True)
class Evaluation:
    ts: float
    requests: int
    errors: int
    ratio: float | None
    condition: bool | None  # None — данных в окне меньше min_requests
    state: str
    event: str | None  # "firing" / "resolved" на переходе, иначе None


def snapshot_from_metrics(text: str, ts: float, rule: HttpErrorRule) -> Snapshot:
    """Суммировать ``http_requests_total`` из текста /metrics.

    Ошибкой считается любой статус 5xx; 4xx — проблема клиента.
    """
    requests = errors = 0.0
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            if sample.name != "http_requests_total":
                continue
            if sample.labels.get("endpoint") in rule.excluded_endpoints:
                continue
            requests += sample.value
            if sample.labels.get("status", "").startswith("5"):
                errors += sample.value
    return Snapshot(ts=ts, requests=requests, errors=errors)


class HttpErrorAlert:
    """Состояние правила между снимками; один экземпляр на один сервис."""

    def __init__(self, rule: HttpErrorRule | None = None) -> None:
        self.rule = rule or HttpErrorRule()
        self.state = STATE_INACTIVE
        self._snapshots: deque[Snapshot] = deque()

    def observe(self, snap: Snapshot) -> Evaluation:
        # Счётчик уменьшился — процесс перезапущен; старые снимки несравнимы.
        if self._snapshots and snap.requests < self._snapshots[-1].requests:
            self._snapshots.clear()
        self._snapshots.append(snap)
        while self._snapshots[0].ts < snap.ts - self.rule.window_seconds:
            self._snapshots.popleft()

        base = self._snapshots[0]
        requests = int(snap.requests - base.requests)
        errors = int(snap.errors - base.errors)
        if requests < self.rule.min_requests:
            # Мало данных: решение не принимается, состояние сохраняется.
            return Evaluation(snap.ts, requests, errors, None, None, self.state, None)

        ratio = errors / requests
        condition = ratio > self.rule.threshold
        event = None
        if condition and self.state == STATE_INACTIVE:
            self.state, event = STATE_FIRING, "firing"
        elif not condition and self.state == STATE_FIRING:
            self.state, event = STATE_INACTIVE, "resolved"
        if event:
            self._log(event, requests, errors, ratio)
        return Evaluation(snap.ts, requests, errors, round(ratio, 4), condition, self.state, event)

    def _log(self, event: str, requests: int, errors: int, ratio: float) -> None:
        logger.log(
            logging.WARNING if event == "firing" else logging.INFO,
            "alert %s %s: 5xx %d/%d = %.1f%% over %ds (threshold > %.0f%%)",
            self.rule.name, event, errors, requests, ratio * 100,
            self.rule.window_seconds, self.rule.threshold * 100,
            extra={
                "event": f"alert_{event}",
                "alert": self.rule.name,
                "requests": requests,
                "errors": errors,
                "ratio": round(ratio, 4),
                "threshold": self.rule.threshold,
                "window_s": self.rule.window_seconds,
            },
        )


def main(argv: list[str] | None = None) -> None:
    from app.logging_setup import configure_logging

    parser = argparse.ArgumentParser(description="Evaluate the HTTP 5xx alert against /metrics.")
    parser.add_argument("--url", default="http://127.0.0.1:5001/metrics")
    parser.add_argument("--interval", type=float, default=30.0, help="seconds between scrapes")
    parser.add_argument("--window", type=float, default=HttpErrorRule.window_seconds)
    parser.add_argument("--min-requests", type=int, default=HttpErrorRule.min_requests)
    parser.add_argument("--threshold", type=float, default=HttpErrorRule.threshold)
    parser.add_argument("--iterations", type=int, default=0, help="0 = run until interrupted")
    args = parser.parse_args(argv)

    configure_logging(os.environ.get("LOG_FORMAT", "text"), os.environ.get("LOG_LEVEL", "INFO"))
    alert = HttpErrorAlert(HttpErrorRule(
        threshold=args.threshold, window_seconds=args.window, min_requests=args.min_requests,
    ))
    done = 0
    while not args.iterations or done < args.iterations:
        if done:
            time.sleep(args.interval)
        done += 1
        try:
            with urllib.request.urlopen(args.url, timeout=5) as resp:
                text = resp.read().decode("utf-8")
        except OSError as exc:
            # Недоступный /metrics — отдельный сигнал, а не срабатывание правила.
            logger.warning("metrics scrape failed: %s", exc, extra={"event": "scrape_failed"})
            continue
        result = alert.observe(snapshot_from_metrics(text, time.time(), alert.rule))
        print(json.dumps(asdict(result)), flush=True)


if __name__ == "__main__":
    main()
