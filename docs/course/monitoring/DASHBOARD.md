# Эскиз дашборда технического состояния сервиса (#15)

Учебный эскиз для В12, слайд 41. Это не развёрнутый Grafana dashboard: панели
описывают, **что** смотреть и **откуда** берутся данные. Все числовые
показатели уже выдаёт `GET /metrics` (код: [app/instrumentation.py](../../../app/instrumentation.py));
выражения PromQL приведены как способ построения в Prometheus/Grafana и в
этой задаче против настоящего Prometheus не запускались. Качество
ML-модели (FPR, recall, дрейф) сюда намеренно не входит — это #16.

```text
┌─ DB Monitoring · сервис (app) ─────────────── окно: 5 мин · обновление: 30 с ┐
│                                                                               │
│ [1] Health        [2] Запросов/с     [3] Доля 5xx       [4] Latency p95       │
│ ┌─────────────┐  ┌──────────────┐   ┌────────────────┐ ┌───────────────────┐  │
│ │  ● OK       │  │    0,10      │   │   0,0 %  ✓     │ │   12 ms           │  │
│ │ monitor_db ok│  │ ▁▂▁▂▁▁▂▁▂▁   │   │ порог > 10 %   │ │ p50 4 ms          │  │
│ │ target_db ok │  └──────────────┘   └────────────────┘ └───────────────────┘  │
│ └─────────────┘                                                               │
│                                                                               │
│ [5] Состояние алерта HttpErrorRatioHigh                                       │
│ inactive ████████████░░░░░░░░████████████████  firing ░  (по логу app.alerts)  │
│                                                                               │
│ [6] Запросы по коду статуса, шт/мин     [7] Latency по endpoint, p95          │
│  200 ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇                     health          ▇▇ 15 ms              │
│  404 ▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂                     dashboard.ov…   ▇▇▇▇▇ 48 ms           │
│  503 ______▅▅▅▅▅▅___                     api.metrics     ▇▇▇ 30 ms             │
│                                                                               │
│ [8] Топ endpoint по 5xx (таблица)        [9] Неудачные входы /мин  [10] Сборщик│
│  endpoint │ 5xx │ всего │ доля           ▁▁▁▁▁▅▁▁▁▁                ok 96 err 0│
│  health   │  6  │  30   │ 20 %                                                │
└───────────────────────────────────────────────────────────────────────────────┘
```

Значения на рисунке иллюстративные. Фактические числа прогона — в
[README.md](README.md#фактические-результаты) и [demo.json](demo.json).

## Панели

`endpoint!="metrics"` во всех выражениях: опросы самого сборщика не являются
трафиком сервиса.

| № | Панель | Источник | Как получить | Интерпретация |
|---|---|---|---|---|
| 1 | Health | `GET /healthz` (JSON: `status`, `checks.*.status`) — [app/health.py](../../../app/health.py) | Опрос HTTP (Docker HEALTHCHECK каждые 30 с; в Prometheus — blackbox-проба) | `ok` / 200 — зависимости доступны; `degraded` / 503 — какая-то из `monitor_db`, `target_db`, `ratelimit_storage` в состоянии `down`. `n/a` для in-memory хранилища лимитов — норма учебного стенда |
| 2 | Запросов в секунду | `http_requests_total{method,endpoint,status}` | `sum(rate(http_requests_total{endpoint!="metrics"}[5m]))` | Нагрузка. Резкое падение до нуля при работающем health — проблема на входе (прокси, сеть); рост — нагрузка или бот |
| 3 | Доля ошибок 5xx | `http_requests_total`, `status=~"5.."` | `sum(increase(http_requests_total{endpoint!="metrics",status=~"5.."}[5m])) / sum(increase(http_requests_total{endpoint!="metrics"}[5m]))` | Основной сигнал технического сбоя; на нём построен алерт (порог > 10 % при ≥ 10 запросах за 5 мин). 4xx в долю не входит: это ошибки клиента |
| 4 | Latency p50/p95 | `http_request_duration_seconds` (гистограмма, корзины 10 мс … 10 с) | `histogram_quantile(0.95, sum by (le) (rate(http_request_duration_seconds_bucket{endpoint!="metrics"}[5m])))` | Скорость ответа. p95 оценивается интерполяцией внутри корзины, точность ограничена её границами. Рост p95 при нормальной доле 5xx — медленная БД или тяжёлые запросы |
| 5 | Состояние алерта | Записи логгера `app.alerts` (`event`: `alert_firing` / `alert_resolved`) от [app/http_alert.py](../../../app/http_alert.py) | `python -m app.http_alert --url …/metrics` и просмотр его вывода; в Grafana — панель логов (Loki) с фильтром `logger="app.alerts"` | Отрезки `firing` — интервалы технического инцидента |
| 6 | Запросы по коду статуса | `http_requests_total` | `sum by (status) (increase(http_requests_total{endpoint!="metrics"}[1m]))` | Одна линия на конкретный код (`200`, `404`, `503` …), не на класс 2xx/4xx/5xx: лейбл `status` хранит код. Видно, какой ответ вырос; всплеск 404 обычно означает сканирование адресов, а не сбой сервиса. Сумма по классу 5xx — панель 3 |
| 7 | Latency по endpoint | `http_request_duration_seconds_bucket` | `histogram_quantile(0.95, sum by (le, endpoint) (rate(http_request_duration_seconds_bucket{endpoint!="metrics"}[5m])))` | Какой маршрут медленный. В лейбле — имя Flask-endpoint, не путь: нет взрыва кардинальности и персональных данных |
| 8 | Топ endpoint по 5xx | `http_requests_total` | `topk(5, sum by (endpoint) (increase(http_requests_total{status=~"5.."}[5m])))` | С чего начинать разбор при срабатывании алерта |
| 9 | Неудачные входы | `failed_login_attempts_total` | `rate(failed_login_attempts_total[5m]) * 60` | Подбор паролей. Не входит в алерт #15, показатель безопасности |
| 10 | Сборщик | `collector_runs_total{result}` | `sum by (result) (increase(collector_runs_total[1h]))` | Работает ли фоновый сбор (`ok` / `error`). Существующий показатель, отдельным требованием #15 не является |

## Как пользоваться на защите

1. **Норма:** Health зелёный, доля 5xx около 0, p95 стабильна, алерт `inactive`.
2. **Сбой целевой БД:** Health → `degraded`, `target_db: down`; на панели 6
   появляется код 503, панель 8 показывает `health`; когда доля превышает 10 %,
   панель 5 переходит в `firing`.
3. **Восстановление:** Health снова `ok`, новые 5xx прекращаются; алерт
   становится `resolved`, когда ошибки выходят из 5-минутного окна.

Этот же сценарий воспроизводится командой из [README.md](README.md#воспроизведение).
