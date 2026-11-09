# В12: логирование, метрики, эскиз дашборда и технический алерт (#15)

Основание: [В12 «Мониторинг и поддержка в продакшене», слайд 41](https://netology.ru/profile/program/bhemai-25-iair-3/lessons/658302/lesson_items/3699550) —
логирование, метрики сервиса, эскиз дашборда с описанием. Для итогового
проекта по согласованию 08.11.2026 добавлен один проверяемый технический алерт.
Объём минимальный: полный стек Prometheus + Grafana не разворачивается, мониторинг
качества ML-модели относится к #16 и здесь не повторяется.

Документы: [эскиз дашборда](DASHBOARD.md) · [отчёт прогона алерта](demo.json).

## Что было и что добавлено

Проверено на `master` `8512ab5` (09.11.2026).

| Механизм | До #15 | Изменение в #15 |
|---|---|---|
| JSON-логи, `request_id`, `X-Request-Id` | Работали: [app/logging_setup.py](../../../app/logging_setup.py), хуки в [app/app.py](../../../app/app.py) | Без изменений |
| Маскирование секретов в логах | Работало для `msg`/`args`: [app/security.py](../../../app/security.py) (DSN, Telegram, AWS, Bearer, `token=`/`password=`), Sentry `before_send`. **Текст traceback из `exc_info` не маскировался** — секрет из сообщения исключения попадал в лог (воспроизведено тестом до исправления, см. раздел 5) | Шаблоны не менялись. `ScrubbingFormatter` в [app/logging_setup.py](../../../app/logging_setup.py) пропускает отформатированный traceback и `stack_info` через существующий `scrub_value` в `LOG_FORMAT=json` и `text` |
| Лог HTTP-запросов | **Только строка dev-сервера Werkzeug**: без `request_id` и latency, с ANSI-кодами и **сырым путём** — токен из `/auth/reset-password/<token>` попадал в лог (проверено запуском `python -m app.app`, тестовым токеном) | Одна строка `app.http` на запрос в существующем хуке [app/instrumentation.py](../../../app/instrumentation.py): метод, имя endpoint, статус, `duration_ms`; 5xx — уровень ERROR. Строка Werkzeug отключена (`AppLoggedRequestHandler`), остальные сообщения Werkzeug сохранены |
| Лог необработанного исключения | Стандартный `Flask.log_exception` писал `Exception on <request.path> [GET]` — токен из `/auth/reset-password/<token>` попадал в лог (воспроизведено регрессионным тестом до исправления, см. раздел 5) | `EndpointLoggingFlask` в [app/logging_setup.py](../../../app/logging_setup.py) переопределяет только `log_exception`: имя endpoint вместо пути; уровень ERROR, логгер Flask, `exc_info` с traceback сохранены |
| `/metrics`: запросы, ошибки, latency | Работали: `http_requests_total`, `http_request_duration_seconds` | Без изменений |
| `/healthz` с пробами зависимостей | Работал: [app/health.py](../../../app/health.py) | Без изменений |
| Технический алерт | Не было (Telegram в проекте — для аномалий данных) | [app/http_alert.py](../../../app/http_alert.py): правило доли 5xx; сценарий проверки [scripts/check_http_error_alert.py](../../../scripts/check_http_error_alert.py) |

## 1. Логирование

**Где смотреть.** Логи пишутся в stdout процесса: локально — в терминал
`python -m app.app`, в Compose — `docker compose logs -f app`. JSON-формат
включается `LOG_FORMAT=json` (в `.env.example` не задан, по умолчанию `text`).

**Что логируется.**

| Событие | Логгер, уровень | Поля |
|---|---|---|
| Каждый HTTP-запрос | `app.http`, INFO (< 500) / ERROR (5xx) | `event=http_request`, `method`, `endpoint`, `status`, `duration_ms`, `request_id` |
| Необработанное исключение | логгер Flask (`app` при `python -m app.app`), ERROR + `exc_info` | `Exception on <endpoint> [<метод>]`, traceback, `request_id` (JSON); в Sentry — при заданном `SENTRY_DSN` |
| Отказ зависимости в `/healthz` | `app.health`, WARNING | компонент, время, первая строка ошибки |
| Срабатывание / восстановление алерта | `app.alerts`, WARNING / INFO | `event`, `alert`, `requests`, `errors`, `ratio`, `threshold`, `window_s` |
| Старт, планировщик, сборщик | `app.startup`, `collectors.*`, `apscheduler.*` | сообщения о запуске заданий и ошибках сбора |

Сырой путь и query string в `app.http` и в строке необработанного исключения не пишутся:
вместо них имя Flask-endpoint (`auth.reset_password`, а не `/auth/reset-password/<token>`).

**Фактические записи** (dev-сервер `python -m app.app`, `LOG_FORMAT=json`,
09.11.2026; запрос к сбросу пароля шёл с тестовым токеном в пути и в `?token=`):

```json
{"timestamp": "2026-11-09T02:17:48.000+00:00", "level": "INFO", "logger": "app.http", "message": "GET auth.reset_password -> 400 in 22.0 ms", "request_id": "5c84d1bb630d404ea8d3f212b318c36f", "event": "http_request", "method": "GET", "endpoint": "auth.reset_password", "status": 400, "duration_ms": 22.0}
{"timestamp": "2026-11-09T02:17:47.908+00:00", "level": "WARNING", "logger": "app.health", "message": "healthz: target_db failed in 2ms: (sqlite3.OperationalError) unable to open database file\n(Background on this error at: https://sqlalche.me/e/20/e3q8)", "request_id": "c8ed286801f64bd9aee9ab081d41a9ac"}
{"timestamp": "2026-11-09T02:17:47.917+00:00", "level": "ERROR", "logger": "app.http", "message": "GET health -> 503 in 15.0 ms", "request_id": "c8ed286801f64bd9aee9ab081d41a9ac", "event": "http_request", "method": "GET", "endpoint": "health", "status": 503, "duration_ms": 15.0}
```

Отказ зависимости и ответ 503 связаны общим `request_id`.

## 2. Метрики

Источник — `GET /metrics` (формат Prometheus, без авторизации, лимит 60/мин).
Получение: `curl -s http://127.0.0.1:5001/metrics` или scrape Prometheus.

| Метрика | Источник | Как получить показатель | Интерпретация |
|---|---|---|---|
| Количество запросов — `http_requests_total{method,endpoint,status}` (counter) | хук `after_request`, [app/instrumentation.py](../../../app/instrumentation.py) | Разность счётчика за окно; PromQL `sum(rate(http_requests_total{endpoint!="metrics"}[5m]))` | Нагрузка на сервис; `endpoint="unknown"` — запросы к несуществующим маршрутам |
| Ошибки / доля ошибок — тот же счётчик, `status=~"5.."` | тот же хук | `increase(5xx) / increase(всех)` за окно; так считает алерт | Серверные сбои. 4xx — ошибки клиента, в долю не входят |
| Время ответа — `http_request_duration_seconds{method,endpoint}` (histogram, корзины 0,01…10 с) | `before_request` / `after_request`, `time.perf_counter()` | `_sum / _count` — среднее; `histogram_quantile(0.95, …_bucket)` — p95 | Скорость ответа; p95 точна до границ корзины |
| `failed_login_attempts_total`, `collector_runs_total{result}` | `app/auth.py`, `collectors/per_project.py` | `increase(...)` | Подбор паролей; исправность фонового сбора. В требования #15 не входят |

Фактический фрагмент ответа (тестовый клиент, 3 × `/healthz` и один 404):

```text
200 text/plain; version=1.0.0; charset=utf-8
http_requests_total{endpoint="health",method="GET",status="200"} 3.0
http_requests_total{endpoint="unknown",method="GET",status="404"} 1.0
http_request_duration_seconds_bucket{endpoint="health",le="0.05",method="GET"} 3.0
http_request_duration_seconds_count{endpoint="health",method="GET"} 3.0
http_request_duration_seconds_sum{endpoint="health",method="GET"} 0.022163244080729783
```

## 3. Эскиз дашборда

[DASHBOARD.md](DASHBOARD.md): 10 панелей (health, RPS, доля 5xx, p50/p95,
состояние алерта, статусы, latency по endpoint, топ 5xx, неудачные входы,
сборщик), для каждой указаны источник, выражение и интерпретация.

## 4. Технический алерт `HttpErrorRatioHigh`

| Параметр | Значение |
|---|---|
| Метрика | `http_requests_total` из `/metrics`, без `endpoint="metrics"` (опросы сборщика) |
| Условие | доля 5xx за окно **> 10 %** |
| Порог, обоснование | В норме 5xx около 0. При минимальном объёме (10 запросов) нужно ≥ 2 ошибки, поэтому одиночный сбой алерт не поднимает. Полный отказ зависимости (`/healthz` 503) при одном трафике healthcheck даёт 100 %; при 3 запросах на тик, как в сценарии, — около 33 % |
| Окно | 5 минут, разность счётчиков между самым старым снимком окна и последним (как `increase()` в Prometheus, без экстраполяции) |
| Минимум наблюдений | **≥ 10 запросов** в окне. Docker HEALTHCHECK опрашивает `/healthz` раз в 30 с — около 10 запросов за 5 минут и без пользователей, но **не гарантированно**: в зависимости от того, как опросы наблюдателя ложатся относительно проверок, и от задержек/таймаутов healthcheck в окне может оказаться 9. При меньшем объёме решение не принимается, состояние сохраняется до следующего опроса |
| Сигнал | Запись `app.alerts`: WARNING `alert_firing`, INFO `alert_resolved`; поле `event` в JSON-строке результата оценки |
| Канал / способ наблюдения | stdout наблюдателя `python -m app.http_alert` (или `docker compose logs`, если запустить его рядом с сервисом). **Доставка во внешний канал (Telegram, почта) не реализована и не проверялась** |
| Ответственная роль | DevOps / дежурный инженер (по [MLOPS_ROADMAP](../MLOPS_ROADMAP.md) — роль DevOps) |
| Сброс счётчиков | Если уменьшился общий счётчик запросов **или** счётчик 5xx (перезапуск процесса), история окна сбрасывается: отрицательная доля не вычисляется, `resolved` по несравнимым снимкам не выдаётся (при `firing` состояние сохраняется), оценка возобновляется, когда в новом окне набирается ≥ 10 запросов |

**При срабатывании.** 1) Открыть `/healthz`: какой компонент `down`.
2) Найти в логах `app.http` с `level=ERROR` и `app.health` за последние 5 минут;
связать по `request_id`. 3) Если причина — БД: `docker compose ps`,
`docker compose logs postgres timescaledb`, перезапуск сервиса БД; если 5xx
не от `/healthz` — traceback в логах `app` / Sentry. 4) Зафиксировать время
начала и причину.

**При восстановлении.** Дождаться `alert_resolved` (ошибки вышли из окна),
проверить `/healthz` = 200 и отсутствие новых 5xx, дописать в запись
об инциденте время восстановления и действия.

## Воспроизведение

Из корня репозитория с установленными зависимостями (`pip install -r requirements-dev.txt`).

**A. Детерминированный сценарий «норма → сбой → восстановление»** —
настоящее Flask-приложение в режиме TESTING, тестовый клиент, временные SQLite,
реальные хуки и реальный текст `/metrics`; моделируются только часы
опроса (один `/metrics` на тик 30 с). Сбой — недоступная целевая БД (`/healthz` → 503).
Sentry, Telegram, LLM и SMTP принудительно выключены; сеть не используется.

```bash
python -m scripts.check_http_error_alert > /tmp/alert-demo.json 2> /tmp/alert-logs.jsonl
echo $?   # 0 — все проверки прошли
python -m pytest -q tests/test_http_error_alert.py
```

Ожидается: в фазе `normal` (10 тиков) состояние всё время `inactive`;
в `failure` (6 тиков) — событие `firing`; в `recovery` (12 тиков) — `resolved`
и итоговое `inactive`; тестовые токены не встречаются ни в логах, ни в `/metrics`.

**B. Наблюдатель против работающего сервиса** (реальный HTTP и реальное время;
окно сокращено до 30 с, чтобы прогон занял около двух минут):

```bash
mkdir -p /tmp/alert-live && cd /tmp/alert-live   # пустой каталог: без .env
# сбой с самого старта: каталога для целевой SQLite ещё нет
LOG_FORMAT=json FLASK_DEBUG=0 PORT=5918 PYTHONPATH=<repo> \
  DATABASE_URL=sqlite:////tmp/alert-live/missing/target.db \
  MONITOR_DB_URL=sqlite:////tmp/alert-live/mon.db python -m app.app > server.log 2>&1 &
PYTHONPATH=<repo> LOG_FORMAT=json python -m app.http_alert \
  --url http://127.0.0.1:5918/metrics --interval 3 --window 30 --iterations 36 > watcher.log &
# трафик: curl /healthz, /, /auth/reset-password/<тестовый токен> раз в секунду;
# через 40 с — восстановление: mkdir -p /tmp/alert-live/missing
```

Сбой включается с первой секунды: после первого успешного подключения SQLite
держит соединение в пуле, и удаление каталога на лету отказа не вызывает.

## Фактические результаты

09.11.2026, Python 3.12.3, локально без Docker.

**A** ([demo.json](demo.json), `passed: true`, все 8 проверок `true`):

| Тики | Фаза | Запросов / 5xx в окне | Состояние |
|---|---|---|---|
| 1–2 | normal | 4, 7 / 0 | данных мало (< 10), `inactive` |
| 3–10 | normal | 10…31 / 0 | `inactive` |
| 11–13 | failure | 30 / 1…3 (≤ 10 %) | `inactive` |
| **14** | failure | 30 / 4 (13,3 %) | **`firing`** — на 4-м тике сбоя |
| 15–22 | failure → recovery | 30 / 5–6, затем 4 | `firing`, пока ошибки в окне |
| **23** | recovery | 30 / 3 (10,0 %) | **`resolved`** — на 7-м тике восстановления |
| 24–28 | recovery | 30 / 2…0 | `inactive` |

**B** (dev-сервер `python -m app.app` и наблюдатель по HTTP):

| Время UTC | Событие | Окно 30 с: запросов / 5xx |
|---|---|---|
| 02:17:47 | старт со сбоем, `/healthz` → 503 | — |
| 02:17:54 | `alert_firing` | 15 / 5 (33,3 %) |
| 02:18:27 | восстановление (каталог создан) | — |
| 02:18:48 | `alert_resolved` | 75 / 5 (6,7 %) |
| 02:18:51–02:19:27 | `inactive` | 75–78 / 2…0 |

В `server.log` 2 строки от `werkzeug` (баннер запуска), строк доступа Werkzeug нет;
тестовые токены в `server.log` и `watcher.log` не найдены (`grep -c` = 0).

**Доработка по второму ревью** (09.11.2026, Python 3.12.3): новые регрессионные тесты
до исправления — 5 failed (ложный `resolved`: `[None, 'firing', 'resolved']`; тестовые
`token=`, пароль DSN и Bearer — в traceback в JSON и text), после — `tests/test_http_error_alert.py`
15 passed. Сценарий A повторён: те же 8 проверок `true`, `firing` на тике 14,
`resolved` на тике 23, итог `inactive` — результаты совпадают с [demo.json](demo.json).

**Исправление лога исключения** (доработка по ревью, 09.11.2026): dev-сервер
`python -m app.app`, `FLASK_DEBUG=0`, `MONITOR_DB_URL` в несуществующем каталоге —
запрос `/auth/reset-password/<тестовый токен>?token=<тестовое значение>` падает с
необработанным `sqlalchemy.exc.OperationalError` внутри `auth.reset_password`.
Ответ 500 в `LOG_FORMAT=json` и `text`; тестовые значения и `/auth/reset-password`
в логе сервера не найдены (`grep -c` = 0). Фактические записи (JSON; `timestamp` опущен, `exc_info` сокращён):

```json
{"level": "ERROR", "logger": "app", "message": "Exception on auth.reset_password [GET]", "request_id": "rid-live-json", "exc_info": "Traceback (most recent call last): … sqlalchemy.exc.OperationalError: (sqlite3.OperationalError) unable to open database file …"}
{"level": "ERROR", "logger": "app.http", "message": "GET auth.reset_password -> 500 in 12.2 ms", "request_id": "rid-live-json", "event": "http_request", "method": "GET", "endpoint": "auth.reset_password", "status": 500, "duration_ms": 12.2}
```

## 5. Безопасность

- **Секреты в логах.** Тестовые значения (токен в пути сброса пароля,
  `?token=`, заголовок `Authorization: Bearer`) не попали ни в логи, ни в
  `/metrics` в A и B; автоматическая проверка — `fake_secrets_absent_from_logs_and_metrics`
  и `tests/test_http_error_alert.py`. Существующие тесты маскирования (`tests/test_secrets_scrub.py`,
  `tests/test_logging_json.py`, `tests/test_sentry.py`) проходят.
- **Необработанное исключение с секретом в пути** (замечание ревью, исправлено).
  Источник — `Flask.log_exception` (Flask 3.1.3), вызываемый из `handle_exception`
  при `FLASK_DEBUG=0`: сообщение строилось из `request.path`. До исправления тест
  записывал `Exception on /auth/reset-password/<тестовый токен> [GET]` в JSON и text.
  После: `Exception on auth.reset_password [GET]`, ответ — штатный 500
  `Internal Server Error`, traceback и `request_id` сохранены. Проверка:
  `python -m pytest -q tests/test_http_error_alert.py -k unhandled` — сбой хранилища
  внутри `auth.reset_password`, `PROPAGATE_EXCEPTIONS=False`, токен в пути и в `?token=`,
  оба формата логов.
- **Секреты в traceback** (замечание второго ревью, исправлено). Фабрика `LogRecord`
  маскировала только `msg` и `args`; текст, который форматтер строит из `exc_info`
  (сообщение исключения, строки исходного кода), шёл в лог как есть. Пример — существующий
  обработчик `_probe_iceberg` в [app/connections.py](../../../app/connections.py):
  `logger.warning("iceberg probe failed: %s", exc, exc_info=True)` — сообщение было
  замаскировано, а traceback содержал `token=…`, пароль DSN и Bearer-токен. Теперь
  `ScrubbingFormatter` (базовый для `JsonFormatter` и текстовый форматтер `configure_logging`)
  применяет тот же `scrub_value` к traceback и `stack_info`; traceback, тип исключения,
  уровень и `request_id` сохраняются, известные форматы заменяются на `***`. Проверка:
  `python -m pytest -q tests/test_http_error_alert.py -k "traceback or unhandled"` —
  реальный путь `_probe_iceberg` и необработанное исключение Flask, оба формата логов.
- **Токен сброса пароля в событии Sentry** (замечание ревью, исправлено). Токен стоит в
  сегменте пути `/auth/reset-password/<token>`, а не в `?token=`, и прежний `before_send`
  его не распознавал. Проверено с настоящим `FlaskIntegration` и локальным транспортом
  (без отправки в Sentry): до исправления тестовый токен был в `request.url` и в
  переменных кадров (`environ`: `PATH_INFO`, `REQUEST_URI`, `RAW_URI`; repr `Request`).
  Теперь `before_send` в [app/sentry.py](../../../app/sentry.py) заменяет этот сегмент
  на `[REDACTED]` во всех строках события, включая токен в процентной кодировке:
  `https://example.test/auth/reset-password/[REDACTED]`. Маршрут, тип исключения,
  стек и остальные поля сохраняются, другие URL не меняются. Если очистка события
  падает, вместо исходного события (раньше оно уходило без маскирования) отправляется
  заглушка с тем же `event_id`, тегом `scrub_failed` и без данных запроса. Проверка:
  `python -m pytest -q tests/test_sentry.py` (до исправления — 6 failed).
- **Внешние сервисы.** Сценарий A работает без сети: Sentry, Telegram, LLM, SMTP пустые;
  `/healthz` помечает SMTP и in-memory лимиты как `n/a` и отвечает 200.
  Наблюдатель при недоступном `/metrics` пишет `scrape_failed` и продолжает работу.
- **`/metrics` не нужно открывать публично.** Наблюдатель и Prometheus опрашивают сервис
  изнутри хоста или сети Compose. Endpoint без авторизации (стандарт для scrape);
  лейблы — метод, имя endpoint и статус, без путей и пользовательских данных.
  В Compose порт приложения по умолчанию слушает `0.0.0.0` — снаружи учебного стенда
  его следует ограничить `BIND_ADDRESS=127.0.0.1` или сетевым ACL.
- **Регрессии.** Аутентификация и API не менялись; полный офлайн-набор тестов — в описании PR.

## Ограничения

- Алерт проверен как правило + сигнал в логе. Внешняя доставка не выполнялась;
  наблюдатель не запускается автоматически в Compose — его нужно запустить отдельно.
- В A время опросов смоделировано; реальное время проверено только в B
  с сокращённым до 30 с окном. Полный 5-минутный прогон в реальном времени не выполнялся.
- Prometheus/Grafana не разворачивались; выражения PromQL в [DASHBOARD.md](DASHBOARD.md)
  на настоящем Prometheus не проверялись.
- Счётчики живут в памяти одного процесса (`python -m app.app`); для нескольких
  worker-процессов нужен multiprocess-режим `prometheus_client`.
- Состояние алерта хранится в памяти наблюдателя; после его перезапуска окно набирается заново.
- Строка необработанного исключения больше не содержит путь, а traceback проходит
  маскирование, но только **существующими шаблонами** (`token=`, `password=`, DSN, Bearer,
  Telegram, AWS): секрет другого формата в сообщении исключения или в строке исходного
  кода попадёт в лог. Универсальной защиты от произвольных секретов нет. Маскирование
  traceback действует в форматтерах `configure_logging`; сторонний обработчик со своим
  форматтером его не получает.
- Сброс счётчиков распознаётся по уменьшению любого из них между соседними опросами;
  если за время между опросами процесс перезапустился и оба счётчика уже превысили
  прежние значения, сброс незаметен (как и у `increase()` в Prometheus).
- Режим `FLASK_DEBUG=1` (интерактивный отладчик Werkzeug, локальная разработка) Flask
  обрабатывает без `log_exception`; в этой задаче не проверялся. Docker-образ
  запускается с `FLASK_DEBUG=0`.
- Sentry: токен маскируется только для маршрута `/auth/reset-password/<token>` в
  буквальном виде. Если сам префикс маршрута записан в процентной кодировке (например,
  `reset%2Dpassword`), в «сырых» полях `environ` он не распознаётся. Секреты других
  форматов в путях не распознаются. Реальная отправка в Sentry не проверялась —
  только локальный транспорт SDK. Настройки SDK не менялись.
- В текстовом формате логов (`LOG_FORMAT=text`) `request_id` не выводится — так
  было и до #15; корреляция по `request_id` доступна в `LOG_FORMAT=json`.
- Docker-сборка и Compose-прогон в этой задаче не выполнялись: Docker и API не изменялись.
