# Проверки FastAPI-сервиса — 02.11.2026

Исходный HEAD PR #39: `783e83facc16538b5654fccf45f05803976d0f5e`.
Проверочный GitHub merge: `de21179775e1b8233cb7a9a757cb7bbec893130a`, родители
`bbef1cd819b4cfc2bf955cb66de7dbf02321d41c` (master) и исходный HEAD.
Merge в публикуемую ветку не выполнялся. Локально проверялось объединённое
дерево, затем оно же с добавленным GET и регрессионными тестами. Исходные файлы
master восстановлены после испытаний; изменения Issue #11 не входят в доработку.
Окончательные SHA, результаты CI и статусы приёмки приведены в описании PR #39.

## Изменения

Требуемый `GET /v1/model` ранее возвращал 405. Добавлен вариант с шестью
query-параметрами, без GET body. Существующий POST сохранён. Оба пути используют
одинаковые проверки bearer-токена, полного source, версий и реального артефакта.
Неизвестные, отсутствующие и повторные параметры GET дают 422, чужой source —
403, несовместимая схема — 409, отсутствующая модель — 503. Обучения нет.

Добавлены 14 регрессионных случаев GET и проверка в Docker image. Добавлен
интеграционный сценарий обновления существующей TimescaleDB hypertable с данными
и старой таблицы scores; проверяются сохранение chunks/строк, повторная миграция,
legacy-маркеры и версионированный upsert. Изменений алгоритма миграции нет.

## Локальные испытания

Python 3.12.3, Linux x86_64, `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`.
Flask 3.1.3, FastAPI 0.162.0, SQLAlchemy 2.0.49, HTTPX 0.28.1,
scikit-learn 1.6.1, Playwright 1.61.0, Chromium 141.0.7390.37.

| Проверка | Фактический результат |
|---|---|
| Профиль после исправления: `test_db_utils.py`, `test_inference_service.py`, `test_inference_migration.py`, `test_testing_pipeline.py`, `test_openapi_contract.py` | **363 passed**, 42,87 с; общая `offline_httpx` из master включена |
| E2E исходного объединённого дерева | **19 passed**, 57,59 с |
| E2E после доработки, на публикуемой ветке | **19 passed**, 52,55 с; master-only job в PR этим не заменяется |
| Ruff, `pip check`, оба OpenAPI-документа, `git diff --check` | PASS |
| PostgreSQL migration через testcontainers | 2 ошибки запуска: Docker socket отсутствует; миграция локально не выполнена |
| Python 3.11/3.13, полный Docker integration, новый TimescaleDB-сценарий | Локальных интерпретаторов/Docker нет; подтверждение — CI нового HEAD, ссылка в PR |

Полный диагностический прогон объединённого дерева **до исправления GET**, с
новыми регрессионными тестами: **1483 passed, 25 failed, 53 deselected**, 436,49 с,
branch coverage 87% (включая исполнение упавших тестов). Причины:

- 14 новых GET-проверок воспроизвели 405; все прошли после исправления.
- 7 `test_db_utils.py` упали из-за `DATABASE_URL=sqlite://` вместо штатного
  `postgresql://test:test@localhost/test`; повтор с конфигурацией CI прошёл.
- 4 `test_backup_restore.py` требуют отсутствующий `sqlite3` CLI. Тесты не
  отключались и не изменялись; локальный полный набор зелёным не объявляется.

Новые фикстуры, `test_testing_pipeline.py`, покрытие `ml/` и Playwright 1.61.0
из Issue #11 использовались при проверке объединённого дерева. Реальных вызовов
NIM/Telegram не было. E2E-job в PR пропускается условием master-only; это не PASS.

## Матрица доказательств девяти критериев

Полные исходные формулировки и окончательные отметки — в описании PR.

| № | Требование → реализация | Испытание → результат → доказательство |
|---|---|---|
| 1 | FastAPI, три эндпоинта → `inference/service.py`, OpenAPI | Настоящий Uvicorn; GET health/model и POST predict вернули 200 в Swagger. Импорт сервиса не загрузил Flask/scheduler |
| 2 | Интерактивный Swagger → `scripts/inference_smoke.py` | Шесть реальных Execute: health, GET/POST model, predict 200/503/422; запросы/ответы в `evidence/final-smoke.json` |
| 3 | Версии в ключе/upsert → `metrics_storage.py`, `source_storage.py`, SQL schemas | `exercise_migration`: SQLite PASS; PostgreSQL и TimescaleDB требуют integration CI |
| 4 | Сохранение legacy → `_migrate_inference_columns`, registry validation, dashboard | SQLite migration и тесты registry/контракта: старые строки сохраняют `legacy-unknown`, опубликованный ID не может иметь этот маркер |
| 5 | История, идемпотентность, выбор/откат → adapter, storage, API/dashboard | SQLite migration, `test_flask_adapter_dashboard_two_versions_and_idempotence`, `test_review_version_switch_and_rollback_alert_only_fresh_events`, `test_review_quality_gate_compares_times_not_strings` — PASS; PostgreSQL требует CI |
| 6 | Отдельный сервис при сохранённом Flask → `inference/service.py`, ADR-001 | Реальный запуск без Flask/scheduler; сверено с опубликованными условиями В09, слайд 41, в `netology/ASSIGNMENTS.md` |
| 7 | Выбранный артефакт, валидация, ошибки → service, registry | Настоящие scaler/IsolationForest; `fit` запрещён тестом; GET/POST совпадают, 401/403/409/422/503 контролируемы — PASS |
| 8 | Изоляция, пустая история, отсутствие модели/LLM → adapter, API, LLM | Inference/LLM/Telegram-тесты диагностического набора и профильные контрактные тесты — PASS; внешние вызовы подменены на транспортной границе |
| 9 | Воспроизводимые примеры/latency → smoke и JSON evidence | 100/100 тёплых последовательных запросов, p95 34,364 мс; параметры и сырые samples сохранены; SLA не заявляется |

Два прежних дефекта не воспроизводятся: проходят
`test_source_failure_does_not_block_other_published_model`,
`test_duplicate_metrics_returns_conflict`,
`test_explanation_conflict_precedes_cache_and_llm`. Ошибка первой модели не
прерывает второй источник; неоднозначная история возвращает 409 до кэша/LLM.

## Swagger и производительность

[Новые запросы, ответы и замеры](evidence/final-smoke.json) получены настоящим
Uvicorn с временным реестром, экспортированным из неизменяемого архива #9.
Модель явно работает с `demo=true`, `quality_admitted=false`: качество этой
модели не допущено для продуктивного применения. `tested_code_sha256` связывает
результаты с точными байтами сервиса, контракта и smoke-скрипта.

Существующие `swagger-health.png`, `swagger-inspectModel.png`,
`swagger-predictAnomaly.png` изучены и сохранены как исторические примеры.
Повторный браузерный запуск использует те же запросы POST и артефакт; новый GET
и ошибки также выполнены в интерфейсе Swagger. Дубликаты изображений не добавлены.

Воспроизведение: `python -m scripts.inference_smoke --output /tmp/inference-evidence --swagger`.
Один worker, localhost HTTP, последовательные запросы, один холодный запрос
измерен отдельно, затем 100 тёплых; p95 — nearest rank. Число CPU/RAM хоста и
версии пакетов сохранены в JSON. Скрипт не задаёт лимиты ресурсов; унаследованные
лимиты не проверены. Замеры этого учебного стенда не являются SLA и не включают
Flask, чтение истории, LLM или Telegram. Старые значения на новый код не перенесены.

## Сохраняющиеся ограничения

- Для одноимённых таблиц необходим binding каждого обслуживаемого источника.
  Смешанный bound/unbound режим явно отмечается как неподдерживаемый;
  legacy-графики/прогноз/дрейф объединяют такие таблицы. Версионированные scores,
  объяснения и уведомления полностью привязанных источников изолированы.
- Ручная оценка неактивной версии ещё не обработанной точки может подавить
  последующее уведомление о ней. История версий сохраняется; доставка уведомлений
  не является критерием качества модели или гарантией exactly-once.
- Outbox, межпроцессная сериализация SQLite, восстановление после частичного DDL,
  сжатые TimescaleDB chunks и эксплуатационная репетиция backup/restore остаются
  отдельной работой по [runbook](migrations.md). Новая проверка обычной заполненной
  hypertable не подменяет эти испытания.
