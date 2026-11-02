# Тестовая стратегия · Issue №11

Цель — проверять поведение DB Monitoring на границах данных, модели, хранилища
и HTTP, переиспользуя существующий pytest-набор. Зелёный технический тест не
означает допуск качества модели: метрики и ограничения — в [протоколе качества](../quality/README.md).

| Риск | Уровень и существующие/добавленные проверки | Команда |
|---|---|---|
| Неверное выравнивание, дельты, нехватка истории | Unit: `test_openapi_contract.py` (alignment, missing previous), `test_anomaly_detector.py` | `pytest tests/test_anomaly_detector.py tests/test_openapi_contract.py` |
| Изменение знака/округления score, переобучение scaler при inference | Unit + API с mock-моделью: `test_testing_pipeline.py::test_mock_model_postprocessing_persists_through_api` | `pytest tests/test_testing_pipeline.py` |
| Несовместимость настоящей модели с БД и API | Локальная интеграция: настоящие IsolationForest, StandardScaler, joblib и временная SQLite → `/api/anomalies`; без маркера `integration`, идёт в обычном наборе | `pytest tests/test_testing_pipeline.py` |
| Ошибки диалектов SQL, миграций и доступности образов | Docker integration: PostgreSQL/MySQL/ClickHouse/TimescaleDB/Iceberg + MinIO, `tests/integration/` | `make test-integration` (нужен Docker) |
| Регрессии контракта, авторизации и изоляции проектов | Unit/HTTP: `test_openapi_contract.py`, `test_tenant_isolation.py`, `test_security.py`; два OpenAPI-документа №4 | `pytest tests/test_openapi_contract.py tests/test_tenant_isolation.py tests/test_security.py` |
| Утечка данных/платные обращения NIM и Telegram | Unit: `test_llm.py`, `test_telegram.py`, `test_llm_audit.py`; общая fixture блокирует внешние sync/async HTTPX transport, моки остаются на границе | `pytest tests/test_llm.py tests/test_telegram.py tests/test_llm_audit.py tests/test_testing_pipeline.py` |
| Поломка пользовательских сценариев и JS | E2E: Chromium → настоящий Flask + SQLite, introspection целевой БД подменён; `tests/e2e/` | `make test-e2e` |

## Запуск и CI

В Python 3.11–3.13: `python -m venv venv`, активировать окружение,
`pip install -r requirements-dev.txt`. Для backup/restore нужен системный `sqlite3` CLI.
Для браузера: `playwright install --with-deps chromium` (Playwright закреплён на 1.61.0).
`pytest` исключает **оба** маркера `integration` и `e2e`;
`make test-coverage` запускает тот же набор с branch coverage `app, collectors, ml`.
`make lint` и `git diff --check` — перед коммитом. Все маркеры разом:
`pytest -o addopts=''` (нужны Docker и браузер).

По неизменённым `.github/workflows/tests.yml` и `lint.yml`:
**PR и push master** — `test (3.11)`, `test (3.12)`, `test (3.13)`, `integration`, `ruff`;
**только push master** — `e2e`. В PR `e2e` пропускается условием; его зелёный
результат для нового изменения требует запуска на master. Такой запуск для базы
`ac0ae04` завершился ошибкой на установке браузера: разбор и повторная проверка
после merge описаны в [VERIFICATION.md](VERIFICATION.md).
Python 3.12 публикует `coverage-report`, E2E при падении — `e2e-screenshots`.
Порога coverage в CI нет. Рекомендуемые обязательные checks перечислены выше,
но обязательность не настроена: публичная проверка 2026-10-31 вернула
`master.protected=false`, пустые required contexts и пустой список rulesets.
Авторизованный endpoint protection недоступен (401); право менять настройки не проверялось.
Зелёный PR поэтому не является техническим запретом merge.

## Найденные пробелы и границы

Ранее `.coveragerc` исключал весь `ml/`; добавлен в отчёт без искусственного порога.
Отдельная связка настоящей ML-модели с SQLite и API добавлена к уже существующим
проверкам модели и Postgres → collector → API. Сквозные реальные NIM/Telegram
намеренно не проверяются; браузер не проверяет реальные внешние СУБД.
Недоступный Docker документируется как «не запускался», без новых skip.
Фактические результаты и соответствие пяти критериям — в [VERIFICATION.md](VERIFICATION.md).
