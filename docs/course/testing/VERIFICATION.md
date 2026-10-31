# Проверка Issue №11

Дата: 2026-10-31. База: `ac0ae04fda308faaf613a3880be84b2d6463f88e` (`master`).
Разработка и проверки выполнены в зарегистрированной сессии
`s20261031-170510-e7e320`. Это снимок **до публикации**: результаты GitHub Actions
нового коммита необходимо сверять отдельно по SHA опубликованного PR.
Машиночитаемый снимок: [verification.json](verification.json).

## Матрица пяти критериев

| № | Критерий Issue №11 | Свидетельство и границы |
|---|---|---|
| 1 | Риски, уровни, команды и PR/master | [STRATEGY.md](STRATEGY.md); unit/HTTP + локальная SQLite-интеграция в default suite, Docker отдельно, E2E только master в CI |
| 2 | 3+ pre/post проверки, API + mock, настоящая модель + тестовая БД | Переиспользованы `test_proposed_features_follow_existing_alignment_and_delta_semantics`, `test_missing_previous_aligned_point_is_not_zero_filled` (2 случая) из `test_openapi_contract.py`, `test_parse_nim_response_clamps_confidence` и invalid JSON из `test_llm.py`. Добавлены `test_mock_model_postprocessing_persists_through_api`, `test_real_isolation_forest_sqlite_to_api` в `test_testing_pipeline.py` |
| 3 | Контракты №4 и замоканные внешние сервисы | `test_openapi_contract.py` проверяет оба OpenAPI-документа и реализованный Flask HTTP. `/v1/predict` пока только проект контракта, реализация относится к №12. Существующие NIM/Telegram-моки сохранены; `tests/conftest.py::offline_httpx` запрещает внешние реальные HTTPX transport, sync/async запрет подтверждён 4 тестами |
| 4 | Фактические проверки, имена checks, доступность защиты | Результаты ниже; имена и условия CI проверены по текущим workflow. GitHub Actions нового SHA на момент этого снимка ещё не запускались. Настройки защиты прочитаны публично, административный доступ не подтверждён |
| 5 | Найденный пробел и честное описание merge gate | `ml/` отсутствовал в `.coveragerc`: добавлен с branch coverage. Не было отдельной связки настоящая модель → SQLite → API: добавлена. Порог coverage отсутствует, master не защищён по доступным данным; эти тесты сами по себе не блокируют merge |

## Воспроизводимые проверки

Окружение: Python 3.12.3; зависимости из `requirements-dev.txt`;
Playwright 1.61.0 + pytest-playwright 0.7.0; Chromium 149.0.7827.55;
scikit-learn 1.6.1, NumPy 2.5.4, pytest 8.3.4, Ruff 0.9.4, coverage 7.13.5.
Использован собственный браузер Playwright 1.61 в игнорируемом `venv/ms-playwright`,
без подмены устаревшим Chromium из общего кеша.

| Проверка | Команда после активации venv | Результат |
|---|---|---|
| Профильная | `pytest tests/test_testing_pipeline.py tests/test_anomaly_detector.py tests/test_openapi_contract.py tests/test_llm.py tests/test_llm_audit.py tests/test_telegram.py -q` | 181 passed, 3 xfailed |
| Полный default suite с branch coverage | `venv/bin/coverage run -m pytest -q` | **1239 passed, 6 failed, 3 xfailed, 48 deselected**, 356,57 с; ошибки окружения разобраны ниже |
| Перепроверка двух CLI-тестов | `PATH="$PWD/venv/bin:$PATH" pytest tests/test_live_demo.py::test_live_demo_cli_requires_connection_id_for_non_legacy tests/test_seed_metrics_db.py::test_seed_cli_help_contains_project_id -q` | **2 passed**, 2,06 с |
| Отчёт | `coverage report -m && coverage xml` | **86%** суммарно (с ветвями), `ml/anomaly_detector.py` 84%, `ml/features.py` 89%; отчёт включает исполнение упавших тестов, порога нет |
| Браузерный E2E | `pytest -m e2e -q --tb=short` | **17 passed**, 1279 deselected, 39,63 с |
| Docker integration | `pytest -m integration -v` | **Не запускался:** отсутствуют Docker CLI и daemon socket; новых skip не добавлено |
| Ruff | `ruff check .` | PASS |
| Зависимости | `pip check` | PASS |
| Whitespace | `git diff --check` | PASS |

В локальной среде нет `make`: команды новой цели `make test-coverage` запускаются
непосредственно. По умолчанию pytest исключает integration и e2e; это не результат
запуска этих уровней. Модель/SQLite/API тест, напротив, входит в default suite.
Тестовые модели сохраняются во временном каталоге, веса не включаются в коммит.

Полный локальный набор **не зелёный**. Четыре падения в `test_backup_restore.py`
вызваны отсутствием `sqlite3` CLI (`backup.sh: line 94: sqlite3: command not found`).
Ещё два — `test_live_demo_cli_requires_connection_id_for_non_legacy` и
`test_seed_cli_help_contains_project_id` — не нашли команду `python`: исходный
запуск использовал `venv/bin/coverage` без добавления `venv/bin` в PATH.
После исправления PATH оба прошли без изменения исходников. Полный набор
повторно не запускался; результат 1239/6 выше сохраняет фактический исход
первоначального запуска, а не выдаёт объединённые прогоны за один зелёный.

Три существующих strict xfail из `test_llm_audit.py` связаны с №12/№18:
схема prompt читается из legacy и кеш объяснений не изолирует проекты (2 случая).
Они не исправлены и не скрыты этой задачей. Низкий FPR синтетического теста не
является валидацией качества на реальных данных. Внешние NIM и Telegram не вызывались.

## CI и фактическая защита ветки

Файлы `.github/workflows/` не изменялись. PR и push master запускают
`test (3.11)`, `test (3.12)`, `test (3.13)`, `integration`, `ruff`.
`integration` зависит от успешных `test`. `e2e` допускается только для
`push` в `refs/heads/master`; в новом PR он будет **SKIPPED**.
Успешность E2E нового изменения в GitHub Actions здесь не заявлена: нужна
отдельная проверка соответствующего запуска. Локальный E2E и старый master-run
не заменяют такое подтверждение.

Публичное чтение GitHub REST 2026-10-31 (без изменения настроек):

- [`branches/master`](https://api.github.com/repos/aleksandr-novikov/ai-integration-development/branches/master): HTTP 200, `protected=false`, `protection.enabled=false`, `enforcement_level=off`, `contexts=[]`, `checks=[]`.
- [`rulesets`](https://api.github.com/repos/aleksandr-novikov/ai-integration-development/rulesets): HTTP 200, `[]`.
- `branches/master/protection`: HTTP 401 `Requires authentication`. Нет оснований заявлять доступ к административным настройкам.

**Рекомендуемые**, но не настроенные обязательные checks: три `test`, `integration`,
`ruff`. Не называть их действующими merge gate. Master-only `e2e` не является
проверкой PR. Порог `coverage --fail-under` также не настроен.
Исторический MinIO-блокер из Issue №11 устранён в №27; успех старого запуска
не переносится на этот SHA. Свежий Docker-результат нужно получить из нового PR CI.
