# Проверка архитектуры по задаче #3

Дата: 13.10.2026. Проверяемый код: `59c54fa3f5de502126cbbbf0617688e1b9ead4eb`. Основной документ — [Architecture.md](../Architecture.md), решение по сервису — [ADR-001](../adr/001-flask-and-inference-service.md). Этот протокол содержит доказательства, а не альтернативную архитектуру.

## Сверка с реализацией

| Утверждение | Источник в коде |
|---|---|
| Flask-фабрика, HTML login gate, запуск scheduler в процессе | [app/app.py](../../app/app.py): `create_app` |
| Текущий проект выбирается среди доступных пользователю | [app/projects.py](../../app/projects.py): `load_current_project_into_g` |
| Отдельный сбор по каждому подключению, интервал и защита от наложения тиков | [collectors/per_project.py](../../collectors/per_project.py): `collect_for_connection`, `add_job_for_connection`; [app/connections.py](../../app/connections.py): диапазон 5–1440 |
| Legacy-сбор, cron обучения, обход проектов с историей | [collectors/scheduler.py](../../collectors/scheduler.py): `start_scheduler`, `_iter_retrain_projects`, `retrain_forecasts`, `retrain_anomaly_detectors` |
| PostgreSQL/TimescaleDB и SQLite, миграции, project-scoped чтение | [app/metrics_storage.py](../../app/metrics_storage.py): `_apply_schema`, `get_metrics`, `get_anomaly_scores`; [timescale_schema.sql](../../scripts/timescale_schema.sql) |
| Общие признаки при обучении и скоринге, scaler и модель на диске | [ml/anomaly_detector.py](../../ml/anomaly_detector.py): `_load_features`, `train`, `score_table`, `_model_path` |
| Обучение прогноза на запросе; обработка change-point и резкого падения | [ml/forecast.py](../../ml/forecast.py): `forecast`, `train`, `_anchor_shift`, `_normalize_forecast_point` |
| Чтение scores и синхронный прогноз; схема/список таблиц через адаптер | [app/api.py](../../app/api.py): `anomalies`, `forecast_endpoint`, `tables`, `schema` |
| Поконнектный скоринг зависит от Telegram; фильтры и throttle | [collectors/per_project.py](../../collectors/per_project.py): `_maybe_notify_anomalies`; [telegram.py](../../app/notifications/telegram.py) |
| LLM использует контекст метрик, имеет шаблонный fallback | [app/llm.py](../../app/llm.py): `_build_prompt`, `explain_anomaly` |
| Техническая наблюдаемость уже существует | [app/health.py](../../app/health.py), [app/instrumentation.py](../../app/instrumentation.py), [app/logging_setup.py](../../app/logging_setup.py), [app/sentry.py](../../app/sentry.py) |
| Учебные MLflow-runs отделены от приложения | [ml/course_experiments.py](../../ml/course_experiments.py), [протокол экспериментов](experiments/PROTOCOL.md) |

## Обнаруженные ограничения и дальнейшие задачи

1. **#4/#12: граница API и аутентификация.** `_current_project_id` сохраняет fallback `legacy`; HTML login gate не является общей защитой `/api`. `/api/tables` и `/api/schema` используют адаптер источника. Контракт должен описать реальное поведение и проверять запрет чужого контекста, включая отсутствие текущего проекта.
2. **#12/#18: кэш объяснений.** `get_cached_explanation`/`save_explanation` используют ключ `(table_name, metric, ts)` без `project_id`. `POST /api/explain` возвращает cache hit до проверки принадлежности таблицы проекту. В задаче архитектуры это обнаружено статически, не исправлено; нужны миграция ключа и регрессионный тест с одинаковой таблицей/временем у двух проектов.
3. **#5/#12: публикация модели.** В рабочем joblib payload есть модель, scaler и время обучения, но нет всей цепочки code/data/run_id. Экспериментальные skops-артефакты не становятся автоматически рабочими joblib-моделями; формат и загрузку нужно согласовать, обучение исключить из целевого inference-запроса.
4. **#13: сборка Docker.** `requirements-dev.txt` включает `requirements-course.txt`, но Dockerfile до `pip install` копирует только `requirements.txt` и `requirements-dev.txt`. При чистой сборке включённого файла ещё нет. Это конкретный статический блокер; чистая сборка в этой документационной задаче не выполнялась.
5. **#13/#15: фоновые задания и ресурсы.** Scheduler живёт внутри каждого процесса приложения, распределённой блокировки нет. Не обещать горизонтальное масштабирование простым увеличением workers. Полный SQL-сбор может быть дорогим; режимы и ограничения подтверждаются для конкретного адаптера.
6. **#16: свежесть и идентичность данных.** Поконнектный оперативный IF-скоринг пропускается при ненастроенном Telegram. Также модель и история привязаны к проекту/имени таблицы; одинаковые имена в разных подключениях одного проекта требуют отдельного сценария проверки. Это не проверено тестом межпроектной изоляции.

Перечень является результатом анализа исходного кода, не заявлением о выполненном исправлении. Прежние обещания «нет авторизации», «TimescaleDB позже», «уведомлений нет» сняты; ограничения прогнозирования обновлены по фактическим функциям и тестам.

## Проверки

Уточнение по [ревью Раиля от 14.10.2026](https://github.com/aleksandr-novikov/ai-integration-development/pull/28#issuecomment-6175228687): ADR-001 теперь задаёт опубликованный манифест MLflow как источник активной модели, кандидатов от ночного обучения, единый путь скоринга API/дашборда и хранение версии результата. Историю читает Flask-адаптер, общая чистая функция строит признаки для обучения и инференса, FastAPI только валидирует, масштабирует и предсказывает. Это выбранная целевая граница для #4/#5/#12, не утверждение о выполненной миграции. Повторное ревью запрошено; прежние статические ограничения остаются актуальными.

Уточнение по [повторному ревью от 15.10.2026](https://github.com/aleksandr-novikov/ai-integration-development/pull/28#issuecomment-6195626826): текущий ключ `anomaly_scores` и upsert в `save_anomaly_scores` не различают версии; `warmup_anomalies` в [warmup_ml.py](../../scripts/warmup_ml.py) пересчитывает 14 дней. ADR и критерии #12 теперь явно требуют миграцию ключа с `model_version`/`feature_schema_version`, сохранение старых строк как неизвестной legacy-версии и проверки повторного скоринга/отката без потери истории. Схема базы и код здесь не изменены. В журнале ИИ удалены пустые строки, разрывавшие Markdown-таблицу.

Запуск существующих регрессионных тестов:

```bash
venv/bin/python -m pytest -q \
  tests/test_forecast.py tests/test_tenant_isolation.py \
  tests/test_scheduler_ml_retrain.py tests/test_per_project.py
```

Результат 13.10.2026: **52 passed in 24.41s** (Python 3.12). Эти проверки покрывают прогноз, выбор проекта, хранение метрик и расписание ML; они не доказывают устранение перечисленных пробелов всего API.

Относительные ссылки на файлы проверены автоматически; схема Mermaid сверена с описанными путями исполнения и проверена в GitHub Preview; CI PR отдельно проверяет обычные и интеграционные тесты. Новые программные тесты не добавлялись: меняется документация. Командное ревью архитектуры и ADR запрашивается через PR; одобрение не предполагается заранее.
