# Эксперименты Isolation Forest в MLflow

Задача [#9](https://github.com/aleksandr-novikov/ai-integration-development/issues/9). [Протокол](PROTOCOL.md) фиксирует baseline, пять конфигураций, метрики и правило выбора до просмотра test. Учебный пайплайн переиспользует подготовку признаков и scaler из #7; рабочий сервис DB Monitoring не изменяется.

**Фактический результат:** [5 экспериментов и финальная оценка](RESULTS.md). На test обнаружены 9/9 инцидентов, но FPR 13,27% превышает цель 10%.

## Повторить серию

Python 3.12 рекомендуется для совпадения с зафиксированным окружением; CI проверяет код также на 3.11/3.13. Из корня репозитория:

```bash
python -m pip install -r requirements-course.txt
python -m scripts.generate_ml_dataset --output data/course
python -m scripts.run_ml_experiments --dataset data/course --output artifacts/course-experiments/run-01
```

Каталог результата должен быть пустым, рабочее дерево Git — чистым. Генератор не перезаписывает существующий датасет; если он уже есть, повторять первую команду генерации не нужно. Для следующего запуска выбирайте новый каталог. Скрипт сам проверяет данные и сохраняет отчёт перед созданием MLflow runs. При провале валидации обучение не начинается.

Пять runs содержат оценку на validation; отдельный `final_test` — оценку выбранной конфигурации. Каждая конфигурация состоит из трёх моделей по таблицам. Сохраняются параметры, метрики и знаменатели, предсказания, найденные/пропущенные события, Git SHA, версии библиотек, dataset_id и манифест, модели StandardScaler + IsolationForest в формате skops. Сигнатура модели — массив четырёх чисел в порядке `row_count, null_rate, d_row_count, d_null_rate`; `predict` возвращает +1 (норма) / −1 (аномалия), `decision_function < 0` означает сигнал. В итоговом run есть также сравнительная таблица и график validation.

После выполнения можно открыть локальный интерфейс:

```bash
MLFLOW_DISABLE_TELEMETRY=true mlflow ui --backend-store-uri sqlite:///artifacts/course-experiments/run-01/mlflow.db --host 127.0.0.1 --port 5001
```

Открыть http://127.0.0.1:5001 и эксперимент `course-isolation-forest`. Работа с [локальным SQLite](https://www.mlflow.org/docs/latest/ml/tracking/tutorials/local-database/) и [формат sklearn-моделей](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.sklearn.html) описаны в документации MLflow.

## Открыть готовые результаты без обучения

Архив `mlflow-export.zip` содержит JSON-экспорт шести завершённых runs и все артефакты. Он не требует доступности исходного компьютера. Скрипт импорта создаёт новые run IDs, сохраняет исходные в тегах/артефактах и пишет `run-id-map.json`. Времена импортированных runs относятся к импорту; исходные времена доступны в `original/run.json`.

```bash
python -m scripts.import_ml_experiments --archive docs/course/experiments/mlflow-export.zip --output artifacts/course-experiments/imported
MLFLOW_DISABLE_TELEMETRY=true mlflow ui --backend-store-uri sqlite:///artifacts/course-experiments/imported/mlflow.db --host 127.0.0.1 --port 5001
```

Каталог импорта должен быть пуст. В архиве находятся только артефакты и метаданные; исходный SQLite с абсолютными локальными путями не переносится. Импорт не загружает модели и не переобучает их. Открывайте модели из доверенных экспортов. JSON-метрики с null (например, задержка при отсутствии обнаружений) сохраняются в артефактах и не подменяются нулём в числовых метриках MLflow.

## Проверка реализации

```bash
python -m pytest tests/test_course_experiments.py tests/test_validate_ml_dataset.py tests/test_generate_ml_dataset.py tests/test_seed_metrics_db.py -q
ruff check .
```

Тесты проверяют арифметику метрик, привязку событий к объектам и интервалам, правило выбора, обучение только на train, блокировку плохих данных, последовательность 5 validation → выбор → 1 test, экспорт/импорт всех runs и совпадение score/предсказаний загруженной модели. Для тестовой серии используется отдельный маленький набор с seed=999; это не результаты зачётного эксперимента.
