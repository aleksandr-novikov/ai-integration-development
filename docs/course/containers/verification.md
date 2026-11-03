# Проверка контейнеризации #13

Дата обследования: 03.11.2026. База разработки — актуальный `master`
`46712fb54b77bc45e9f89693214e5a467d264270`. Issue #13 открыта, комментариев нет;
зависимости #7 и #12 закрыты. На начало разработки открытого PR по #13 нет.
Точная редакция файлов локальных проверок фиксируется SHA-256 в
[verification.json](verification.json); SHA опубликованного коммита находится
в PR. Это проверка рабочего дерева на указанной базе, не заявление о запуске
неизменённого master.

## Исходное состояние и изменения

Уже существовали Python 3.12 Dockerfile, PostgreSQL 18, TimescaleDB pg16,
раздельные DSN, volumes `pg_data`/`timescale_data`, healthchecks, scheduler,
seed/warmup, FastAPI и версионированный реестр. Историческая ошибка копирования
`requirements-course.txt` исправлена в #12: `COPY requirements*.txt` находится
до `pip install`. Dockerfile для этого не изменялся.

Исправлены рассогласованные инструкции SQLite/TimescaleDB и host-команды
демонстрации, отсутствие явной подготовки постоянного Fernet-ключа,
включение локальных секретов/данных/моделей в build context. Параметризованы
имена, host-порты, образ и UID/GID основного стенда с сохранением прежних
значений по умолчанию. Добавлен опциональный Compose-профиль существующего
FastAPI на том же образе. Обучение, хранение, API и допуск качества модели
не перерабатывались. Backup/Iceberg/ClickHouse-профили в новом маршруте не
включены и отдельно не испытаны.

## Матрица критериев и доступные доказательства

| № | Исходный критерий | Статус | Реализация, проверка и недостающее доказательство |
|---|---|---|---|
| 1 | Подготовлены Dockerfile и docker-compose.yml проекта. | PARTIAL | Dockerfile сохранён; YAML, разделение DSN, profiles, build args и routing команд проверены offline. Docker/Compose config, создание и запуск текущего образа здесь не выполнены. |
| 2 | При наличии Docker проверена сборка; при отсутствии Docker представлены корректные файлы и описание проверки, как прямо допускает слайд. | PASS — вариант без Docker | Конфигурация и зависимости статически проверены; точные команды чистой сборки, импорта, запуска и readiness приведены в [маршруте](README.md). Docker CLI отсутствует; текущая реальная сборка не заявляется. |
| 3 | Чистый checkout запускается по инструкции; .env.example достаточен без доступа к личной Supabase и платному LLM. | PARTIAL | Инициализация конфигурации без pip, локальные DSN, пустые внешние ключи, команды seed внутри Compose. Полный запуск чистого checkout без Docker не подтверждён. |
| 4 | БД, хранилище метрик, модель и приложение получают согласованные настройки; readiness/healthchecks и сохранность volumes проверены. | PARTIAL | Offline-проверки DSN, постоянного ключа между процессами, неизменных volumes и readonly-реестра. Live readiness и сохранность после restart/recreate требуют Docker. |
| 5 | Seed и прогрев ML обеспечивают демонстрацию без ожидания 200 точек в реальном времени. | PARTIAL | Существующий backfill создаёт 336 точек/до 335 векторов; маршрут связывает реальный source с архивным манифестом и demo-версией. Seed/warmup и фактическое обнаружение текущим стендом не выполнены. |
| 6 | Секреты не входят в образ и репозиторий; внешние уведомления в учебном режиме явно настраиваются. | PARTIAL | Расширены git/build исключения, добавлены регрессии Docker SDK и проверки образа; `.env` создаётся 0600 и ключи не печатаются. Новые SDK/образные проверки и инспекция реального образа здесь не выполнены. |
| 7 | Сохранён протокол запуска другим участником: нормальные данные → инцидент → обнаружение → дашборд/алерт. Ограничения учебной среды описаны. | NOT VERIFIED | Маршрут и форма результатов подготовлены. Ни полный текущий прогон, ни участие второго проверяющего не подтверждены. |

## Фактически выполнено

Окружение: Linux, Python 3.12.3, системные PyYAML и cryptography; проектные
зависимости не устанавливались. Docker CLI, pip, pytest, Ruff и Make отсутствуют.

| Команда / проверка | Полученный результат |
|---|---|
| `python3 -m unittest discover -s tests -p test_container_setup.py -v` | 15 тестов PASS: безопасное создание `.env`, отказ от перезаписи/symlink, ключ между процессами, независимость конфигураций, DSN/порты/volumes/profiles, routing рецептов Makefile. |
| Первый запуск этого набора | 14 PASS, 1 ERROR: `make` не найден. Проверка routing исправлена на статическое чтение рецептов, соответствующее назначению offline-набора; исполнение GNU Make не заявляется. Повторный запуск — 15 PASS. |
| `python3 -m pip --version`, `python3 -m pytest --version`, `python3 -m ruff --version` | Соответствующие модули отсутствуют. Это ограничения окружения, не PASS регрессии/линтера. |
| `command -v docker`, `command -v make` | Команды не найдены. Установка и изменение системных прав не выполнялись. |
| Граф включений requirements, `ast.parse`, ссылки новых документов, `git diff --check` | Результаты и количество файлов записаны в verification.json. Это статические проверки, не запуск приложения. |

Изменённые SDK-проверки контекста и интеграционный тест образа исполняются
существующим CI: `pytest -q tests/test_inference_packaging.py` и
`pytest -m integration -v tests/integration/test_inference_image.py`. Последний
теперь запрашивает `nocache=True`. Здесь они **не выполнены** из-за отсутствия
pytest, Docker SDK и Docker Engine. После появления окружения выполнить также
профиль seed/warmup/live-demo, миграции, регрессию и Ruff:

```bash
pytest -q tests/test_container_setup.py tests/test_inference_packaging.py \
  tests/test_demo_prepare.py tests/test_seed_target_db.py tests/test_seed_metrics_db.py \
  tests/test_warmup_ml.py tests/test_live_demo.py tests/test_health.py \
  tests/test_inference_service.py tests/test_inference_migration.py
pytest -m integration -v tests/integration/test_inference_image.py \
  tests/integration/test_inference_migration.py tests/integration/test_metrics_storage_timescale.py
ruff check .
pytest -m e2e -v
```

Тяжёлые наборы запускать последовательно. E2E требует Playwright/Chromium;
SKIPPED не считается успехом. Существующий Actions E2E-job исполняется только
при push в master.

## Историческое CI, без переноса результатов на новый стенд

[PR #39](https://github.com/aleksandr-novikov/ai-integration-development/pull/39),
HEAD `5d1f9116d26e9f21d9bb4be6f9073ba4af3ac258`:
[tests run 41504334279](https://github.com/aleksandr-novikov/ai-integration-development/actions/runs/41504334279)
имеет SUCCESS для Python 3.11/3.12/3.13 и integration, E2E — SKIPPED.
Integration включает реальную сборку/запуск образа и миграции TimescaleDB;
тогда build не требовал `nocache=True`. Это не подтверждает текущий чистый
Compose-прогон, перезапуск volumes или новый демонстрационный маршрут.
Статус нового PR нужно читать отдельно по его SHA.

## Форма независимого воспроизведения

На момент подготовки все поля ниже **не подтверждены**. Заполнить фактическими
результатами другого участника, не заменяя ожидание результатом:

| Шаг и команда из [маршрута](README.md) | Ожидается | Фактически / доказательство |
|---|---|---|
| Проверяющий, дата, `git rev-parse HEAD`, чистый status | Указан другой участник и полный SHA | Не подтверждено |
| `docker version`, `docker compose version`, список конфликтов, image ID/digest | Engine/Compose доступны; изолированные имена/порты | Не выполнено |
| `config --quiet`, `build --no-cache app`, import smoke | Код 0; зависимости установлены; OpenAPI в образе | Не выполнено |
| БД `up --wait`, два `pg_isready` | Обе БД готовы | Не выполнено |
| seed target, demo-prepare, stage-export | Данные/история/артефакт; demo остаётся явным | Не выполнено |
| `/v1/model`, `/healthz`, вход в Flask | Артефакт доступен, monitor backend PostgreSQL, сессия работает | Не выполнено |
| normal live-demo и SQL | Реальные нормальные метрики и фактический score | Не выполнено |
| incident live-demo и SQL | Инъекция 10× пакета; новая оценка выбранного source/версии | Не выполнено |
| `/dashboard/inference`, снимок экрана | Та же оценка/время/версия; demo виден | Не выполнено |
| restart БД, сравнение контрольных значений, recreate app и новый tick | Данные/артефакт сохранены, DSN расшифровывается | Не выполнено |
| `stop` | Остановлены только тестовые сервисы, volumes сохранены | Не выполнено |

Нужны реальные score/is_anomaly и наблюдения UI. Если baseline выдаёт ложную
аномалию в нормальном режиме или не обнаруживает инцидент, зафиксировать FAIL
этого шага. Синтетические seed-уведомления не подтверждают детектор/доставку.
Качество #9 (FPR 13,27%) не улучшалось и не допускается к production.
Итог по #13: **PARTIAL**, связь PR — `Refs #13` до дополнительных проверок.
