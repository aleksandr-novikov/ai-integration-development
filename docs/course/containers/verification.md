# Проверка контейнеризации #13

Дата обследования: 03.11.2026. База разработки — актуальный `master`
`46712fb54b77bc45e9f89693214e5a467d264270`. Issue #13 открыта, комментариев нет;
зависимости #7 и #12 закрыты. На начало разработки открытого PR по #13 нет.
Точная редакция файлов локальных проверок фиксируется SHA-256 в
[verification.json](verification.json); SHA опубликованного коммита находится
в PR. Дополнительно проверена отдельная чистая локальная копия коммита
`fb795a38003c5acd3dbc457b6d6c3e5521744c06`: 15 offline-тестов PASS, статические
проверки PASS, создание `.env` с версией этого коммита PASS, повторная
инициализация отказала без изменения файла. Начальный и конечный git status
чистые; `.env` проигнорирован. Это не подтверждение полного Compose-запуска.
Следующий коммит дополняет только этот протокол и JSON-доказательства.

## Текущий статус: доработка 04.11.2026 (demo-модель)

Исходная точка — HEAD PR `104d41bef6d684dc05d5b4711bdc27c8d90c5c75`,
[run 41845706680](https://github.com/aleksandr-novikov/ai-integration-development/actions/runs/41845706680):
Ruff, Python 3.11/3.12/3.13 и обычные integration — SUCCESS; Compose
`technical_pass=true`, 31 шаг и cleanup PASS; job упал только на
`check-outcome`: архивная #9 дала score −0.077182 и `is_anomaly=true` всем трём
нормальным тикам и инциденту. Причины и решение — в
[протоколе demo-модели](demo-model/PROTOCOL.md), метрики — в
[результатах](demo-model/RESULTS.md). Успех прежних SHA на новый код не
переносится: статусы ниже, помеченные CI, относятся к прогону, указанному в PR.

| № | Критерий | Реализация | Способ проверки | Доказательство | Статус |
|---|---|---|---|---|---|
| 1 | Dockerfile и docker-compose.yml | Прежний Dockerfile; Compose с изоляцией, профилем `inference`, readonly-реестром | Compose CI-сценарий на образе `nocache=True` | `technical_pass=true` на `51c0da6`, `104d41b`; для нового HEAD — run из PR | PASS на прежних SHA; новый HEAD — по CI PR |
| 2 | Сборка проверена при наличии Docker | `tests/integration/test_inference_image.py` с `nocache=True` | integration job | 35 integration PASS на `ddb29e1`; SUCCESS на `104d41b` | PASS на прежних SHA; новый HEAD — по CI PR |
| 3 | Чистый checkout по инструкции, `.env.example` без Supabase/LLM | `init_compose_demo` строит `.env` из `.env.example`; NIM/Telegram пусты; БД локальные | CI из чистого checkout (`git status` пуст, `.env` отсутствует); локальный clone HEAD + init без Docker | CI `technical_pass`, integration SUCCESS на `7b068b7`; локальная проверка — ниже | PASS по автоматической проверке (см. «Уточнение статусов №3 и №6»); ручной браузерный прогон по маршруту не выполнен |
| 4 | Согласованные настройки, readiness, volumes | Раздельные DSN, постоянный FERNET_KEY, healthchecks, restart/recreate | Compose CI: хэши строк и артефакта до/после | PASS на `51c0da6`, `104d41b`; новый HEAD — по CI PR | PASS на прежних SHA |
| 5 | Seed и прогрев ML; демонстрация | seed + warmup + отдельная demo-модель `contamination_005`; реальный FastAPI-инференс | Независимый test (480 эпизодов); CI `check-outcome` (норма без тревог, инцидент найден, цели test внутри образа, все четыре сравнения `demo_model_reproduced`) | test: FPR 1.66%, recall 1.0, Compose-сценарий 190/200; CI — по PR | Офлайн PASS; CI нового HEAD — по PR |
| 6 | Секреты не в образе/репозитории; уведомления учебного режима | `.dockerignore`, `.env` 0600, пустые внешние ключи; Telegram только из настроек проекта | Проверка образа в CI; поиск шаблонов секретов в diff; `container_state` требует пустые NIM/Telegram | CI-проверки образа PASS, integration SUCCESS на `7b068b7`; поиск по diff и отслеживаемым файлам — ниже | PASS в установленном объёме (см. «Уточнение статусов №3 и №6»); фактическая доставка Telegram не проверялась и критерием не требуется |
| 7 | Протокол запуска другим участником | [Маршрут](README.md), краткое задание и форма ниже | Независимый прогон человеком, не участвовавшим в разработке | Нет. Ручное тестирование выполнял участник разработки; оно не считается независимым прогоном другого человека | NOT VERIFIED |

CI на HEAD `53b7b825767bfe8d73f6de933a90c612f781509a`
([run 41858632502](https://github.com/aleksandr-novikov/ai-integration-development/actions/runs/41858632502),
обезличенный отчёт получен пользователем): Ruff и Python — SUCCESS; Compose —
подготовка, обучение demo-модели, readiness, normal, incident, dashboard,
restart/recreate БД с совпадением хэшей и пересоздание app/inference — PASS;
`demo_model_reproduced` — все четыре сравнения true; `model_scenario.passed=true`
(3 нормальных тика: score +0.0749, +0.0372, +0.1551, без тревог; инцидент
−0.0472, обнаружен). `technical_pass=false`: отказ только на `after-recreate`
(`command_failed`, код 1), его наблюдения не сохранены, `check-outcome` не
выполнялся. Локальный повтор той же последовательности на PostgreSQL 16 с
настоящим перезапуском БД и нового процесса FastAPI отказ не воспроизвёл
(оценка инцидента совпала с CI: −0.04718365). Причина не установлена; добавлена
диагностика `observe_diagnostics` фиксированными категориями.

Учебный «алерт» — отметка «Да» в колонке аномалии на `/dashboard/inference`
для новой версионной оценки. Без Telegram-конфигурации проекта
`_notify_prediction` ничего не отправляет; seed-уведомления demo-prepare —
синтетические строки, не доказательство доставки.

### Локальные проверки доработки (VDS2, без Docker)

Окружение: Linux, Python 3.12.3; зависимости проекта установлены во временный
venv вне репозитория из `requirements.txt`/`requirements-dev.txt` (PyPI), без
sudo. Docker Engine отсутствует, поэтому Compose здесь не запускался.

| Проверка | Результат |
|---|---|
| `pytest -q` до изменений, HEAD `104d41b` | 1574 PASS, 6 FAIL: 4 — нет CLI `sqlite3`, 2 — нет `python` в PATH (окружение) |
| `pytest -q` после изменений, venv в PATH | **1594 PASS**, 4 FAIL — те же тесты `test_backup_restore.py` без CLI `sqlite3`; 54 deselected (integration/e2e) |
| `tests/test_demo_model.py` | 10 PASS, включая совпадение симулятора с реальными INSERT сидера/live-demo и `MetricsCollector` на SQLite |
| `tests/test_compose_verification.py`, `tests/test_model_registry.py` | 53 и 49 PASS: этап `demo_model`, схема отчёта, `check-outcome` с целями test, `histories`/`provenance`, неизменный `dataset_id` одной истории |
| `ruff check .`, `git diff --check`, относительные ссылки документов | PASS |
| Build context текущего дерева, Docker SDK `exclude_paths` | 151 файл; `scripts/demo_model.py` и контракт OpenAPI включены; env/ключи/модели/архивы/тесты/документы demo-модели исключены |
| Поиск шаблонов токенов, ключей, DSN с паролем в diff | Совпадений нет (учебные `postgres:dev@localhost`, `demo12345` и тестовые фикстуры известны и допустимы) |
| Чистый clone `af8bfd0`, `python3 -m scripts.init_compose_demo` | `.env` 0600 из `.env.example`; NIM/Telegram пусты; DSN только `localhost`; Supabase нет; `git status` пуст до и после; повторный вызов — код 2 без перезаписи |
| Тот же clone, `scripts.demo_model train` | SHA train/test, SHA score и метрики test совпали с [evaluation.json](demo-model/evaluation.json); `evaluation` манифеста пуст |

Это не заменяет Compose CI нового HEAD, браузерную проверку и независимый прогон.

### Доработка 06.11.2026: воспроизводимость score demo-модели

Исходная точка — HEAD PR `1076d63b7f6dce17835c3e24746a80df9782f5e6`.
Повтор обучения дал SHA score `210f256d…7324` вместо исторического
`0667678…8346`. При этом данные и метрики совпали. Отчёт записывал
`demo_model_reproduced.scores_sha256=false`, но `check-outcome` это значение не
проверял. Сравнение метрик `test` тоже не проверялось.

- `check-outcome` теперь требует все четыре сравнения `true`. Отсутствие
  сравнения считается ошибкой.
- SHA score сверяется с эталоном своего окружения из
  [scores-reference.json](demo-model/scores-reference.json): python 3.12,
  scikit-learn, numpy, skops и уровень NumPy SIMD (`environment.numpy_simd`).
  Если для окружения эталона нет, проверка не проходит.
- Причина расхождения установлена экспериментально: score зависят от уровня
  SIMD-диспетчеризации NumPy, модель от него не зависит. Подробности — в
  [RESULTS.md](demo-model/RESULTS.md#воспроизводимость).
- Исторический SHA сохранён как эталон `X86_V4`. Для `X86_V3`/`X86_V2`
  добавлен эталон: на каждом уровне выполнено по два независимых обучения.
  Хост без AVX-512 физически не проверялся.
- Алгоритм, конфигурация модели, порог и данные не изменялись. Workflow не
  изменялся.

### Уточнение статусов №3 и №6 (06.11.2026)

Проверяемый HEAD — `7b068b70a7901b9b40ee7f1624de454802b01589`,
[run 42214390628](https://github.com/aleksandr-novikov/ai-integration-development/actions/runs/42214390628):
Python 3.11/3.12/3.13 и integration — SUCCESS, E2E — SKIPPED (не PASS);
[lint](https://github.com/aleksandr-novikov/ai-integration-development/actions/runs/42214390586)
— SUCCESS. Compose проверяется на дереве `github.sha` (merge-ref PR); база PR —
`46712fb`. Шаги `run` и `check-outcome` выполняются без условий. Любая
проверка `run` и `technical_pass=false` завершают job ошибкой. Поэтому
SUCCESS означает, что все перечисленные ниже утверждения выполнены.
Артефакт `ci-report.json` этого run здесь не читался.
Код, тесты, модель и workflow при уточнении не менялись.

**№3 — PASS по автоматической проверке.** Что подтверждено CI:

- перед запуском `.env` отсутствует, `git status --porcelain` пуст, HEAD
  совпадает с проверяемым SHA;
- `.env` создаётся только из `.env.example` через `create_environment` —
  ту же функцию, что вызывает `python3 -m scripts.init_compose_demo` из
  [маршрута](README.md);
- стенд работает в сети Compose с `internal: true`, поэтому внешние сервисы
  недоступны. `container_state` требует, чтобы `DATABASE_URL` и
  `MONITOR_DB_URL` указывали на сервисы `postgres` и `timescaledb`, а
  `NIM_API_KEY`, `TELEGRAM_BOT_TOKEN` и `TELEGRAM_CHAT_ID` были пустыми.
  Личная Supabase и платный LLM не нужны;
- те же модули, что в маршруте (`seed_target_db`, `demo_prepare`,
  `demo_model`), подготавливают данные внутри образа. Затем проходят
  healthchecks Flask и FastAPI, readiness, нормальный тик без тревоги и
  обнаруженный инцидент.

Граница доказательства: CI — автоматический эквивалент маршрута, а не
буквальное выполнение команд README. Образ собирает integration-тест с
`nocache=True`, а не `docker compose build`; подготовка вызывает модули одним
скриптом. **Не выполнено:** ручной прогон маршрута человеком и браузерная
проверка входа и `/dashboard/inference`.

**№6 — PASS в установленном объёме.** Что подтверждено:

- образ (CI): в `docker image inspect` нет значений `SECRET_KEY`,
  `FERNET_KEY` и `INFERENCE_ADAPTER_TOKEN` этого запуска. В `/app` нет
  `.env`, `.env.local`, `backups`, `models/registry` и
  `docs/course/experiments`;
- build context (test jobs): `tests/test_inference_packaging.py` проверяет
  исключение приватных файлов через Docker SDK `exclude_paths`;
- репозиторий: `.gitignore` и `.dockerignore` исключают `.env*`, кроме
  `.env.example`. Из env-, key- и db-подобных файлов отслеживается только
  `.env.example`. Поиск шаблонов токенов и приватных ключей по diff PR ничего
  не нашёл. По всем отслеживаемым файлам он нашёл только заведомо фиктивные
  значения (плейсхолдер в `app/settings.py`, тестовые фикстуры маскирования),
  которые этот PR не меняет;
- уведомления: в `.env.example` пустые `NIM_API_KEY` и `TELEGRAM_*`,
  `FF_LLM_NOTIFICATIONS=0`. Регрессия
  `test_no_paid_or_external_notifications_by_default` проверяет это
  значение по умолчанию. Включение возможно только явной настройкой проекта.

**Не проверялось и критерием не требуется:** фактическая доставка в Telegram и
вызов NIM с реальным ключом. Слои образа целиком не сканировались: проверка
ограничена метаданными образа и перечисленными путями.

**№7 остаётся NOT VERIFIED.** Автоматические проверки не заменяют
независимый прогон другим участником.

## Продолжение: Docker-проверка PR #41

На HEAD `ddb29e15941be52301b47646700406ef6f2487fd` получены окончательные
результаты [run 41569760737](https://github.com/aleksandr-novikov/ai-integration-development/actions/runs/41569760737):
Python 3.11/3.12/3.13 — по **1524 PASS**, 54 deselected; integration —
**35 PASS**, 1543 deselected; Ruff — SUCCESS; E2E — SKIPPED.
Журнал image-теста подтверждает реальную сборку `nocache=True`, запуск
контейнера без сети, наличие контракта, отсутствие локальных файлов и проверки
ASGI-маршрутов. Полный Compose этим запуском не проверялся.

Для нового коммита добавлен Compose-этап в существующий integration job,
переиспользующий тот же image ID. Docker на машине разработки не устанавливался.
В исходном коммите `9c7fc98` проверены 15 прежних и 6 новых offline-тестов;
после исправления замечаний P1 — 15 прежних и 12 тестов CI-проверки. Новый реальный результат
нужно читать в artifact `compose-evidence-<SHA>` соответствующего run, не
переносить на него результаты предыдущего HEAD. Матрица ниже сохраняет
предыдущий срез до получения Compose-доказательств; итоговый статус отражается
в описании PR со ссылкой на конкретный run и SHA.

Исправления P1 сохраняют checkout объединённого дерева (`github.sha`) и
`persist-credentials: false`. `COMPOSE_SMOKE_SHA`, APP_VERSION и проверка
версии образа/приложения соответствуют `checkout_sha`; исходный HEAD PR
сохраняется только в `pr_head_sha`. Отчёт теперь записывается через явную
схему разрешённых полей и значений. Произвольные исключения, аргументы,
вывод команд и контейнерные журналы не сохраняются. Отрицательные тесты
проверяют отсутствие тестовых секретов и сохранность ограничений очистки.
Эти изменения проверены локально; новый Docker/Compose-прогон не выполнен.

### Краткое задание другому участнику

1. Для повторения CI взять `checkout_sha` из отчёта проверяемого run; HEAD PR
   сверять отдельно с `pr_head_sha`. В новой копии получить merge-ref командой
   `git fetch origin refs/pull/41/merge`, затем выполнить
   `git switch --detach "<checkout_sha>"` и сверить `git rev-parse HEAD`.
   Если нужный объект недоступен или merge-ref уже изменился, не подменять SHA
   новым деревом: зафиксировать отсутствие точного воспроизведения. Сохранить
   чистый `git status --porcelain` и версии Docker/Compose.
2. Пройти [маршрут](README.md) на разрешённом изолированном Docker-стенде;
   не использовать CI-режим с подставленной runner-идентичностью.
3. Сохранить нормальные и инцидентные признаки/оценки, source, время, версии,
   `demo=true`/`quality_admitted=false`, проверку входа и снимок реального UI.
   Ожидание — нормальный пакет без тревоги и обнаруженный инцидент. При
   несовпадении записать FAIL, не менять модель/данные/порог ради результата.
4. Сопоставить БД и артефакт до/после restart/recreate, проверить дешифрование
   подключения и новый tick. Завершить через `stop` без удаления volumes.
5. Заполнить форму в конце документа: участник/дата/SHA, точные команды,
   фактические результаты, ссылки на обезличенные JSON/снимки и ограничения.
   До этого критерий №7 остаётся неподтверждённым независимо от CI.

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

## Матрица критериев: исторический срез 03.11.2026

Заменена таблицей «Текущий статус» выше; сохранена как история.


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
| Чистая локальная копия `fb795a38003c5acd3dbc457b6d6c3e5521744c06`, тот же unittest и `python3 -m scripts.init_compose_demo` | 15 PASS; исходные отпечатки совпали; `.env` создан 0600 без прежней настройки; повторный вызов завершился кодом 2 и сохранил файл. Полный Docker-запуск не выполнялся. |
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
| Проверяющий, дата, `git rev-parse HEAD`, чистый status | Указан другой участник, не участвовавший в разработке, и полный SHA | Не подтверждено |
| `docker version`, `docker compose version`, список конфликтов, image ID/digest | Engine/Compose доступны; изолированные имена/порты | Не выполнено |
| `config --quiet`, `build --no-cache app`, import smoke | Код 0; зависимости установлены; OpenAPI в образе | Не выполнено |
| БД `up --wait`, два `pg_isready` | Обе БД готовы | Не выполнено |
| seed target, demo-prepare, `scripts.demo_model train` | Данные/история/артефакт; SHA данных и метрики test совпадают с RESULTS.md; SHA score совпадает с эталоном своего `numpy_simd`; `targets_passed=True`; demo остаётся явным | Не выполнено |
| `/v1/model`, `/healthz`, вход в Flask | Артефакт доступен, monitor backend PostgreSQL, сессия работает | Не выполнено |
| базовый сбор, normal live-demo и SQL | Реальные нормальные метрики и score, `is_anomaly=0` (≈95% показов) | Не выполнено |
| incident live-demo и SQL | Инъекция 10× пакета; новая оценка выбранного source/версии, `is_anomaly=1`; тик после инцидента записать отдельно | Не выполнено |
| `/dashboard/inference`, снимок экрана | Та же оценка/время/версия; demo виден | Не выполнено |
| restart БД, сравнение контрольных значений, recreate app и новый tick | Данные/артефакт сохранены, DSN расшифровывается | Не выполнено |
| `stop` | Остановлены только тестовые сервисы, volumes сохранены | Не выполнено |

Нужны реальные score/is_anomaly и наблюдения UI. Если demo-модель выдаёт ложную
аномалию в нормальном режиме или не обнаруживает инцидент, зафиксировать FAIL
этого шага. Синтетические seed-уведомления не подтверждают детектор/доставку.
Архивная #9 (FPR 13,27%) не изменялась; demo-модель допускается только с
`demo=true` и не предназначена для production.
Итог по #13: **PARTIAL**. Критерии №1–6 подтверждены автоматическими
проверками; №7 — NOT VERIFIED до независимого прогона. Ручная браузерная
проверка не выполнена. Связь PR — `Refs #13`.
