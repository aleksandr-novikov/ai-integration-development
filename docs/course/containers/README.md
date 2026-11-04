# Контейнерный учебный стенд (#13)

Существующий Compose запускает PostgreSQL-источник, TimescaleDB-хранилище и
Flask с scheduler. Профиль `inference` добавляет отдельный FastAPI на том же
образе с реестром только для чтения. Профили `iceberg`, `clickhouse`, `backup`
сохранены; этот маршрут их не включает. Их фиксированные имена/порты требуют
отдельной проверки изоляции.

Нужны Linux, Git, Python 3, Docker Engine и Compose v2 с `up --wait` и
`--wait-timeout`. Make необязателен. Python-зависимости устанавливаются внутри
образа; первая сборка с MLflow/Prophet может быть долгой. Доступ к registry/PyPI
нужен для сборки; Supabase, платный LLM и Telegram для демонстрации не нужны.
Фактические результаты и непроверенные шаги: [протокол](verification.md).

### Автоматическая проверка на GitHub runner

В `tests.yml` после integration выполняется отдельный Compose-этап на том же
Linux runner: он использует образ, уже собранный image-тестом с `nocache=True`.
Вторая сборка и перенос многогигабайтного образа между jobs не нужны.
Для события PR job проверяет объединённое дерево PR и master: checkout и
`COMPOSE_SMOKE_SHA` используют `github.sha`. Этот же SHA задаёт APP_VERSION
образа и проверяется в образе и `/healthz`; в отчёте он называется
`checkout_sha`. Исходный HEAD PR записывается отдельно как `pr_head_sha`,
не подменяет проверяемое дерево и может отличаться от него. На push
`checkout_sha` — фактический commit события. Checkout не сохраняет credentials.

`python -m scripts.verify_compose_demo run` создаёт новую конфигурацию и
изолированный проект; проверяет Compose config, БД, Flask/FastAPI, доступ
к артефакту, seed/warmup, реальные нормальные/инцидентные оценки, HTTP HTML/API
дашборда с обычным входом и CSRF, restart/recreate с полными хэшами тестовых
строк и артефакта. Стартовая инъекция и модель используют существующие скрипты;
выбор данных, порогов и версии ради успешного исхода не выполняется.

Отдельная команда `check-outcome` завершает job ошибкой при ложной тревоге,
необнаружении или отсутствии результатов, даже если инфраструктура исправна.
Отчёт различает `technical_pass` и `model_scenario.passed`. В нём сохраняются
source, timestamp, версии, четыре признака, score/is_anomaly, `demo=true`,
`quality_admitted=false`. Это не независимое воспроизведение человеком и не
проверка отрисовки графиков браузером; снимок UI остаётся ручным этапом ниже.

Job ограничен 30 минутами, контейнеры — CPU/RAM, Docker network — `internal`.
В такой сети опубликованные порты недоступны с runner, поэтому `/healthz`,
`/v1/model` и HTTP-проверки дашборда выполняются из одноразового контейнера
`app` внутри той же сети по именам сервисов `http://app:5001` и
`http://inference:8000`; версию из `/healthz` с `checkout_sha` сравнивает
host-процесс. Доступ к `127.0.0.1` с хоста CI не проверяет — это часть
ручного маршрута ниже.
NIM/Telegram не настроены. В `always()` удаляются только контейнеры и два
тестовых volume записанного проекта этого workflow-run. Такое удаление
разрешено только для одноразового runner, не для существующих данных.
Host-команды `run`/`cleanup` отказывают за пределами Linux GitHub Actions.
Публикуемый `ci-report.json` формируется по явной схеме `REPORT_SCHEMA`:
разрешены результаты проверок, SHA, ограниченные поля наблюдений/хэшей и
диагностика «этап, код завершения, категория ошибки». Неизвестные поля и
недопустимые значения отбрасываются. Тексты исключений, stdout/stderr,
аргументы команд, значения окружения, произвольные URL и журналы контейнеров
в артефакт не включаются; журналы для отчёта не запрашиваются.
Артефакт `compose-evidence-<checkout_sha>` хранится 14 дней; перед истечением
срока сохранить проверенный JSON как свидетельство.

## 1. Чистая копия и конфигурация

Использовать отдельный каталог и новый Compose project, а не рабочие данные:

```bash
git clone https://github.com/aleksandr-novikov/ai-integration-development.git course13-check
cd course13-check
git switch --detach "<проверяемый-SHA-из-PR>"
git rev-parse HEAD
git status --porcelain
# Вывод status должен быть пустым.

# В отдельной shell стенда убрать унаследованные Compose overrides.
unset COMPOSE_FILE COMPOSE_PROFILES COMPOSE_PROJECT_NAME PORT POSTGRES_PORT TIMESCALE_PORT
unset INFERENCE_PORT APP_IMAGE APP_VERSION APP_UID APP_GID BIND_ADDRESS FERNET_KEY INFERENCE_CONFIG
python3 -m scripts.init_compose_demo
```

Последняя команда создаёт `.env` с правами 0600, случайными SECRET_KEY,
FERNET_KEY и токеном инференса; значения не печатаются. Отдельное имя проекта
задаёт имена трёх основных контейнеров, сети, образа и volumes. UID/GID берутся
у непривилегированного владельца checkout: он сможет писать в `models/`.
Повторная подготовка **отказывается перезаписывать** `.env`, включая symlink.
Не удалять файл ради повторного запуска.

Порты по умолчанию: Flask 15001, PostgreSQL 15432, TimescaleDB 15433,
FastAPI 18000, только `127.0.0.1`. При конфликте до создания `.env` выбрать
другие: `python3 -m scripts.init_compose_demo --ports 25001 25432 25433 28000`.
Дальнейшие URL ниже предполагают стандартные порты. В существующей конфигурации
сохранить действующие ключи; FERNET_KEY из `.env.local` перенести в `.env`,
**не генерировать замену** для уже зашифрованных подключений.

| Настройка | Python на хосте (созданный `.env`) | Внутри Compose |
|---|---|---|
| `DATABASE_URL` | `localhost:15432/monitor` | `postgres:5432/monitor` |
| `MONITOR_DB_URL` | `localhost:15433/metrics` | `timescaledb:5432/metrics` |
| HTTP Flask | `127.0.0.1:15001` | `0.0.0.0:5001` |
| FastAPI для адаптера | опубликованный порт 18000 | `http://inference:8000` |

В исходном `.env.example` SQLite остаётся вариантом одиночного Python-запуска.
Генератор стенда выбирает TimescaleDB и для host-команд. Compose явно задаёт
оба внутренних DSN, постоянный Fernet-ключ и внутренний HTTP-порт 5001.
Не монтировать `monitor.db` в контейнер приложения.

## 2. Предварительные проверки, сборка и БД

```bash
docker version
docker compose version
docker ps -a --format '{{.Names}} {{.Ports}}'
python3 - <<'PY'
import socket
from contextlib import ExitStack
from pathlib import Path
env = dict(line.split('=', 1) for line in Path('.env').read_text().splitlines()
           if line and not line.startswith('#'))
with ExitStack() as stack:
    for name in ('PORT', 'POSTGRES_PORT', 'TIMESCALE_PORT', 'INFERENCE_PORT'):
        sock = stack.enter_context(socket.socket())
        sock.bind(('127.0.0.1', int(env[name])))
print('Четыре порта свободны; проверьте также имена контейнеров.')
PY
docker compose --profile inference config --quiet
docker compose build --no-cache app
docker compose up -d --wait --wait-timeout 120 postgres timescaledb
docker compose exec -T postgres pg_isready -U postgres -d monitor
docker compose exec -T timescaledb pg_isready -U postgres -d metrics
docker compose run --rm --no-deps app python -c \
  "import app.app, inference.service; from ml.inference_contract import validator; validator('PredictionRequest'); print('imports-and-contract-ok')"
```

При совпадении имени/порта с существующим сервисом остановиться и выбрать
свободную конфигурацию. Не останавливать чужие контейнеры. Проверка портов
предварительная: между ней и запуском порт может занять другой процесс.
`config --quiet` не раскрывает секреты; полный `config` не публиковать.
Не заменять `--no-cache` одним только синтаксическим анализом.

## 3. Seed, прогрев и явный выбор demo-версии

Следующие команды предназначены **только для новых изолированных данных**.
`demo_prepare` очищает часть таблиц истории и создаёт синтетические события/
уведомления. Такие строки не являются доказательством реального обнаружения
или доставки Telegram. Не запускать его повторно после записи доказательств.

```bash
docker compose run --rm --no-deps app python -m scripts.seed_target_db \
  --users 500 --products 50 --orders 1000 --events 2000
docker compose run --rm --no-deps app python -m scripts.demo_prepare --slug retail-postgres
# Эквивалент второй команды: make demo-prepare ARGS="--slug retail-postgres"
```

Ожидаются 336 часовых точек за 14 дней, до 335 согласованных векторов на таблицу
и результаты warmup без ожидания 200 реальных тиков. Сохранить summary; при
нулевых счётчиках или ошибках не считать прогрев успешным. Сам warmup обучает
унаследованные IF/прогнозы; следующий шаг отдельно выбирает **архивную #9
модель**, а не приписывает ей результаты этого обучения.

Подготовить source для реального demo-подключения и отключить подключения к
не поднятым бонусным БД. Команда проверяет DSN без его печати:

```bash
docker compose run --rm -T --no-deps app python - <<'PY'
import json
from pathlib import Path
from app import crypto, metrics_storage as ms
from app.config import settings
user = ms.get_user_by_email('demo@dbmonitor.app')
project = ms.get_project_by_slug(user['id'], 'retail-postgres')
connection = next(c for c in ms.list_connections_for_project(project['id'])
                  if c['name'] == 'Local Postgres')
assert crypto.decrypt_dsn(connection['dsn_encrypted']) == settings.DATABASE_URL
for email in ('demo@dbmonitor.app', 'lake@dbmonitor.app'):
    for p in ms.list_projects_for_user(ms.get_user_by_email(email)['id']):
        if p['id'] != project['id']:
            for c in ms.list_connections_for_project(p['id']):
                ms.set_connection_active(p['id'], c['id'], False)
source = dict(project_id=project['id'], connection_id=connection['id'],
              schema_name=connection['schema_name'], table_name='events')
directory = Path('models/demo')
directory.mkdir(parents=True, exist_ok=True)
(directory / 'source.json').write_text(json.dumps(source))
print(json.dumps(source))
PY
docker compose run --rm --no-deps \
  -v "$PWD/docs/course/experiments/mlflow-export.zip:/tmp/mlflow-export.zip:ro" \
  app python -m scripts.model_versions --registry models/registry stage-export \
  --source models/demo/source.json --archive /tmp/mlflow-export.zip > models/demo/manifest.json
```

Это явный импорт неизменяемого доверенного экспорта, без обучения, активации
или допуска новой модели. Архив и реестр не входят в образ. Исходный SHA-256
архива проверяет существующий реестр. Модель имеет FPR 13,27% при цели ≤10%
и допустима только с `demo=true`. Не менять порог или score ради успеха показа.

Создать согласованные конфигурации из фактического манифеста:

```bash
python3 - <<'PY'
import json
from pathlib import Path
directory = Path('models/demo')
source = json.loads((directory / 'source.json').read_text())
manifest = json.loads((directory / 'manifest.json').read_text())
assert manifest['source'] == source
binding = dict(source=source, model_version=manifest['model_version'],
               feature_schema_version=manifest['feature_schema_version'])
service = dict(registry_path='models/registry', demo=True,
               principals=[dict(token_env='INFERENCE_ADAPTER_TOKEN', sources=[source])])
adapter = dict(url='http://inference:8000', token_env='INFERENCE_ADAPTER_TOKEN',
               demo=True, bindings=[binding])
for name, value in (('service', service), ('adapter', adapter)):
    (directory / (name + '.json')).write_text(json.dumps(value, indent=2))
with Path('.env').open('a') as stream:
    stream.write('\nINFERENCE_ADAPTER_CONFIG=/app/models/demo/adapter.json\n')
print('Selected demo version:', manifest['model_version'])
PY
docker compose --profile inference up -d --no-build --wait --wait-timeout 120 inference
docker compose run --rm -T --no-deps app python - <<'PY'
import json, os
from pathlib import Path
import httpx
binding = json.loads(Path('models/demo/adapter.json').read_text())['bindings'][0]
params = {**binding['source'], **{k: binding[k] for k in ('model_version', 'feature_schema_version')}}
r = httpx.get('http://inference:8000/v1/model', params=params,
              headers={'Authorization': 'Bearer ' + os.environ['INFERENCE_ADAPTER_TOKEN']})
assert r.status_code == 200, r.status_code
print(r.json())
PY
docker compose up -d --no-build --wait --wait-timeout 120 app
curl --fail http://127.0.0.1:15001/healthz
```

У Flask должны быть `target_db=ok`, `monitor_db=ok`, backend `postgresql`.
FastAPI `/healthz` подтверждает процесс; авторизованный `/v1/model` выше
дополнительно подтверждает загрузку выбранного артефакта. Отсутствующий токен
даёт 401, отсутствующая модель — 503; их нельзя считать готовностью.
Первый `/v1/model` явно прогревает lazy-cache локального артефакта до запуска
приложения; обучения или скачивания модели при запросе нет. Для выбранного
bound-source HTTP-запросы не вызывают fit. Legacy-маршруты сохраняют прежнее
поведение и не являются проверкой жизненного цикла нового сервиса.
Старые результаты warmup остаются с `legacy-unknown`; они не становятся
версионными оценками #9. Подробнее: [инференс](../inference/README.md).

## 4. Нормальные данные → инцидент → обнаружение → дашборд

Получить ID из напечатанного source (либо `make demo-ids`) и задать переменные:

```bash
PROJECT_ID="<project_id-из-source.json>"
CONNECTION_ID="<connection_id-из-source.json>"
mkdir -p artifacts/compose-demo
set -o pipefail
docker compose run --rm --no-deps app python -m scripts.live_demo \
  --project-id "$PROJECT_ID" --connection-id "$CONNECTION_ID" \
  --ticks 3 --interval 5 2>&1 | tee artifacts/compose-demo/normal.log
docker compose exec -T postgres psql -U postgres -d monitor -c \
  'SELECT count(*) AS rows, count(*) FILTER (WHERE ip_address IS NULL) AS null_ips FROM events;'
docker compose exec -T timescaledb psql -U postgres -d metrics -c \
  "SELECT ts,score,is_anomaly,model_version,feature_schema_version,is_demo AS demo FROM anomaly_scores WHERE project_id='$PROJECT_ID' AND connection_id='$CONNECTION_ID' AND table_name='events' ORDER BY ts DESC LIMIT 5;"
docker compose run --rm --no-deps app python -m scripts.live_demo \
  --project-id "$PROJECT_ID" --connection-id "$CONNECTION_ID" \
  --ticks 3 --interval 5 --incident-at 2 2>&1 | tee artifacts/compose-demo/incident.log
docker compose exec -T postgres psql -U postgres -d monitor -c \
  'SELECT count(*) AS rows, count(*) FILTER (WHERE ip_address IS NULL) AS null_ips FROM events;'
docker compose exec -T timescaledb psql -U postgres -d metrics -c \
  "SELECT ts,score,is_anomaly,model_version,feature_schema_version,is_demo AS demo FROM anomaly_scores WHERE project_id='$PROJECT_ID' AND connection_id='$CONNECTION_ID' AND table_name='events' ORDER BY ts DESC LIMIT 10;"
```

`pipefail` не позволяет `tee` скрывать ошибки команды.
Нормальный тик добавляет 200 строк с ожидаемыми ~2% NULL IP в новом пакете;
инцидентный — 2000 строк с ~40% NULL и смещением server_id. Значения случайные,
общая доля NULL включает старые данные. `★ INCIDENT` подтверждает инъекцию,
**не обнаружение**. Детектор подтверждается новой оценкой того же source/
версии/времени с `is_anomaly=1`, полученной от FastAPI, а не seed-уведомлением.
Нормальный режим проверять также по фактическим scores: возможны ложные
срабатывания. Если инцидент не обнаружен или нормальные точки аномальны,
сохранить результат как ограничение/неуспех сценария; не подменять значения.

Войти на `http://127.0.0.1:15001/auth/login` как `demo@dbmonitor.app` с учебным
паролем `demo12345`, выбрать Retail Postgres. На `/dashboard/inference`
проверить source, пару версий, отметку demo и новую оценку. На обзоре/странице
events сопоставить время и изменение метрик; сохранить снимок экрана.
Не публиковать такой учебный аккаунт в интернете. CSRF и авторизация сохранены.
Telegram по умолчанию не отправляется; дашборд достаточен для этого маршрута.
LLM при пустом ключе использует правила, без платного запроса. Настройка
внешних уведомлений — явно в настройках проекта; seed-строки `sent` не доказывают доставку.

## 5. Перезапуск, сохранность и завершение

Перед фиксацией контрольных сумм остановить **только приложение этого проекта**,
чтобы scheduler не менял данные. One-off команды live-demo уже завершены.

```bash
docker compose stop app
docker compose exec -T postgres psql -At -U postgres -d monitor -c \
  'SELECT count(*),min(id::text),max(id::text) FROM events;' > artifacts/compose-demo/target.before
docker compose exec -T timescaledb psql -At -U postgres -d metrics -c \
  'SELECT count(*),min(ts),max(ts) FROM metrics; SELECT count(*) FROM anomaly_scores;' > artifacts/compose-demo/metrics.before
sha256sum models/registry/versions/*/pipeline.skops > artifacts/compose-demo/models.before
docker compose restart postgres timescaledb
docker compose up -d --wait --wait-timeout 120 postgres timescaledb
docker compose exec -T postgres psql -At -U postgres -d monitor -c \
  'SELECT count(*),min(id::text),max(id::text) FROM events;' > artifacts/compose-demo/target.after
docker compose exec -T timescaledb psql -At -U postgres -d metrics -c \
  'SELECT count(*),min(ts),max(ts) FROM metrics; SELECT count(*) FROM anomaly_scores;' > artifacts/compose-demo/metrics.after
diff -u artifacts/compose-demo/target.before artifacts/compose-demo/target.after
diff -u artifacts/compose-demo/metrics.before artifacts/compose-demo/metrics.after
sha256sum -c artifacts/compose-demo/models.before
docker compose up -d --force-recreate --no-build --wait --wait-timeout 120 app
curl --fail http://127.0.0.1:15001/healthz
```

Ожидаются одинаковые контрольные значения, доступный артефакт и успешное
расшифрование подключения после recreate. Повторить авторизованный `/v1/model`
и короткий нормальный live-demo после `up`, проверить успешный collector run
и новую версионную оценку: один healthcheck не проверяет шифрование DSN.
Повторный `up --no-build --wait app` должен использовать те же volumes и ключи.
Seed повторно не запускать: он добавляет строки, а demo-prepare сбрасывает историю.

После этих проверок штатно завершить работу:

```bash
docker compose --profile inference stop app inference postgres timescaledb
```

Volumes и `models/` сохраняются. `down -v`,
`make db-reset`, `make reset-db`, `make reset-metrics`, `make seed` и volume prune
не входят в маршрут: они могут удалить данные. Не удалять `.env` и реестр.
Перед обновлением существующего хранилища читать [миграции](../inference/migrations.md).

Диагностика: `docker compose ps -a`, `docker compose logs --tail 100 app`,
`docker compose --profile inference logs --tail 100 inference`, `pg_isready`.
При 401 проверить наличие токена без его печати; при 503 модели — config,
манифест, полный source и доступ к реестру; при ошибке расшифрования — прежний
FERNET_KEY. Не публиковать `.env`, полный Compose config, DSN или сырые логи
без проверки на секреты. Образы `python:3.12-slim`, `postgres:18` и TimescaleDB
с mutable tags не гарантируют побайтово одинаковую будущую сборку; фиксировать
фактические image ID/digest и версии из своего запуска.

Другой участник заполняет [таблицу протокола](verification.md) своим SHA,
командами, результатами и ссылками на обезличенные доказательства.
