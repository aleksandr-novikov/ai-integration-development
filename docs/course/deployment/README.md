# В10: стратегия деплоя и отката (#14)

ДЗ: [стратегия](STRATEGY.md), [версия PDF на две страницы](STRATEGY.pdf).
Основание: [презентация В10](../netology/presentations/В10_Стратегии_деплоя.pdf), слайд 45
(схема, описание стратегии, критериев успешности, порогов и плана отката; 2 страницы) и
практика — слайды 33–38.

Документ — проектное решение. Он не меняет код, workflows, Compose и модели; деплой,
переключение трафика и откат не выполнялись. Технические утверждения сверены с `master`
`f891234` (10.11.2026) по исходникам и документации:
[release.yml](../../../.github/workflows/release.yml), [promote.yml](../../../.github/workflows/promote.yml),
[tests.yml](../../../.github/workflows/tests.yml), [runbook отката](../../runbooks/rollback.md),
[docker-compose.yml](../../../docker-compose.yml), [app/health.py](../../../app/health.py),
[журнал сборщика](../../../app/metrics_storage.py) (`collector_runs`), [флаги](../../../app/feature_flags.py),
[подготовка `.env` стенда](../../../scripts/init_compose_demo.py),
[реестр моделей](../models/README.md), [инференс и привязки](../inference/README.md),
[миграции](../inference/migrations.md), мониторинг [сервиса](../monitoring/README.md) и
[модели](../monitoring/model/README.md), [MLOps-roadmap](../MLOPS_ROADMAP.md).

## Сборка PDF

PDF строится из `STRATEGY.md`; разрыв страницы задан комментарием `<!-- pagebreak -->`,
ширины колонок таблиц — комментариями `<!-- widths: … -->`. Нужны ReportLab и Unicode
TTF-шрифты (пути передаются аргументами):

```bash
python docs/course/deployment/render_pdf.py \
  --font /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf \
  --bold-font /usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf \
  --mono-font /usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf
```

Проверено 10.11.2026: ReportLab 4.2.5 во временном venv вне репозитория, DejaVu Sans 2.37.
Результат — A4, **2 страницы** (подсчёт pypdf и PyMuPDF), обе страницы отрисованы и
просмотрены: текст и таблицы не обрезаны, схема помещается по ширине. Повторная сборка
даёт побайтово тот же файл (`rl_config.invariant`). Если после правки `STRATEGY.md`
страниц станет больше двух, текст нужно сократить, а не уменьшать шрифт.
