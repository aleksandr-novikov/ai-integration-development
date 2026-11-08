# План итогового проекта

Предмет: **Интеграция искусственного интеллекта в разработку**.
Кейс: обнаружение аномалий в метриках мониторинга данных DB Monitoring с помощью Isolation Forest. Описание кейса, критерии успеха и границы «унаследовано / выполнено в рамках курса» — в [CASE.md](CASE.md).

[Канбан-доска](https://github.com/users/aleksandr-novikov/projects/2/views/2) · [Все задачи](https://github.com/aleksandr-novikov/ai-integration-development/issues) · [Этапы](https://github.com/aleksandr-novikov/ai-integration-development/milestones) · [Описание кейса](CASE.md)

## Основа и граница работ

Проект основан на [https://github.com/aleksandr-novikov/db-monitoring](https://github.com/aleksandr-novikov/db-monitoring), коммит [`1a869e4`](https://github.com/aleksandr-novikov/ai-integration-development/commit/1a869e4bb24c2b0e238249e04ca46f97534a2a1f) от 11 июня 2026.
Сохранена история основной ветки. Это отдельная копия в аккаунте владельца исходного проекта: GitHub не поддерживает форк собственного репозитория в тот же личный аккаунт.

В исходном проекте уже есть Flask API, сбор метрик, Isolation Forest, Prophet/линейный прогноз, PSI/KS, change-point и schema drift, LLM-объяснения, Docker, тесты, CI, Prometheus и уведомления. Их наличие не означает, что задания курса автоматически выполнены: каждая карточка требует проверяемый учебный результат.

Основная новая работа: воспроизводимая оценка ML, пять экспериментов в MLflow, версия данных/модели, валидация, актуальные контракты и документация, доказательства работы деплоя/мониторинга, журнал ИИ-ассистентов, отчёт и защита.

## Основание плана

Проверены презентации В01–В12 и два файла итогового задания. Уточнение от 21.10.2026: оригиналы В09–В12 сохранены локально в [netology/presentations](netology/presentations), условия и источники выписаны в [netology/ASSIGNMENTS.md](netology/ASSIGNMENTS.md). Оригиналы и условия опубликованы вместе с обновлением плана от 21.10.2026. В13 сверен по презентации 25.10.2026; В14 и В15 — 08.11.2026 по тексту слайдов, предоставленному участником команды (оригиналы не сохранялись). Подробности — в [ASSIGNMENTS.md](netology/ASSIGNMENTS.md).

| Вебинар | Результат по доступным материалам |
|---|---|
| В01, слайд 62 | Одна страница: 7 этапов жизненного цикла и схема |
| В02, слайд 51 | Архитектурная схема, компоненты, потоки и стек; около 2 страниц |
| В03, слайд 49 | Валидная OpenAPI-спецификация 1–2 эндпоинтов |
| В04, слайд 43 | MLOps-roadmap на 6 месяцев, 1 страница |
| В05, слайд 44 | **5 экспериментов** с разными гиперпараметрами в MLflow и вывод |
| В06, слайд 45 | Скрипт валидации: минимум 5 проверок и отчёт о проблемах |
| В07, слайд 42 | 5+ pytest-тестов модели и найденные слабые места |
| В08, слайд 40 | Тестовая стратегия unit/integration/e2e, 1 страница |
| В09, слайд 41 | FastAPI-сервис модели, минимум 3 эндпоинта, проверка через Swagger UI |
| В10, слайд 45 | Стратегия деплоя: схема, критерии успеха, пороги и план отката; 2 страницы |
| В11, слайд 46 | Dockerfile и docker-compose.yml; проверка сборки либо описание проверки при отсутствии Docker |
| В12, слайд 41 | Логирование, метрики сервиса и эскиз дашборда с текстовым описанием |
| В13, слайд 42 | Стратегия обнаружения дрейфа и переобучения: признаки, метод, порог, стратегия; 1–2 страницы |
| В14, слайд 49 | Архитектура ИИ-агента: пайплайн обработки запроса, инструменты, память между шагами; 2 страницы |
| В15, слайд 41 | Задача средней сложности с ИИ-ассистентом; отчёт: промпты, корректировки, выводы; 1–2 страницы |

Итоговое задание требует минимум 3 запуска; план использует более конкретное ДЗ В05: 5 запусков. Результаты нельзя заменять наличием нескольких алгоритмов в исходном коде.

## Этапы

1. **М1 — Архитектура и роль ИИ-инженера (В01–В04).** Кейс, роли, жизненный цикл, архитектура, контракт, версии и roadmap.
2. **М2 — Эксперименты, качество данных и тестирование (В05–В08).** Воспроизводимый датасет, валидация, MLflow, качество модели, стратегия и CI.
3. **М3 — Деплой, API и мониторинг (В09–В12).** Требования В09–В12 подтверждены 21.10.2026: FastAPI с минимум тремя эндпоинтами и Swagger UI; стратегия деплоя на две страницы; Docker/Compose; логирование, метрики и эскиз дашборда. Задачи #12–#15 соответствуют этим работам. #16 остаётся задачей проекта по мониторингу данных и модели, но не является отдельным явно сформулированным ДЗ В12.
4. **М4 — ИИ-агенты, GenAI и работа с ИИ-ассистентами (В13–В16).** Название взято из В01, слайд 7. Требования В13–В15 подтверждены (см. таблицу выше и [покрытие М4](#м4-покрытие-заданий-в13в16)): ДЗ В13 выполнено в #17; для В14 и В15 нужны отдельные задачи. Работающий ИИ-агент не является обязательным результатом. В16 ожидается: задание по догадке не добавляется.
5. **М5 — Защита (В17–В18).** Отчёт 5–7 страниц, презентация 10–15 слайдов, 5 минут выступления и 2 минуты вопросов.

Метка `needs-lecture` означает предварительную декомпозицию, а не известное домашнее задание ещё не опубликованной лекции. На 21.10.2026 в ЛМС подтверждены сроки: тест М3 — 31.10, тест М4 — 28.11, итоговое ДЗ — 05.12.2026. Защиты назначены на 21.11 и 05.12; дата команды здесь не подтверждена. Отдельные сроки сдачи домашних работ В09–В12 в слайдах не указаны. Карточки доски синхронизируются с подтверждёнными условиями В09–В12 и сроками ЛМС.

## Задачи и зависимости

| Задача | Этап | Основание объёма | Зависимости |
|---|---|---|---|
| [#1](https://github.com/aleksandr-novikov/ai-integration-development/issues/1) Зафиксировать кейс, состав команды и критерии успеха | М1 | Подтверждено материалами | — |
| [#2](https://github.com/aleksandr-novikov/ai-integration-development/issues/2) В01 · Описать 7 этапов жизненного цикла ИИ-системы | М1 | Подтверждено материалами | [#1](https://github.com/aleksandr-novikov/ai-integration-development/issues/1) |
| [#3](https://github.com/aleksandr-novikov/ai-integration-development/issues/3) В02 · Актуализировать архитектуру и обоснование стека | М1 | Подтверждено материалами | [#1](https://github.com/aleksandr-novikov/ai-integration-development/issues/1) |
| [#4](https://github.com/aleksandr-novikov/ai-integration-development/issues/4) В03 · Оформить OpenAPI-контракт 1–2 ML-эндпоинтов | М1 | Подтверждено материалами | [#3](https://github.com/aleksandr-novikov/ai-integration-development/issues/3) |
| [#5](https://github.com/aleksandr-novikov/ai-integration-development/issues/5) В03 · Связать версии кода, данных и моделей | М1 | Подтверждено материалами | [#3](https://github.com/aleksandr-novikov/ai-integration-development/issues/3), [#7](https://github.com/aleksandr-novikov/ai-integration-development/issues/7) |
| [#6](https://github.com/aleksandr-novikov/ai-integration-development/issues/6) В04 · Составить MLOps-roadmap на 6 месяцев | М1 | Подтверждено материалами | [#3](https://github.com/aleksandr-novikov/ai-integration-development/issues/3) |
| [#7](https://github.com/aleksandr-novikov/ai-integration-development/issues/7) Подготовить воспроизводимый датасет инцидентов и временное разбиение | М2 | Подтверждено материалами | — |
| [#8](https://github.com/aleksandr-novikov/ai-integration-development/issues/8) В06 · Валидация датасета: минимум 5 проверок и отчёт | М2 | Подтверждено материалами | [#7](https://github.com/aleksandr-novikov/ai-integration-development/issues/7) |
| [#9](https://github.com/aleksandr-novikov/ai-integration-development/issues/9) В05 · Провести 5 экспериментов Isolation Forest в MLflow | М2 | Подтверждено материалами | [#7](https://github.com/aleksandr-novikov/ai-integration-development/issues/7), [#8](https://github.com/aleksandr-novikov/ai-integration-development/issues/8) |
| [#10](https://github.com/aleksandr-novikov/ai-integration-development/issues/10) В07 · Проверить качество модели: 5+ тестов и слабые места | М2 | Подтверждено материалами | [#7](https://github.com/aleksandr-novikov/ai-integration-development/issues/7), [#8](https://github.com/aleksandr-novikov/ai-integration-development/issues/8) |
| [#11](https://github.com/aleksandr-novikov/ai-integration-development/issues/11) В08 · Описать тестовую стратегию и проверить CI | М2 | Подтверждено материалами | — |
| [#12](https://github.com/aleksandr-novikov/ai-integration-development/issues/12) М3 / В09 · FastAPI-сервис модели: минимум 3 эндпоинта и Swagger UI | М3 | Подтверждено В09, слайд 41 | [#4](https://github.com/aleksandr-novikov/ai-integration-development/issues/4), [#5](https://github.com/aleksandr-novikov/ai-integration-development/issues/5), [#9](https://github.com/aleksandr-novikov/ai-integration-development/issues/9), [#11](https://github.com/aleksandr-novikov/ai-integration-development/issues/11) |
| [#13](https://github.com/aleksandr-novikov/ai-integration-development/issues/13) М3 / В11 · Dockerfile, Compose и описание проверки | М3 | Подтверждено В11, слайд 46 | [#7](https://github.com/aleksandr-novikov/ai-integration-development/issues/7), [#12](https://github.com/aleksandr-novikov/ai-integration-development/issues/12) |
| [#14](https://github.com/aleksandr-novikov/ai-integration-development/issues/14) М3 / В10 · Стратегия деплоя: схема, пороги и откат, 2 страницы | М3 | Подтверждено В10, слайд 45 | [#5](https://github.com/aleksandr-novikov/ai-integration-development/issues/5), [#11](https://github.com/aleksandr-novikov/ai-integration-development/issues/11), [#13](https://github.com/aleksandr-novikov/ai-integration-development/issues/13) |
| [#15](https://github.com/aleksandr-novikov/ai-integration-development/issues/15) М3 / В12 · Логирование, метрики и эскиз дашборда | М3 | Подтверждено В12, слайд 41 | [#13](https://github.com/aleksandr-novikov/ai-integration-development/issues/13) |
| [#16](https://github.com/aleksandr-novikov/ai-integration-development/issues/16) М3 · Мониторинг данных и качества самой ML-модели | М3 | Задача проекта; не отдельное ДЗ В12 | [#9](https://github.com/aleksandr-novikov/ai-integration-development/issues/9), [#10](https://github.com/aleksandr-novikov/ai-integration-development/issues/10), [#13](https://github.com/aleksandr-novikov/ai-integration-development/issues/13) |
| [#17](https://github.com/aleksandr-novikov/ai-integration-development/issues/17) В13 · Стратегия обнаружения дрейфа и переобучения — 1–2 страницы | М4 | Подтверждено В13, слайд 42; закрыта, PR #38 слит | [#5](https://github.com/aleksandr-novikov/ai-integration-development/issues/5), [#9](https://github.com/aleksandr-novikov/ai-integration-development/issues/9), [#16](https://github.com/aleksandr-novikov/ai-integration-development/issues/16) |
| [#18](https://github.com/aleksandr-novikov/ai-integration-development/issues/18) Вести журнал ИИ-ассистентов и оценить ограничения LLM-объяснений | М4 | Подтверждено материалами; закрыта. Журнал без промптов, ДЗ В15 не покрывает | — |
| [#19](https://github.com/aleksandr-novikov/ai-integration-development/issues/19) Подготовить итоговый отчёт на 5–7 страниц | М5 / защита | Подтверждено материалами | [#2](https://github.com/aleksandr-novikov/ai-integration-development/issues/2), [#3](https://github.com/aleksandr-novikov/ai-integration-development/issues/3), [#4](https://github.com/aleksandr-novikov/ai-integration-development/issues/4), [#5](https://github.com/aleksandr-novikov/ai-integration-development/issues/5), [#6](https://github.com/aleksandr-novikov/ai-integration-development/issues/6), [#8](https://github.com/aleksandr-novikov/ai-integration-development/issues/8), [#9](https://github.com/aleksandr-novikov/ai-integration-development/issues/9), [#10](https://github.com/aleksandr-novikov/ai-integration-development/issues/10), [#11](https://github.com/aleksandr-novikov/ai-integration-development/issues/11), [#14](https://github.com/aleksandr-novikov/ai-integration-development/issues/14), [#15](https://github.com/aleksandr-novikov/ai-integration-development/issues/15), [#16](https://github.com/aleksandr-novikov/ai-integration-development/issues/16), [#17](https://github.com/aleksandr-novikov/ai-integration-development/issues/17), [#18](https://github.com/aleksandr-novikov/ai-integration-development/issues/18) |
| [#20](https://github.com/aleksandr-novikov/ai-integration-development/issues/20) Подготовить 10–15 слайдов и отрепетировать защиту | М5 / защита | Подтверждено материалами | [#9](https://github.com/aleksandr-novikov/ai-integration-development/issues/9), [#13](https://github.com/aleksandr-novikov/ai-integration-development/issues/13), [#15](https://github.com/aleksandr-novikov/ai-integration-development/issues/15), [#16](https://github.com/aleksandr-novikov/ai-integration-development/issues/16), [#18](https://github.com/aleksandr-novikov/ai-integration-development/issues/18), [#19](https://github.com/aleksandr-novikov/ai-integration-development/issues/19) |
| [#21](https://github.com/aleksandr-novikov/ai-integration-development/issues/21) Проверить итоговый комплект по критериям курса | М5 / защита | Подтверждено материалами | [#19](https://github.com/aleksandr-novikov/ai-integration-development/issues/19), [#20](https://github.com/aleksandr-novikov/ai-integration-development/issues/20) |
| [#22](https://github.com/aleksandr-novikov/ai-integration-development/issues/22) Уточнить оставшийся план после выхода лекций В13–В16 | М4 | В09–В12 уточнены 21.10, В13 — 25.10, В14–В15 — 08.11; В16 ожидается | — |

### М4: покрытие заданий В13–В16

| Вебинар | Задача | Состояние | Зависимости |
|---|---|---|---|
| В13 | [#17](https://github.com/aleksandr-novikov/ai-integration-development/issues/17) | ДЗ подготовлено в [docs/course/drift](drift/README.md); PR #38 слит, задача закрыта | [#5](https://github.com/aleksandr-novikov/ai-integration-development/issues/5), [#9](https://github.com/aleksandr-novikov/ai-integration-development/issues/9), [#16](https://github.com/aleksandr-novikov/ai-integration-development/issues/16) |
| В14 | Отдельная задача — **ещё не создана** | Существующие задачи архитектуру агента не покрывают | [#3](https://github.com/aleksandr-novikov/ai-integration-development/issues/3) — архитектура системы; [#18](https://github.com/aleksandr-novikov/ai-integration-development/issues/18) — ограничения LLM для раздела рисков |
| В15 | Отдельная задача — **ещё не создана** | [#18](https://github.com/aleksandr-novikov/ai-integration-development/issues/18) закрыта, но журнал не содержит промптов и корректировок; новую практику нужно записывать с начала | [#18](https://github.com/aleksandr-novikov/ai-integration-development/issues/18) — формат записи в журнале; для варианта с pytest — [#10](https://github.com/aleksandr-novikov/ai-integration-development/issues/10), [#11](https://github.com/aleksandr-novikov/ai-integration-development/issues/11) |
| В16 | — | Ожидается открытие 14.11.2026; задание неизвестно | Уточняется в [#22](https://github.com/aleksandr-novikov/ai-integration-development/issues/22) |

Результаты В14 и В15 — исходные материалы для разделов про ИИ в [#19](https://github.com/aleksandr-novikov/ai-integration-development/issues/19) и [#20](https://github.com/aleksandr-novikov/ai-integration-development/issues/20); после создания задач их добавляют в зависимости #19. Отдельные сроки не назначены: на слайдах их нет, 28.11 — срок теста модуля 4.

## Начало работы

Можно независимо начать #1 (кейс и роли), #7 (датасет), #11 (стратегия тестов) и #18 (журнал ИИ).
Ключевая цепочка: #7 → #8 → #9 → #12 → #13 → #15/#16 → #19 → #20/#21; параллельно закрываются архитектура, контракт, версии и тесты.

Колонки: **Бэклог → Можно брать → В работе → На проверке → Готово**. Перед переводом в «Можно брать» проверить зависимости. Перед закрытием приложить PR/документ, команду или способ проверки и фактический результат; проверяющий — другой участник.

## Роли и зоны ответственности

| Участник | GitHub | Зона ответственности |
|---|---|---|
| Александр | [aleksandr-novikov](https://github.com/aleksandr-novikov) | Архитектура, MLflow, версии моделей и интеграция |
| Раиль | [rail-ss](https://github.com/rail-ss) | Постановка задачи, roadmap, отчёт и презентация |
| Виталий | [vtm9](https://github.com/vtm9) | Тесты, CI, Docker, выпуск и эксплуатация |
| Оксана | [ksdergach](https://github.com/ksdergach) | Данные, валидация, API и коллекторы |

Роли согласованы командой. Это распределение зон ответственности, а не утверждение о выполненном вкладе: роли могут пересекаться, а исполнители конкретных задач назначаются при взятии задач.

## Что считается завершением

Архитектура (20%), качество/тесты (25%), MLOps (20%), деплой/мониторинг (15%), документация/защита (20%) подтверждены ссылками на результаты. Тема согласована, роли указаны, преподаватель имеет доступ к репозиторию и итоговым артефактам. Полноценное облачное production-развёртывание не является обязательным условием.
