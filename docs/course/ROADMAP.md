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

Проверены презентации В01–В08 и два файла итогового задания. Материалы лекций не публикуются в этом репозитории; в карточках указаны названия файлов и номера слайдов.

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

Итоговое задание требует минимум 3 запуска; план использует более конкретное ДЗ В05: 5 запусков. Результаты нельзя заменять наличием нескольких алгоритмов в исходном коде.

## Этапы

1. **М1 — Архитектура и роль ИИ-инженера (В01–В04).** Кейс, роли, жизненный цикл, архитектура, контракт, версии и roadmap.
2. **М2 — Эксперименты, качество данных и тестирование (В05–В08).** Воспроизводимый датасет, валидация, MLflow, качество модели, стратегия и CI.
3. **М3 — Деплой, API и мониторинг (В09–В12).** API-сервис, Docker и демо с нуля, выпуск/откат, технический мониторинг, качество данных и модели. Пять отдельных задач #12–#16; названия и объём уточнить по новым презентациям.
4. **М4 — ИИ-агенты, GenAI и работа с ИИ-ассистентами (В13–В16).** Название взято из В01, слайд 7. Пока запланированы подтверждённые итоговым заданием дрейф и использование ИИ; специфичные задания по агентам/GenAI добавляются по мере публикации лекций.
5. **М5 — Защита (В17–В18).** Отчёт 5–7 страниц, презентация 10–15 слайдов, 5 минут выступления и 2 минуты вопросов.

Метка `needs-lecture` означает предварительную декомпозицию, а не известное домашнее задание ещё не опубликованной лекции. Даты не назначены: расписание и дедлайн сдачи пока не зафиксированы.

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
| [#12](https://github.com/aleksandr-novikov/ai-integration-development/issues/12) М3 / В09 · Подготовить учебный API-сервис инференса | М3 | Предварительно | [#4](https://github.com/aleksandr-novikov/ai-integration-development/issues/4), [#5](https://github.com/aleksandr-novikov/ai-integration-development/issues/5), [#9](https://github.com/aleksandr-novikov/ai-integration-development/issues/9), [#11](https://github.com/aleksandr-novikov/ai-integration-development/issues/11) |
| [#13](https://github.com/aleksandr-novikov/ai-integration-development/issues/13) М3 / В10–В11 · Воспроизводимый запуск в Docker и демо с нуля | М3 | Предварительно | [#7](https://github.com/aleksandr-novikov/ai-integration-development/issues/7), [#12](https://github.com/aleksandr-novikov/ai-integration-development/issues/12) |
| [#14](https://github.com/aleksandr-novikov/ai-integration-development/issues/14) М3 / В10–В11 · Проверить выпуск версии и откат | М3 | Предварительно | [#5](https://github.com/aleksandr-novikov/ai-integration-development/issues/5), [#11](https://github.com/aleksandr-novikov/ai-integration-development/issues/11), [#13](https://github.com/aleksandr-novikov/ai-integration-development/issues/13) |
| [#15](https://github.com/aleksandr-novikov/ai-integration-development/issues/15) М3 / В12 · Технический мониторинг, алерты и инструкция реакции | М3 | Предварительно | [#13](https://github.com/aleksandr-novikov/ai-integration-development/issues/13) |
| [#16](https://github.com/aleksandr-novikov/ai-integration-development/issues/16) М3 / В12 · Мониторинг данных и качества самой ML-модели | М3 | Предварительно | [#9](https://github.com/aleksandr-novikov/ai-integration-development/issues/9), [#10](https://github.com/aleksandr-novikov/ai-integration-development/issues/10), [#13](https://github.com/aleksandr-novikov/ai-integration-development/issues/13) |
| [#17](https://github.com/aleksandr-novikov/ai-integration-development/issues/17) Описать стратегию дрейфа и проверяемого переобучения | М4 | Предварительно | [#5](https://github.com/aleksandr-novikov/ai-integration-development/issues/5), [#9](https://github.com/aleksandr-novikov/ai-integration-development/issues/9), [#16](https://github.com/aleksandr-novikov/ai-integration-development/issues/16) |
| [#18](https://github.com/aleksandr-novikov/ai-integration-development/issues/18) Вести журнал ИИ-ассистентов и оценить ограничения LLM-объяснений | М4 | Подтверждено материалами | — |
| [#19](https://github.com/aleksandr-novikov/ai-integration-development/issues/19) Подготовить итоговый отчёт на 5–7 страниц | М5 / защита | Подтверждено материалами | [#2](https://github.com/aleksandr-novikov/ai-integration-development/issues/2), [#3](https://github.com/aleksandr-novikov/ai-integration-development/issues/3), [#4](https://github.com/aleksandr-novikov/ai-integration-development/issues/4), [#5](https://github.com/aleksandr-novikov/ai-integration-development/issues/5), [#6](https://github.com/aleksandr-novikov/ai-integration-development/issues/6), [#8](https://github.com/aleksandr-novikov/ai-integration-development/issues/8), [#9](https://github.com/aleksandr-novikov/ai-integration-development/issues/9), [#10](https://github.com/aleksandr-novikov/ai-integration-development/issues/10), [#11](https://github.com/aleksandr-novikov/ai-integration-development/issues/11), [#14](https://github.com/aleksandr-novikov/ai-integration-development/issues/14), [#15](https://github.com/aleksandr-novikov/ai-integration-development/issues/15), [#16](https://github.com/aleksandr-novikov/ai-integration-development/issues/16), [#17](https://github.com/aleksandr-novikov/ai-integration-development/issues/17), [#18](https://github.com/aleksandr-novikov/ai-integration-development/issues/18) |
| [#20](https://github.com/aleksandr-novikov/ai-integration-development/issues/20) Подготовить 10–15 слайдов и отрепетировать защиту | М5 / защита | Подтверждено материалами | [#9](https://github.com/aleksandr-novikov/ai-integration-development/issues/9), [#13](https://github.com/aleksandr-novikov/ai-integration-development/issues/13), [#15](https://github.com/aleksandr-novikov/ai-integration-development/issues/15), [#16](https://github.com/aleksandr-novikov/ai-integration-development/issues/16), [#18](https://github.com/aleksandr-novikov/ai-integration-development/issues/18), [#19](https://github.com/aleksandr-novikov/ai-integration-development/issues/19) |
| [#21](https://github.com/aleksandr-novikov/ai-integration-development/issues/21) Проверить итоговый комплект по критериям курса | М5 / защита | Подтверждено материалами | [#19](https://github.com/aleksandr-novikov/ai-integration-development/issues/19), [#20](https://github.com/aleksandr-novikov/ai-integration-development/issues/20) |
| [#22](https://github.com/aleksandr-novikov/ai-integration-development/issues/22) Уточнить план после выхода лекций В09–В16 | М4 | Предварительно | — |

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
