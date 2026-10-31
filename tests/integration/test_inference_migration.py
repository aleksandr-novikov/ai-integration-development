"""Real PostgreSQL migration of pre-version scores and unowned LLM cache."""

import pytest

from tests.integration.conftest import use_metrics_db
from tests.test_inference_migration import exercise_migration

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("concurrent", [False, True])
def test_existing_postgres_is_migrated_without_loss(concurrent):
    from testcontainers.postgres import PostgresContainer

    from app import metrics_storage as storage

    with PostgresContainer("postgres:16-alpine") as database:
        with use_metrics_db(database.get_connection_url()):
            engine = storage._new_engine()
            storage._engine = engine
            storage._initialized = True
            exercise_migration(engine, postgres=True, concurrent=concurrent)
