"""Tests for live_demo._run_collector_tick tenant scoping (#138)."""
from unittest.mock import MagicMock, patch


def test_run_collector_tick_with_connection_calls_per_project():
    """_run_collector_tick с connection_id вызывает collect_for_connection."""
    mock_latest = {"value": 42.0}
    with (
        patch("collectors.per_project.collect_for_connection") as mock_collect,
        patch("app.metrics_storage.get_latest_metric", return_value=mock_latest),
    ):
        from scripts.live_demo import _run_collector_tick
        result = _run_collector_tick("proj-a", "conn-a")

    mock_collect.assert_called_once_with("proj-a", "conn-a")
    assert result == 42


def test_run_collector_tick_with_connection_reads_from_project():
    """_run_collector_tick с connection_id читает метрику из project_id, не из legacy."""
    with (
        patch("collectors.per_project.collect_for_connection"),
        patch("app.metrics_storage.get_latest_metric", return_value=None) as mock_get,
    ):
        from scripts.live_demo import _run_collector_tick
        _run_collector_tick("proj-a", "conn-a")

    args = mock_get.call_args
    assert args[0][2] == "proj-a", "должен читать метрику из proj-a, не из legacy"


def test_run_collector_tick_no_connection_uses_global_collector():
    """_run_collector_tick без connection_id использует глобальный collect_all_tables."""
    with (
        patch("collectors.scheduler.collect_all_tables") as mock_collect,
        patch("app.metrics_storage.get_latest_metric", return_value=None),
    ):
        from scripts.live_demo import _run_collector_tick
        _run_collector_tick("legacy", None)

    mock_collect.assert_called_once()


def test_run_collector_tick_no_connection_reads_from_legacy():
    """_run_collector_tick без connection_id читает метрику из 'legacy', не из project_id."""
    with (
        patch("collectors.scheduler.collect_all_tables"),
        patch("app.metrics_storage.get_latest_metric", return_value=None) as mock_get,
    ):
        from scripts.live_demo import _run_collector_tick
        _run_collector_tick("some-project", None)

    args = mock_get.call_args
    assert args[0][2] == "legacy", "без connection_id collector пишет в legacy"


def test_live_demo_cli_requires_connection_id_for_non_legacy():
    """--project-id != legacy без --connection-id завершается с ошибкой."""
    import subprocess
    result = subprocess.run(
        ["python", "-m", "scripts.live_demo",
         "--project-id", "proj-x", "--dry-run"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "connection-id" in result.stderr.lower()


def _engine(dialect="postgresql", version=(18, 0), privilege=True):
    engine = MagicMock()
    engine.dialect.name = dialect
    conn = engine.connect.return_value.__enter__.return_value
    conn.dialect.server_version_info = version
    conn.execute.return_value.scalar.return_value = privilege
    return engine, conn


def test_stats_flush_is_not_requested_for_other_dialects_or_old_postgres():
    """Без поддержки pg_stat_force_next_flush() вставка ведёт себя как раньше."""
    from scripts.live_demo import _stats_flush_supported

    engine, _ = _engine(dialect="sqlite")
    assert _stats_flush_supported(engine) is False
    engine.connect.assert_not_called()
    engine, conn = _engine(version=(14, 11))
    assert _stats_flush_supported(engine) is False
    conn.execute.assert_not_called()


def test_stats_flush_requires_execute_privilege_on_postgres_15_or_later():
    from scripts.live_demo import _stats_flush_supported

    for privilege in (True, False):
        engine, conn = _engine(version=(15, 0), privilege=privilege)
        assert _stats_flush_supported(engine) is privilege
        assert "has_function_privilege('pg_stat_force_next_flush()', 'EXECUTE')" in str(conn.execute.call_args.args[0])


def test_stats_flush_check_does_not_hide_database_errors():
    import pytest

    from scripts.live_demo import _stats_flush_supported

    engine, conn = _engine()
    conn.execute.side_effect = RuntimeError("connection lost")
    with pytest.raises(RuntimeError):
        _stats_flush_supported(engine)


def test_insert_batch_requests_stats_flush_inside_the_insert_transaction():
    """Флаг ставится в той же транзакции и применяется сервером при её COMMIT."""
    from scripts.live_demo import _INSERT_SQL, _insert_batch

    for flush_stats, expected in ((True, 2), (False, 1)):
        engine = MagicMock()
        conn = engine.begin.return_value.__enter__.return_value
        _insert_batch(engine, [{"id": "row"}], flush_stats=flush_stats)
        engine.begin.assert_called_once()
        assert conn.execute.call_count == expected
        assert conn.execute.call_args_list[0].args == (_INSERT_SQL, [{"id": "row"}])
        if flush_stats:
            assert str(conn.execute.call_args_list[1].args[0]) == "SELECT pg_stat_force_next_flush()"
