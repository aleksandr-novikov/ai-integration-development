"""Compose demo model: generator fidelity, split hygiene, metrics and registry status."""

import json
import random
import re
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest
from faker import Faker
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

from app import db
from collectors.metrics_collector import MetricsCollector
from ml import model_registry as registry
from scripts import demo_model, live_demo, seed_target_db

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/course/containers/demo-model"
SOURCE = {"project_id": "p", "connection_id": "c", "schema_name": "public", "table_name": "events"}


def _schema_columns():
    sql = (ROOT / "scripts/schema.sql").read_text()
    body = re.search(r"CREATE TABLE IF NOT EXISTS events \((.*?)\n\);", sql, re.S).group(1)
    columns = {}
    for line in body.splitlines():
        match = re.match(r"\s+([a-z_0-9]+)\s+(.*)", line)
        if match:
            columns[match.group(1)] = match.group(2).upper()
    return columns


def test_simulated_columns_and_null_defaults_follow_the_events_schema():
    columns = _schema_columns()
    assert tuple(columns) == demo_model.EVENTS_COLUMNS
    nullable_without_default = {
        name for name, spec in columns.items()
        if "NOT NULL" not in spec and "PRIMARY KEY" not in spec and "DEFAULT" not in spec
    }
    assert nullable_without_default == demo_model.NULL_WHEN_OMITTED


class _SQLiteAdapter:
    """Only what MetricsCollector's full null-rate path needs, via the generic SQL."""

    def quote_ident(self, identifier):
        return f'"{identifier}"'

    def table_schema(self, table_name, schema):
        return [{"name": name, "type": "text"} for name in demo_model.EVENTS_COLUMNS]

    def column_nulls(self, table_name, schema):
        return db._column_nulls_generic(self, table_name, schema)

    def column_distribution(self, table_name, schema, top_n=20):
        return []


def _sqlite_events():
    engine = create_engine("sqlite://", poolclass=StaticPool)
    defaults = {"id": "DEFAULT (lower(hex(randomblob(16))))", "events_in_session": "DEFAULT 1",
                "ip_events_last_1h": "DEFAULT 0", "server_id": "DEFAULT 'server-1'",
                "device_type": "DEFAULT 'desktop'", "is_bot_suspected": "DEFAULT 0",
                "created_at": "DEFAULT CURRENT_TIMESTAMP"}
    columns = ", ".join(f"{name} {defaults.get(name, '')}" for name in demo_model.EVENTS_COLUMNS)
    with engine.begin() as conn:
        conn.execute(text(f"CREATE TABLE events ({columns})"))
    return engine


def test_simulator_matches_real_inserts_and_the_collector_null_rate():
    """Real seeder/live-demo INSERT statements and the collector's averaging."""
    engine, table = _sqlite_events(), demo_model._Table()
    original = seed_target_db._event_row
    captured = []

    def capture(*args):
        captured.append(original(*args))
        return captured[-1]

    random.seed(7)
    with patch.object(seed_target_db, "_event_row", side_effect=capture):
        seed_target_db._seed_events(engine, Faker(), ["u1", "u2"], 700)
    table.insert(captured)
    observations = []
    for incident in (False, False, True, False):
        count = 150 * (live_demo._INCIDENT_ROW_MULTIPLIER if incident else 1)
        rows = [live_demo._generate_event_row(["u1"], incident) for _ in range(count)]
        live_demo._insert_batch(engine, rows)
        table.insert(rows)
        with db.using_engine(engine, _SQLiteAdapter()):
            metrics = MetricsCollector(schema="main")._collect_null_rate_full("events", datetime.now(UTC))
        actual = next(m["value"] for m in metrics if m["metric_name"] == "null_rate")
        with engine.connect() as conn:
            total = conn.execute(text("SELECT count(*) FROM events")).scalar_one()
        observations.append(((total, actual), table.observe()))
    for real, simulated in observations:
        assert real == simulated
    # The incident batch (40% NULL ip_address) raises the table null rate.
    rates = [real[1] for real, _ in observations]
    assert rates[2] > max(rates[:2]) + 0.002


def test_live_demo_rows_carry_seeder_like_prev_event_fields_and_keep_the_incident_signal():
    random.seed(11)
    normal = [live_demo._generate_event_row(["u"], False) for _ in range(5000)]
    incident = [live_demo._generate_event_row(["u"], True) for _ in range(5000)]
    for rows in (normal, incident):
        missing = sum(row["prev_event_type"] is None for row in rows) / len(rows)
        assert 0.17 < missing < 0.23
        assert all((row["prev_event_type"] is None) == (row["prev_event_gap_s"] is None) for row in rows)
    ip_null = [sum(row["ip_address"] is None for row in rows) / len(rows) for rows in (normal, incident)]
    assert ip_null[0] < 0.04 and 0.36 < ip_null[1] < 0.44
    assert {":prev_event_type", ":prev_event_gap_s"} <= set(re.findall(r":\w+", str(live_demo._INSERT_SQL)))


def test_simulation_is_deterministic_and_restores_the_global_random_state():
    episode = demo_model.Episode("general", 5, 1200, 8, 150, False, 4)
    random.seed(99)
    before = random.random()
    random.seed(99)
    first = demo_model.simulate(episode)
    assert random.random() == before
    assert demo_model.simulate(episode) == first
    assert first["labels"] == ["normal"] * 3 + ["incident"] + ["aftermath"] * 4
    increments = [b["value"] - a["value"] for a, b in zip(first["row_counts"], first["row_counts"][1:])]
    assert increments[3] == 1500
    assert all(rows == 0 or 100 <= rows <= 300 for i, rows in enumerate(increments) if i != 3)


def test_splits_use_disjoint_seeds_and_train_has_no_incidents():
    splits = demo_model.splits()
    seeds = {name: {episode.seed for episode in episodes} for name, episodes in splits.items()}
    assert not seeds["train"] & seeds["validation"]
    assert not seeds["train"] & seeds["test"]
    assert not seeds["validation"] & seeds["test"]
    assert all(e.incident_at is None for e in splits["train"])
    assert len(splits["train"]) * 25 >= 200
    compose = [e for e in splits["test"] if e.family == "compose"]
    assert len(compose) == 200
    assert {(e.initial_rows, e.rows_per_tick, e.ticks, e.incident_at, e.fixed_rows) for e in compose} == {
        (2000, 200, 4, 4, True)}
    readme = [e for e in splits["test"] if e.family == "readme"]
    assert {(e.ticks, e.incident_at) for e in readme} == {(6, 5)}


def test_metrics_follow_the_protocol_and_exclude_aftermath_from_fpr():
    episodes = [demo_model.Episode("compose", 1, 2000, 4, 200, True, 4),
                demo_model.Episode("compose", 2, 2000, 4, 200, True, 4),
                demo_model.Episode("general", 3, 1000, 3, 200, False, 2)]
    labels = ["normal"] * 3 + ["incident"] + ["normal"] * 3 + ["incident"] + ["normal", "incident", "aftermath"]
    owners = [0] * 4 + [1] * 4 + [2] * 3
    flags = [False, False, False, True, True, False, False, True, False, False, True]
    result = demo_model.evaluate(episodes, labels, owners, flags)
    metrics = result["metrics"]
    assert (metrics["normal_points"], metrics["false_alarms"]) == (7, 1)
    assert (metrics["incidents"], metrics["detected"]) == (3, 2)
    assert (metrics["aftermath_points"], metrics["aftermath_alarms"]) == (1, 1)
    assert metrics["false_positive_rate"] == round(1 / 7, 6)
    assert metrics["event_recall"] == round(2 / 3, 6)
    compose = result["families"]["compose"]
    assert (compose["scenarios"], compose["scenarios_passed"]) == (2, 1)
    assert result["targets"]["event_recall"] is False
    assert result["targets_passed"] is False


def test_selection_and_test_evidence_are_consistent_with_the_code():
    selection = json.loads((EVIDENCE / "selection.json").read_text())
    evaluation = json.loads((EVIDENCE / "evaluation.json").read_text())
    assert selection["selected"] == demo_model.SELECTED == evaluation["configuration"]
    assert {name: c["parameters"] for name, c in selection["candidates"].items()} == demo_model.CONFIGS
    assert selection["regime"] == evaluation["regime"] == demo_model.REGIME
    assert selection["train_data_sha256"] == evaluation["train_data_sha256"]
    assert evaluation["test"]["targets_passed"] is True
    assert evaluation["demo"] is True and evaluation["quality_admitted"] is False
    # The archived #9 model scored the same test vectors for comparison.
    assert evaluation["reference_archive_9"]["test"]["families"]["compose"]["scenarios_passed"] == 0


def test_generator_still_reproduces_the_recorded_training_data():
    _, histories, digest = demo_model.generate("train")
    assert digest == json.loads((EVIDENCE / "selection.json").read_text())["train_data_sha256"]
    assert sum(len(h["labels"]) for h in histories) == 4000


def test_train_and_evaluate_uses_the_registry_and_stays_demo_only(tmp_path, monkeypatch):
    small = {
        "train": demo_model.family("train", 10, 11, ticks=25, incident=False),
        "test": demo_model.family("general", 2, 21, ticks=12)
        + demo_model.family("compose", 2, 31, compose=(4, 4)),
    }
    monkeypatch.setattr(demo_model, "splits", lambda: small)
    manifest, evaluation = demo_model.train_and_evaluate(tmp_path, SOURCE, "a" * 40)
    training = manifest["training"]
    assert training["n_histories"] == 10 and training["n_points"] == 250
    assert training["parameters"] == demo_model.CONFIGS[demo_model.SELECTED]
    assert training["provenance"]["train_data_sha256"] == evaluation["train_data_sha256"]
    assert set(training["provenance"]["generator_sha256"]) == set(demo_model.SOURCE_FILES)
    assert manifest["evaluation"] is None
    assert manifest["origin"] == {"kind": "local-candidate", "automatic_activation": False}
    assert evaluation["model_version"] == manifest["model_version"]
    assert evaluation["test_vectors"] == 2 * 12 + 2 * 4
    registry.load_version(tmp_path, manifest["model_version"], source=SOURCE, demo=True)
    with pytest.raises(registry.RegistryError, match="admission"):
        registry.load_version(tmp_path, manifest["model_version"], source=SOURCE)


def test_cli_rejects_an_abbreviated_commit(tmp_path, capsys):
    source = tmp_path / "source.json"
    source.write_text(json.dumps(SOURCE))
    with pytest.raises(SystemExit):
        demo_model.main(["train", "--source", str(source), "--git-sha", "8fb49e5",
                         "--evaluation-out", str(tmp_path / "e.json")])
    assert "full commit SHA" in capsys.readouterr().err
