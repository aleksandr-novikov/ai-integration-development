"""Offline safety/result-policy tests; no simulated result is CI evidence."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import yaml

from scripts import verify_compose_demo as verify


class ComposeVerificationTests(unittest.TestCase):
    def test_false_alarm_or_missed_incident_is_not_a_success(self):
        for normal, incident in (([True], [True]), ([False], [False]), ([], [True]), ([False], [])):
            with self.subTest(normal=normal, incident=incident):
                result = verify.outcome([{"is_anomaly": x} for x in normal],
                                        [{"is_anomaly": x} for x in incident])
                self.assertFalse(result["passed"])
        self.assertTrue(verify.outcome([{"is_anomaly": False}], [{"is_anomaly": True}])["passed"])

    def test_report_drops_arbitrary_text_and_invalid_scalar_values_at_every_level(self):
        private = "fixture-private-token postgresql://fixture:fixture-password@host/db https://fixture.invalid/private"
        report = {
            "technical_pass": False, "diagnostic_tail": private, "environment": {"token": private},
            "failure": {"stage": "normal", "exit_code": 1, "error_type": "unexpected_error", "message": private},
            "steps": [{"step": "normal", "exit_code": 1, "seconds": 0.1,
                       "command": private, "stdout": private, "stderr": private}],
            "docker": private, "compose": private, "checkout_sha": private, "pr_head_sha": private,
            "prepared": {"warmup": {"retail-postgres": {"metrics": 10, "log": private}}},
            "readiness": {"model": {"model_version": private, "source": {"table_name": private}, "error": private}},
            "normal": [{"score": 0.5, "ts": private, "features": {"row_count": 2000, "null_rate": private}, "url": private}],
            "restart_persistence": {"artifacts": {private: private}, "target": {"events": {"rows": 1, "digest": private}}},
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "report.json"
            with patch.object(verify, "REPORT", path):
                verify.save_report(report)
            content = path.read_text()
            for fragment in ("fixture-private-token", "fixture-password", "fixture.invalid", "postgresql://", "diagnostic_tail", "stdout", "stderr", "command", "environment"):
                self.assertNotIn(fragment, content)
            public = json.loads(content)
            self.assertEqual(public["failure"], {"stage": "normal", "exit_code": 1, "error_type": "unexpected_error"})
            self.assertEqual(public["prepared"]["warmup"]["retail-postgres"], {"metrics": 10})
            self.assertEqual(public["normal"][0]["features"], {"row_count": 2000})

    def test_observation_fields_remain_available_without_copying_extra_fields(self):
        row = {"source": {"project_id": "a" * 32, "connection_id": "b" * 32,
                          "schema_name": "public", "table_name": "events"},
               "model_version": "if-" + "c" * 64, "feature_schema_version": "1.0.0",
               "ts": "2026-11-03T03:00:00+00:00", "score": -0.2, "is_anomaly": True,
               "demo": True, "quality_admitted": False,
               "features": {"row_count": 2000, "null_rate": 0.1, "d_row_count": 200, "d_null_rate": 0.01}}
        self.assertEqual(verify.public_report({"incident": [row]})["incident"], [row])

    def test_failed_commands_and_timeouts_never_persist_arguments_or_output(self):
        private = "fixture-private-token https://fixture.invalid/private"
        for result in (subprocess.CompletedProcess([private], 17, stdout=private, stderr=private),
                       subprocess.TimeoutExpired([private], 1, output=private, stderr=private)):
            with self.subTest(kind=type(result).__name__), tempfile.TemporaryDirectory() as temporary:
                run = object.__new__(verify.ComposeRun)
                run.report = {"technical_pass": False, "steps": []}
                path = Path(temporary) / "report.json"
                options = {"side_effect": result} if isinstance(result, Exception) else {"return_value": result}
                with patch.object(verify, "REPORT", path), patch.object(verify.subprocess, "run", **options):
                    with self.assertRaises(verify.CommandFailure) as raised:
                        run.command("normal", [private])
                    run.report["failure"] = verify.failure_fields(raised.exception, "normal")
                    verify.save_report(run.report)
                self.assertNotIn("fixture", path.read_text())
                failure = json.loads(path.read_text())["failure"]
                self.assertEqual(failure["stage"], "normal")
                self.assertEqual(failure["error_type"], "timeout" if isinstance(result, Exception) else "command_failed")
                self.assertEqual(failure["exit_code"], None if isinstance(result, Exception) else 17)

    def test_main_failure_does_not_collect_logs_or_serialize_exception_text(self):
        fake = Mock(stage="dashboard", report={"technical_pass": False, "steps": []})
        fake.run.side_effect = RuntimeError("fixture-private-token https://fixture.invalid/private")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "report.json"
            with patch.object(verify, "REPORT", path), patch.object(verify, "ComposeRun", return_value=fake), \
                    patch.object(verify.sys, "argv", ["verify", "run"]):
                with self.assertRaises(SystemExit):
                    verify.main()
            self.assertNotIn("fixture", path.read_text())
            self.assertEqual(json.loads(path.read_text())["failure"],
                             {"stage": "dashboard", "exit_code": None, "error_type": "unexpected_error"})
            fake.compose.assert_not_called()

    def test_host_execution_refuses_without_runner_identity_before_any_command(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(verify.subprocess, "run") as run:
            with self.assertRaises(RuntimeError):
                verify.ComposeRun()
            run.assert_not_called()

    def test_cleanup_refuses_a_different_workflow_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "state.json"
            state.write_text(json.dumps({"run_id": "other-run", "project": "course13-0123456789ab"}))
            with patch.object(verify, "STATE", state), patch.dict(os.environ, {
                "GITHUB_ACTIONS": "true", "RUNNER_OS": "Linux", "GITHUB_RUN_ID": "current-run",
            }), patch.object(verify.subprocess, "run") as run:
                with self.assertRaises(AssertionError):
                    verify.cleanup()
                run.assert_not_called()

    def test_cleanup_refuses_foreign_project_names_or_volumes(self):
        project = "course13-0123456789ab"
        for recorded, configured, volumes in (("foreign", "foreign", ""),
                                               (project, "course13-aaaaaaaaaaaa", ""),
                                               (project, project, "foreign_volume")):
            with self.subTest(recorded=recorded, configured=configured, volumes=volumes), \
                    tempfile.TemporaryDirectory() as temporary:
                state, report = (Path(temporary) / name for name in ("state.json", "report.json"))
                state.write_text(json.dumps({"run_id": "current-run", "project": recorded}))
                report.write_text('{"technical_pass": false, "steps": []}')
                fake = Mock(values={"COMPOSE_PROJECT_NAME": configured})
                fake.command.return_value = volumes
                with patch.object(verify, "STATE", state), patch.object(verify, "REPORT", report), \
                        patch.object(verify, "ComposeRun", return_value=fake), patch.dict(os.environ, {
                            "GITHUB_ACTIONS": "true", "RUNNER_OS": "Linux", "GITHUB_RUN_ID": "current-run"}):
                    with self.assertRaises(AssertionError):
                        verify.cleanup()
                fake.compose.assert_not_called()

    def test_cleanup_accepts_only_recorded_project_resources_and_keeps_report(self):
        project = "course13-0123456789ab"
        with tempfile.TemporaryDirectory() as temporary:
            state, report = (Path(temporary) / name for name in ("state.json", "report.json"))
            state.write_text(json.dumps({"run_id": "current-run", "project": project}))
            report.write_text('{"technical_pass": true, "steps": []}')
            fake = Mock(values={"COMPOSE_PROJECT_NAME": project})
            fake.command.return_value = f"{project}_pg_data\n{project}_timescale_data\n"
            with patch.object(verify, "STATE", state), patch.object(verify, "REPORT", report), \
                    patch.object(verify, "ComposeRun", return_value=fake), patch.dict(os.environ, {
                        "GITHUB_ACTIONS": "true", "RUNNER_OS": "Linux", "GITHUB_RUN_ID": "current-run"}):
                verify.cleanup()
            fake.compose.assert_called_once_with("remove only runner test resources", "down", "--volumes", "--remove-orphans", timeout=120)
            self.assertEqual(json.loads(report.read_text()), {"technical_pass": True, "steps": [], "cleanup": "PASS"})

    def test_environment_creation_uses_checked_tree_sha_instead_of_pr_head(self):
        tree_sha, pr_head = "a" * 40, "b" * 40
        with tempfile.TemporaryDirectory() as temporary, patch.object(verify, "ROOT", Path(temporary)):
            run = object.__new__(verify.ComposeRun)
            run.command = Mock(side_effect=["", tree_sha])
            with patch.dict(os.environ, {"COMPOSE_SMOKE_SHA": tree_sha, "COMPOSE_SMOKE_PR_HEAD": pr_head}), \
                    patch("scripts.init_compose_demo.create_environment", side_effect=RuntimeError("stop before writes")) as create:
                with self.assertRaisesRegex(RuntimeError, "stop before writes"):
                    run.run()
            self.assertEqual(create.call_args.kwargs["version"], tree_sha)

    def test_outcome_command_fails_for_a_technically_working_but_misclassifying_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "report.json"
            report.write_text(json.dumps({"technical_pass": True, "model_scenario": {"passed": False}}))
            with patch.object(verify, "REPORT", report), patch.object(verify.sys, "argv", ["verify", "check-outcome"]):
                with self.assertRaises(SystemExit):
                    verify.main()

    def test_ci_reuses_one_image_and_uploads_only_the_sanitized_report(self):
        job = yaml.safe_load((verify.ROOT / ".github/workflows/tests.yml").read_text())["jobs"]["integration"]
        self.assertEqual(job["timeout-minutes"], 30)
        self.assertIn("COMPOSE_SMOKE_IMAGE", job["env"])
        self.assertEqual(job["permissions"], {"contents": "read"})
        self.assertEqual(job["steps"][0]["with"]["ref"], "${{ github.sha }}")
        self.assertIs(job["steps"][0]["with"]["persist-credentials"], False)
        self.assertEqual(job["env"]["COMPOSE_SMOKE_SHA"], "${{ github.sha }}")
        self.assertEqual(job["env"]["COMPOSE_SMOKE_PR_HEAD"], "${{ github.event.pull_request.head.sha }}")
        upload = next(s for s in job["steps"] if s.get("uses") == "actions/upload-artifact@v4")
        self.assertEqual(upload["with"]["path"], "artifacts/compose-demo/ci-report.json")
        self.assertEqual(upload["with"]["name"], "compose-evidence-${{ github.sha }}")
        self.assertEqual(upload["if"], "always()")
        cleanup = next(s for s in job["steps"] if s.get("run") == "python -m scripts.verify_compose_demo cleanup")
        self.assertEqual(cleanup["if"], "always()")


class PrepareDiagnosticsTests(unittest.TestCase):
    private = "fixture-private-token postgresql://fixture:fixture-password@fixture.invalid/db https://fixture.invalid/private"

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        (self.root / "models").mkdir()
        self.path = self.root / "models/.compose-diagnostics/prepare.json"
        for name, value in (("ROOT", self.root), ("PREPARE_DIAGNOSTICS", self.path),
                            ("REPORT", self.root / "report.json"), ("DEMO", self.root / "models/demo")):
            replacement = patch.object(verify, name, value)
            replacement.start()
            self.addCleanup(replacement.stop)
        verify.create_prepare_exchange()

    def preflight(self):
        with patch.dict(os.environ, {"COMPOSE_DEMO_TEST": "1"}, clear=True), \
                patch.object(verify.Path, "exists", return_value=True):
            verify.prepare_exchange_preflight()

    def assert_private_report(self, diagnostics):
        verify.save_report({"prepare_diagnostics": diagnostics})
        content = verify.REPORT.read_text()
        for fragment in ("fixture", "postgresql://", "https://", "password", "stdout", "stderr", "environment"):
            self.assertNotIn(fragment, content)
        return json.loads(content)

    def run_prepare(self, failed=None):
        """Mock business dependencies only; execute real prepare/checkpoint/file code."""
        source = {"project_id": "a" * 32, "connection_id": "b" * 32,
                  "schema_name": "public", "table_name": "events"}
        calls = []

        def action(stage, result=None):
            def execute(*args, **kwargs):
                calls.append(stage)
                if stage == failed:
                    raise RuntimeError(self.private)
                return result
            return execute

        app, ml, registry, scripts = (ModuleType(name) for name in ("app", "ml", "ml.model_registry", "scripts"))
        app.metrics_storage = Mock()
        app.metrics_storage.get_user_by_email.return_value = {"id": "fixture-user"}
        app.metrics_storage.list_projects_for_user.return_value = [{"id": source["project_id"]}, {"id": "other"}]
        app.metrics_storage.list_connections_for_project.return_value = [{"id": "disabled"}]
        manifest = {"model_version": "if-" + "c" * 64, "feature_schema_version": "1.0.0"}
        registry.stage_export = Mock(side_effect=action("registry_export", manifest))
        scripts.seed_target_db = SimpleNamespace(main=Mock(side_effect=action("target_seed")))
        summary = {"retail-postgres": {"metrics": 336, "anomaly_scores": 335, "forecast_models": 1}}
        if failed == "summary_check":
            summary["retail-postgres"]["anomaly_scores"] = 0
        scripts.demo_prepare = SimpleNamespace(main=Mock(side_effect=action("demo_prepare", summary)))
        modules = {"app": None if failed == "imports" else app, "ml": ml,
                   "ml.model_registry": registry, "scripts": scripts}
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, modules))
            stack.enter_context(patch.dict(os.environ, {} if failed == "guard" else {"COMPOSE_DEMO_TEST": "1"}, clear=True))
            stack.enter_context(patch.object(verify.Path, "exists", return_value=True))
            stack.enter_context(patch.object(verify, "container_state", side_effect=action("source_check", source)))
            if failed == "config_write":
                stack.enter_context(patch.object(verify, "save", side_effect=PermissionError(self.private)))
            verify.prepare()
        scripts.seed_target_db.main.assert_called_once_with(n_users=500, n_products=50, n_orders=1000, n_events=2000)
        self.assertEqual(calls, ["target_seed", "demo_prepare", "source_check", "registry_export"])
        registry.stage_export.assert_called_once_with(self.root / "models/registry", Path("/tmp/mlflow-export.zip"), source=source)
        self.assertEqual(app.metrics_storage.set_connection_active.call_count, 2)
        prepared = verify.read(verify.DEMO / "prepared.json")
        self.assertTrue(prepared["demo"])
        self.assertFalse(prepared["quality_admitted"])
        self.assertEqual(prepared["baseline_fpr"], 0.1327)

    def test_preflight_round_trip_is_private_and_requires_a_fresh_exchange(self):
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.path.parent.stat().st_mode & 0o777, 0o700)
        self.assertFalse(verify.read_checkpoints()["mount_verified"])
        self.preflight()
        self.assertTrue(verify.read_checkpoints()["mount_verified"])
        with self.assertRaises(ValueError):
            self.preflight()
        with self.assertRaises(FileExistsError):
            verify.create_prepare_exchange()

    def test_preflight_guard_and_missing_preflight_stop_before_business_operations(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(RuntimeError):
            verify.prepare_exchange_preflight()
        with patch.object(verify, "container_state") as state, self.assertRaises(ValueError):
            verify.prepare()
        state.assert_not_called()
        self.assertEqual(verify.collect_prepare_diagnostics()["availability"], "not_started")

    def test_prepare_success_preserves_business_calls_and_all_eight_checkpoints(self):
        self.preflight()
        self.run_prepare()
        diagnostics = verify.collect_prepare_diagnostics()
        self.assertEqual(diagnostics["events"], [{"stage": stage, "state": status}
                                                for stage in verify.PREPARE_STAGES for status in ("started", "passed")])
        self.assertEqual(diagnostics["last_successful_stage"], "config_write")
        self.assertIsNone(diagnostics["failed_stage"])
        self.assertIsNone(diagnostics["active_stage"])
        self.assertEqual(self.assert_private_report(diagnostics)["prepare_diagnostics"], diagnostics)

    def test_prepare_failure_at_each_stage_records_last_success_and_safe_category(self):
        for index, stage in enumerate(verify.PREPARE_STAGES):
            with self.subTest(stage=stage):
                verify.write_checkpoints({"version": 1, "mount_verified": True, "events": []})
                with self.assertRaises((RuntimeError, ImportError, AssertionError, PermissionError)):
                    self.run_prepare(failed=stage)
                diagnostics = verify.collect_prepare_diagnostics()
                self.assertEqual(diagnostics["failed_stage"], stage)
                self.assertEqual(diagnostics["last_successful_stage"], verify.PREPARE_STAGES[index - 1] if index else None)
                self.assertIsNone(diagnostics["active_stage"])
                category = {"imports": "import_error", "summary_check": "assertion_failed", "config_write": "permission_denied"}.get(stage, "unexpected_error")
                self.assertEqual(diagnostics["error_type"], category)
                self.assert_private_report(diagnostics)
                self.assertNotIn("fixture", self.path.read_text())

    def test_process_exit_and_interrupt_are_recorded_without_exit_message(self):
        for error, category in ((SystemExit(self.private), "process_exit"),
                                (KeyboardInterrupt(self.private), "interrupted")):
            with self.subTest(category=category):
                verify.write_checkpoints({"version": 1, "mount_verified": True, "events": []})
                progress = verify.PrepareProgress()
                with self.assertRaises(type(error)), progress.step("guard"):
                    raise error
                diagnostics = verify.collect_prepare_diagnostics()
                self.assertEqual(diagnostics["error_type"], category)
                self.assert_private_report(diagnostics)

    def test_hard_exit_before_first_checkpoint_or_mid_stage_is_not_reported_as_success(self):
        self.preflight()
        diagnostics = verify.collect_prepare_diagnostics()
        self.assertEqual(diagnostics["availability"], "not_started")
        self.assertIsNone(diagnostics["last_successful_stage"])
        # Simulate a process dying after its atomic started write, without an exception handler.
        verify.write_checkpoints({"version": 1, "mount_verified": True, "events": [{"stage": "guard", "state": "started"}]})
        diagnostics = verify.collect_prepare_diagnostics()
        self.assertEqual(diagnostics["active_stage"], "guard")
        self.assertIsNone(diagnostics["failed_stage"])
        self.assertIsNone(diagnostics["error_type"])
        self.assert_private_report(diagnostics)

    def test_error_categories_never_depend_on_exception_messages(self):
        class DatabaseError(Exception):
            pass

        with patch.dict(sys.modules, {"sqlalchemy.exc": SimpleNamespace(SQLAlchemyError=DatabaseError)}):
            cases = ((ImportError, "import_error"), (PermissionError, "permission_denied"),
                     (FileNotFoundError, "file_missing"), (DatabaseError, "database_error"),
                     (AssertionError, "assertion_failed"), (OSError, "os_error"),
                     (ValueError, "invalid_data"), (RuntimeError, "unexpected_error"))
            for kind, category in cases:
                self.assertEqual(verify.prepare_error(kind(self.private)), category)

    def test_invalid_file_schema_sequence_and_secret_fields_are_rejected(self):
        start = {"stage": "guard", "state": "started"}
        failed = {"stage": "guard", "state": "failed", "error_type": "unexpected_error"}
        valid = {"version": 1, "mount_verified": True, "events": [start, failed]}
        invalid = [
            {**valid, "environment": self.private}, {**valid, "version": True},
            {**valid, "mount_verified": False}, {**valid, "events": [start, {**failed, "stderr": self.private}]},
            {**valid, "events": [{**start, "stage": self.private}]},
            {**valid, "events": [{**start, "state": self.private}]},
            {**valid, "events": [start, {**failed, "error_type": self.private}]},
            {**valid, "events": [failed]}, {**valid, "events": [start, start]},
            {**valid, "events": [start, failed, start]}, {**valid, "events": [start] * 17},
        ]
        raw_cases = [json.dumps(value) for value in invalid]
        raw_cases += ['{"version":1,"version":1,"mount_verified":true,"events":[]}',
                      self.private, self.private * 1000, '[' * 1500 + ']' * 1500]
        for raw in raw_cases:
            with self.subTest(index=raw_cases.index(raw)):
                self.path.write_text(raw)
                diagnostics = verify.collect_prepare_diagnostics()
                self.assertEqual(diagnostics, verify.empty_diagnostics("invalid", "invalid_data"))
                self.assert_private_report(diagnostics)

    def test_public_schema_rejects_extra_fields_or_forged_summaries(self):
        self.preflight()
        valid = verify.collect_prepare_diagnostics()
        for invalid in ({**valid, "message": self.private}, {**valid, "active_stage": self.private},
                        {**valid, "events": [{"stage": "guard", "state": "started", "stdout": self.private}]},
                        {**valid, "error_type": self.private}, {**valid, "last_successful_stage": "config_write"}):
            self.assertEqual(self.assert_private_report(invalid), {})

    def test_missing_unreadable_and_symlink_exchange_never_publish_file_contents(self):
        self.path.chmod(0o644)
        self.assertEqual(verify.collect_prepare_diagnostics()["availability"], "invalid")
        self.path.chmod(0o600)
        self.path.parent.chmod(0o755)
        self.assertEqual(verify.collect_prepare_diagnostics()["availability"], "invalid")
        self.path.parent.chmod(0o700)
        self.path.unlink()
        self.assertEqual(verify.collect_prepare_diagnostics(), verify.empty_diagnostics("unavailable", "file_missing"))
        other = self.root / "private-fixture.json"
        other.write_text(self.private)
        self.path.symlink_to(other)
        self.assertEqual(verify.collect_prepare_diagnostics()["availability"], "unavailable")
        self.assert_private_report(verify.collect_prepare_diagnostics())
        self.assertEqual(other.read_text(), self.private)

    def test_parent_collects_checkpoints_even_when_container_fails_or_times_out(self):
        # Exercise parent finally, using the same exchange file as the mocked container.
        self.path.unlink()
        self.path.parent.rmdir()
        run = object.__new__(verify.ComposeRun)
        for error in (verify.CommandFailure("seed and warmup", 1, "command_failed"),
                      verify.CommandFailure("seed and warmup", None, "timeout")):
            with self.subTest(category=error.error_type):
                run.report = {"steps": []}

                def container(label, *args, **kwargs):
                    if label == "prepare diagnostics preflight":
                        self.preflight()
                    else:
                        verify.write_checkpoints({"version": 1, "mount_verified": True, "events": [
                            {"stage": "guard", "state": "started"}, {"stage": "guard", "state": "passed"},
                            {"stage": "imports", "state": "started"},
                            {"stage": "imports", "state": "failed", "error_type": "import_error"}]})
                        raise error

                run.compose = Mock(side_effect=container)
                with self.assertRaises(verify.CommandFailure):
                    run.prepare_in_container()
                diagnostics = json.loads(verify.REPORT.read_text())["prepare_diagnostics"]
                self.assertEqual(diagnostics["last_successful_stage"], "guard")
                self.assertEqual(diagnostics["failed_stage"], "imports")
                self.path.unlink()
                self.path.parent.rmdir()

    def test_parent_stops_before_seed_if_mount_acknowledgment_is_not_visible(self):
        self.path.unlink()
        self.path.parent.rmdir()
        run = object.__new__(verify.ComposeRun)
        run.report = {"steps": []}
        run.compose = Mock()
        with self.assertRaises(AssertionError):
            run.prepare_in_container()
        self.assertEqual(run.compose.call_count, 1)
        diagnostics = json.loads(verify.REPORT.read_text())["prepare_diagnostics"]
        self.assertEqual(diagnostics["availability"], "not_started")
        self.assertFalse(diagnostics["mount_verified"])

    def test_parent_retains_not_started_when_container_exits_before_first_checkpoint(self):
        self.path.unlink()
        self.path.parent.rmdir()
        run = object.__new__(verify.ComposeRun)
        run.report = {"steps": []}

        def container(label, *args, **kwargs):
            if label == "prepare diagnostics preflight":
                self.preflight()
            else:
                raise verify.CommandFailure("seed and warmup", 137, "command_failed")

        run.compose = Mock(side_effect=container)
        with self.assertRaises(verify.CommandFailure):
            run.prepare_in_container()
        diagnostics = json.loads(verify.REPORT.read_text())["prepare_diagnostics"]
        self.assertEqual(diagnostics["availability"], "not_started")
        self.assertTrue(diagnostics["mount_verified"])
        self.assertIsNone(diagnostics["failed_stage"])

    def test_parent_accepts_complete_preparation_and_rejects_a_success_exit_without_checkpoints(self):
        self.path.unlink()
        self.path.parent.rmdir()
        for complete in (True, False):
            with self.subTest(complete=complete):
                run = object.__new__(verify.ComposeRun)
                run.report = {"steps": []}

                def container(label, *args, **kwargs):
                    if label == "prepare diagnostics preflight":
                        self.preflight()
                    elif complete:
                        self.run_prepare()

                run.compose = Mock(side_effect=container)
                if complete:
                    run.prepare_in_container()
                    self.assertEqual(run.report["prepare_diagnostics"]["last_successful_stage"], "config_write")
                else:
                    with self.assertRaises(AssertionError):
                        run.prepare_in_container()
                    self.assertEqual(run.report["prepare_diagnostics"]["availability"], "not_started")
                self.assertEqual(run.compose.call_count, 2)
                self.path.unlink()
                self.path.parent.rmdir()

    def test_failed_diagnostic_write_preserves_original_error_and_last_atomic_record(self):
        self.preflight()
        progress = verify.PrepareProgress()
        original = RuntimeError(self.private)
        write = verify.write_checkpoints

        def unwritable(value):
            if value["events"][-1]["state"] == "failed":
                raise PermissionError(self.private)
            write(value)

        with patch.object(verify, "write_checkpoints", side_effect=unwritable), \
                self.assertRaises(RuntimeError) as raised, progress.step("guard"):
            raise original
        self.assertIs(raised.exception, original)
        diagnostics = verify.collect_prepare_diagnostics()
        self.assertEqual(diagnostics["active_stage"], "guard")
        self.assertIsNone(diagnostics["failed_stage"])
        self.assert_private_report(diagnostics)

    def test_existing_compose_mount_and_uid_gid_support_the_exchange(self):
        # ROOT is patched for fixtures; use the checked-in Compose file explicitly.
        compose = yaml.safe_load((Path(__file__).resolve().parents[1] / "docker-compose.yml").read_text())
        app = compose["services"]["app"]
        self.assertIn("./models:/app/models", app["volumes"])
        self.assertEqual(app["user"], "${APP_UID:-1000}:${APP_GID:-1000}")
        self.assertEqual(compose["services"]["inference"]["volumes"], ["./models:/app/models:ro"])


if __name__ == "__main__":
    unittest.main()
