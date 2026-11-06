"""Offline safety/result-policy tests; no simulated result is CI evidence."""

import copy
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from datetime import datetime, timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import yaml

from ml.inference_contract import canonical_ts
from scripts import verify_compose_demo as verify

# Committed offline evaluation, projected onto the report fields it may expose. It
# predates the numpy_simd field; its scores belong to the X86_V4 reference.
_EVIDENCE = verify.read(verify.DEMO_MODEL_EVIDENCE)
DEMO_EVALUATION = {key: value for key, value in _EVIDENCE.items() if key in verify.DEMO_MODEL}
DEMO_EVALUATION["environment"] = {**_EVIDENCE["environment"], "numpy_simd": "X86_V4"}
REPRODUCED_ALL = dict.fromkeys(verify.REPRODUCED, True)


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

    def test_outcome_command_also_requires_the_demo_model_test_targets(self):
        failing = copy.deepcopy(DEMO_EVALUATION)
        failing["test"]["targets_passed"] = False
        for demo_model, passes in ((DEMO_EVALUATION, True), (failing, False), (None, False)):
            with self.subTest(passes=passes), tempfile.TemporaryDirectory() as temporary:
                report = Path(temporary) / "report.json"
                value = {"technical_pass": True, "model_scenario": {"passed": True}, "prepared": {},
                         "demo_model_reproduced": REPRODUCED_ALL}
                if demo_model is not None:
                    value["prepared"]["demo_model"] = demo_model
                report.write_text(json.dumps(value))
                with patch.object(verify, "REPORT", report), \
                        patch.object(verify.sys, "argv", ["verify", "check-outcome"]), redirect_stdout(io.StringIO()):
                    if passes:
                        verify.main()
                    else:
                        with self.assertRaisesRegex(SystemExit, "independent test split"):
                            verify.main()

    def test_committed_demo_model_evidence_fits_the_report_schema_and_reproduction_check(self):
        evidence = verify.read(verify.DEMO_MODEL_EVIDENCE)
        public = verify.public_report({"prepared": {"demo_model": evidence}})["prepared"]["demo_model"]
        self.assertEqual(public, {key: value for key, value in evidence.items() if key in verify.DEMO_MODEL})
        self.assertTrue(public["test"]["targets_passed"])
        self.assertEqual(verify.public_report({"prepared": {"demo_model": DEMO_EVALUATION}})
                         ["prepared"]["demo_model"], DEMO_EVALUATION)
        self.assertEqual(verify.reproduced(DEMO_EVALUATION), REPRODUCED_ALL)
        changed = {**DEMO_EVALUATION, "scores_sha256": "0" * 64}
        self.assertFalse(verify.reproduced(changed)["scores_sha256"])
        # Without a recorded dispatch level no reference applies.
        self.assertFalse(verify.reproduced(evidence)["scores_sha256"])
        metrics = copy.deepcopy(DEMO_EVALUATION)
        metrics["test"]["metrics"]["false_alarms"] += 1
        self.assertEqual(verify.reproduced(metrics), {**REPRODUCED_ALL, "test": False})
        with self.assertRaisesRegex(AssertionError, "recorded generator"):
            verify.reproduced({**evidence, "test_data_sha256": "0" * 64})

    def test_scores_match_only_the_reference_of_their_numerical_environment(self):
        references = verify.read(verify.DEMO_SCORE_REFERENCES)
        historical = references["historical"]["scores_sha256"]
        # The historical result is kept and remains the reference of its own environment.
        self.assertEqual(historical, _EVIDENCE["scores_sha256"])
        self.assertEqual(verify.reference_scores(DEMO_EVALUATION["environment"]), historical)
        hashes = {entry["scores_sha256"] for entry in references["references"]}
        self.assertEqual(len(hashes), 2)
        for entry in references["references"]:
            with self.subTest(numpy_simd=entry["environment"]["numpy_simd"]):
                environment = {**entry["environment"], "python": entry["environment"]["python"] + ".16"}
                actual = {**DEMO_EVALUATION, "environment": environment, "scores_sha256": entry["scores_sha256"]}
                self.assertEqual(verify.reproduced(actual), REPRODUCED_ALL)
                for other in hashes - {entry["scores_sha256"]}:
                    self.assertFalse(verify.reproduced({**actual, "scores_sha256": other})["scores_sha256"])
        for change in ({"numpy_simd": "other"}, {"numpy": "2.5.5"}, {"scikit-learn": "1.7.0"},
                       {"skops": "0.17.0"}, {"python": "3.13.1"}):
            with self.subTest(change=change):
                environment = {**DEMO_EVALUATION["environment"], **change}
                self.assertIsNone(verify.reference_scores(environment))
                self.assertFalse(verify.reproduced({**DEMO_EVALUATION, "environment": environment})["scores_sha256"])
        without = {key: value for key, value in DEMO_EVALUATION.items() if key != "environment"}
        self.assertFalse(verify.reproduced(without)["scores_sha256"])

    def test_outcome_command_fails_when_the_demo_model_is_not_reproduced(self):
        for reproduction, passes in ((REPRODUCED_ALL, True), ({**REPRODUCED_ALL, "scores_sha256": False}, False),
                                     ({**REPRODUCED_ALL, "test": False}, False),
                                     ({key: True for key in verify.REPRODUCED if key != "scores_sha256"}, False),
                                     (None, False)):
            with self.subTest(reproduction=reproduction), tempfile.TemporaryDirectory() as temporary:
                report = Path(temporary) / "report.json"
                value = {"technical_pass": True, "model_scenario": {"passed": True},
                         "prepared": {"demo_model": DEMO_EVALUATION}}
                if reproduction is not None:
                    value["demo_model_reproduced"] = reproduction
                report.write_text(json.dumps(value))
                with patch.object(verify, "REPORT", report), \
                        patch.object(verify.sys, "argv", ["verify", "check-outcome"]), redirect_stdout(io.StringIO()):
                    if passes:
                        verify.main()
                    else:
                        with self.assertRaisesRegex(SystemExit, "did not reproduce"):
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
        evaluation = {**DEMO_EVALUATION, "log": self.private}
        registry.stage_export = Mock(side_effect=AssertionError("#9 export is no longer the demo model"))
        scripts.demo_model = SimpleNamespace(
            train_and_evaluate=Mock(side_effect=action("demo_model", (manifest, evaluation))))
        scripts.seed_target_db = SimpleNamespace(main=Mock(side_effect=action("target_seed")))
        summary = {"retail-postgres": {"metrics": 336, "anomaly_scores": 335, "forecast_models": 1}}
        if failed == "summary_check":
            summary["retail-postgres"]["anomaly_scores"] = 0
        scripts.demo_prepare = SimpleNamespace(main=Mock(side_effect=action("demo_prepare", summary)))
        modules = {"app": None if failed == "imports" else app, "ml": ml,
                   "ml.model_registry": registry, "scripts": scripts}
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, modules))
            stack.enter_context(patch.dict(os.environ, {} if failed == "guard" else
                                           {"COMPOSE_DEMO_TEST": "1", "APP_VERSION": "d" * 40}, clear=True))
            stack.enter_context(patch.object(verify.Path, "exists", return_value=True))
            stack.enter_context(patch.object(verify, "container_state", side_effect=action("source_check", source)))
            if failed == "config_write":
                stack.enter_context(patch.object(verify, "save", side_effect=PermissionError(self.private)))
            verify.prepare()
        scripts.seed_target_db.main.assert_called_once_with(n_users=500, n_products=50, n_orders=1000, n_events=2000)
        self.assertEqual(calls, ["target_seed", "demo_prepare", "source_check", "demo_model"])
        scripts.demo_model.train_and_evaluate.assert_called_once_with(self.root / "models/registry", source, "d" * 40)
        registry.stage_export.assert_not_called()
        self.assertEqual(app.metrics_storage.set_connection_active.call_count, 2)
        prepared = verify.read(verify.DEMO / "prepared.json")
        self.assertTrue(prepared["demo"])
        self.assertFalse(prepared["quality_admitted"])
        self.assertEqual(prepared["binding"]["model_version"], manifest["model_version"])
        self.assertNotIn("baseline_fpr", prepared)
        self.assertNotIn("log", prepared["demo_model"])
        self.assertEqual(verify.public_report({"prepared": prepared})["prepared"]["demo_model"], DEMO_EVALUATION)
        self.assertEqual(verify.read(verify.DEMO / "evaluation.json")["log"], self.private)

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

    def run_demo_fixture(self, checkpoint=None, failed=None):
        """Real demo orchestrator with isolated, mocked business dependencies."""
        class DatabaseError(Exception):
            pass

        error = DatabaseError(self.private)
        error.orig = SimpleNamespace(pgcode="42703", sqlstate=None, statement=self.private, params=self.private)
        calls = []

        def operation(stage, result=None):
            def execute(*args, **kwargs):
                calls.append(stage)
                if stage == failed:
                    raise error
                return result
            return Mock(side_effect=execute)

        sqlalchemy = ModuleType("sqlalchemy")
        sqlalchemy.text = Mock()
        storage = ModuleType("app.metrics_storage")
        storage.get_engine = Mock(side_effect=AssertionError("no real database in offline tests"))
        workspace = ModuleType("scripts.seed_demo_workspace")
        workspace.seed_demo_workspace = operation("workspace")
        scripts = ModuleType("scripts")
        scripts.seed_metrics_db = SimpleNamespace(main=operation("seed_metrics", {"rows": 1}))
        scripts.warmup_ml = SimpleNamespace(main=operation("warmup_ml", {}))
        modules = {"sqlalchemy": sqlalchemy, "sqlalchemy.exc": SimpleNamespace(SQLAlchemyError=DatabaseError, DBAPIError=DatabaseError),
                   "app": ModuleType("app"), "app.metrics_storage": storage,
                   "scripts": scripts, "scripts.seed_demo_workspace": workspace}
        path = Path(__file__).resolve().parents[1] / "scripts/demo_prepare.py"
        demo = ModuleType("fixture_demo_prepare")
        demo.__file__ = str(path)
        with patch.dict(sys.modules, modules), patch("builtins.print"), patch("logging.basicConfig"):
            exec(compile(path.read_text(), str(path), "exec"), demo.__dict__)
            demo.logger = Mock()
            demo._resolve_project_ids = operation("resolve_project", {"retail-postgres": "a" * 32})
            demo.verify_project = operation("verify_project", {"metrics": 1, "anomaly_scores": 1, "changepoints": 0,
                                                              "drift_reports": 0, "notifications": 0, "forecast_models": 1})
            # The real outer handler runs while the fake SQLAlchemy types are installed.
            if checkpoint is None:
                result = demo.main()
            else:
                progress = checkpoint
                with progress.step("demo_prepare"):
                    result = demo.main(checkpoint=progress.demo_checkpoint)
        scripts.seed_metrics_db.main.assert_called_once_with(days=14, interval_minutes=60, reset=True, project_id="a" * 32)
        scripts.warmup_ml.main.assert_called_once_with(project_id="a" * 32)
        self.assertEqual(calls, list(verify.DEMO_PREPARE_STAGES))
        return result

    def demo_progress(self):
        verify.write_checkpoints({"version": 1, "mount_verified": True, "events": []})
        progress = verify.PrepareProgress()
        for stage in ("guard", "imports", "target_seed"):
            with progress.step(stage):
                pass
        return progress

    def test_demo_callbacks_preserve_default_operations_and_report_last_internal_stage(self):
        self.run_demo_fixture()  # Default entrypoint remains usable without diagnostics.
        progress = self.demo_progress()
        self.run_demo_fixture(checkpoint=progress)
        diagnostics = verify.collect_prepare_diagnostics()
        self.assertEqual(diagnostics["last_successful_stage"], "demo_prepare")
        self.assertEqual(diagnostics["demo_prepare_stage"], "verify_project")
        self.assertIsNone(diagnostics["failed_stage"])
        self.assertIsNone(diagnostics["sqlstate"])
        self.assert_private_report(diagnostics)

    def test_demo_failure_reports_each_internal_stage_and_only_allowlisted_sqlstate(self):
        for stage in verify.DEMO_PREPARE_STAGES:
            with self.subTest(stage=stage):
                progress = self.demo_progress()
                with self.assertRaises(Exception):
                    self.run_demo_fixture(checkpoint=progress, failed=stage)
                diagnostics = verify.collect_prepare_diagnostics()
                self.assertEqual(diagnostics["failed_stage"], "demo_prepare")
                self.assertEqual(diagnostics["last_successful_stage"], "target_seed")
                self.assertEqual(diagnostics["demo_prepare_stage"], stage)
                self.assertEqual(diagnostics["error_type"], "database_error")
                self.assertEqual(diagnostics["sqlstate"], "42703")
                self.assert_private_report(diagnostics)
                self.assertNotIn("fixture", self.path.read_text())

    def test_sqlstate_extraction_ignores_arbitrary_driver_text_and_non_dbapi_errors(self):
        class DatabaseError(Exception):
            pass

        with patch.dict(sys.modules, {"sqlalchemy.exc": SimpleNamespace(DBAPIError=DatabaseError)}):
            for value in (self.private, "ADMIN", "ZZ123", "42703 extra", 42703, None):
                error = DatabaseError(self.private)
                error.orig = SimpleNamespace(sqlstate=value, pgcode=value)
                self.assertIsNone(verify.prepare_sqlstate(error))
            for field in ("sqlstate", "pgcode"):
                error = DatabaseError(self.private)
                error.orig = SimpleNamespace(**{field: "42P01"})
                self.assertEqual(verify.prepare_sqlstate(error), "42P01")
            other = RuntimeError(self.private)
            other.orig = SimpleNamespace(pgcode="42P01")
            self.assertIsNone(verify.prepare_sqlstate(other))

    def test_internal_stage_and_sqlstate_fields_are_strictly_scoped_and_filtered(self):
        progress = self.demo_progress()
        with progress.step("demo_prepare"):
            progress.demo_checkpoint("workspace")
        value = verify.read_checkpoints()
        start, passed = value["events"][-2:]
        invalid_pairs = [
            ({**start, "substage": self.private}, passed),
            ({**start, "sqlstate": "42P01"}, passed),
            (start, {**passed, "substage": "workspace"}),
            (start, {"stage": "demo_prepare", "state": "failed", "error_type": "database_error", "sqlstate": self.private}),
            (start, {"stage": "demo_prepare", "state": "failed", "error_type": "unexpected_error", "sqlstate": "42P01"}),
        ]
        for pair in invalid_pairs:
            self.path.write_text(json.dumps({**value, "events": value["events"][:-2] + list(pair)}))
            diagnostics = verify.collect_prepare_diagnostics()
            self.assertEqual(diagnostics["availability"], "invalid")
            self.assert_private_report(diagnostics)
        valid = verify.summarize_checkpoints(value)
        for forged in ({**valid, "sqlstate": self.private}, {**valid, "demo_prepare_stage": self.private}):
            self.assertEqual(self.assert_private_report(forged), {})

    def test_demo_checkpoint_requires_an_active_demo_stage_and_known_identifier(self):
        progress = self.demo_progress()
        with self.assertRaises(ValueError):
            progress.demo_checkpoint("workspace")
        with progress.step("demo_prepare"):
            with self.assertRaises(ValueError):
                progress.demo_checkpoint(self.private)
        self.assertNotIn("fixture", self.path.read_text())


SOURCE = {"project_id": "a" * 32, "connection_id": "b" * 32, "schema_name": "public", "table_name": "events"}
BINDING = {"source": SOURCE, "model_version": "if-" + "c" * 64, "feature_schema_version": "1.0.0"}
ROW = {**BINDING, "ts": "2026-11-03T03:00:00+00:00", "score": -0.2, "is_anomaly": True,
       "demo": True, "quality_admitted": False,
       "features": {"row_count": 2000, "null_rate": 0.1, "d_row_count": 200, "d_null_rate": 0.01}}


class NetworkProbeTests(unittest.TestCase):
    """HTTP checks run inside the internal Compose network, never via host-published ports."""

    private = "fixture-private-token https://fixture.invalid/private"
    token = "fixture-adapter-token"
    sha = "a" * 40

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        for name, value in (("ROOT", self.root), ("DEMO", self.root / "models/demo"),
                            ("STATE", self.root / "state.json"), ("OVERRIDE", self.root / "override.json"),
                            ("REPORT", self.root / "report.json")):
            replacement = patch.object(verify, name, value)
            replacement.start()
            self.addCleanup(replacement.stop)
        verify.save(verify.DEMO / "prepared.json", {"binding": BINDING, "warmup": {}, "demo": True,
                                                    "quality_admitted": False, "fernet_key_shared": True,
                                                    "demo_model": DEMO_EVALUATION})
        self.calls = []

    def fake_httpx(self, handler):
        calls = self.calls

        class Response:
            def __init__(self, status, body=None, text="", url=""):
                self.status_code, self.body, self.text, self.url = status, body, text, url

            def json(self):
                return self.body

        class Client:
            def __init__(self, base_url="", **kwargs):
                self.base_url = base_url
                assert kwargs.get("trust_env") is False

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def request(self, method, url, headers=None, **kwargs):
                calls.append((method, self.base_url + url))
                return Response(*handler(method, self.base_url + url, headers or {}))

            def get(self, url, **kwargs):
                return self.request("GET", url, **kwargs)

            def post(self, url, **kwargs):
                return self.request("POST", url, **kwargs)

        module = ModuleType("httpx")
        module.Client = Client
        return module

    def inside(self, handler):
        stack = ExitStack()
        stack.enter_context(patch.dict(sys.modules, {"httpx": self.fake_httpx(handler)}))
        stack.enter_context(patch.dict(os.environ, {"COMPOSE_DEMO_TEST": "1", "INFERENCE_ADAPTER_TOKEN": self.token}, clear=True))
        stack.enter_context(patch.object(verify.Path, "exists", return_value=True))
        return stack

    def service(self, health=200, unauthorized=401, model=None):
        model = {**BINDING, "demo": True, "quality_admitted": False, "error": self.private} if model is None else model

        def handler(method, url, headers):
            if url == "http://app:5001/healthz":
                return health, {"status": "ok", "version": self.sha, "checks": {
                    "target_db": {"status": "ok"}, "monitor_db": {"status": "ok", "backend": "postgresql"}}}
            if url == "http://inference:8000/v1/model":
                if headers.get("Authorization") == "Bearer " + self.token:
                    return 200, model
                return unauthorized, {"error": {"code": "authentication_required"}}
            raise AssertionError("unexpected URL")
        return handler

    def dashboard_service(self, rows, logged_url="http://app:5001/dashboard"):
        page = " ".join((SOURCE["connection_id"], BINDING["model_version"], rows[-1]["ts"], str(rows[-1]["score"])))

        def handler(method, url, headers):
            responses = {
                ("GET", "http://app:5001/auth/login"): (200, None, '<input name="csrf_token" type="hidden" value="fixture-csrf">'),
                ("POST", "http://app:5001/auth/login"): (200, None, "", logged_url),
                ("POST", "http://app:5001/projects/retail-postgres/switch"): (200,),
                ("GET", "http://app:5001/dashboard/inference"): (200, None, page),
                ("GET", "http://app:5001/api/inference/scores"): (200, [{"ts": rows[-1]["ts"], "score": rows[-1]["score"]}]),
            }
            return responses[(method, url)]
        return handler

    def test_readiness_uses_service_names_and_keeps_only_contract_fields(self):
        with self.inside(self.service()):
            verify.readiness()
        self.assertEqual([url for _, url in self.calls], ["http://app:5001/healthz", "http://inference:8000/v1/model",
                                                          "http://inference:8000/v1/model"])
        content = (verify.DEMO / "readiness.json").read_text()
        self.assertNotIn(self.token, content)
        self.assertNotIn("fixture", content)
        self.assertEqual(json.loads(content), {
            "version": self.sha, "target_db": "ok", "monitor_db": "ok", "monitor_backend": "postgresql",
            "unauthorized_model_status": 401, "model": {**BINDING, "demo": True, "quality_admitted": False}})

    def test_readiness_keeps_status_authorization_and_demo_assertions(self):
        cases = {"unhealthy": {"health": 503}, "unauthenticated_access": {"unauthorized": 200},
                 "admitted_model": {"model": {**BINDING, "demo": True, "quality_admitted": True}},
                 "not_demo": {"model": {**BINDING, "demo": False, "quality_admitted": False}}}
        for name, options in cases.items():
            with self.subTest(name), self.inside(self.service(**options)):
                with self.assertRaises(AssertionError):
                    verify.readiness()
                self.assertFalse((verify.DEMO / "readiness.json").is_file())

    def test_http_probes_refuse_outside_the_isolated_test_container(self):
        with patch.dict(sys.modules, {"httpx": self.fake_httpx(self.service())}), patch.dict(os.environ, {}, clear=True):
            for probe in (verify.readiness, lambda: verify.dashboard("incident")):
                with self.assertRaises(RuntimeError):
                    probe()
        self.assertEqual(self.calls, [])

    def test_dashboard_reads_each_phase_and_writes_a_separate_result(self):
        for phase in ("incident", "after-recreate"):
            with self.subTest(phase):
                rows = [{**ROW, "ts": f"2026-11-03T0{index}:00:00+00:00", "score": -0.1 * index}
                        for index in (1, 2 if phase == "incident" else 3)]
                verify.save(verify.DEMO / (phase + ".json"), rows)
                self.calls.clear()
                with self.inside(self.dashboard_service(rows)):
                    verify.dashboard(phase)
                self.assertTrue(all(url.startswith("http://app:5001/") for _, url in self.calls))
                self.assertEqual(verify.read(verify.DEMO / ("dashboard-" + phase + ".json")), {
                    "status": 200, "csrf_login": True, "source": SOURCE, "version": BINDING["model_version"],
                    "observed_at": rows[-1]["ts"], "score_visible": True, "scope": "HTTP HTML/API, not browser rendering"})

    def test_dashboard_keeps_login_and_visibility_assertions(self):
        rows = [ROW]
        verify.save(verify.DEMO / "incident.json", rows)
        with self.inside(self.dashboard_service(rows, logged_url="http://app:5001/auth/login")):
            with self.assertRaises(AssertionError):
                verify.dashboard("incident")
        with self.inside(self.dashboard_service([{**ROW, "score": 0.9}])):
            with self.assertRaises(AssertionError):
                verify.dashboard("incident")
        with self.inside(self.dashboard_service(rows)), self.assertRaises(ValueError):
            verify.dashboard("../prepared")
        self.assertFalse((verify.DEMO / "dashboard-incident.json").is_file())

    def test_inside_modes_dispatch_to_container_probes(self):
        for argument, name, args in (("inside-readiness", "readiness", ()),
                                     ("inside-dashboard-incident", "dashboard", ("incident",)),
                                     ("inside-dashboard-after-recreate", "dashboard", ("after-recreate",))):
            with self.subTest(argument), patch.object(verify, name) as probe, \
                    patch.object(verify.sys, "argv", ["verify", argument]):
                verify.main()
            probe.assert_called_once_with(*args)

    def host_run(self, readiness_version=None):
        """Execute the real host run() with Docker commands mocked; container probes write their files."""
        image_id = "sha256:" + "d" * 64
        outputs = {
            "readiness": {"version": readiness_version or self.sha, "target_db": "ok", "monitor_db": "ok",
                          "monitor_backend": "postgresql", "unauthorized_model_status": 401,
                          "model": {**BINDING, "demo": True, "quality_admitted": False}},
            "normal": [{**ROW, "is_anomaly": False}], "incident": [ROW], "after-recreate": [ROW],
            "snapshot": {"source": SOURCE, "fernet_decryption": True, "target": {}, "store": {}, "artifacts": {}},
        }
        dashboard = {"status": 200, "csrf_login": True, "source": SOURCE, "version": BINDING["model_version"],
                     "observed_at": ROW["ts"], "score_visible": True, "scope": "HTTP HTML/API, not browser rendering"}
        outputs["dashboard-incident"] = outputs["dashboard-after-recreate"] = dashboard
        commands = []

        def docker(args, **kwargs):
            # The real command()/compose()/probe() set the stage label before each call.
            label = run.stage
            commands.append((label, args))
            inside = next((arg.removeprefix("inside-") for arg in args if arg.startswith("inside-")), None)
            if inside is not None:
                verify.save(verify.DEMO / (inside + ".json"), outputs[inside])
            stdout = {"checkout SHA": self.sha, "Docker version": "28.0.0 / 28.0.0", "Compose version": "2.30.0",
                      "inspect image": json.dumps([{"Id": image_id, "Config": {"Env": ["APP_VERSION=" + self.sha]}}]),
                      "app container ID": "app-id", "inference container ID": "inference-id",
                      "app image ID": image_id, "inference image ID": image_id}.get(label, "")
            return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

        run = object.__new__(verify.ComposeRun)
        run.values, run.stage = {}, "verification"
        run.report = {"technical_pass": False, "steps": [], "manual_browser_and_second_participant": "NOT VERIFIED"}
        run.prepare_in_container = Mock()
        self.host = run
        host_httpx = ModuleType("httpx")
        host_httpx.Client = Mock(side_effect=AssertionError("host HTTP request"))
        with patch.dict(os.environ, {"COMPOSE_SMOKE_SHA": self.sha, "COMPOSE_SMOKE_IMAGE": "course13-ci:1",
                                     "COMPOSE_SMOKE_PR_HEAD": "b" * 40, "GITHUB_RUN_ID": "1"}), \
                patch.dict(sys.modules, {"httpx": host_httpx}), patch.object(verify.subprocess, "run", side_effect=docker):
            try:
                run.run()
            finally:
                host_httpx.Client.assert_not_called()
        return run, commands

    def test_host_run_uses_container_probes_and_no_published_host_ports(self):
        run, commands = self.host_run()
        self.assertTrue(run.report["technical_pass"])
        self.assertEqual(run.report["demo_model_reproduced"], REPRODUCED_ALL)
        self.assertEqual(verify.read(verify.OVERRIDE)["networks"], {"default": {"internal": True}})
        probes = [(label, args[args.index("scripts.verify_compose_demo") + 1]) for label, args in commands
                  if "scripts.verify_compose_demo" in args and label != "prepare diagnostics preflight"]
        self.assertEqual(probes, [("readiness", "inside-readiness"), ("normal", "inside-normal"),
                                  ("incident", "inside-incident"), ("dashboard", "inside-dashboard-incident"),
                                  ("snapshot", "inside-snapshot"), ("snapshot", "inside-snapshot"),
                                  ("snapshot", "inside-snapshot"), ("after-recreate", "inside-after-recreate"),
                                  ("dashboard after recreate", "inside-dashboard-after-recreate")])
        self.assertFalse(any("127.0.0.1" in str(args) for _, args in commands))
        self.assertEqual(run.report["readiness"], {"target_db": "ok", "monitor_db": "ok", "monitor_backend": "postgresql",
                                                   "unauthorized_model_status": 401,
                                                   "model": {**BINDING, "demo": True, "quality_admitted": False}})
        public = verify.public_report(run.report)
        for key in ("readiness", "dashboard", "dashboard_after_recreate"):
            self.assertEqual(public[key], run.report[key])

    def test_failed_observation_probe_adds_its_diagnostics_to_the_report(self):
        run = object.__new__(verify.ComposeRun)
        run.report = {"technical_pass": False, "steps": []}

        def failing_probe(mode, label=None):
            verify.save(verify.DEMO / (mode + "-diagnostics.json"),
                        {"check": "window", "error_type": "AssertionError", "message": "fixture-private-text"})
            raise verify.CommandFailure(mode, 1, "command_failed")

        run.probe = failing_probe
        with self.assertRaises(verify.CommandFailure):
            run.observe_probe("after-recreate")
        self.assertEqual(run.report["observe_diagnostics"]["after-recreate"]["check"], "window")
        content = verify.REPORT.read_text()
        self.assertNotIn("fixture-private-text", content)
        self.assertEqual(json.loads(content)["observe_diagnostics"],
                         {"after-recreate": {"check": "window", "error_type": "AssertionError"}})

    def test_host_run_still_requires_the_checked_application_version(self):
        with self.assertRaises(AssertionError) as raised:
            self.host_run(readiness_version="f" * 40)
        self.assertEqual(verify.failure_fields(raised.exception, self.host.stage),
                         {"stage": "readiness", "exit_code": None, "error_type": "assertion_failed"})
        self.assertNotIn("readiness", self.host.report)


class ObservationPhaseTests(unittest.TestCase):
    """Each phase keeps exactly its own ticks; n_live_tup is never treated as COUNT(*)."""

    start = "2026-11-04T10:00:10.000000+00:00"
    end = "2026-11-04T10:00:20.000000+00:00"
    previous = "2026-11-04T10:00:05.000000+00:00"
    baseline = "2026-11-04T10:00:07.000000+00:00"
    ticks = ("2026-11-04T10:00:11.000000+00:00", "2026-11-04T10:00:13.000000+00:00",
             "2026-11-04T10:00:15.000000+00:00")
    late = "2026-11-04T10:00:25.000000+00:00"
    stored_ticks = None

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        demo = Path(directory.name) / "models/demo"
        replacement = patch.object(verify, "DEMO", demo)
        replacement.start()
        self.addCleanup(replacement.stop)
        verify.save(demo / "prepared.json", {"binding": BINDING})
        verify.save(demo / "adapter.json", {"url": "http://inference:8000"})

    def observe(self, phase, values, score_ts, counts, *, baseline=True, prior=None, served=None, live_demo_exit=None):
        """Run the real observe() with storage, collector, subprocess, clock and HTTP replaced.

        Seeded history is never part of the source; a normal run starts with no
        source metrics unless ``prior`` is set, and the incident follows a scored tick.
        """
        prior = phase != "normal" if prior is None else prior
        order = (self.previous, *self.ticks, self.late)
        row_counts, null_rates = [], []
        self.collected = []

        def collect(timestamp, value):
            row_counts.append({"ts": timestamp, "value": value})
            null_rates.append({"ts": timestamp, "value": 0.02})

        if prior:
            collect(self.previous, 2000)

        def baseline_collection(project_id, connection_id):
            self.collected.append((project_id, connection_id, len(run.call_args_list)))
            if baseline:
                collect(self.baseline, 2000)

        def live_demo(command, **kwargs):
            # Optionally store the ticks where collections after the window wait land.
            stamps = dict(zip(self.ticks, self.stored_ticks(sum(self.slept)))) if self.stored_ticks else {}
            renamed = {stamps.get(ts, ts): {**row, "ts": stamps.get(ts, ts)} for ts, row in scores.items()}
            scores.clear()
            scores.update(renamed)
            for timestamp, value in zip((*self.ticks[:len(values)], self.late), (*values, 9999), strict=True):
                collect(stamps.get(timestamp, timestamp), value)
            if live_demo_exit is not None:
                raise subprocess.CalledProcessError(live_demo_exit, command)

        scored = ((self.previous,) if prior else ()) + (*score_ts, self.late)
        scores = {ts: {**BINDING, "ts": ts, "score": -0.1 - order.index(ts) / 10,
                       "is_anomaly": order.index(ts) % 2 == 0, "demo": True} for ts in scored}
        storage = SimpleNamespace(metrics=lambda source, name, **kwargs: sorted(
                                      row_counts if name == "row_count" else null_rates, key=lambda row: row["ts"]),
                                  scores=lambda source, version, schema: [scores[ts] for ts in sorted(scores)])
        self.requests = []

        class Response:
            def __init__(self, body, status_code=200):
                self.body = body
                self.status_code = status_code

            def json(self):
                return self.body

        class Client:
            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def get(self, url, **kwargs):
                requests.append(("GET", url))
                return Response({}, 200)

            def post(self, url, json, headers):
                assert url == "http://inference:8000/v1/predict"
                requests.append(("POST", json["observed_at"]))
                if json["observed_at"] not in (served or scores):
                    return Response({}, 422)
                row = (served or scores)[json["observed_at"]]
                self.previous_seen.append(json["previous_observed_at"])
                return Response({**BINDING, "demo": True, "quality_admitted": False, "observed_at": row["ts"],
                                 "score": row["score"], "is_anomaly": row["is_anomaly"]})

        requests = self.requests
        Client.previous_seen = self.previous_seen = []
        clock = Mock()
        clock.now.side_effect = [datetime.fromisoformat(self.start), datetime.fromisoformat(self.end)]
        self.slept = []
        modules = {"app": SimpleNamespace(source_storage=storage),
                   "httpx": SimpleNamespace(Client=Client),
                   "collectors.per_project": SimpleNamespace(collect_for_connection=baseline_collection),
                   "scripts.live_demo": SimpleNamespace(_BASE_ROWS_PER_TICK=200, _INCIDENT_ROW_MULTIPLIER=10)}
        with patch.dict(sys.modules, modules), patch.object(verify, "container_state", return_value=SOURCE), \
                patch.object(verify, "table_count", side_effect=counts), patch.object(verify, "datetime", clock), \
                patch.object(verify.subprocess, "run", side_effect=live_demo) as run, \
                patch.object(verify.time, "sleep", side_effect=self.slept.append), \
                patch.dict(os.environ, {"INFERENCE_ADAPTER_TOKEN": "fixture-adapter-token"}):
            self.run = run  # inspected after a refused phase
            verify.observe(phase)
        self.command = run.call_args.args[0]
        return verify.read(verify.DEMO / (phase + ".json"))

    def test_each_normal_tick_maps_to_its_exact_count_inside_the_run_window(self):
        rows = self.observe("normal", (2200, 2400, 2600), self.ticks, (2000, 2600))
        self.assertEqual([row["ts"] for row in rows], list(self.ticks))
        self.assertEqual([row["exact_row_count"] for row in rows], [2200, 2400, 2600])
        self.assertEqual([row["features"]["row_count"] for row in rows], [2200, 2400, 2600])
        self.assertEqual(self.command[self.command.index("--ticks") + 1], "3")
        self.assertNotIn("--incident-at", self.command)

    def test_window_opens_on_the_next_whole_second_after_waiting_for_it(self):
        for now, opened, waited in (("2026-11-04T10:00:10.600000+00:00", "2026-11-04T10:00:11.000000+00:00", 0.4),
                                    ("2026-11-04T10:00:10.000000+00:00", "2026-11-04T10:00:11.000000+00:00", 1.0),
                                    ("2026-11-04T10:00:59.999999+00:00", "2026-11-04T10:01:00.000000+00:00", 1e-06)):
            with self.subTest(now), patch.object(verify, "datetime", Mock(now=Mock(return_value=datetime.fromisoformat(now)))), \
                    patch.object(verify.time, "sleep") as sleep:
                self.assertEqual(verify.window_start(), opened)
                self.assertAlmostEqual(sleep.call_args.args[0], waited)

    def test_first_tick_in_the_starting_second_stays_inside_the_window(self):
        # CI run 42239736474: the window opened mid-second and a first tick
        # collected in that second was stored before it, losing one observation.
        from app.metrics_storage import _iso

        self.start = "2026-11-04T10:00:10.600000+00:00"
        self.baseline = "2026-11-04T10:00:10.000000+00:00"  # collected at 10:00:10.2

        def stored_ticks(slept):
            # live_demo collects 0.35 s after the window opens, then every 2 s;
            # storage keeps whole seconds only.
            opened = datetime.fromisoformat(self.start) + timedelta(seconds=slept)
            return tuple(canonical_ts(_iso(opened + timedelta(seconds=0.35 + 2 * tick))) for tick in range(3))

        self.stored_ticks = stored_ticks
        rows = self.observe("normal", (2200, 2400, 2600), self.ticks, (2000, 2600))
        self.assertAlmostEqual(sum(self.slept), 0.4)
        self.assertEqual([row["ts"] for row in rows], ["2026-11-04T10:00:11.000000+00:00",
                                                       "2026-11-04T10:00:13.000000+00:00",
                                                       "2026-11-04T10:00:15.000000+00:00"])
        self.assertEqual(self.previous_seen[0], self.baseline)
        self.assertEqual([row["exact_row_count"] for row in rows], [2200, 2400, 2600])

    def test_normal_collects_one_unscored_baseline_of_the_source_before_inserting(self):
        rows = self.observe("normal", (2200, 2400, 2600), self.ticks, (2000, 2600))
        self.assertEqual(self.collected, [(SOURCE["project_id"], SOURCE["connection_id"], 0)])
        self.assertNotIn(self.baseline, [row["ts"] for row in rows])
        self.assertEqual(self.previous_seen, [self.baseline, *self.ticks[:2]])
        self.assertEqual([row["features"]["d_row_count"] for row in rows], [200, 200, 200])

    def test_normal_stops_before_inserting_without_a_baseline_or_with_foreign_history(self):
        for name, options in {"baseline_missing": {"baseline": False},
                              "unexpected_history": {"prior": True}}.items():
            with self.subTest(name):
                with self.assertRaises(AssertionError):
                    self.observe("normal", (2200, 2400, 2600), self.ticks, (2000, 2600), **options)
                self.assertEqual(self.run.call_count, 0)
                self.assertFalse((verify.DEMO / "normal.json").is_file())

    def test_incident_follows_the_last_tick_without_a_new_baseline(self):
        rows = self.observe("incident", (4600,), self.ticks[:1], (2600, 4600))
        self.assertEqual(self.collected, [])
        self.assertEqual(self.previous_seen, [self.previous])
        self.assertNotIn(self.previous, [row["ts"] for row in rows])

    def test_service_result_must_match_each_stored_score(self):
        served = {ts: {**BINDING, "ts": ts, "score": -0.5, "is_anomaly": False} for ts in self.ticks}
        with self.assertRaises(AssertionError):
            self.observe("normal", (2200, 2400, 2600), self.ticks, (2000, 2600), served=served)
        self.assertFalse((verify.DEMO / "normal.json").is_file())

    def test_lagging_estimate_is_kept_beside_the_exact_count_without_changing_phase_or_result(self):
        rows = self.observe("normal", (2000, 2200, 2400), self.ticks, (2000, 2600))
        self.assertEqual([row["ts"] for row in rows], list(self.ticks))
        self.assertEqual([row["features"]["row_count"] for row in rows], [2000, 2200, 2400])
        self.assertEqual([row["exact_row_count"] for row in rows], [2200, 2400, 2600])
        self.assertEqual([row["is_anomaly"] for row in rows], [False, True, False])

    def test_incident_is_one_tick_of_ten_batches(self):
        rows = self.observe("incident", (4600,), self.ticks[:1], (2600, 4600))
        self.assertEqual([(row["ts"], row["exact_row_count"]) for row in rows], [(self.ticks[0], 4600)])
        self.assertEqual(self.command[self.command.index("--ticks") + 1:], ["1", "--incident-at", "1"])

    def test_missing_or_uncounted_observations_fail_the_phase(self):
        cases = {"missing_score": ((2200, 2400, 2600), self.ticks[:2], (2000, 2600)),
                 "count_mismatch": ((2200, 2400, 2600), self.ticks, (2000, 2400))}
        for name, (values, score_ts, counts) in cases.items():
            with self.subTest(name):
                with self.assertRaises(AssertionError):
                    self.observe("normal", values, score_ts, counts)
                self.assertFalse((verify.DEMO / "normal.json").is_file())

    def test_scheduler_collection_inside_the_window_is_not_attributed_to_a_tick(self):
        extra = "2026-11-04T10:00:16.000000+00:00"
        self.ticks = (*self.ticks, extra)
        with self.assertRaises(AssertionError):
            self.observe("normal", (2200, 2400, 2600, 2600), self.ticks, (2000, 2600))
        self.assertFalse((verify.DEMO / "normal.json").is_file())

    def diagnostics(self, phase="after-recreate"):
        path = verify.DEMO / (phase + "-diagnostics.json")
        content = path.read_text()
        self.assertNotIn("fixture-adapter-token", content)
        self.assertNotIn("http://", content)
        return json.loads(content)

    def test_successful_phase_writes_no_diagnostics(self):
        self.observe("after-recreate", (2200, 2400, 2600), self.ticks, (2000, 2600))
        self.assertFalse((verify.DEMO / "after-recreate-diagnostics.json").exists())

    def test_live_demo_failure_is_told_apart_from_observation_checks(self):
        with self.assertRaises(subprocess.CalledProcessError):
            self.observe("after-recreate", (2200,), self.ticks[:1], (2000, 2200), live_demo_exit=1)
        value = self.diagnostics()
        self.assertEqual((value["check"], value["error_type"], value["live_demo_exit_code"]),
                         ("live_demo", "CalledProcessError", 1))
        self.assertEqual({key: value["counts"][key] for key in ("expected_ticks", "rows_before", "rows_after")},
                         {"expected_ticks": 3, "rows_before": 2000, "rows_after": 2200})
        self.assertFalse((verify.DEMO / "after-recreate.json").exists())

    def test_unscored_ticks_are_counted_and_rechecked_by_status_code_only(self):
        with self.assertRaisesRegex(AssertionError, "phase observations differ"):
            self.observe("after-recreate", (2200, 2400, 2600), self.ticks[:2], (2000, 2600, 2600))
        value = self.diagnostics()
        self.assertEqual((value["check"], value["error_type"]), ("window", "AssertionError"))
        # Three ticks plus the late collection are observed; two ticks and the late one are scored.
        self.assertEqual(value["counts"], {"expected_ticks": 3, "rows_before": 2000, "rows_after": 2600,
                                           "observations_in_window": 4, "scores_in_window": 3})
        self.assertEqual(value["unscored_statuses"], [422])
        self.assertEqual(value["inference"], {"healthz": 200, "model": 200})
        self.assertIn(("POST", self.ticks[2]), self.requests)
        self.assertNotIn("collector_runs", value)  # store unavailable in this fixture: omitted, not invented

    def test_unreachable_inference_is_recorded_as_status_zero(self):
        class HTTPError(Exception):
            pass

        class Client:
            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def get(self, url, **kwargs):
                raise HTTPError

            post = get

        history = {"row_count": {self.previous: 2000, self.ticks[0]: 2200},
                   "null_rate": {self.previous: 0.02, self.ticks[0]: 0.02}}
        diagnostics = {}
        context = {"source": SOURCE, "binding": BINDING}
        with patch.dict(sys.modules, {"httpx": SimpleNamespace(Client=Client, HTTPError=HTTPError)}), \
                patch.dict(os.environ, {"INFERENCE_ADAPTER_TOKEN": "fixture-adapter-token"}):
            verify.inference_evidence(context, diagnostics, [self.previous, self.ticks[0]], [self.ticks[0]],
                                      set(), history)
        self.assertEqual(diagnostics, {"inference": {"healthz": 0, "model": 0}, "unscored_statuses": [0]})

    def test_service_mismatch_names_only_the_differing_field(self):
        served = {ts: {**BINDING, "ts": ts, "score": -0.5, "is_anomaly": False} for ts in self.ticks}
        with self.assertRaises(AssertionError):
            self.observe("after-recreate", (2200, 2400, 2600), self.ticks, (2000, 2600), served=served)
        value = self.diagnostics()
        self.assertEqual((value["check"], value["response_field"], value["http_status"]), ("response", "score", 200))

    def test_diagnostics_reach_the_report_only_through_the_fixed_schema(self):
        private = "fixture-private-token postgresql://fixture:fixture-password@host/db"
        value = {"check": "window", "error_type": "AssertionError", "message": private, "http_status": 503,
                 "response_field": private, "counts": {"scores_in_window": 2, "sql": private, "rows_after": -1},
                 "collector_runs": {"success": 3, private: 1}, "unscored_statuses": [503, private],
                 "inference": {"healthz": 200, "model": "503"}, "live_demo_exit_code": "1"}
        public = verify.public_report({"observe_diagnostics": {"after-recreate": value, "other": value}})
        self.assertEqual(public["observe_diagnostics"], {"after-recreate": {
            "check": "window", "error_type": "AssertionError", "http_status": 503,
            "counts": {"scores_in_window": 2}, "collector_runs": {"success": 3},
            "unscored_statuses": [503], "inference": {"healthz": 200}}})
        self.assertNotIn("fixture", json.dumps(public))

    def test_report_keeps_exact_count_only_as_a_non_negative_integer(self):
        row = {**ROW, "exact_row_count": 2200}
        self.assertEqual(verify.public_report({"normal": [row]})["normal"], [row])
        self.assertNotIn("exact_row_count", verify.public_report({"normal": [{**ROW, "exact_row_count": "2200"}]})["normal"][0])


if __name__ == "__main__":
    unittest.main()
