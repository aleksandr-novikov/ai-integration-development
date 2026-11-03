"""Offline safety/result-policy tests; no simulated result is CI evidence."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
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


if __name__ == "__main__":
    unittest.main()
