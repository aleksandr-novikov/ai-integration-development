"""Offline safety/result-policy tests; no simulated result is CI evidence."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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

    def test_diagnostics_redact_generated_keys_and_credential_dsns(self):
        values = {"SECRET_KEY": "fixture-session-key", "FERNET_KEY": "fixture-fernet-key",
                  "INFERENCE_ADAPTER_TOKEN": "fixture-inference-token"}
        message = " ".join(values.values()) + " postgresql+psycopg2://demo:fixture-password@postgres/db"
        clean = verify.redact(message, values)
        for secret in (*values.values(), "fixture-password"):
            self.assertNotIn(secret, clean)
        self.assertIn("<redacted>@postgres/db", clean)

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
        self.assertEqual(job["steps"][0]["with"]["ref"], "${{ github.event.pull_request.head.sha || github.sha }}")
        upload = next(s for s in job["steps"] if s.get("uses") == "actions/upload-artifact@v4")
        self.assertEqual(upload["with"]["path"], "artifacts/compose-demo/ci-report.json")
        self.assertEqual(upload["if"], "always()")
        cleanup = next(s for s in job["steps"] if s.get("run") == "python -m scripts.verify_compose_demo cleanup")
        self.assertEqual(cleanup["if"], "always()")


if __name__ == "__main__":
    unittest.main()
