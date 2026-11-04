"""Offline configuration checks; these do not claim a Docker/Compose runtime test.

Also runnable without pytest: python3 -m unittest discover -s tests
-p test_container_setup.py -v. PyYAML is already a project dependency.
"""

import base64
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from scripts.init_compose_demo import DEFAULT_PORTS, ROOT, create_environment


def read_values(path):
    return dict(line.split("=", 1) for line in path.read_text().splitlines()
                if line and not line.startswith("#"))


class DemoEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.destination = Path(self.directory.name) / ".env"

    def create(self, **overrides):
        options = {"version": "a" * 40, "uid": 1001, "gid": 1001, **overrides}
        create_environment(self.destination, **options)
        return read_values(self.destination)

    def test_private_persistent_keys_and_unprivileged_identity(self):
        values = self.create()
        self.assertEqual(self.destination.stat().st_mode & 0o777, 0o600)
        self.assertEqual(len(base64.urlsafe_b64decode(values["FERNET_KEY"])), 32)
        self.assertNotEqual(values["SECRET_KEY"], "change-me-to-something-random")
        self.assertGreaterEqual(len(values["INFERENCE_ADAPTER_TOKEN"]), 32)
        self.assertEqual(values["APP_UID"], "1001")
        self.assertEqual(values["APP_GID"], "1001")
        self.assertEqual(values["APP_VERSION"], "a" * 40)

    def test_no_paid_or_external_notifications_by_default(self):
        values = self.create()
        for name in ("NIM_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
            self.assertEqual(values[name], "")
        self.assertEqual(values["FF_LLM_NOTIFICATIONS"], "0")
        self.assertNotIn("INFERENCE_ADAPTER_CONFIG", values)

    def test_host_dsns_match_published_ports_and_use_metrics_postgres(self):
        values = self.create(ports=(25001, 25432, 25433, 28000))
        self.assertEqual(values["DATABASE_URL"], "postgresql://postgres:dev@localhost:25432/monitor")
        self.assertEqual(values["MONITOR_DB_URL"], "postgresql://postgres:dev@localhost:25433/metrics")
        self.assertEqual(values["BIND_ADDRESS"], "127.0.0.1")
        self.assertEqual(values["PORT"], "25001")
        self.assertEqual(values["POSTGRES_PORT"], "25432")
        self.assertEqual(values["TIMESCALE_PORT"], "25433")
        self.assertEqual(values["INFERENCE_PORT"], "28000")

    def test_existing_environment_is_never_replaced(self):
        self.create()
        original = self.destination.read_bytes()
        with self.assertRaises(FileExistsError):
            self.create()
        self.assertEqual(self.destination.read_bytes(), original)

    def test_symlink_is_not_followed(self):
        target = self.destination.with_name("existing")
        target.write_text("keep this configuration")
        self.destination.symlink_to(target)
        with self.assertRaises(FileExistsError):
            self.create()
        self.assertEqual(target.read_text(), "keep this configuration")

    def test_new_checkouts_get_different_keys_names_and_images(self):
        first = self.create()
        self.destination = self.destination.with_name("second.env")
        second = self.create()
        for key in ("FERNET_KEY", "SECRET_KEY", "INFERENCE_ADAPTER_TOKEN", "COMPOSE_PROJECT_NAME", "APP_IMAGE"):
            self.assertNotEqual(first[key], second[key])
        self.assertTrue(first["APP_IMAGE"].startswith(first["COMPOSE_PROJECT_NAME"] + ":"))

    def test_generated_fernet_key_decrypts_in_a_different_process(self):
        values = self.create()
        environment = {**os.environ, "FERNET_KEY": values["FERNET_KEY"]}
        encrypt = subprocess.run(
            [sys.executable, "-c", "from app.crypto import encrypt_dsn; print(encrypt_dsn('demo-dsn').decode())"],
            cwd=ROOT, env=environment, capture_output=True, text=True, check=True,
        )
        decrypt = subprocess.run(
            [sys.executable, "-c", "import sys; from app.crypto import decrypt_dsn; print(decrypt_dsn(sys.stdin.buffer.read().strip()))"],
            cwd=ROOT, env=environment, input=encrypt.stdout, capture_output=True, text=True, check=True,
        )
        self.assertEqual(decrypt.stdout.strip(), "demo-dsn")

    def test_invalid_ports_root_or_sha_leave_no_file(self):
        for overrides in (
            {"ports": (1, 1, 2, 3)}, {"ports": (0, 2, 3, 4)},
            {"ports": (65536, 2, 3, 4)}, {"uid": 0}, {"version": "a\nPORT=1"},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.create(**overrides)
            self.assertFalse(self.destination.exists())

    def test_cli_does_not_print_keys_and_rejects_second_run(self):
        command = [sys.executable, "-m", "scripts.init_compose_demo", "--output", str(self.destination)]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        if os.getuid() == 0:
            self.assertEqual(result.returncode, 2)
            self.assertFalse(self.destination.exists())
            return
        self.assertEqual(result.returncode, 0, result.stderr)
        for name in ("FERNET_KEY", "SECRET_KEY", "INFERENCE_ADAPTER_TOKEN"):
            self.assertNotIn(read_values(self.destination)[name], result.stdout + result.stderr)
        before = self.destination.read_bytes()
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(self.destination.read_bytes(), before)


class ComposeConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
        cls.services = cls.compose["services"]

    def test_databases_are_distinct_and_waited_for(self):
        app = self.services["app"]
        self.assertEqual(app["environment"]["DATABASE_URL"], "postgresql://postgres:dev@postgres:5432/monitor")
        self.assertEqual(app["environment"]["MONITOR_DB_URL"], "postgresql://postgres:dev@timescaledb:5432/metrics")
        for name in ("postgres", "timescaledb"):
            self.assertEqual(app["depends_on"][name]["condition"], "service_healthy")
            self.assertIn("pg_isready", self.services[name]["healthcheck"]["test"][1])
        self.assertFalse(any("monitor.db" in path for path in app["volumes"]))

    def test_db_healthchecks_wait_for_tcp_not_the_temporary_init_socket(self):
        for name, database in (("postgres", "monitor"), ("timescaledb", "metrics")):
            with self.subTest(service=name):
                kind, command = self.services[name]["healthcheck"]["test"]
                self.assertEqual(kind, "CMD-SHELL")
                args = shlex.split(command)
                self.assertEqual(args[0], "pg_isready")
                self.assertEqual(args[args.index("-h") + 1], "127.0.0.1")
                self.assertEqual(args[args.index("-p") + 1], "5432")
                self.assertEqual(args[args.index("-d") + 1], database)

    def test_persistence_and_legacy_profiles_are_preserved(self):
        self.assertIn("pg_data:/var/lib/postgresql", self.services["postgres"]["volumes"])
        self.assertIn("timescale_data:/var/lib/postgresql/data", self.services["timescaledb"]["volumes"])
        for service, profile in (("minio", "iceberg"), ("iceberg-rest", "iceberg"),
                                 ("clickhouse", "clickhouse"), ("backup", "backup")):
            self.assertEqual(self.services[service]["profiles"], [profile])

    def test_container_ports_do_not_follow_host_port(self):
        app = self.services["app"]
        self.assertEqual(app["environment"]["PORT"], "5001")
        self.assertEqual(app["environment"]["FLASK_DEBUG"], "0")
        self.assertTrue(app["ports"][0].endswith(":5001"))
        self.assertIn("${FERNET_KEY:?", app["environment"]["FERNET_KEY"])

    def test_inference_uses_same_build_and_readonly_registry(self):
        inference = self.services["inference"]
        self.assertEqual(inference["profiles"], ["inference"])
        self.assertEqual(inference["build"], self.services["app"]["build"])
        self.assertEqual(inference["image"], self.services["app"]["image"])
        self.assertEqual(inference["volumes"], ["./models:/app/models:ro"])
        self.assertNotIn("env_file", inference)
        self.assertEqual(set(inference["environment"]), {"INFERENCE_CONFIG", "INFERENCE_ADAPTER_TOKEN"})
        self.assertIn("inference.service:create_app", inference["command"])
        self.assertIn("8000/healthz", inference["healthcheck"]["test"][-1])

    def test_isolated_names_ports_and_users_are_configurable(self):
        for name in ("postgres", "timescaledb", "app"):
            self.assertIn("COMPOSE_PROJECT_NAME", self.services[name]["container_name"])
            self.assertIn("BIND_ADDRESS", self.services[name]["ports"][0])
        for name in ("app", "inference"):
            self.assertIn("APP_UID", self.services[name]["user"])
        self.assertEqual(len(set(DEFAULT_PORTS)), 4)

    def test_demo_make_targets_run_inside_compose(self):
        makefile = (ROOT / "Makefile").read_text()
        for target, module in (("demo-prepare", "demo_prepare"), ("live-demo", "live_demo")):
            recipe = makefile.split(f"\n{target}:\n", 1)[1].split("\n\n", 1)[0]
            self.assertIn(f"docker compose run --rm --no-deps app python -m scripts.{module}", recipe)


if __name__ == "__main__":
    unittest.main()
