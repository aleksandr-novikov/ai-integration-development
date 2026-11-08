"""Create a private, isolated Compose demo configuration without overwriting .env.

Standard library only; does not install tools, start containers or change data.
Existing deployments must retain their current Fernet key instead of using this.
"""

from __future__ import annotations

import argparse
import base64
import os
import secrets
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PORTS = (15001, 15432, 15433, 18000)


def create_environment(
    destination: Path,
    *,
    version: str,
    uid: int,
    gid: int,
    ports: tuple[int, int, int, int] = DEFAULT_PORTS,
) -> None:
    """Write an exclusive 0600 file. Keys and Compose project persist in this file."""
    if uid <= 0 or gid < 0:
        raise ValueError("run as a non-root user who owns the checkout and models directory")
    if len(ports) != 4 or len(set(ports)) != 4 or any(not 1 <= p <= 65535 for p in ports):
        raise ValueError("provide four distinct ports between 1 and 65535")
    if len(version) != 40 or any(c not in "0123456789abcdef" for c in version):
        raise ValueError("version must be the full checkout commit SHA")
    app_port, postgres_port, metrics_port, inference_port = ports
    project = "course13-" + secrets.token_hex(6)
    replacements = {
        "SECRET_KEY": secrets.token_urlsafe(48),
        "DATABASE_URL": f"postgresql://postgres:dev@localhost:{postgres_port}/monitor",
        "MONITOR_DB_URL": f"postgresql://postgres:dev@localhost:{metrics_port}/metrics",
    }
    lines = []
    for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        name = line.partition("=")[0]
        lines.append(f"{name}={replacements[name]}" if name in replacements else line)
    values = {
        "FERNET_KEY": base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("ascii"),
        "INFERENCE_ADAPTER_TOKEN": secrets.token_urlsafe(48),
        "COMPOSE_PROJECT_NAME": project,
        "APP_IMAGE": project + ":demo",
        "APP_VERSION": version,
        "APP_UID": uid,
        "APP_GID": gid,
        "BIND_ADDRESS": "127.0.0.1",
        "PORT": app_port,
        "POSTGRES_PORT": postgres_port,
        "TIMESCALE_PORT": metrics_port,
        "INFERENCE_PORT": inference_port,
    }
    content = "\n".join(lines) + "\n\n# Isolated Compose demo; retain on restart.\n"
    content += "".join(f"{name}={value}\n" for name, value in values.items())
    # O_EXCL also rejects symlinks; never truncate a user's configuration.
    fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(content)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".env")
    parser.add_argument("--ports", type=int, nargs=4, default=DEFAULT_PORTS,
                        metavar=("APP", "POSTGRES", "METRICS", "INFERENCE"))
    args = parser.parse_args()
    version = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip()
    try:
        create_environment(args.output, version=version, uid=os.getuid(), gid=os.getgid(),
                           ports=tuple(args.ports))
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Configuration not created: {exc}\n")
    print(f"Created {args.output.name}; keys are not printed. Check ports before starting Compose.")


if __name__ == "__main__":
    main()
