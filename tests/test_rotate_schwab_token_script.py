from __future__ import annotations

import os
import stat
import subprocess
import textwrap
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "rotate_schwab_token.sh"
BING_REPO_ROOT = SCRIPT_PATH.parents[1]
FIXED_TIMESTAMP = "2026-04-10T12:34:56Z"
NEW_REFRESH_TOKEN = "new-shared-refresh-token"
BING_REPO_NAME = "example/BingBingBot"
STOCK_REPO_NAME = "example/StockOpportunityScanner"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(textwrap.dedent(content), encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _parse_refresh_token(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("SCHWAB_REFRESH_TOKEN="):
            return line.split("=", 1)[1].strip().strip("'\"")
    raise AssertionError(f"SCHWAB_REFRESH_TOKEN not found in {path}")


def _build_stub_bin(tmp_path: Path, stock_root: Path) -> tuple[Path, Path]:
    bin_dir = tmp_path / "bin"
    state_dir = tmp_path / "state"
    log_path = tmp_path / "stub.log"
    bin_dir.mkdir()
    state_dir.mkdir()
    log_path.write_text("", encoding="utf-8")

    common_stub_helpers = """
from pathlib import Path
import os

LOG_PATH = Path(os.environ["ROTATE_TEST_LOG"])
STATE_DIR = Path(os.environ["ROTATE_TEST_STATE_DIR"])

def log(message: str) -> None:
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(f"{message}\\n")

def sanitize_repo(repo: str) -> str:
    return repo.replace("/", "__").replace(":", "__")
"""

    _write_executable(
        bin_dir / "git",
        f"""#!/usr/bin/env python3
import sys

repo_root = ""
args = sys.argv[1:]
if len(args) >= 2 and args[0] == "-C":
    repo_root = args[1]
    args = args[2:]

if args == ["config", "--get", "remote.origin.url"]:
    if repo_root == {str(BING_REPO_ROOT)!r}:
        print("https://github.com/{BING_REPO_NAME}.git")
    elif repo_root == {str(stock_root)!r}:
        print("https://github.com/{STOCK_REPO_NAME}.git")
    raise SystemExit(0)

raise SystemExit(f"unexpected git args: {{sys.argv[1:]}}")
""",
    )

    _write_executable(
        bin_dir / "gh",
        f"""#!/usr/bin/env python3
import sys
{common_stub_helpers}

args = sys.argv[1:]
if args[:2] == ["auth", "status"]:
    log("auth_status")
    raise SystemExit(0)

if args[:1] == ["api"]:
    repo_path = args[1]
    parts = repo_path.split("/")
    repo = "/".join(parts[1:3]) if len(parts) >= 3 else ""
    updated_file = STATE_DIR / f"{{sanitize_repo(repo)}}.updated_at"
    if updated_file.exists():
        print(updated_file.read_text(encoding="utf-8"), end="")
    raise SystemExit(0)

if args[:2] == ["secret", "set"]:
    name = args[2]
    repo = ""
    for idx, arg in enumerate(args):
        if arg == "--repo" and idx + 1 < len(args):
            repo = args[idx + 1]
            break
    token = sys.stdin.read()
    log(f"secret_set|{{repo}}|{{name}}|{{token}}")
    updated_file = STATE_DIR / f"{{sanitize_repo(repo)}}.updated_at"
    updated_file.write_text(f"updated-{{repo.replace('/', '-')}}", encoding="utf-8")
    raise SystemExit(0)

if args[:2] == ["variable", "set"]:
    name = args[2]
    repo = ""
    body = ""
    for idx, arg in enumerate(args):
        if arg == "--repo" and idx + 1 < len(args):
            repo = args[idx + 1]
        if arg == "--body" and idx + 1 < len(args):
            body = args[idx + 1]
    log(f"variable_set|{{repo}}|{{name}}|{{body}}")
    raise SystemExit(0)

if args[:2] == ["repo", "view"]:
    print("fallback/repo")
    raise SystemExit(0)

raise SystemExit(f"unexpected gh args: {{args}}")
""",
    )

    _write_executable(
        bin_dir / "curl",
        f"""#!/usr/bin/env python3
import json
import sys
{common_stub_helpers}

args = sys.argv[1:]
output_file = ""
for idx, arg in enumerate(args):
    if arg == "-o" and idx + 1 < len(args):
        output_file = args[idx + 1]
        break

if not output_file:
    raise SystemExit("missing curl output file")

payload = {{
    "access_token": "access-token",
    "expires_in": 1800,
    "refresh_token": {NEW_REFRESH_TOKEN!r},
    "refresh_token_expires_in": 604800,
}}
Path(output_file).write_text(json.dumps(payload), encoding="utf-8")
log("curl_call")
sys.stdout.write("200")
""",
    )

    _write_executable(
        bin_dir / "xdg-open",
        f"""#!/usr/bin/env python3
import sys
{common_stub_helpers}
log(f"browser_open|{{sys.argv[1] if len(sys.argv) > 1 else ''}}")
""",
    )

    _write_executable(
        bin_dir / "open",
        f"""#!/usr/bin/env python3
import sys
{common_stub_helpers}
log(f"browser_open|{{sys.argv[1] if len(sys.argv) > 1 else ''}}")
""",
    )

    _write_executable(
        bin_dir / "date",
        f"""#!/usr/bin/env python3
import sys
if sys.argv[1:] and sys.argv[1] == "-u":
    print({FIXED_TIMESTAMP!r})
    raise SystemExit(0)
raise SystemExit(f"unexpected date args: {{sys.argv[1:]}}")
""",
    )

    _write_executable(
        bin_dir / "sleep",
        """#!/usr/bin/env python3
raise SystemExit(0)
""",
    )

    return bin_dir, log_path


def _run_script(tmp_path: Path, *, extra_args: list[str] | None = None) -> tuple[subprocess.CompletedProcess[str], Path, Path, Path]:
    bing_env = tmp_path / "bing.env"
    bing_env.write_text(
        "\n".join(
            [
                "SCHWAB_CLIENT_ID='client-id'",
                "SCHWAB_CLIENT_SECRET='client-secret'",
                "SCHWAB_REFRESH_TOKEN='old-bing-token'",
                "",
            ]
        ),
        encoding="utf-8",
    )

    stock_root = tmp_path / "StockOpportunityScanner"
    stock_root.mkdir()
    stock_env = stock_root / ".env.local"
    stock_env.write_text("SCHWAB_REFRESH_TOKEN='old-stock-token'\n", encoding="utf-8")

    bin_dir, log_path = _build_stub_bin(tmp_path, stock_root)

    cmd = [
        str(SCRIPT_PATH),
        "--env-file",
        str(bing_env),
        "--stock-root",
        str(stock_root),
        "--no-open",
    ]
    if extra_args:
        cmd.extend(extra_args)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "ROTATE_TEST_LOG": str(log_path),
            "ROTATE_TEST_STATE_DIR": str(tmp_path / "state"),
        }
    )

    completed = subprocess.run(
        cmd,
        input="https://127.0.0.1?code=demo-auth-code\n",
        text=True,
        capture_output=True,
        cwd=BING_REPO_ROOT,
        env=env,
        check=False,
    )
    return completed, bing_env, stock_env, log_path


def test_rotate_script_updates_bing_and_stock_targets(tmp_path: Path) -> None:
    completed, bing_env, stock_env, log_path = _run_script(tmp_path)

    assert completed.returncode == 0, completed.stderr
    assert _parse_refresh_token(bing_env) == NEW_REFRESH_TOKEN
    assert _parse_refresh_token(stock_env) == NEW_REFRESH_TOKEN

    log_lines = log_path.read_text(encoding="utf-8").splitlines()
    assert log_lines.count("curl_call") == 1
    assert f"secret_set|{BING_REPO_NAME}|SCHWAB_REFRESH_TOKEN|{NEW_REFRESH_TOKEN}" in log_lines
    assert f"secret_set|{STOCK_REPO_NAME}|SCHWAB_REFRESH_TOKEN|{NEW_REFRESH_TOKEN}" in log_lines
    assert f"variable_set|{BING_REPO_NAME}|SCHWAB_REFRESH_TOKEN_ROTATED_AT_UTC|{FIXED_TIMESTAMP}" in log_lines
    assert f"variable_set|{STOCK_REPO_NAME}|SCHWAB_REFRESH_TOKEN_ROTATED_AT_UTC|{FIXED_TIMESTAMP}" in log_lines


def test_rotate_script_no_stock_sync_updates_only_bing(tmp_path: Path) -> None:
    completed, bing_env, stock_env, log_path = _run_script(tmp_path, extra_args=["--no-stock-sync"])

    assert completed.returncode == 0, completed.stderr
    assert _parse_refresh_token(bing_env) == NEW_REFRESH_TOKEN
    assert _parse_refresh_token(stock_env) == "old-stock-token"

    log_lines = log_path.read_text(encoding="utf-8").splitlines()
    assert log_lines.count("curl_call") == 1
    assert f"secret_set|{BING_REPO_NAME}|SCHWAB_REFRESH_TOKEN|{NEW_REFRESH_TOKEN}" in log_lines
    assert f"variable_set|{BING_REPO_NAME}|SCHWAB_REFRESH_TOKEN_ROTATED_AT_UTC|{FIXED_TIMESTAMP}" in log_lines
    assert not any(STOCK_REPO_NAME in line for line in log_lines if line.startswith("secret_set|"))
    assert not any(STOCK_REPO_NAME in line for line in log_lines if line.startswith("variable_set|"))


def test_rotate_script_fails_before_oauth_when_stock_root_is_invalid(tmp_path: Path) -> None:
    bing_env = tmp_path / "bing.env"
    bing_env.write_text(
        "\n".join(
            [
                "SCHWAB_CLIENT_ID='client-id'",
                "SCHWAB_CLIENT_SECRET='client-secret'",
                "SCHWAB_REFRESH_TOKEN='old-bing-token'",
                "",
            ]
        ),
        encoding="utf-8",
    )

    missing_stock_root = tmp_path / "missing-stock-root"
    bin_dir, log_path = _build_stub_bin(tmp_path, missing_stock_root)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "ROTATE_TEST_LOG": str(log_path),
            "ROTATE_TEST_STATE_DIR": str(tmp_path / "state"),
        }
    )

    completed = subprocess.run(
        [
            str(SCRIPT_PATH),
            "--env-file",
            str(bing_env),
            "--stock-root",
            str(missing_stock_root),
            "--no-open",
        ],
        input="https://127.0.0.1?code=demo-auth-code\n",
        text=True,
        capture_output=True,
        cwd=BING_REPO_ROOT,
        env=env,
        check=False,
    )

    assert completed.returncode != 0
    assert "stock companion root does not exist" in completed.stderr
    assert "curl_call" not in log_path.read_text(encoding="utf-8").splitlines()
