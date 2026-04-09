#!/usr/bin/env bash
set -euo pipefail

SCRIPT_NAME="$(basename "$0")"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCHWAB_OAUTH_AUTHORIZE_URL="${SCHWAB_OAUTH_AUTHORIZE_URL:-https://api.schwabapi.com/v1/oauth/authorize}"
SCHWAB_OAUTH_TOKEN_URL="${SCHWAB_OAUTH_TOKEN_URL:-https://api.schwabapi.com/v1/oauth/token}"
SCHWAB_ROTATION_TIMESTAMP_VAR="${SCHWAB_ROTATION_TIMESTAMP_VAR:-SCHWAB_REFRESH_TOKEN_ROTATED_AT_UTC}"
ENV_FILE="${REPO_ROOT}/.env.schwab.local"

usage() {
  cat <<EOF_USAGE
Usage: ${SCRIPT_NAME} [--env-file PATH] [--redirect-uri URI] [--repo owner/repo] [--no-open] [--print-token] [--no-github-secret]

This script reuses the auth flow from StockOpportunityScanner and rotates SCHWAB_REFRESH_TOKEN.
By default it updates ${ENV_FILE}, GitHub secret SCHWAB_REFRESH_TOKEN, and
GitHub variable ${SCHWAB_ROTATION_TIMESTAMP_VAR} (UTC timestamp of last successful rotation).
Use --no-github-secret to skip secret update.

Required environment variables:
  SCHWAB_CLIENT_ID
  SCHWAB_CLIENT_SECRET

Optional environment variables:
  SCHWAB_REDIRECT_URI (default: https://127.0.0.1)
  GH_REPO             (fallback target repo if --repo is omitted)
  SCHWAB_ROTATION_TIMESTAMP_VAR (default: SCHWAB_REFRESH_TOKEN_ROTATED_AT_UTC)

Examples:
  ${SCRIPT_NAME}
  ${SCRIPT_NAME} --env-file .env.schwab.local --redirect-uri https://127.0.0.1
  ${SCRIPT_NAME} --repo owner/repo --print-token
  ${SCRIPT_NAME} --no-github-secret
EOF_USAGE
}

require_command() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    printf 'ERROR: required command not found: %s\n' "${cmd}" >&2
    exit 1
  fi
}

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    printf 'ERROR: required environment variable is missing: %s\n' "${name}" >&2
    exit 1
  fi
}

load_env_file() {
  local path="$1"
  if [[ -z "${path}" || ! -f "${path}" ]]; then
    return 0
  fi

  # shellcheck disable=SC1090
  set -a
  source "${path}"
  set +a
  printf 'Loaded environment values from %s\n' "${path}"
}

resolve_target_repo() {
  local explicit_repo="$1"
  if [[ -n "${explicit_repo}" ]]; then
    printf '%s\n' "${explicit_repo}"
    return 0
  fi

  local env_repo="${GH_REPO:-}"
  if [[ -n "${env_repo}" ]]; then
    printf '%s\n' "${env_repo}"
    return 0
  fi

  local detected_repo=""
  detected_repo="$(gh repo view --json nameWithOwner -q '.nameWithOwner' 2>/dev/null || true)"
  printf '%s\n' "${detected_repo}"
}

verify_refresh_token_env_var() {
  local path="$1"
  local expected_token="$2"
  python3 - "${path}" "${expected_token}" <<'PY'
import re
import shlex
import sys
from pathlib import Path

env_path = Path(sys.argv[1]).expanduser()
expected = sys.argv[2]
if not env_path.exists():
    raise SystemExit(f"ERROR: env file does not exist after update: {env_path}")

pattern = re.compile(r"^\s*SCHWAB_REFRESH_TOKEN\s*=\s*(.*)\s*$")
for line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
    match = pattern.match(line)
    if not match:
        continue
    raw_value = match.group(1)
    try:
        parsed = shlex.split(f"x={raw_value}", posix=True)[0].split("=", 1)[1]
    except Exception:
        parsed = raw_value
    if parsed != expected:
        raise SystemExit(
            "ERROR: SCHWAB_REFRESH_TOKEN was written, but verification failed for "
            f"{env_path}. Check file contents manually."
        )
    print(f"Verified SCHWAB_REFRESH_TOKEN update in {env_path}")
    raise SystemExit(0)

raise SystemExit(f"ERROR: SCHWAB_REFRESH_TOKEN assignment not found in {env_path}")
PY
}

upsert_refresh_token_env_var() {
  local path="$1"
  local token="$2"
  python3 - "${path}" "${token}" <<'PY'
import re
import shlex
import sys
from pathlib import Path

env_path = Path(sys.argv[1]).expanduser()
token = sys.argv[2]
assignment = f"SCHWAB_REFRESH_TOKEN={shlex.quote(token)}\n"
pattern = re.compile(r"^\s*SCHWAB_REFRESH_TOKEN\s*=")

if env_path.exists():
    lines = env_path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
else:
    lines = []

output = []
replaced = False
for line in lines:
    if not replaced and pattern.match(line):
        output.append(assignment)
        replaced = True
    else:
        output.append(line)

if not replaced:
    if output and not output[-1].endswith("\n"):
        output[-1] += "\n"
    if output and output[-1].strip():
        output.append("\n")
    output.append(assignment)

env_path.parent.mkdir(parents=True, exist_ok=True)
env_path.write_text("".join(output), encoding="utf-8")
PY
}

CLI_REDIRECT_URI=""
CLI_TARGET_REPO=""
OPEN_BROWSER=true
PRINT_TOKEN=false
UPDATE_GITHUB_SECRET=true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --redirect-uri)
      if [[ -z "${2:-}" ]]; then
        printf 'ERROR: --redirect-uri requires a value.\n' >&2
        exit 2
      fi
      CLI_REDIRECT_URI="$2"
      shift 2
      ;;
    --repo)
      if [[ -z "${2:-}" ]]; then
        printf 'ERROR: --repo requires a value.\n' >&2
        exit 2
      fi
      CLI_TARGET_REPO="$2"
      shift 2
      ;;
    --env-file)
      if [[ -z "${2:-}" ]]; then
        printf 'ERROR: --env-file requires a value.\n' >&2
        exit 2
      fi
      ENV_FILE="$2"
      shift 2
      ;;
    --no-open)
      OPEN_BROWSER=false
      shift
      ;;
    --print-token)
      PRINT_TOKEN=true
      shift
      ;;
    --no-github-secret)
      UPDATE_GITHUB_SECRET=false
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'ERROR: unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "${ENV_FILE}" != /* ]]; then
  ENV_FILE="${REPO_ROOT}/${ENV_FILE}"
fi
printf 'Using env file: %s\n' "${ENV_FILE}"

load_env_file "${ENV_FILE}"

require_command python3
require_command curl
require_env SCHWAB_CLIENT_ID
require_env SCHWAB_CLIENT_SECRET

SCHWAB_REDIRECT_URI="${CLI_REDIRECT_URI:-${SCHWAB_REDIRECT_URI:-https://127.0.0.1}}"
TARGET_REPO=""

if [[ "${UPDATE_GITHUB_SECRET}" == "true" ]]; then
  require_command gh
  if ! gh auth status -h github.com >/dev/null 2>&1; then
    printf 'ERROR: GitHub CLI is not authenticated. Run: gh auth login (or use --no-github-secret).\n' >&2
    exit 1
  fi
  TARGET_REPO="$(resolve_target_repo "${CLI_TARGET_REPO}")"
  if [[ -z "${TARGET_REPO}" ]]; then
    printf 'ERROR: unable to resolve target GitHub repo. Pass --repo owner/repo or set GH_REPO.\n' >&2
    exit 1
  fi
  printf 'GitHub secret target repo: %s\n' "${TARGET_REPO}"
fi

AUTH_URL="$(python3 - "${SCHWAB_CLIENT_ID}" "${SCHWAB_REDIRECT_URI}" "${SCHWAB_OAUTH_AUTHORIZE_URL}" <<'PY'
import sys
import urllib.parse as parse

client_id = sys.argv[1]
redirect_uri = sys.argv[2]
authorize_url = sys.argv[3]
query = parse.urlencode(
    {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
    }
)
print(f"{authorize_url}?{query}")
PY
)"

printf 'Open this URL in your browser to authorize:\n%s\n' "${AUTH_URL}"
if [[ "${OPEN_BROWSER}" == "true" ]]; then
  if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "${AUTH_URL}" >/dev/null 2>&1 || true
  elif command -v open >/dev/null 2>&1; then
    open "${AUTH_URL}" >/dev/null 2>&1 || true
  fi
fi

printf 'Paste FULL redirect URL from browser: '
IFS= read -r REDIRECT_URL
if [[ -z "${REDIRECT_URL}" ]]; then
  printf 'ERROR: redirect URL is required.\n' >&2
  exit 1
fi

AUTH_CODE="$(python3 - "${REDIRECT_URL}" <<'PY'
import sys
import urllib.parse as parse

redirect_url = sys.argv[1]
query = parse.parse_qs(parse.urlparse(redirect_url).query)
values = query.get("code")
if not values or not values[0].strip():
    raise SystemExit("ERROR: redirect URL does not include a valid 'code' query parameter.")
print(values[0].strip())
PY
)"

response_file="$(mktemp -t schwab_token_response.XXXXXX)"
cleanup() {
  rm -f "${response_file}"
}
trap cleanup EXIT

http_status="$(curl -sS -o "${response_file}" -w "%{http_code}" \
  -u "${SCHWAB_CLIENT_ID}:${SCHWAB_CLIENT_SECRET}" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -H "Accept: application/json" \
  --data-urlencode "grant_type=authorization_code" \
  --data-urlencode "code=${AUTH_CODE}" \
  --data-urlencode "redirect_uri=${SCHWAB_REDIRECT_URI}" \
  "${SCHWAB_OAUTH_TOKEN_URL}")"

if [[ "${http_status}" -lt 200 || "${http_status}" -ge 300 ]]; then
  printf 'ERROR: Schwab token endpoint returned HTTP %s\n' "${http_status}" >&2
  printf 'Response body:\n%s\n' "$(cat "${response_file}")" >&2
  exit 1
fi

mapfile -t token_fields < <(python3 - "${response_file}" <<'PY'
import json
import sys
from pathlib import Path

raw = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
try:
    payload = json.loads(raw)
except Exception as exc:
    raise SystemExit(f"ERROR: token endpoint returned invalid JSON: {exc}\nBody: {raw[:1000]}")

refresh_token = str(payload.get("refresh_token", "")).strip()
if not refresh_token:
    keys = ",".join(sorted(payload.keys()))
    raise SystemExit(f"ERROR: token endpoint response missing refresh_token. keys={keys}")

access_expires_in = payload.get("expires_in")
refresh_expires_in = payload.get("refresh_token_expires_in")
print(refresh_token)
print("" if access_expires_in is None else str(access_expires_in))
print("" if refresh_expires_in is None else str(refresh_expires_in))
PY
)

new_refresh_token="${token_fields[0]:-}"
access_expires_in="${token_fields[1]:-}"
refresh_expires_in="${token_fields[2]:-}"

if [[ -z "${new_refresh_token}" ]]; then
  printf 'ERROR: no refresh token extracted from token response.\n' >&2
  exit 1
fi

upsert_refresh_token_env_var "${ENV_FILE}" "${new_refresh_token}"
chmod 600 "${ENV_FILE}" 2>/dev/null || true
printf 'Updated SCHWAB_REFRESH_TOKEN in %s\n' "${ENV_FILE}"
verify_refresh_token_env_var "${ENV_FILE}" "${new_refresh_token}"

if [[ "${UPDATE_GITHUB_SECRET}" == "true" ]]; then
  printf '%s' "${new_refresh_token}" | gh secret set SCHWAB_REFRESH_TOKEN --repo "${TARGET_REPO}"
  printf 'Updated GitHub secret SCHWAB_REFRESH_TOKEN for %s\n' "${TARGET_REPO}"

  rotation_timestamp_utc="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  if gh variable set "${SCHWAB_ROTATION_TIMESTAMP_VAR}" --repo "${TARGET_REPO}" --body "${rotation_timestamp_utc}" >/dev/null; then
    printf 'Updated GitHub variable %s for %s (%s)\n' "${SCHWAB_ROTATION_TIMESTAMP_VAR}" "${TARGET_REPO}" "${rotation_timestamp_utc}"
  else
    printf 'Warning: unable to update GitHub variable %s for %s.\n' "${SCHWAB_ROTATION_TIMESTAMP_VAR}" "${TARGET_REPO}" >&2
  fi
fi

if [[ "${PRINT_TOKEN}" == "true" ]]; then
  printf 'SCHWAB_REFRESH_TOKEN=%s\n' "${new_refresh_token}"
fi

if [[ -n "${access_expires_in}" ]]; then
  printf 'access_token_expires_in=%ss\n' "${access_expires_in}"
fi
if [[ -n "${refresh_expires_in}" ]]; then
  printf 'refresh_token_expires_in=%ss\n' "${refresh_expires_in}"
fi
printf 'Rotation complete. Repeat this flow weekly (about every 7 days).\n'
