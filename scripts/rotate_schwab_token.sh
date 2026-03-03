#!/usr/bin/env bash
set -euo pipefail

SCRIPT_NAME="$(basename "$0")"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCHWAB_OAUTH_AUTHORIZE_URL="https://api.schwabapi.com/v1/oauth/authorize"
SCHWAB_OAUTH_TOKEN_URL="https://api.schwabapi.com/v1/oauth/token"
ENV_FILE="${REPO_ROOT}/.env.schwab.local"

usage() {
  cat <<EOF
Usage: ${SCRIPT_NAME} [--repo owner/repo] [--redirect-uri URI] [--env-file PATH] [--no-open]

This script auto-loads ${ENV_FILE} when present.

Required environment variables:
  SCHWAB_CLIENT_ID
  SCHWAB_CLIENT_SECRET

Optional environment variables:
  SCHWAB_REDIRECT_URI (default: https://127.0.0.1)
  GH_REPO             (fallback target repo if --repo is omitted)

Examples:
  ${SCRIPT_NAME} --repo owner/repo
  ${SCRIPT_NAME} --env-file .env.schwab.local --repo owner/repo
  SCHWAB_REDIRECT_URI="https://127.0.0.1" ${SCRIPT_NAME} --repo owner/repo
EOF
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
  if [[ -z "${path}" ]]; then
    return 0
  fi
  if [[ ! -f "${path}" ]]; then
    return 0
  fi

  # shellcheck disable=SC1090
  set -a
  source "${path}"
  set +a
  printf 'Loaded environment values from %s\n' "${path}"
}

CLI_TARGET_REPO=""
CLI_REDIRECT_URI=""
OPEN_BROWSER=true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)
      if [[ -z "${2:-}" ]]; then
        printf 'ERROR: --repo requires a value.\n' >&2
        exit 2
      fi
      CLI_TARGET_REPO="$2"
      shift 2
      ;;
    --redirect-uri)
      if [[ -z "${2:-}" ]]; then
        printf 'ERROR: --redirect-uri requires a value.\n' >&2
        exit 2
      fi
      CLI_REDIRECT_URI="$2"
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

load_env_file "${ENV_FILE}"

TARGET_REPO="${CLI_TARGET_REPO:-${GH_REPO:-}}"
SCHWAB_REDIRECT_URI="${CLI_REDIRECT_URI:-${SCHWAB_REDIRECT_URI:-https://127.0.0.1}}"

require_command python3
require_command curl
require_command gh
require_env SCHWAB_CLIENT_ID
require_env SCHWAB_CLIENT_SECRET

if ! gh auth status -h github.com >/dev/null 2>&1; then
  printf 'ERROR: GitHub CLI is not authenticated. Run: gh auth login\n' >&2
  exit 1
fi

if [[ -z "${TARGET_REPO}" ]]; then
  TARGET_REPO="$(gh repo view --json nameWithOwner -q '.nameWithOwner' 2>/dev/null || true)"
fi
if [[ -z "${TARGET_REPO}" ]]; then
  printf 'ERROR: unable to determine target repo. Pass --repo owner/repo.\n' >&2
  exit 1
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

printf 'Target repo: %s\n' "${TARGET_REPO}"
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

printf '%s' "${new_refresh_token}" | gh secret set SCHWAB_REFRESH_TOKEN --repo "${TARGET_REPO}"

printf 'Updated GitHub secret SCHWAB_REFRESH_TOKEN for %s.\n' "${TARGET_REPO}"
if [[ -n "${access_expires_in}" ]]; then
  printf 'access_token_expires_in=%ss\n' "${access_expires_in}"
fi
if [[ -n "${refresh_expires_in}" ]]; then
  printf 'refresh_token_expires_in=%ss\n' "${refresh_expires_in}"
fi
printf 'Rotation complete. Repeat this flow weekly (about every 7 days).\n'
