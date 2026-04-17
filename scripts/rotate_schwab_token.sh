#!/usr/bin/env bash
set -euo pipefail

SCRIPT_NAME="$(basename "$0")"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_STOCK_ROOT="${REPO_ROOT}/../StockOpportunityScanner"
SCHWAB_OAUTH_AUTHORIZE_URL="${SCHWAB_OAUTH_AUTHORIZE_URL:-https://api.schwabapi.com/v1/oauth/authorize}"
SCHWAB_OAUTH_TOKEN_URL="${SCHWAB_OAUTH_TOKEN_URL:-https://api.schwabapi.com/v1/oauth/token}"
SCHWAB_ROTATION_TIMESTAMP_VAR="${SCHWAB_ROTATION_TIMESTAMP_VAR:-SCHWAB_REFRESH_TOKEN_ROTATED_AT_UTC}"
ENV_FILE="${REPO_ROOT}/.env.schwab.local"
STOCK_ROOT="${DEFAULT_STOCK_ROOT}"
STOCK_ENV_FILE=""

usage() {
  cat <<EOF_USAGE
Usage: ${SCRIPT_NAME} [--env-file PATH] [--redirect-uri URI] [--repo owner/repo] [--stock-root PATH] [--stock-env-file PATH] [--stock-repo owner/repo] [--no-stock-sync] [--no-open] [--print-token] [--no-github-secret]

Rotate SCHWAB_REFRESH_TOKEN once and fan the refreshed token out to BingBingBot and,
by default, the StockOpportunityScanner companion checkout.

Default targets:
  - Bing local env file: ${ENV_FILE}
  - Bing GitHub secret: SCHWAB_REFRESH_TOKEN
  - Bing GitHub variable: ${SCHWAB_ROTATION_TIMESTAMP_VAR}
  - Stock companion root: ${DEFAULT_STOCK_ROOT}
  - Stock local env file: <stock-root>/.env.local
  - Stock GitHub secret: SCHWAB_REFRESH_TOKEN
  - Stock GitHub variable: ${SCHWAB_ROTATION_TIMESTAMP_VAR}

Required environment variables:
  SCHWAB_CLIENT_ID
  SCHWAB_CLIENT_SECRET

Optional environment variables:
  SCHWAB_REDIRECT_URI (default: https://127.0.0.1)
  GH_REPO             (fallback target repo for Bing if --repo is omitted)
  SCHWAB_ROTATION_TIMESTAMP_VAR (default: SCHWAB_REFRESH_TOKEN_ROTATED_AT_UTC)

Behavior:
  1) Resolve and validate every enabled target before opening the browser.
  2) Build the Schwab OAuth authorize URL and prompt for the redirect URL.
  3) Exchange the authorization code once for fresh tokens.
  4) Update SCHWAB_REFRESH_TOKEN in Bing and, unless disabled, Stock env files.
  5) Update GitHub secrets and rotation timestamp variables unless --no-github-secret is used.

Examples:
  ${SCRIPT_NAME}
  ${SCRIPT_NAME} --stock-root ../StockOpportunityScanner
  ${SCRIPT_NAME} --stock-env-file .env.local --stock-repo owner/StockOpportunityScanner
  ${SCRIPT_NAME} --env-file .env.schwab.local --repo owner/BingBingBot --no-stock-sync
  ${SCRIPT_NAME} --no-github-secret --print-token
EOF_USAGE
}

error() {
  printf 'ERROR: %s\n' "$1" >&2
  exit 1
}

warn() {
  printf 'Warning: %s\n' "$1" >&2
}

require_command() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    error "required command not found: ${cmd}"
  fi
}

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    error "required environment variable is missing: ${name}"
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

resolve_path() {
  local base_dir="$1"
  local raw_path="$2"
  python3 - "${base_dir}" "${raw_path}" <<'PY'
from pathlib import Path
import sys

base_dir = Path(sys.argv[1]).expanduser()
raw_path = Path(sys.argv[2]).expanduser()
if not raw_path.is_absolute():
    raw_path = base_dir / raw_path
print(raw_path)
PY
}

resolve_repo_from_origin_remote() {
  local repo_root="$1"
  local remote_url=""
  remote_url="$(git -C "${repo_root}" config --get remote.origin.url 2>/dev/null || true)"
  if [[ -z "${remote_url}" ]]; then
    return 0
  fi

  python3 - "${remote_url}" <<'PY'
import re
import sys

remote_url = sys.argv[1].strip()
patterns = (
    r"^https://github\.com/([^/]+/[^/]+?)(?:\.git)?$",
    r"^git@github\.com:([^/]+/[^/]+?)(?:\.git)?$",
    r"^ssh://git@github\.com/([^/]+/[^/]+?)(?:\.git)?$",
)
for pattern in patterns:
    match = re.match(pattern, remote_url)
    if match:
        print(match.group(1))
        raise SystemExit(0)
raise SystemExit(0)
PY
}

resolve_target_repo() {
  local explicit_repo="$1"
  local repo_root="$2"
  local env_repo="$3"
  if [[ -n "${explicit_repo}" ]]; then
    printf '%s\n' "${explicit_repo}"
    return 0
  fi

  local repo_from_origin=""
  repo_from_origin="$(resolve_repo_from_origin_remote "${repo_root}")"

  if [[ -n "${repo_from_origin}" ]]; then
    if [[ -n "${env_repo}" && "${env_repo}" != "${repo_from_origin}" ]]; then
      warn "GH_REPO=${env_repo} differs from repo origin ${repo_from_origin}; using repo origin. Pass --repo to override."
    fi
    printf '%s\n' "${repo_from_origin}"
    return 0
  fi

  if [[ -n "${env_repo}" ]]; then
    printf '%s\n' "${env_repo}"
    return 0
  fi

  local detected_repo=""
  detected_repo="$(cd "${repo_root}" && gh repo view --json nameWithOwner -q '.nameWithOwner' 2>/dev/null || true)"
  printf '%s\n' "${detected_repo}"
}

github_secret_updated_at() {
  local repo="$1"
  local name="$2"
  gh api "repos/${repo}/actions/secrets/${name}" --jq '.updated_at' 2>/dev/null || true
}

verify_github_secret_update() {
  local repo="$1"
  local name="$2"
  local previous_updated_at="$3"
  local previous_local_token="$4"
  local new_token="$5"
  local current_updated_at=""
  local attempt

  for attempt in 1 2 3 4 5; do
    current_updated_at="$(github_secret_updated_at "${repo}" "${name}")"
    if [[ -n "${current_updated_at}" && "${current_updated_at}" != "${previous_updated_at}" ]]; then
      printf 'Verified GitHub secret %s for %s (updated_at=%s)\n' "${name}" "${repo}" "${current_updated_at}"
      return 0
    fi
    sleep 1
  done

  if [[ "${new_token}" != "${previous_local_token}" ]]; then
    printf 'ERROR: GitHub secret %s for %s did not show a metadata update after gh secret set.\n' "${name}" "${repo}" >&2
    if [[ -n "${previous_updated_at}" ]]; then
      printf 'Previous updated_at: %s\n' "${previous_updated_at}" >&2
    fi
    if [[ -n "${current_updated_at}" ]]; then
      printf 'Current updated_at: %s\n' "${current_updated_at}" >&2
    fi
    printf 'Check gh auth, target repo selection, and GitHub secret permissions.\n' >&2
    exit 1
  fi

  if [[ -n "${current_updated_at}" ]]; then
    warn "GitHub secret ${name} metadata for ${repo} remained ${current_updated_at}; token may be unchanged."
  else
    warn "unable to verify GitHub secret ${name} metadata for ${repo} after update."
  fi
}

read_refresh_token_env_var() {
  local path="$1"
  python3 - "${path}" <<'PY'
import re
import shlex
import sys
from pathlib import Path

env_path = Path(sys.argv[1]).expanduser()
if not env_path.exists():
    raise SystemExit(0)

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
    print(parsed)
    raise SystemExit(0)
PY
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

update_rotation_timestamp_var() {
  local repo="$1"
  local timestamp_utc="$2"
  if gh variable set "${SCHWAB_ROTATION_TIMESTAMP_VAR}" --repo "${repo}" --body "${timestamp_utc}" >/dev/null; then
    printf 'Updated GitHub variable %s for %s (%s)\n' "${SCHWAB_ROTATION_TIMESTAMP_VAR}" "${repo}" "${timestamp_utc}"
  else
    warn "unable to update GitHub variable ${SCHWAB_ROTATION_TIMESTAMP_VAR} for ${repo}."
  fi
}

CLI_REDIRECT_URI=""
CLI_TARGET_REPO=""
CLI_STOCK_REPO=""
OPEN_BROWSER=true
PRINT_TOKEN=false
UPDATE_GITHUB_SECRET=true
STOCK_SYNC=true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --redirect-uri)
      if [[ -z "${2:-}" ]]; then
        error '--redirect-uri requires a value.'
      fi
      CLI_REDIRECT_URI="$2"
      shift 2
      ;;
    --repo)
      if [[ -z "${2:-}" ]]; then
        error '--repo requires a value.'
      fi
      CLI_TARGET_REPO="$2"
      shift 2
      ;;
    --stock-root)
      if [[ -z "${2:-}" ]]; then
        error '--stock-root requires a value.'
      fi
      STOCK_ROOT="$2"
      shift 2
      ;;
    --stock-env-file)
      if [[ -z "${2:-}" ]]; then
        error '--stock-env-file requires a value.'
      fi
      STOCK_ENV_FILE="$2"
      shift 2
      ;;
    --stock-repo)
      if [[ -z "${2:-}" ]]; then
        error '--stock-repo requires a value.'
      fi
      CLI_STOCK_REPO="$2"
      shift 2
      ;;
    --env-file)
      if [[ -z "${2:-}" ]]; then
        error '--env-file requires a value.'
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
    --no-stock-sync)
      STOCK_SYNC=false
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

require_command python3
require_command curl
require_command git

ENV_FILE="$(resolve_path "${REPO_ROOT}" "${ENV_FILE}")"
printf 'Using Bing env file: %s\n' "${ENV_FILE}"
load_env_file "${ENV_FILE}"

require_env SCHWAB_CLIENT_ID
require_env SCHWAB_CLIENT_SECRET

SCHWAB_REDIRECT_URI="${CLI_REDIRECT_URI:-${SCHWAB_REDIRECT_URI:-https://127.0.0.1}}"
BING_TARGET_REPO=""
STOCK_TARGET_REPO=""
BING_PREVIOUS_LOCAL_REFRESH_TOKEN="${SCHWAB_REFRESH_TOKEN:-}"
STOCK_PREVIOUS_LOCAL_REFRESH_TOKEN=""
BING_SECRET_UPDATED_AT_BEFORE=""
STOCK_SECRET_UPDATED_AT_BEFORE=""

if [[ "${STOCK_SYNC}" == "true" ]]; then
  STOCK_ROOT="$(resolve_path "${REPO_ROOT}" "${STOCK_ROOT}")"
  if [[ ! -d "${STOCK_ROOT}" ]]; then
    error "stock companion root does not exist: ${STOCK_ROOT}. Pass --stock-root PATH or use --no-stock-sync."
  fi

  if [[ -z "${STOCK_ENV_FILE}" ]]; then
    STOCK_ENV_FILE="${STOCK_ROOT}/.env.local"
  else
    STOCK_ENV_FILE="$(resolve_path "${STOCK_ROOT}" "${STOCK_ENV_FILE}")"
  fi
  STOCK_PREVIOUS_LOCAL_REFRESH_TOKEN="$(read_refresh_token_env_var "${STOCK_ENV_FILE}")"
fi

if [[ "${UPDATE_GITHUB_SECRET}" == "true" ]]; then
  require_command gh
  if ! gh auth status -h github.com >/dev/null 2>&1; then
    error 'GitHub CLI is not authenticated. Run: gh auth login (or use --no-github-secret).'
  fi

  BING_TARGET_REPO="$(resolve_target_repo "${CLI_TARGET_REPO}" "${REPO_ROOT}" "${GH_REPO:-}")"
  if [[ -z "${BING_TARGET_REPO}" ]]; then
    error 'unable to resolve Bing GitHub repo. Pass --repo owner/repo or set GH_REPO.'
  fi
  BING_SECRET_UPDATED_AT_BEFORE="$(github_secret_updated_at "${BING_TARGET_REPO}" SCHWAB_REFRESH_TOKEN)"

  if [[ "${STOCK_SYNC}" == "true" ]]; then
    STOCK_TARGET_REPO="$(resolve_target_repo "${CLI_STOCK_REPO}" "${STOCK_ROOT}" "")"
    if [[ -z "${STOCK_TARGET_REPO}" ]]; then
      error "unable to resolve Stock GitHub repo from ${STOCK_ROOT}. Pass --stock-repo owner/repo or use --no-stock-sync."
    fi
    STOCK_SECRET_UPDATED_AT_BEFORE="$(github_secret_updated_at "${STOCK_TARGET_REPO}" SCHWAB_REFRESH_TOKEN)"
  fi
fi

printf 'Bing GitHub secret target repo: %s\n' "${BING_TARGET_REPO:-disabled}"
if [[ "${UPDATE_GITHUB_SECRET}" == "false" ]]; then
  printf 'GitHub secret/variable updates disabled via --no-github-secret\n'
fi

if [[ "${STOCK_SYNC}" == "true" ]]; then
  printf 'Using Stock companion root: %s\n' "${STOCK_ROOT}"
  printf 'Using Stock env file: %s\n' "${STOCK_ENV_FILE}"
  if [[ "${UPDATE_GITHUB_SECRET}" == "true" ]]; then
    printf 'Stock GitHub secret target repo: %s\n' "${STOCK_TARGET_REPO}"
  fi
else
  printf 'Stock companion sync disabled via --no-stock-sync\n'
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
  error 'redirect URL is required.'
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
rotation_timestamp_utc="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

if [[ -z "${new_refresh_token}" ]]; then
  error 'no refresh token extracted from token response.'
fi

upsert_refresh_token_env_var "${ENV_FILE}" "${new_refresh_token}"
chmod 600 "${ENV_FILE}" 2>/dev/null || true
printf 'Updated SCHWAB_REFRESH_TOKEN in %s\n' "${ENV_FILE}"
verify_refresh_token_env_var "${ENV_FILE}" "${new_refresh_token}"

if [[ "${STOCK_SYNC}" == "true" ]]; then
  upsert_refresh_token_env_var "${STOCK_ENV_FILE}" "${new_refresh_token}"
  chmod 600 "${STOCK_ENV_FILE}" 2>/dev/null || true
  printf 'Updated SCHWAB_REFRESH_TOKEN in %s\n' "${STOCK_ENV_FILE}"
  verify_refresh_token_env_var "${STOCK_ENV_FILE}" "${new_refresh_token}"
fi

if [[ "${UPDATE_GITHUB_SECRET}" == "true" ]]; then
  printf '%s' "${new_refresh_token}" | gh secret set SCHWAB_REFRESH_TOKEN --repo "${BING_TARGET_REPO}"
  printf 'Updated GitHub secret SCHWAB_REFRESH_TOKEN for %s\n' "${BING_TARGET_REPO}"
  verify_github_secret_update "${BING_TARGET_REPO}" "SCHWAB_REFRESH_TOKEN" "${BING_SECRET_UPDATED_AT_BEFORE}" "${BING_PREVIOUS_LOCAL_REFRESH_TOKEN}" "${new_refresh_token}"

  if [[ "${STOCK_SYNC}" == "true" ]]; then
    printf '%s' "${new_refresh_token}" | gh secret set SCHWAB_REFRESH_TOKEN --repo "${STOCK_TARGET_REPO}"
    printf 'Updated GitHub secret SCHWAB_REFRESH_TOKEN for %s\n' "${STOCK_TARGET_REPO}"
    verify_github_secret_update "${STOCK_TARGET_REPO}" "SCHWAB_REFRESH_TOKEN" "${STOCK_SECRET_UPDATED_AT_BEFORE}" "${STOCK_PREVIOUS_LOCAL_REFRESH_TOKEN}" "${new_refresh_token}"
  fi

  update_rotation_timestamp_var "${BING_TARGET_REPO}" "${rotation_timestamp_utc}"
  if [[ "${STOCK_SYNC}" == "true" ]]; then
    update_rotation_timestamp_var "${STOCK_TARGET_REPO}" "${rotation_timestamp_utc}"
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
