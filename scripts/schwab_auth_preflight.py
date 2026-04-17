#!/usr/bin/env python3
"""Fail fast when Schwab refresh credentials cannot mint an access token."""

from __future__ import annotations

import base64
import gzip
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from urllib import error, parse, request

SCHWAB_BASE_URL = "https://api.schwabapi.com"
SCHWAB_OAUTH_TOKEN_PATH = "/v1/oauth/token"


def _missing_required_env() -> list[str]:
    required = (
        "SCHWAB_REFRESH_TOKEN",
        "SCHWAB_CLIENT_ID",
        "SCHWAB_CLIENT_SECRET",
    )
    return [name for name in required if not str(os.getenv(name, "")).strip()]


def _schwab_base_url() -> str:
    raw = str(os.getenv("SCHWAB_BASE_URL", SCHWAB_BASE_URL)).strip()
    if not raw:
        return SCHWAB_BASE_URL
    return raw.rstrip("/")


def _timeout_seconds() -> int:
    raw = str(os.getenv("SCHWAB_REQUEST_TIMEOUT_SECONDS", "")).strip()
    if not raw:
        return 12
    try:
        parsed = int(raw)
    except ValueError:
        return 12
    return max(3, parsed)


def _max_attempts() -> int:
    raw = str(os.getenv("SCHWAB_FETCH_MAX_ATTEMPTS", "")).strip()
    if not raw:
        return 3
    try:
        parsed = int(raw)
    except ValueError:
        return 3
    return max(1, parsed)


def _retry_backoff_seconds() -> float:
    raw = str(os.getenv("SCHWAB_FETCH_RETRY_BACKOFF_SECONDS", "")).strip()
    if not raw:
        return 0.75
    try:
        parsed = float(raw)
    except ValueError:
        return 0.75
    return max(0.0, parsed)


def _decode_http_body(exc: error.HTTPError) -> str:
    raw_bytes = b""
    try:
        raw_bytes = exc.read()
    except Exception:
        raw_bytes = b""
    if not raw_bytes:
        return str(getattr(exc, "reason", "")).strip()

    decoded = ""
    try:
        decoded = raw_bytes.decode("utf-8", errors="replace")
    except Exception:
        decoded = ""
    if decoded and "\ufffd" not in decoded and decoded.strip():
        return decoded.strip()

    try:
        inflated = gzip.decompress(raw_bytes)
        decoded = inflated.decode("utf-8", errors="replace")
        if decoded.strip():
            return decoded.strip()
    except Exception:
        pass

    return decoded.strip() or repr(raw_bytes[:200])


def _http_status_error_message(exc: error.HTTPError) -> str:
    body = _decode_http_body(exc)
    parsed_json: dict[str, object] | None = None
    try:
        parsed = json.loads(body)
        if isinstance(parsed, dict):
            parsed_json = parsed
    except Exception:
        parsed_json = None

    if parsed_json is not None:
        err = str(parsed_json.get("error", "")).strip()
        desc = str(parsed_json.get("error_description", "")).strip()
        fields = [part for part in [err, desc] if part]
        if fields:
            return " | ".join(fields)

    return body or str(getattr(exc, "reason", "")).strip() or "unknown_error"


def _auth_failure_action(message: str) -> str:
    normalized = str(message).strip().lower()
    if not normalized:
        return ""
    if "refresh_token_authentication_error" in normalized or "unsupported_token_type" in normalized:
        return "rotate_schwab_refresh_token"
    if "invalid_client" in normalized or "unauthorized_client" in normalized:
        return "check_schwab_client_credentials"
    return ""


def _is_truthy(raw_value: str) -> bool:
    return str(raw_value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _export_access_token_if_requested(access_token: str, expires_in: int) -> None:
    if not _is_truthy(str(os.getenv("SCHWAB_PREFLIGHT_EXPORT_ACCESS_TOKEN", ""))):
        return

    output_path = str(os.getenv("GITHUB_ENV", "")).strip()
    if not output_path:
        raise RuntimeError(
            "SCHWAB_PREFLIGHT_EXPORT_ACCESS_TOKEN is enabled but GITHUB_ENV is not set."
        )

    expires_at = datetime.now(timezone.utc) + timedelta(seconds=max(1, int(expires_in)))
    expires_at_text = expires_at.strftime("%Y-%m-%dT%H:%M:%SZ")
    if _is_truthy(str(os.getenv("GITHUB_ACTIONS", ""))):
        print(f"::add-mask::{access_token}")

    with open(output_path, "a", encoding="utf-8") as fh:
        fh.write(f"SCHWAB_ACCESS_TOKEN={access_token}\n")
        fh.write(f"SCHWAB_ACCESS_TOKEN_EXPIRES_IN={int(expires_in)}\n")
        fh.write(f"SCHWAB_ACCESS_TOKEN_EXPIRES_AT_UTC={expires_at_text}\n")

    print(
        "schwab_auth_preflight_exported_access_token "
        f"expires_in={int(expires_in)} "
        f"expires_at_utc={expires_at_text}"
    )


def main() -> int:
    missing = _missing_required_env()
    if missing:
        print(
            "schwab_auth_preflight_failed "
            f"missing_env={','.join(missing)}",
            file=sys.stderr,
        )
        return 1

    refresh_token = str(os.getenv("SCHWAB_REFRESH_TOKEN", "")).strip()
    client_id = str(os.getenv("SCHWAB_CLIENT_ID", "")).strip()
    client_secret = str(os.getenv("SCHWAB_CLIENT_SECRET", "")).strip()
    token_url = f"{_schwab_base_url()}{SCHWAB_OAUTH_TOKEN_PATH}"
    payload = parse.urlencode(
        {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        }
    ).encode("utf-8")
    basic_auth = base64.b64encode(f"{client_id}:{client_secret}".encode("utf-8")).decode("ascii")
    req = request.Request(
        token_url,
        headers={
            "Authorization": f"Basic {basic_auth}",
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "BingBingBot-Schwab-Auth-Preflight/1.0",
        },
        data=payload,
        method="POST",
    )

    attempts = _max_attempts()
    backoff_seconds = _retry_backoff_seconds()
    raw_body = ""
    for attempt in range(attempts):
        try:
            with request.urlopen(req, timeout=_timeout_seconds()) as response:
                raw_body = response.read().decode("utf-8", errors="replace")
            break
        except error.HTTPError as exc:
            status = int(getattr(exc, "code", 0) or 0)
            message = _http_status_error_message(exc)
            action = _auth_failure_action(message)
            retryable_http = status in {408, 429, 500, 502, 503, 504}
            is_last_attempt = attempt >= (attempts - 1)
            if is_last_attempt or not retryable_http:
                action_field = f" action={action}" if action else ""
                print(
                    "schwab_auth_preflight_failed "
                    f"http_status={status}{action_field} message={message}",
                    file=sys.stderr,
                )
                return 1
            print(
                "schwab_auth_preflight_retrying "
                f"attempt={attempt + 1}/{attempts} http_status={status} message={message}",
                file=sys.stderr,
            )
            sleep_seconds = backoff_seconds * (2**attempt)
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
        except Exception as exc:
            is_last_attempt = attempt >= (attempts - 1)
            if is_last_attempt:
                print(
                    "schwab_auth_preflight_failed "
                    f"request_error={exc}",
                    file=sys.stderr,
                )
                return 1
            print(
                "schwab_auth_preflight_retrying "
                f"attempt={attempt + 1}/{attempts} request_error={exc}",
                file=sys.stderr,
            )
            sleep_seconds = backoff_seconds * (2**attempt)
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    try:
        parsed = json.loads(raw_body)
    except Exception as exc:
        print(
            "schwab_auth_preflight_failed "
            f"invalid_json={exc}",
            file=sys.stderr,
        )
        return 1

    access_token = str(parsed.get("access_token", "")).strip()
    expires_in_raw = parsed.get("expires_in")
    try:
        expires_in = int(float(expires_in_raw))
    except (TypeError, ValueError):
        expires_in = -1

    if not access_token or expires_in < 1:
        print(
            "schwab_auth_preflight_failed "
            "missing_required_fields=access_token,expires_in",
            file=sys.stderr,
        )
        return 1

    try:
        _export_access_token_if_requested(access_token, expires_in)
    except Exception as exc:
        print(
            "schwab_auth_preflight_failed "
            f"action=export_access_token message={exc}",
            file=sys.stderr,
        )
        return 1

    print(f"schwab_auth_preflight_ok access_token_len={len(access_token)} expires_in={expires_in}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
