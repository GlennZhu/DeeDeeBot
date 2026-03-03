#!/usr/bin/env python3
"""Fail fast when Schwab refresh credentials cannot mint an access token."""

from __future__ import annotations

import base64
import json
import os
import sys
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

    try:
        with request.urlopen(req, timeout=_timeout_seconds()) as response:
            raw_body = response.read().decode("utf-8", errors="replace")
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else ""
        print(
            "schwab_auth_preflight_failed "
            f"http_status={exc.code} message={body or exc.reason}",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:
        print(
            "schwab_auth_preflight_failed "
            f"request_error={exc}",
            file=sys.stderr,
        )
        return 1

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

    print(f"schwab_auth_preflight_ok access_token_len={len(access_token)} expires_in={expires_in}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
