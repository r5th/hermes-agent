"""State matrix for the single-use-OAuth Credential sources (ADR-0002, cut 2).

Covers ``openai-codex`` and ``xai-oauth`` — the single-use-refresh OAuth
providers that share machinery with anthropic but seed from the device-code
auth-store singleton (inside HERMES_HOME, already isolated by the autouse
``_hermetic_environment`` fixture) rather than env keys. Each test drives the
public ``resolve()`` / ``refresh()`` seam; the pure refresh POST is stubbed so
tests never rotate a real token.
"""
from __future__ import annotations

import time

import pytest

from agent.credential_pool import (
    AUTH_TYPE_OAUTH,
    PooledCredential,
    write_credential_pool,
)
from agent.credential_source import (
    CODEX_DEFAULT_BASE_URL,
    XAI_OAUTH_DEFAULT_BASE_URL,
    CodexCredentialSource,
    XaiOAuthCredentialSource,
)


def _seed_oauth_pool(provider: str, *, access_token: str, refresh_token: str = "r1",
                     base_url: str, expires_at_ms: int) -> None:
    entry = PooledCredential(
        provider=provider,
        id="p1",
        label=f"{provider} oauth",
        auth_type=AUTH_TYPE_OAUTH,
        priority=0,
        source="device_code",
        access_token=access_token,
        refresh_token=refresh_token,
        base_url=base_url,
        expires_at_ms=expires_at_ms,
    )
    write_credential_pool(provider, [entry.to_dict()])


# ── pool-only ───────────────────────────────────────────────────────────────


def test_codex_pool_only_resolves_selected_entry():
    _seed_oauth_pool(
        "openai-codex",
        access_token="codex-access",
        base_url=CODEX_DEFAULT_BASE_URL,
        expires_at_ms=int(time.time() * 1000) + 3_600_000,
    )

    cred = CodexCredentialSource().resolve()

    assert cred is not None
    assert cred.api_key == "codex-access"
    assert cred.base_url == CODEX_DEFAULT_BASE_URL
    assert cred.provenance == "pool"


def test_xai_pool_only_resolves_selected_entry():
    _seed_oauth_pool(
        "xai-oauth",
        access_token="xai-access",
        base_url=XAI_OAUTH_DEFAULT_BASE_URL,
        expires_at_ms=int(time.time() * 1000) + 3_600_000,
    )

    cred = XaiOAuthCredentialSource().resolve()

    assert cred is not None
    assert cred.api_key == "xai-access"
    assert cred.base_url == XAI_OAUTH_DEFAULT_BASE_URL  # validated, falls back to default host
    assert cred.provenance == "pool"


# ── legacy (auth-store runtime resolver, no pool) ────────────────────────────


def test_codex_legacy_when_no_pool(monkeypatch):
    """No pool → the codex reader falls through to the auth.json tokens."""
    monkeypatch.setattr(
        "hermes_cli.auth._read_codex_tokens",
        lambda: {"tokens": {"access_token": "codex-legacy"}},
    )

    cred = CodexCredentialSource().resolve()

    assert cred is not None
    assert cred.api_key == "codex-legacy"
    assert cred.base_url == CODEX_DEFAULT_BASE_URL
    assert cred.provenance == "legacy"


def test_xai_legacy_when_no_pool(monkeypatch):
    """No pool → the xAI runtime resolver supplies (api_key, base_url)."""
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_xai_oauth_runtime_credentials",
        lambda **_: {"api_key": "xai-legacy", "base_url": XAI_OAUTH_DEFAULT_BASE_URL},
    )

    cred = XaiOAuthCredentialSource().resolve()

    assert cred is not None
    assert cred.api_key == "xai-legacy"
    assert cred.base_url == XAI_OAUTH_DEFAULT_BASE_URL
    assert cred.provenance == "legacy"


# ── forced rotation (pool force: rotate even a still-valid token) ─────────────


def test_codex_force_rotates_pool_token(monkeypatch):
    _seed_oauth_pool(
        "openai-codex",
        access_token="codex-valid",
        refresh_token="codex-r1",
        base_url=CODEX_DEFAULT_BASE_URL,
        expires_at_ms=int(time.time() * 1000) + 3_600_000,  # NOT expired
    )
    posted = {}

    def _fake_refresh(access_token, refresh_token, **_):
        posted["access_token"] = access_token
        return {"access_token": "codex-rotated", "refresh_token": "codex-r2", "last_refresh": None}

    monkeypatch.setattr("hermes_cli.auth.refresh_codex_oauth_pure", _fake_refresh)

    cred = CodexCredentialSource().refresh(force=True)

    assert posted["access_token"] == "codex-valid"
    assert cred is not None
    assert cred.api_key == "codex-rotated"
    assert cred.provenance == "pool"


def test_xai_force_rotates_pool_token(monkeypatch):
    _seed_oauth_pool(
        "xai-oauth",
        access_token="xai-valid",
        refresh_token="xai-r1",
        base_url=XAI_OAUTH_DEFAULT_BASE_URL,
        expires_at_ms=int(time.time() * 1000) + 3_600_000,  # NOT expired
    )
    posted = {}

    def _fake_refresh(access_token, refresh_token, **_):
        posted["access_token"] = access_token
        return {"access_token": "xai-rotated", "refresh_token": "xai-r2", "last_refresh": None}

    monkeypatch.setattr("hermes_cli.auth.refresh_xai_oauth_pure", _fake_refresh)

    cred = XaiOAuthCredentialSource().refresh(force=True)

    assert posted["access_token"] == "xai-valid"
    assert cred is not None
    assert cred.api_key == "xai-rotated"
    assert cred.provenance == "pool"


# ── reactive singleton refresh (main-path 401 recovery, ADR-0002 cut 3) ───────
# Singleton-only, NOT pool-force-first: rotate the device_code singleton's token
# only when it is the credential currently in use, guarding against a silent
# mid-conversation account swap. Drives the runtime resolver directly (the same
# auth-store seam the old hand-coded _try_refresh_codex_client_credentials used).


def test_codex_reactive_refresh_rotates_when_active_is_singleton(monkeypatch):
    def _fake_resolve(force_refresh=False, refresh_if_expiring=True, **_):
        # Peek (refresh_if_expiring=False) sees the active token; the force mints a new one.
        return {
            "api_key": "codex-rotated" if force_refresh else "codex-active",
            "base_url": CODEX_DEFAULT_BASE_URL,
        }

    monkeypatch.setattr("hermes_cli.auth.resolve_codex_runtime_credentials", _fake_resolve)

    cred = CodexCredentialSource().reactive_singleton_refresh("codex-active")

    assert cred is not None
    assert cred.api_key == "codex-rotated"
    assert cred.base_url == CODEX_DEFAULT_BASE_URL
    assert cred.provenance == "legacy"


def test_xai_reactive_refresh_rotates_when_active_is_singleton(monkeypatch):
    def _fake_resolve(force_refresh=False, refresh_if_expiring=True, **_):
        return {
            "api_key": "xai-rotated" if force_refresh else "xai-active",
            "base_url": XAI_OAUTH_DEFAULT_BASE_URL,
        }

    monkeypatch.setattr("hermes_cli.auth.resolve_xai_oauth_runtime_credentials", _fake_resolve)

    cred = XaiOAuthCredentialSource().reactive_singleton_refresh("xai-active")

    assert cred is not None
    assert cred.api_key == "xai-rotated"


def test_reactive_refresh_skips_when_active_differs_from_singleton(monkeypatch):
    """Account-swap guard: a non-singleton active credential must NOT rotate the
    singleton — that would spend its single-use refresh token and silently swap
    accounts mid-conversation. The force resolver must never run."""
    force_calls = {"count": 0}

    def _fake_resolve(force_refresh=False, refresh_if_expiring=True, **_):
        if force_refresh:
            force_calls["count"] += 1
        return {"api_key": "singleton-account-token", "base_url": XAI_OAUTH_DEFAULT_BASE_URL}

    monkeypatch.setattr("hermes_cli.auth.resolve_xai_oauth_runtime_credentials", _fake_resolve)

    cred = XaiOAuthCredentialSource().reactive_singleton_refresh("a-different-pool-token")

    assert cred is None
    assert force_calls["count"] == 0


def test_reactive_refresh_none_when_no_new_token_minted(monkeypatch):
    """The runtime resolver returns the same stale token when a refresh fails
    silently — no NEW material to adopt, so the source reports nothing."""
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_codex_runtime_credentials",
        lambda **_: {"api_key": "codex-active", "base_url": CODEX_DEFAULT_BASE_URL},
    )

    cred = CodexCredentialSource().reactive_singleton_refresh("codex-active")

    assert cred is None
