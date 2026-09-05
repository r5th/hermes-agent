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
