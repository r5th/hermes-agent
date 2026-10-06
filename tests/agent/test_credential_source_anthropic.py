"""E2E state matrix for the anthropic ``CredentialSource`` (ADR-0002, first cut).

Each test drives one anthropic auth state through the public
``resolve()`` / ``refresh()`` seam against the per-test isolated ``HERMES_HOME``
(the autouse ``_hermetic_environment`` fixture), asserting the ``Credential``
value object that today's smeared resolution produces. Behaviour is held
identical across the strangler cut by this matrix:

    env-only · pool-only · pool+legacy · expired→refresh · single-use race
"""
from __future__ import annotations

import json
import time

import pytest

from agent.credential_pool import (
    AUTH_TYPE_API_KEY,
    AUTH_TYPE_OAUTH,
    STATUS_DEAD,
    PooledCredential,
    write_credential_pool,
)
from agent.credential_source import (
    ANTHROPIC_DEFAULT_BASE_URL,
    AnthropicCredentialSource,
)


@pytest.fixture(autouse=True)
def _isolate_claude_code_singleton(tmp_path, monkeypatch):
    """Point the shared Claude Code OAuth file at a per-test tmp path.

    The suite's ``_hermetic_environment`` fixture deliberately does NOT redirect
    ``HOME`` (see tests/conftest.py), so ``claude_code_credentials_path()`` would
    otherwise read the developer's REAL ``~/.claude/.credentials.json`` — leaking
    a live token into resolution and, worse, letting the refresh/race slices POST
    and spend the real single-use refresh token. Isolate it for every test here.
    The seeding path (`credential_pool._seed_from_singletons`) reads through these
    same module attributes via function-local imports, so this covers both.
    """
    cc_path = tmp_path / "claude" / ".credentials.json"
    monkeypatch.setattr(
        "agent.anthropic_credentials.claude_code_credentials_path", lambda: cc_path
    )
    monkeypatch.setattr(
        "agent.anthropic_credentials._read_claude_code_credentials_from_keychain",
        lambda: None,
    )
    return cc_path


@pytest.fixture
def anthropic_source() -> AnthropicCredentialSource:
    return AnthropicCredentialSource()


def _no_env_credentials(monkeypatch) -> None:
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(name, raising=False)


def _seed_pool_api_key(api_key: str, *, base_url: str = ANTHROPIC_DEFAULT_BASE_URL) -> None:
    entry = PooledCredential(
        provider="anthropic",
        id="pool1",
        label="anthropic console",
        auth_type=AUTH_TYPE_API_KEY,
        priority=0,
        source="manual",
        access_token=api_key,
        base_url=base_url,
    )
    write_credential_pool("anthropic", [entry.to_dict()])


# ── env-only ──────────────────────────────────────────────────────────────


def test_env_only_resolves_api_key_from_environment(anthropic_source, monkeypatch):
    """With only ``ANTHROPIC_API_KEY`` in the env (no auth.json, no Claude Code
    file), resolve() returns that key at the default base URL. ``load_pool()``
    seeds a pool entry from the env key, so — matching today's ``_try_anthropic``
    — resolution flows through the pool path (provenance ``pool``)."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api-envkey")

    cred = anthropic_source.resolve()

    assert cred is not None
    assert cred.api_key == "sk-ant-api-envkey"
    assert cred.base_url == ANTHROPIC_DEFAULT_BASE_URL
    assert cred.provenance == "pool"


# ── pool-only ─────────────────────────────────────────────────────────────


def test_pool_only_resolves_selected_pool_entry(anthropic_source, monkeypatch):
    """With a usable pool entry and no env credentials, resolve() returns the
    pooled api_key at the entry's base URL, provenance ``pool``."""
    _no_env_credentials(monkeypatch)
    _seed_pool_api_key("sk-ant-api-poolkey")

    cred = anthropic_source.resolve()

    assert cred is not None
    assert cred.api_key == "sk-ant-api-poolkey"
    assert cred.base_url == ANTHROPIC_DEFAULT_BASE_URL
    assert cred.provenance == "pool"


# ── pool + legacy fallback ──────────────────────────────────────────────────


def test_pool_present_but_unusable_falls_back_to_legacy(anthropic_source, monkeypatch):
    """A persisted pool entry that ``select()`` rejects (DEAD, unchanged token so
    reseeding preserves the status) leaves the pool present-but-empty. Resolution
    must fall back to the legacy resolver's standalone env token — parity with
    ``_try_anthropic`` — provenance ``legacy``, not a hard failure."""
    token = "sk-ant-oat-legacytoken"
    monkeypatch.setenv("ANTHROPIC_TOKEN", token)
    dead_entry = PooledCredential(
        provider="anthropic",
        id="dead1",
        label="anthropic oauth",
        auth_type=AUTH_TYPE_OAUTH,
        priority=0,
        source="env:ANTHROPIC_TOKEN",  # non-manual: DEAD is skipped, never revived by resync
        access_token=token,  # unchanged on reseed → _upsert_entry keeps last_status
        last_status=STATUS_DEAD,
        base_url=ANTHROPIC_DEFAULT_BASE_URL,
    )
    write_credential_pool("anthropic", [dead_entry.to_dict()])

    cred = anthropic_source.resolve()

    assert cred is not None
    assert cred.api_key == token
    assert cred.base_url == ANTHROPIC_DEFAULT_BASE_URL
    assert cred.provenance == "legacy"


# ── expired → refresh (pool refresh path) ───────────────────────────────────


def _write_claude_code_file(cc_path, *, access_token, refresh_token, expires_at_ms):
    cc_path.parent.mkdir(parents=True, exist_ok=True)
    cc_path.write_text(
        json.dumps(
            {
                "claudeAiOauth": {
                    "accessToken": access_token,
                    "refreshToken": refresh_token,
                    "expiresAt": expires_at_ms,
                    "scopes": ["user:inference"],
                }
            }
        ),
        encoding="utf-8",
    )


def test_expired_pool_token_is_refreshed_via_pool(
    anthropic_source, monkeypatch, _isolate_claude_code_singleton
):
    """An expired Claude Code OAuth credential seeds an expired pool entry.
    refresh() drives the pool refresh path, which POSTs (stubbed) and returns
    the rotated access token — provenance ``pool``. The real endpoint is never
    called; only the pure refresh primitive is stubbed."""
    # Anthropic must be explicitly configured for the Claude Code singleton to seed.
    monkeypatch.setattr(
        "hermes_cli.auth.is_provider_explicitly_configured", lambda provider: True
    )
    _write_claude_code_file(
        _isolate_claude_code_singleton,
        access_token="sk-ant-oat-EXPIRED",
        refresh_token="refresh-abc",
        expires_at_ms=int(time.time() * 1000) - 60_000,  # already expired
    )

    posted = {}

    def _fake_refresh(refresh_token, *, use_json=False):
        posted["refresh_token"] = refresh_token
        return {
            "access_token": "sk-ant-oat-FRESH",
            "refresh_token": "refresh-def",
            "expires_at_ms": int(time.time() * 1000) + 3_600_000,
        }

    monkeypatch.setattr(
        "agent.anthropic_credentials.refresh_anthropic_oauth_pure", _fake_refresh
    )

    cred = anthropic_source.refresh()

    assert posted["refresh_token"] == "refresh-abc"  # spent the expired pair's refresh token
    assert cred is not None
    assert cred.api_key == "sk-ant-oat-FRESH"
    assert cred.provenance == "pool"


# ── single-use race: no-replay guard ────────────────────────────────────────


def test_spent_uncommitted_refresh_token_is_never_replayed(
    anthropic_source, monkeypatch, _isolate_claude_code_singleton
):
    """The single-use invariant: when a refresh token was already spent by a
    peer whose rotation never committed (spent-rotation sidecar verdict),
    refresh() must NOT POST it again (a replay yields ``invalid_grant``). With no
    other usable credential it fails closed (returns None) rather than replaying.

    This pins, deterministically, the protection the concurrent race relies on —
    inherited by refresh() via the pool refresh machinery — without threads."""
    from agent.anthropic_credentials import mark_rotation_consumed_uncommitted

    monkeypatch.setattr(
        "hermes_cli.auth.is_provider_explicitly_configured", lambda provider: True
    )
    cc_path = _isolate_claude_code_singleton
    _write_claude_code_file(
        cc_path,
        access_token="sk-ant-oat-SPENT",
        refresh_token="refresh-spent",
        expires_at_ms=int(time.time() * 1000) - 60_000,  # expired → would try to refresh
    )
    # A peer spent this pair but never committed the replacement.
    mark_rotation_consumed_uncommitted(
        "refresh-spent", "sk-ant-oat-SPENT", source_path=cc_path
    )

    def _must_not_post(refresh_token, *, use_json=False):
        raise AssertionError(
            f"refresh() replayed a spent single-use refresh token ({refresh_token!r})"
        )

    monkeypatch.setattr(
        "agent.anthropic_credentials.refresh_anthropic_oauth_pure", _must_not_post
    )

    cred = anthropic_source.refresh()

    assert cred is None  # fails closed; no replay, no other usable credential


# ── forced refresh (aux 401 recovery: rotate even when not yet expired) ──────


def test_force_refresh_rotates_a_still_valid_pool_token(
    anthropic_source, monkeypatch, _isolate_claude_code_singleton
):
    """refresh(force=True) is the aux 401-recovery semantic: the current token
    may still be within its expiry window yet is being rejected, so a rotation
    must be forced. A plain resolve() would keep returning the un-expired token;
    force drives the pool's force-refresh, POSTs (stubbed), and returns the
    rotated token."""
    monkeypatch.setattr(
        "hermes_cli.auth.is_provider_explicitly_configured", lambda provider: True
    )
    _write_claude_code_file(
        _isolate_claude_code_singleton,
        access_token="sk-ant-oat-VALID",
        refresh_token="refresh-1",
        expires_at_ms=int(time.time() * 1000) + 3_600_000,  # NOT expired
    )

    posted = {}

    def _fake_refresh(refresh_token, *, use_json=False):
        posted["refresh_token"] = refresh_token
        return {
            "access_token": "sk-ant-oat-FORCED",
            "refresh_token": "refresh-2",
            "expires_at_ms": int(time.time() * 1000) + 3_600_000,
        }

    monkeypatch.setattr(
        "agent.anthropic_credentials.refresh_anthropic_oauth_pure", _fake_refresh
    )

    # Non-forcing refresh leaves the still-valid token alone (no POST).
    plain = anthropic_source.refresh()
    assert plain is not None and plain.api_key == "sk-ant-oat-VALID"
    assert "refresh_token" not in posted

    # Forced refresh rotates it.
    forced = anthropic_source.refresh(force=True)
    assert posted["refresh_token"] == "refresh-1"
    assert forced is not None
    assert forced.api_key == "sk-ant-oat-FORCED"
    assert forced.provenance == "pool"
