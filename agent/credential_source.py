"""Credential sources: per-provider strategies that resolve and refresh a Credential.

ADR-0002 consolidates the smeared per-provider credential logic behind a
``CredentialSource`` ABC returning a ``Credential`` value object that hides
whether the material came from the pool or a legacy resolver. This is the
first strangler cut — anthropic only. See ``agent/CONTEXT.md`` for the
vocabulary (Credential, Credential source, Credential resolution, Credential
pool) and ``agent/docs/adr/0002`` for the boundary: a source owns credential
*acquisition and refresh only*; SDK client construction stays in
``client_lifecycle``.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional

ANTHROPIC_DEFAULT_BASE_URL = "https://api.anthropic.com"
# Canonical values live in hermes_cli.auth_constants (DEFAULT_CODEX_BASE_URL /
# DEFAULT_XAI_OAUTH_BASE_URL); mirrored here to avoid an import at class-def time.
CODEX_DEFAULT_BASE_URL = "https://chatgpt.com/backend-api/codex"
XAI_OAUTH_DEFAULT_BASE_URL = "https://api.x.ai/v1"


@dataclass(frozen=True)
class Credential:
    """The material one model request authenticates with: an ``api_key`` and its
    ``base_url``, plus ``provenance`` — where it came from (``pool`` or ``legacy``).
    A single value regardless of origin."""

    api_key: str
    base_url: str
    provenance: str


class CredentialSource(abc.ABC):
    """Per-provider strategy that produces and refreshes a Credential for one
    provider. Owns acquisition and refresh only — not client construction."""

    @abc.abstractmethod
    def resolve(self) -> Optional[Credential]:
        """The Credential for this provider's next request, or None if none is available."""

    @abc.abstractmethod
    def refresh(self, *, force: bool = False) -> Optional[Credential]:
        """Fresh material for this provider, or None if refresh is impossible.

        ``force`` rotates even a not-yet-expired credential (401 recovery: the
        server rejected a token still inside its expiry window). Without it,
        refresh re-resolves and only refreshes on expiry.
        """


class AnthropicCredentialSource(CredentialSource):
    def resolve(self) -> Optional[Credential]:
        from agent.auxiliary_client import (
            _pool_runtime_api_key,
            _pool_runtime_base_url,
            _select_pool_entry,
        )

        pool_present, entry = _select_pool_entry("anthropic")
        if pool_present and entry is not None:
            api_key = _pool_runtime_api_key(entry)
            if api_key:
                base_url = _pool_runtime_base_url(entry, ANTHROPIC_DEFAULT_BASE_URL)
                return Credential(api_key=api_key, base_url=base_url, provenance="pool")
        # Pool absent/empty/unusable: legacy resolver so a dead pool entry can't
        # wedge resolution when a standalone credential exists (parity with _try_anthropic).
        from agent.anthropic_credentials import resolve_anthropic_token

        token = resolve_anthropic_token()
        if not token:
            return None
        return Credential(
            api_key=token, base_url=ANTHROPIC_DEFAULT_BASE_URL, provenance="legacy"
        )

    def refresh(self, *, force: bool = False) -> Optional[Credential]:
        # Re-resolve, auto-refreshing: the pool's select() refreshes an expired
        # (single-use) OAuth entry on the way out, and the legacy resolver
        # auto-refreshes an expired Claude Code token. Both surface the rotated
        # material through the same shaping as resolve(). Mirrors
        # _try_refresh_anthropic_client_credentials (re-run resolution, adopt the
        # new token); client construction stays in client_lifecycle (ADR-0002).
        if not force:
            return self.resolve()
        return self._force_refresh()

    def _force_refresh(self) -> Optional[Credential]:
        # Aux 401 recovery: rotate even a still-valid token. Pool force first
        # (mirrors _refresh_xai_oauth_credentials: select() then the force
        # try_refresh_current()), then the legacy single-use OAuth refresh, then
        # a plain re-resolve as a last resort.
        from agent.auxiliary_client import _pool_runtime_api_key, _pool_runtime_base_url

        try:
            from agent.credential_pool import load_pool

            pool = load_pool("anthropic")
        except Exception:
            pool = None
        if pool is not None and pool.has_credentials():
            pool.select()
            refreshed = pool.try_refresh_current()
            api_key = _pool_runtime_api_key(refreshed) if refreshed is not None else ""
            if api_key:
                base_url = _pool_runtime_base_url(refreshed, ANTHROPIC_DEFAULT_BASE_URL)
                return Credential(api_key=api_key, base_url=base_url, provenance="pool")

        # No pooled entry to force (e.g. anthropic not explicitly configured, so
        # the Claude Code singleton never seeds): force the legacy single-use
        # OAuth refresh directly off the credential file.
        from agent.anthropic_credentials import (
            _refresh_oauth_token,
            read_claude_code_credentials,
        )

        creds = read_claude_code_credentials()
        token = (
            _refresh_oauth_token(creds)
            if isinstance(creds, dict) and creds.get("refreshToken")
            else None
        )
        token = str(token or "").strip()
        if token:
            return Credential(
                api_key=token, base_url=ANTHROPIC_DEFAULT_BASE_URL, provenance="legacy"
            )
        return self.resolve()


class _SingleUseOAuthCredentialSource(CredentialSource):
    """Shared base for single-use-refresh OAuth providers (openai-codex,
    xai-oauth): pool-first acquisition with an auth-store runtime-resolver
    fallback; force rotates via the pool then the legacy runtime resolver.

    Subclasses set ``PROVIDER`` / ``DEFAULT_BASE_URL`` and supply the pool-entry
    base_url resolution and the two legacy (non-force / force) resolvers.
    """

    PROVIDER: str = ""
    DEFAULT_BASE_URL: str = ""

    def resolve(self) -> Optional[Credential]:
        from agent.auxiliary_client import _pool_runtime_api_key, _select_pool_entry

        pool_present, entry = _select_pool_entry(self.PROVIDER)
        if pool_present and entry is not None:
            api_key = _pool_runtime_api_key(entry)
            if api_key:
                return Credential(
                    api_key=api_key, base_url=self._pool_base_url(entry), provenance="pool"
                )
        # Pool absent/empty/unusable: the auth-store runtime resolver.
        return self._legacy_credential()

    def refresh(self, *, force: bool = False) -> Optional[Credential]:
        if not force:
            return self.resolve()
        return self._force_refresh()

    def _force_refresh(self) -> Optional[Credential]:
        # Pool force first (mirrors _refresh_xai_oauth_credentials: select() then
        # the force try_refresh_current()), then the legacy force resolver.
        from agent.auxiliary_client import _pool_runtime_api_key

        try:
            from agent.credential_pool import load_pool

            pool = load_pool(self.PROVIDER)
        except Exception:
            pool = None
        if pool is not None and pool.has_credentials():
            pool.select()
            refreshed = pool.try_refresh_current()
            api_key = _pool_runtime_api_key(refreshed) if refreshed is not None else ""
            if api_key:
                return Credential(
                    api_key=api_key, base_url=self._pool_base_url(refreshed), provenance="pool"
                )
        return self._legacy_force_credential()

    def reactive_singleton_refresh(
        self, active_api_key: str, *, force: bool = True
    ) -> Optional[Credential]:
        """Main-path 401 recovery for the singleton-only case (ADR-0002): rotate
        the device_code singleton's token, but ONLY when it is the credential
        currently in use. A non-singleton active credential (a manual pool entry
        or an explicit ``api_key=``) must not silently adopt the singleton's
        tokens mid-conversation — the pool owns reactive recovery for those, and
        rotating here would spend the singleton's single-use refresh token for an
        account the agent wasn't even using.

        Deliberately singleton-only (via the auth-store runtime resolver), NOT
        the pool-force-first path of ``refresh(force=True)``. Returns fresh
        material only when a NEW token was minted, else ``None``."""
        resolve = self._runtime_resolver()
        active = str(active_api_key or "").strip()
        try:
            singleton_now = resolve(refresh_if_expiring=False)
        except Exception:
            return None
        singleton_key = str((singleton_now or {}).get("api_key") or "").strip()
        if singleton_key and active and singleton_key != active:
            # Active credential isn't the singleton: skip to avoid a silent account swap.
            return None
        try:
            creds = resolve(force_refresh=force)
        except Exception:
            return None
        cred = self._credential_from_runtime(creds)
        if cred is None:
            return None
        # No NEW token minted (the resolver returns the same stale token when a
        # refresh fails silently) → nothing to adopt.
        if active and cred.api_key.strip() == active:
            return None
        return cred

    # ── per-provider hooks ──
    def _pool_base_url(self, entry) -> str:
        raise NotImplementedError

    def _runtime_resolver(self):
        """The auth-store runtime resolver callable for this provider, resolved
        at call time so tests can monkeypatch ``hermes_cli.auth``."""
        raise NotImplementedError

    def _legacy_credential(self) -> Optional[Credential]:
        raise NotImplementedError

    def _legacy_force_credential(self) -> Optional[Credential]:
        raise NotImplementedError

    def _credential_from_runtime(self, creds) -> Optional[Credential]:
        """Wrap a runtime-credentials dict ({api_key, base_url}) as a legacy
        Credential; base_url falls back to the provider default."""
        if not isinstance(creds, dict):
            return None
        api_key = str(creds.get("api_key") or "").strip()
        if not api_key:
            return None
        base_url = str(creds.get("base_url") or "").strip().rstrip("/") or self.DEFAULT_BASE_URL
        return Credential(api_key=api_key, base_url=base_url, provenance="legacy")


class CodexCredentialSource(_SingleUseOAuthCredentialSource):
    PROVIDER = "openai-codex"
    DEFAULT_BASE_URL = CODEX_DEFAULT_BASE_URL

    def _pool_base_url(self, entry) -> str:
        from agent.auxiliary_client import _pool_runtime_base_url

        return _pool_runtime_base_url(entry, self.DEFAULT_BASE_URL) or self.DEFAULT_BASE_URL

    def _runtime_resolver(self):
        from hermes_cli import auth as _auth

        return _auth.resolve_codex_runtime_credentials

    def _legacy_credential(self) -> Optional[Credential]:
        # Reuses the aux reader (auth.json tokens + JWT-expiry skip); base_url is
        # the codex default, matching _build_codex_client's legacy branch.
        from agent.auxiliary_client import _read_codex_access_token

        token = _read_codex_access_token()
        if not token:
            return None
        return Credential(api_key=token, base_url=self.DEFAULT_BASE_URL, provenance="legacy")

    def _legacy_force_credential(self) -> Optional[Credential]:
        from hermes_cli.auth import resolve_codex_runtime_credentials

        return self._credential_from_runtime(resolve_codex_runtime_credentials(force_refresh=True))


class XaiOAuthCredentialSource(_SingleUseOAuthCredentialSource):
    PROVIDER = "xai-oauth"
    DEFAULT_BASE_URL = XAI_OAUTH_DEFAULT_BASE_URL

    def _pool_base_url(self, entry) -> str:
        # xAI honours HERMES_XAI_BASE_URL / XAI_BASE_URL overrides and validates
        # the host (mirrors _resolve_xai_oauth_for_aux).
        import os

        from hermes_cli.auth import DEFAULT_XAI_OAUTH_BASE_URL, _xai_validate_inference_base_url

        def _url(v):
            return str(v or "").strip().rstrip("/")

        return _xai_validate_inference_base_url(
            _url(os.getenv("HERMES_XAI_BASE_URL", ""))
            or _url(os.getenv("XAI_BASE_URL", ""))
            or _url(getattr(entry, "runtime_base_url", None))
            or _url(getattr(entry, "base_url", None)),
            fallback=DEFAULT_XAI_OAUTH_BASE_URL,
        )

    def _runtime_resolver(self):
        from hermes_cli import auth as _auth

        return _auth.resolve_xai_oauth_runtime_credentials

    def _legacy_credential(self) -> Optional[Credential]:
        from hermes_cli.auth import resolve_xai_oauth_runtime_credentials

        return self._credential_from_runtime(resolve_xai_oauth_runtime_credentials())

    def _legacy_force_credential(self) -> Optional[Credential]:
        from hermes_cli.auth import resolve_xai_oauth_runtime_credentials

        return self._credential_from_runtime(
            resolve_xai_oauth_runtime_credentials(force_refresh=True)
        )
