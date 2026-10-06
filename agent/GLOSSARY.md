# Agent Core

The model-facing agent loop and the machinery it drives: talking to model providers over
several wire protocols, resolving the credentials those calls need, and compressing context.
This glossary is the canonical vocabulary for that area — use these terms exactly; prefer them
over the listed synonyms.

## Language

### Providers & wire protocols

**Provider**:
A model backend Hermes can talk to (anthropic, bedrock, vertex, azure, nous, openai-codex, …).
Identified by a provider string.
_Avoid_: vendor, backend service.

**api_mode**:
The wire protocol a request is spoken in — one of `chat_completions`, `anthropic_messages`,
`bedrock_converse`, `codex_responses`. Derived from provider + host, not chosen directly.
_Avoid_: format, dialect, protocol (unqualified).

**Transport**:
The module that owns one api_mode's *data path* — message conversion, tool conversion,
request-kwargs assembly, and response normalization. It deliberately does **not** own client
construction, streaming, credentials, caching, or retries; those stay on the agent.
_Avoid_: adapter (a Transport is one, but reserve "adapter" for the role at a seam), client.

### Credentials

**Credential**:
The material one model request authenticates with — an api_key and its base_url, plus where it
came from (provenance). A single value regardless of whether it originated in the pool or a
legacy resolver.
_Avoid_: token, key, secret (each names only a part).

**Credential source**:
The per-provider strategy that produces and refreshes a Credential for one provider. The unit
that knows a provider's auth-store layout, single-use-refresh rule, and persistable sources.
_Avoid_: resolver, provider (overloaded), auth handler.

**Credential resolution**:
Obtaining the Credential for a given provider and request — distinct from *credential rotation*
(swapping to a different pooled credential after exhaustion) and from *client construction*
(building the SDK client the Credential feeds).
_Avoid_: auth, login, credential lookup.

**Credential pool**:
The store of interchangeable credentials for a provider, from which one is selected per request
and rotated on exhaustion. A Credential source may draw from it; the pool is not itself the
source.
_Avoid_: key pool, credential cache.

### Context compression

**Compression engine**:
The module that transforms a message list into a shorter one (prune, protect head/tail,
summarize the middle) behind a single `compress` verb. Owns no session lifecycle.
_Avoid_: compressor (ambiguous), summarizer.

**Compression orchestrator**:
The session-lifecycle layer around the engine — leases, session rotation, commit fences,
cooldowns, telemetry. Drives the engine; is not the engine.
_Avoid_: compression manager, controller.
