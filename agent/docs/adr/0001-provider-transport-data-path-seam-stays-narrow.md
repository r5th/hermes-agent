# The Transport seam owns the provider data path only — not streaming, construction, or credentials

`agent/transports/` already gives each api_mode a deep Transport (message/tool conversion,
kwargs assembly, response normalization) behind `get_transport(api_mode)`. We considered widening
that seam to also absorb the open-coded `if/elif api_mode` chains in streaming dispatch and client
construction, and deliberately declined: streaming differs in *kind* per api_mode (codex is always
internally streaming, bedrock uses a separate SDK, the rest share one path), so folding it in would
muddy a currently-clean data-path seam. Streaming, client construction, credentials, caching, and
retries stay on `AIAgent` by design (see `agent/transports/base.py:1-4`).

**Consequence:** a future architecture review will see the remaining api_mode branches and be
tempted to re-propose a single "unify all provider dispatch" Transport. Don't — the data path is
already unified; the rest is intentionally split. The remaining main-vs-auxiliary duplication is
addressed by the credential-resolution work (ADR-0002), not by growing Transport.
