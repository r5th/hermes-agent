# Unify credential resolution behind a CredentialSource ABC returning a Credential value object

Credential logic is currently smeared across six independent resolvers, a per-provider switchboard
inside the pool, four non-derivable per-provider constant tables
(`_SINGLE_USE_REFRESH_PROVIDERS`, `_RESYNC_SOURCE`, `_PERSISTABLE_PROVIDER_SOURCES`,
`_TOKENS_SINGLETON_PROVIDERS`), and six `_try_refresh_*` methods dispatched by two separate
if-ladders — duplicated again between the main and auxiliary request paths. We will consolidate it
behind a **Credential source** ABC + provider-keyed registry (mirroring the existing
`ProviderBase`/`ProviderRegistry` idiom) whose `resolve()`/`refresh()` return a **Credential** value
object that hides whether the material came from the pool or a legacy resolver.

**Boundary:** the Credential source owns credential *acquisition and refresh* only. The
client-construction tail (rebuilding the OpenAI/Anthropic SDK client) stays in `client_lifecycle`,
respecting the same split as ADR-0001 — refresh returns fresh material, the caller rebuilds the
client. The per-provider constant tables dissolve into data on each source.

**Rollout:** strangler, one provider per cut, starting with anthropic (the hardest case: pool +
legacy fallback + single-use refresh), routing **both** the main and auxiliary anthropic paths
through the new `resolve()` in the first cut. Behaviour is held identical via an E2E anthropic
state matrix (env-only · pool-only · pool+legacy · expired→refresh · single-use race) driven
through `resolve()`/`refresh()` against a real temp `HERMES_HOME`.

**Considered and rejected:** a single `CredentialResolver` class (keeps a provider switchboard
inside its implementation) and a plain module-level `resolve()` function (no seam for the
per-provider data to live on). The ABC won because the seam is empirically real — six distinct
resolvers exist today — and the repo already uses this exact provider-registry idiom elsewhere.
