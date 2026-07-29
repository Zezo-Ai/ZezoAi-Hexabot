# RAG helpers

Hexabot ships with three database-owned RAG helpers:

- `fulltext-search` is the default. PostgreSQL uses a GIN expression index over
  the canonical `contents.searchText`; SQLite uses an FTS5 table maintained by
  database triggers.
- `pgvector` is PostgreSQL-only. It chunks `searchText`, calls an
  OpenAI-compatible embedding endpoint, and stores exact-search vectors in
  PostgreSQL. A trigger-backed, leased work queue makes indexing durable across
  API restarts and supports multiple API nodes.
- `sqlite-vector` is SQLite-only. It uses the same chunking and embedding
  settings, stores vectors through sqlite-vec, and indexes directly from CMS
  lifecycle hooks.

The selected helper is controlled by
`global_settings.default_rag_helper`. RAG retrieval is always available through
that helper; there is no feature enable/disable switch.

## Custom helper consistency

A custom RAG helper must extend `BaseRagHelper` and implement `retrieve`.
`index`, `remove`, and `reindex` are optional.

The CMS lifecycle hooks forwarded to `index` and `remove` are best-effort
latency signals. They are not durable change capture: a process failure after
the CMS transaction commits, direct SQL, or another writer can cause an
external index to drift. A custom helper that needs correctness should use a
database outbox, database-native change capture, or an equivalent durable
queue. It should also implement a complete, idempotent `reindex()` so
administrators can reconcile it through `POST /content/rag/reindex`.

## v3.4.0 migration and rollback

The v3.4.0 migration preserves the old LlamaIndex settings and storage for
rollback. It copies embedding configuration into the `pgvector` helper, but
selects `pgvector` only when PostgreSQL, the vector extension, and an API key
are all available. Every other legacy installation moves safely to
`fulltext-search`.

On **SQLite**, the previous version stored its RAG data as tables and triggers
_inside the main database_ (`content_chunks`, `content_embeddings`, the
`content_chunks_fts` mirror, and AFTER triggers on `contents` that wrote into
them). The migration drops these automatically: left in place, the triggers
fire on every content write, and dropping the tables by hand while the triggers
remained made every content write fail with `no such table: content_chunks`.
No manual SQLite cleanup is required.

On **PostgreSQL**, the dormant LlamaIndex structures (including
`llamaindex_embedding` and its document/index-store tables) are preserved for
rollback and are _not_ automated away. After the new helper has been verified
and the rollback window has closed, operators may remove them manually.

## Indexing only active content

Both vector helpers expose `index_only_active_content` (default `true`). When
enabled, inactive content is not embedded and is removed from the vector index.
Pgvector reconciles through its durable queue; sqlite-vector updates directly
from CMS lifecycle hooks. Toggling the setting reindexes the corpus. When
disabled, all content is embedded, while retrieval still excludes inactive
content unless explicitly requested.

## Testing

Unit tests run against the default SQLite config:

```sh
pnpm --filter @hexabot-ai/api test
```

The sqlite-vector integration suite runs in normal CI when the sqlite-vec
binary is available.

The `pgvector` helper also has an integration suite
(`pgvector.integration.spec.ts`) that exercises the real provisioning DDL,
status-aware trigger, leased work queue, and cosine search against PostgreSQL.
It is `describe.skip` unless `TEST_PGVECTOR_DATABASE_URL` is set, and it also
requires `DB_TYPE=postgres` — the entities' `DatetimeColumn` decorator picks its
SQL type from `DB_TYPE` at import time, so without it `createdAt` resolves to the
SQLite `datetime` type and `DataSource.initialize` fails on PostgreSQL. CI runs
it this way (see `.github/workflows/main-ci.yml`):

```sh
docker run -d --name pgv -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=hexabot_test \
  -p 5432:5432 pgvector/pgvector:0.8.2-pg16

DB_TYPE=postgres \
TEST_PGVECTOR_DATABASE_URL=postgres://postgres:postgres@127.0.0.1:5432/hexabot_test \
  pnpm --filter @hexabot-ai/api exec jest --runInBand \
  --runTestsByPath src/extensions/helpers/pgvector/pgvector.integration.spec.ts
```
