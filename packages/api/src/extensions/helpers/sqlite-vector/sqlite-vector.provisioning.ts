/*
 * Hexabot — Fair Core License (FCL-1.0-ALv2)
 * Copyright (c) 2026 Hexastack.
 * Full terms: see LICENSE.md.
 */

import { QueryRunner } from 'typeorm';

// Schema object names and DDL for the SQLite vector RAG helper. This is the
// single source of truth for provisioning, run on demand by the store
// (idempotent self-heal) so a database gains the structures the first time the
// helper is used, with no migration to schedule.
export const SQLITE_VECTOR_DOCUMENTS_TABLE = 'rag_sqlite_vector_documents';

export const SQLITE_VECTOR_CHUNKS_TABLE = 'rag_sqlite_vector_chunks';

export const SQLITE_VECTOR_JOBS_TABLE = 'rag_sqlite_vector_jobs';

export const SQLITE_VECTOR_JOBS_INDEX = 'rag_sqlite_vector_jobs_available_idx';

export const SQLITE_VECTOR_INSERT_TRIGGER =
  'contents_enqueue_sqlite_vector_insert';

export const SQLITE_VECTOR_UPDATE_TRIGGER =
  'contents_enqueue_sqlite_vector_update';

const CONTENTS_TABLE = 'contents';

/**
 * SQLite expression producing the same lexicographically-sortable UTC format as
 * JavaScript's `Date#toISOString`, so timestamps written by triggers and by the
 * worker compare correctly against each other.
 */
export const SQLITE_NOW = `strftime('%Y-%m-%dT%H:%M:%fZ','now')`;

/** Whether the connection is one of the SQLite drivers TypeORM exposes. */
export const isSqliteDatabase = (type: string): boolean =>
  type === 'better-sqlite3' || type === 'sqlite';

/**
 * Reports whether the vector infrastructure is already fully in place: the
 * three helper tables and both enqueue triggers.
 */
export async function isSqliteVectorProvisioned(
  queryRunner: QueryRunner,
): Promise<boolean> {
  const expected = [
    SQLITE_VECTOR_DOCUMENTS_TABLE,
    SQLITE_VECTOR_CHUNKS_TABLE,
    SQLITE_VECTOR_JOBS_TABLE,
    SQLITE_VECTOR_INSERT_TRIGGER,
    SQLITE_VECTOR_UPDATE_TRIGGER,
  ];
  const rows = await queryRunner.query(
    `SELECT "name" FROM "sqlite_master" ` +
      `WHERE ("type" = 'table' AND "name" IN (?, ?, ?)) ` +
      `OR ("type" = 'trigger' AND "name" IN (?, ?))`,
    expected,
  );
  const present = new Set(
    (rows as Array<{ name: string }>).map((row) => row.name),
  );

  return expected.every((name) => present.has(name));
}

/**
 * Creates (or repairs) the entire vector RAG infrastructure: the
 * document/chunk/job tables, the enqueue triggers, and an initial backfill of
 * the job queue. Every statement is idempotent, so this is safe to re-run.
 *
 * Embeddings are stored as `vec_f32` BLOBs in an ordinary table rather than in
 * a `vec0` virtual table. A `vec0` table must declare a fixed dimension up
 * front, which the profile design deliberately keeps fluid, and it cannot
 * participate in the foreign keys that make content deletion cascade. Since
 * sqlite-vec's `vec_distance_cosine` is a scalar function usable on any BLOB
 * column, nothing is lost: the search is an exact (brute-force) cosine scan
 * either way.
 *
 * Timestamps are ISO-8601 UTC TEXT so they sort lexicographically and compare
 * directly against values produced by `Date#toISOString` in the store.
 *
 * It intentionally does NOT catch errors: the caller decides how to react.
 */
export async function provisionSqliteVectorInfrastructure(
  queryRunner: QueryRunner,
): Promise<void> {
  const contents = `"${CONTENTS_TABLE}"`;
  const documents = `"${SQLITE_VECTOR_DOCUMENTS_TABLE}"`;
  const chunks = `"${SQLITE_VECTOR_CHUNKS_TABLE}"`;
  const jobs = `"${SQLITE_VECTOR_JOBS_TABLE}"`;

  await queryRunner.query(
    `CREATE TABLE IF NOT EXISTS ${documents} (` +
      `"content_id" varchar NOT NULL, ` +
      `"profile" varchar(64) NOT NULL, ` +
      `"source_text" text NOT NULL, ` +
      `"created_at" text NOT NULL, ` +
      `"updated_at" text NOT NULL, ` +
      `PRIMARY KEY ("content_id", "profile"), ` +
      `FOREIGN KEY ("content_id") REFERENCES ${contents}("id") ON DELETE CASCADE` +
      `)`,
  );
  await queryRunner.query(
    `CREATE TABLE IF NOT EXISTS ${chunks} (` +
      `"content_id" varchar NOT NULL, ` +
      `"profile" varchar(64) NOT NULL, ` +
      `"chunk_index" integer NOT NULL, ` +
      `"chunk_text" text NOT NULL, ` +
      `"embedding" blob NOT NULL, ` +
      `PRIMARY KEY ("content_id", "profile", "chunk_index"), ` +
      `FOREIGN KEY ("content_id", "profile") ` +
      `REFERENCES ${documents}("content_id", "profile") ON DELETE CASCADE` +
      `)`,
  );
  await queryRunner.query(
    `CREATE TABLE IF NOT EXISTS ${jobs} (` +
      `"content_id" varchar PRIMARY KEY, ` +
      `"revision" integer NOT NULL DEFAULT 1, ` +
      `"attempts" integer NOT NULL DEFAULT 0, ` +
      `"available_at" text NOT NULL, ` +
      `"locked_at" text NULL, ` +
      `"locked_by" varchar NULL, ` +
      `"last_error" text NULL, ` +
      `"updated_at" text NOT NULL, ` +
      `FOREIGN KEY ("content_id") REFERENCES ${contents}("id") ON DELETE CASCADE` +
      `)`,
  );
  await queryRunner.query(
    `CREATE INDEX IF NOT EXISTS "${SQLITE_VECTOR_JOBS_INDEX}" ` +
      `ON ${jobs} ("available_at", "updated_at")`,
  );

  // Re-enqueue on inserts, on source-text edits, and on status flips. The
  // status column matters because the `index_only_active_content` setting can
  // exclude inactive content from the index: activating content must schedule
  // its embedding, and deactivating it must schedule removal. The worker (not
  // these triggers, which cannot read app settings) decides embed vs. remove.
  // SQLite has no per-statement trigger carrying the operation, so the work is
  // split across an insert trigger and an update trigger narrowed to real
  // changes of the columns the worker cares about.
  const enqueueBody =
    `DELETE FROM ${documents} WHERE "content_id" = NEW."id"; ` +
    `INSERT INTO ${jobs} ` +
    `("content_id", "revision", "attempts", "available_at", "locked_at", "locked_by", "last_error", "updated_at") ` +
    `VALUES (NEW."id", 1, 0, ${SQLITE_NOW}, NULL, NULL, NULL, ${SQLITE_NOW}) ` +
    `ON CONFLICT ("content_id") DO UPDATE SET ` +
    `"revision" = ${jobs}."revision" + 1, ` +
    `"attempts" = 0, "available_at" = excluded."available_at", ` +
    `"locked_at" = NULL, "locked_by" = NULL, "last_error" = NULL, ` +
    `"updated_at" = excluded."updated_at"; `;

  await queryRunner.query(
    `DROP TRIGGER IF EXISTS "${SQLITE_VECTOR_INSERT_TRIGGER}"`,
  );
  await queryRunner.query(
    `DROP TRIGGER IF EXISTS "${SQLITE_VECTOR_UPDATE_TRIGGER}"`,
  );
  await queryRunner.query(
    `CREATE TRIGGER "${SQLITE_VECTOR_INSERT_TRIGGER}" ` +
      `AFTER INSERT ON ${contents} BEGIN ${enqueueBody}END`,
  );
  await queryRunner.query(
    `CREATE TRIGGER "${SQLITE_VECTOR_UPDATE_TRIGGER}" ` +
      `AFTER UPDATE OF "searchText", "status" ON ${contents} ` +
      `WHEN NEW."searchText" IS NOT OLD."searchText" ` +
      `OR NEW."status" IS NOT OLD."status" ` +
      `BEGIN ${enqueueBody}END`,
  );
  // The SELECT needs a WHERE clause: without one SQLite cannot tell the upsert
  // clause apart from a join's ON clause.
  await queryRunner.query(
    `INSERT INTO ${jobs} ` +
      `("content_id", "revision", "attempts", "available_at", "updated_at") ` +
      `SELECT "id", 1, 0, ${SQLITE_NOW}, ${SQLITE_NOW} FROM ${contents} ` +
      `WHERE 1 = 1 ON CONFLICT ("content_id") DO NOTHING`,
  );
}
