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

// Removed when upgrading from the worker-based implementation.
export const SQLITE_VECTOR_JOBS_TABLE = 'rag_sqlite_vector_jobs';

export const SQLITE_VECTOR_JOBS_INDEX = 'rag_sqlite_vector_jobs_available_idx';

export const SQLITE_VECTOR_INSERT_TRIGGER =
  'contents_enqueue_sqlite_vector_insert';

export const SQLITE_VECTOR_UPDATE_TRIGGER =
  'contents_enqueue_sqlite_vector_update';

const CONTENTS_TABLE = 'contents';

/** Whether the connection is one of the SQLite drivers TypeORM exposes. */
export const isSqliteDatabase = (type: string): boolean =>
  type === 'better-sqlite3' || type === 'sqlite';

/** Checks the direct-index tables and absence of legacy worker objects. */
export async function isSqliteVectorProvisioned(
  queryRunner: QueryRunner,
): Promise<boolean> {
  const vectorTables = [
    SQLITE_VECTOR_DOCUMENTS_TABLE,
    SQLITE_VECTOR_CHUNKS_TABLE,
  ];
  const workerObjects = [
    SQLITE_VECTOR_JOBS_TABLE,
    SQLITE_VECTOR_INSERT_TRIGGER,
    SQLITE_VECTOR_UPDATE_TRIGGER,
  ];
  const inspectedObjects = [...vectorTables, ...workerObjects];
  const rows = await queryRunner.query(
    `SELECT "name" FROM "sqlite_master" ` +
      `WHERE ("type" = 'table' AND "name" IN (?, ?, ?)) ` +
      `OR ("type" = 'trigger' AND "name" IN (?, ?))`,
    inspectedObjects,
  );
  const present = new Set(
    (rows as Array<{ name: string }>).map((row) => row.name),
  );

  return (
    vectorTables.every((name) => present.has(name)) &&
    workerObjects.every((name) => !present.has(name))
  );
}

/**
 * Creates the vector tables and removes legacy worker objects.
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
    `DROP TRIGGER IF EXISTS "${SQLITE_VECTOR_INSERT_TRIGGER}"`,
  );
  await queryRunner.query(
    `DROP TRIGGER IF EXISTS "${SQLITE_VECTOR_UPDATE_TRIGGER}"`,
  );
  await queryRunner.query(`DROP INDEX IF EXISTS "${SQLITE_VECTOR_JOBS_INDEX}"`);
  await queryRunner.query(`DROP TABLE IF EXISTS "${SQLITE_VECTOR_JOBS_TABLE}"`);
}
