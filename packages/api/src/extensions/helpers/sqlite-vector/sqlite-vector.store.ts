/*
 * Hexabot — Fair Core License (FCL-1.0-ALv2)
 * Copyright (c) 2026 Hexastack.
 * Full terms: see LICENSE.md.
 */

import { QueryRunner } from 'typeorm';

import { RagHelperUnavailableError } from '@/cms/errors/rag.errors';
import {
  ContentSearchHit,
  ContentSearchOptions,
  ContentSearchStore,
  quoteIdentifier,
} from '@/helper/lib/content-search.store';

import {
  isSqliteDatabase,
  isSqliteVectorProvisioned,
  provisionSqliteVectorInfrastructure,
  SQLITE_VECTOR_CHUNKS_TABLE,
  SQLITE_VECTOR_DOCUMENTS_TABLE,
  SQLITE_VECTOR_JOBS_TABLE,
} from './sqlite-vector.provisioning';

/** Truncation bound for the error text persisted on a failed job. */
export const MAX_ERROR_LENGTH = 4000;

/** How long a claimed job may stay locked before another worker may steal it. */
export const LEASE_TIMEOUT_MINUTES = 5;

/** Upper bound of the exponential retry backoff. */
export const MAX_RETRY_DELAY_SECONDS = 900;

export interface SqliteVectorJob {
  contentId: string;
  revision: number;
  attempts: number;
}

export interface SqliteVectorContent {
  id: string;
  searchText: string;
  status: boolean;
}

export interface SqliteVectorEmbeddedChunk {
  index: number;
  text: string;
  embedding: number[];
}

export type SqliteVectorSearchOptions = ContentSearchOptions;

/** Minimal shape of the raw better-sqlite3 connection TypeORM holds. */
interface SqliteConnection {
  loadExtension?: (path: string) => void;
}

const nowIso = (): string => new Date().toISOString();
const documents = quoteIdentifier(SQLITE_VECTOR_DOCUMENTS_TABLE);
const chunks = quoteIdentifier(SQLITE_VECTOR_CHUNKS_TABLE);
const jobs = quoteIdentifier(SQLITE_VECTOR_JOBS_TABLE);

/**
 * SQLite persistence for the sqlite-vector helper, backed by sqlite-vec.
 *
 * Owns queue claiming, revision-guarded writes, reconciliation, and exact
 * cosine search; the schema DDL lives in {@link ./sqlite-vector.provisioning}.
 *
 * Two SQLite traits shape the queries below. Writers are serialized, so a
 * transaction alone gives the isolation the revision/lease guards need and no
 * row-locking clause is required. And there is no server-side clock the helper
 * can rely on for ordering against trigger-written rows, so every timestamp is
 * an ISO-8601 UTC string produced by the process, and the retry backoff is
 * computed in JavaScript.
 */
export class SqliteVectorStore extends ContentSearchStore {
  private infrastructureReady = false;

  private extensionLoaded = false;

  get databaseType(): string {
    return this.dataSource.options.type;
  }

  async assertInfrastructure(): Promise<void> {
    if (!isSqliteDatabase(this.databaseType)) {
      throw new RagHelperUnavailableError(
        `The sqlite-vector RAG helper requires SQLite, but the database is "${this.databaseType}". ` +
          'Use the pgvector helper on PostgreSQL.',
      );
    }
    if (this.infrastructureReady) {
      return;
    }

    await this.loadVectorExtension();

    const queryRunner = this.dataSource.createQueryRunner();
    await queryRunner.connect();
    try {
      if (!(await isSqliteVectorProvisioned(queryRunner))) {
        await queryRunner.startTransaction();
        try {
          await provisionSqliteVectorInfrastructure(queryRunner);
          await queryRunner.commitTransaction();
        } catch (error) {
          await queryRunner.rollbackTransaction();
          const reason = error instanceof Error ? error.message : String(error);
          throw new RagHelperUnavailableError(
            'The sqlite-vector RAG helper could not provision its tables. ' +
              `Provisioning failed: ${reason}`,
          );
        }
      }
    } finally {
      await queryRunner.release();
    }

    this.infrastructureReady = true;
  }

  /**
   * Loads sqlite-vec into the connection TypeORM already owns.
   *
   * The extension is a per-connection facility and the better-sqlite3 driver
   * keeps exactly one connection, so loading it once here covers every query
   * runner. Doing it lazily (rather than through the DataSource's
   * `prepareDatabase` hook) keeps the core database configuration free of any
   * knowledge about this helper: a deployment that never selects it never loads
   * the extension.
   */
  private async loadVectorExtension(): Promise<void> {
    if (this.extensionLoaded) {
      return;
    }

    const connection = (
      this.dataSource.driver as unknown as {
        databaseConnection?: SqliteConnection;
      }
    ).databaseConnection;
    if (!connection?.loadExtension) {
      throw new RagHelperUnavailableError(
        'The sqlite-vector RAG helper could not access the underlying SQLite connection to load sqlite-vec.',
      );
    }

    try {
      const sqliteVec = await import('sqlite-vec');
      sqliteVec.load(connection as Parameters<typeof sqliteVec.load>[0]);
      await this.dataSource.query(`SELECT vec_version()`);
    } catch (error) {
      const reason = error instanceof Error ? error.message : String(error);
      throw new RagHelperUnavailableError(
        'The sqlite-vector RAG helper requires the sqlite-vec extension, which could not be loaded on this platform. ' +
          `Loading failed: ${reason}`,
      );
    }

    this.extensionLoaded = true;
  }

  async enqueueAll(): Promise<void> {
    await this.assertInfrastructure();
    const table = this.contentTable;
    const idColumn = quoteIdentifier(this.columnName('id'));
    const now = nowIso();

    await this.dataSource.query(
      `INSERT INTO ${jobs} ` +
        `("content_id", "revision", "attempts", "available_at", "locked_at", "locked_by", "last_error", "updated_at") ` +
        `SELECT content.${idColumn}, 1, 0, ?, NULL, NULL, NULL, ? FROM ${table} content ` +
        `WHERE 1 = 1 ` +
        `ON CONFLICT ("content_id") DO UPDATE SET ` +
        `"revision" = ${jobs}."revision" + 1, ` +
        `"attempts" = 0, "available_at" = excluded."available_at", ` +
        `"locked_at" = NULL, "locked_by" = NULL, "last_error" = NULL, ` +
        `"updated_at" = excluded."updated_at"`,
      [now, now],
    );
  }

  /**
   * Enqueues rows whose index state does not match what should be stored,
   * without resetting retries of already-queued work.
   *
   * With `activeOnly`, "needs work" means an active row missing a fresh
   * document (embed it) or an inactive row that still has any document (remove
   * it), so the reconciliation converges the index onto active content only.
   * Otherwise it is simply any row missing a fresh document.
   */
  async enqueueMissing(profile: string, activeOnly: boolean): Promise<void> {
    await this.assertInfrastructure();
    const table = this.contentTable;
    const idColumn = quoteIdentifier(this.columnName('id'));
    const textColumn = quoteIdentifier(this.columnName('searchText'));
    const statusColumn = quoteIdentifier(this.columnName('status'));
    const missingFreshDocument =
      `NOT EXISTS (` +
      `SELECT 1 FROM ${documents} document ` +
      `WHERE document."content_id" = content.${idColumn} ` +
      `AND document."profile" = ? ` +
      `AND document."source_text" = content.${textColumn}` +
      `)`;
    const hasAnyDocument =
      `EXISTS (` +
      `SELECT 1 FROM ${documents} document ` +
      `WHERE document."content_id" = content.${idColumn}` +
      `)`;
    // SQLite has no native boolean: the content status is stored as 0/1.
    const needsWork = activeOnly
      ? `((content.${statusColumn} = 1 AND ${missingFreshDocument}) ` +
        `OR (content.${statusColumn} = 0 AND ${hasAnyDocument}))`
      : missingFreshDocument;
    const now = nowIso();

    await this.dataSource.query(
      `INSERT INTO ${jobs} ` +
        `("content_id", "revision", "attempts", "available_at", "updated_at") ` +
        `SELECT content.${idColumn}, 1, 0, ?, ? FROM ${table} content ` +
        `WHERE ${needsWork} AND NOT EXISTS (` +
        `SELECT 1 FROM ${jobs} job WHERE job."content_id" = content.${idColumn}` +
        `) ON CONFLICT ("content_id") DO NOTHING`,
      [now, now, profile],
    );
  }

  async wakePendingRetries(): Promise<void> {
    await this.assertInfrastructure();
    const now = nowIso();
    await this.dataSource.query(
      `UPDATE ${jobs} ` +
        `SET "available_at" = ?, "last_error" = NULL, "updated_at" = ? ` +
        `WHERE "locked_at" IS NULL`,
      [now, now],
    );
  }

  /**
   * Claims due and lease-expired jobs in one statement. SQLite serializes
   * writers, so a single `UPDATE ... RETURNING` is already atomic against other
   * claimers and needs no equivalent of `FOR UPDATE SKIP LOCKED`.
   */
  async claimJobs(workerId: string, limit: number): Promise<SqliteVectorJob[]> {
    await this.assertInfrastructure();
    const now = nowIso();
    const staleBefore = new Date(
      Date.now() - LEASE_TIMEOUT_MINUTES * 60_000,
    ).toISOString();
    const rows = await this.dataSource.query(
      `UPDATE ${jobs} ` +
        `SET "locked_at" = ?, "locked_by" = ?, "updated_at" = ? ` +
        `WHERE "content_id" IN (` +
        `SELECT "content_id" FROM ${jobs} ` +
        `WHERE ("locked_at" IS NULL AND "available_at" <= ?) ` +
        `OR "locked_at" < ? ` +
        `ORDER BY "available_at" ASC, "updated_at" ASC LIMIT ?` +
        `) RETURNING "content_id" AS "contentId", "revision", "attempts"`,
      [now, workerId, now, now, staleBefore, Math.max(1, Math.floor(limit))],
    );

    return (Array.isArray(rows) ? rows : []).map(
      (row: Record<string, unknown>) => ({
        contentId: String(row.contentId),
        revision: Number(row.revision),
        attempts: Number(row.attempts),
      }),
    );
  }

  async loadContent(
    contentId: string,
  ): Promise<SqliteVectorContent | undefined> {
    const table = this.contentTable;
    const idColumn = quoteIdentifier(this.columnName('id'));
    const textColumn = quoteIdentifier(this.columnName('searchText'));
    const statusColumn = quoteIdentifier(this.columnName('status'));
    const rows = await this.dataSource.query(
      `SELECT ${idColumn} AS "id", ${textColumn} AS "searchText", ` +
        `${statusColumn} AS "status" FROM ${table} WHERE ${idColumn} = ?`,
      [contentId],
    );
    if (!rows[0]) {
      return undefined;
    }

    return {
      id: String(rows[0].id),
      searchText: String(rows[0].searchText ?? ''),
      status: Boolean(rows[0].status),
    };
  }

  /**
   * Removes any embeddings for a content row and releases the claimed job.
   *
   * Used when `index_only_active_content` is enabled and the claimed job turns
   * out to reference inactive content. The delete only proceeds while the
   * claimed revision and lease still hold, so a concurrent reactivation (which
   * bumps the revision via the content trigger) is left untouched and
   * reprocessed rather than having its fresh embeddings removed. Returns
   * whether the content was discarded.
   */
  async discardInactive(
    job: SqliteVectorJob,
    workerId: string,
  ): Promise<boolean> {
    const queryRunner = this.dataSource.createQueryRunner();
    await queryRunner.connect();
    await queryRunner.startTransaction();
    try {
      const currentJob = await this.readCurrentJob(queryRunner, job.contentId);
      if (
        currentJob?.revision !== job.revision ||
        currentJob?.workerId !== workerId
      ) {
        await queryRunner.commitTransaction();

        return false;
      }

      await queryRunner.query(
        `DELETE FROM ${documents} WHERE "content_id" = ?`,
        [job.contentId],
      );
      await queryRunner.query(
        `DELETE FROM ${jobs} ` +
          `WHERE "content_id" = ? AND "revision" = ? AND "locked_by" = ?`,
        [job.contentId, job.revision, workerId],
      );
      await queryRunner.commitTransaction();

      return true;
    } catch (error) {
      await queryRunner.rollbackTransaction();
      throw error;
    } finally {
      await queryRunner.release();
    }
  }

  /**
   * Atomically stores embeddings only when the claimed job and live source
   * revision still match. Documents from prior profiles are removed after the
   * new profile succeeds, never before.
   */
  async save(
    job: SqliteVectorJob,
    workerId: string,
    profile: string,
    sourceText: string,
    embeddedChunks: SqliteVectorEmbeddedChunk[],
  ): Promise<boolean> {
    const queryRunner = this.dataSource.createQueryRunner();
    await queryRunner.connect();
    await queryRunner.startTransaction();
    try {
      const currentSource = await this.readCurrentSource(
        queryRunner,
        job.contentId,
      );
      const currentJob = await this.readCurrentJob(queryRunner, job.contentId);
      if (
        currentSource !== sourceText ||
        currentJob?.revision !== job.revision ||
        currentJob?.workerId !== workerId
      ) {
        await queryRunner.commitTransaction();

        return false;
      }

      const now = nowIso();
      await queryRunner.query(
        `DELETE FROM ${documents} WHERE "content_id" = ? AND "profile" = ?`,
        [job.contentId, profile],
      );
      await queryRunner.query(
        `INSERT INTO ${documents} ` +
          `("content_id", "profile", "source_text", "created_at", "updated_at") ` +
          `VALUES (?, ?, ?, ?, ?)`,
        [job.contentId, profile, sourceText, now, now],
      );

      for (const chunk of embeddedChunks) {
        await queryRunner.query(
          `INSERT INTO ${chunks} ` +
            `("content_id", "profile", "chunk_index", "chunk_text", "embedding") ` +
            `VALUES (?, ?, ?, ?, vec_f32(?))`,
          [
            job.contentId,
            profile,
            chunk.index,
            chunk.text,
            JSON.stringify(chunk.embedding),
          ],
        );
      }

      await queryRunner.query(
        `DELETE FROM ${documents} WHERE "content_id" = ? AND "profile" <> ?`,
        [job.contentId, profile],
      );
      await queryRunner.query(
        `DELETE FROM ${jobs} ` +
          `WHERE "content_id" = ? AND "revision" = ? AND "locked_by" = ?`,
        [job.contentId, job.revision, workerId],
      );
      await queryRunner.commitTransaction();

      return true;
    } catch (error) {
      await queryRunner.rollbackTransaction();
      throw error;
    } finally {
      await queryRunner.release();
    }
  }

  async fail(
    job: SqliteVectorJob,
    workerId: string,
    error: unknown,
  ): Promise<void> {
    const now = Date.now();
    const availableAt = new Date(
      now + this.retryDelaySeconds(job.attempts) * 1000,
    ).toISOString();
    await this.dataSource.query(
      `UPDATE ${jobs} SET ` +
        `"attempts" = "attempts" + 1, "available_at" = ?, ` +
        `"locked_at" = NULL, "locked_by" = NULL, ` +
        `"last_error" = ?, "updated_at" = ? ` +
        `WHERE "content_id" = ? AND "revision" = ? AND "locked_by" = ?`,
      [
        availableAt,
        this.errorMessage(error),
        new Date(now).toISOString(),
        job.contentId,
        job.revision,
        workerId,
      ],
    );
  }

  async search(
    embedding: number[],
    profile: string,
    options: SqliteVectorSearchOptions = {},
  ): Promise<ContentSearchHit[]> {
    await this.assertInfrastructure();
    const table = this.contentTable;
    const idColumn = quoteIdentifier(this.columnName('id'));
    const titleColumn = quoteIdentifier(this.columnName('title'));
    const textColumn = quoteIdentifier(this.columnName('searchText'));
    const statusColumn = quoteIdentifier(this.columnName('status'));
    const contentTypeColumn = quoteIdentifier(this.contentTypeColumnName);
    const vector = JSON.stringify(embedding);
    // The distance is referenced twice (score and window ordering) and `?`
    // placeholders are positional, so the vector is bound twice.
    const params: unknown[] = [vector, vector, profile];
    const conditions = [
      `document."profile" = ?`,
      `document."source_text" = content.${textColumn}`,
    ];

    if (typeof options.status === 'boolean') {
      params.push(options.status ? 1 : 0);
      conditions.push(`content.${statusColumn} = ?`);
    }
    if (options.contentTypeId) {
      params.push(options.contentTypeId);
      conditions.push(`content.${contentTypeColumn} = ?`);
    }

    const distance = `vec_distance_cosine(chunk."embedding", vec_f32(?))`;
    const rows = await this.dataSource.query(
      `SELECT "contentId", "title", "text", "contentTypeId", "score" ` +
        `FROM (` +
        `SELECT content.${idColumn} AS "contentId", ` +
        `content.${titleColumn} AS "title", chunk."chunk_text" AS "text", ` +
        `content.${contentTypeColumn} AS "contentTypeId", ` +
        `1 - ${distance} AS "score", ` +
        `ROW_NUMBER() OVER (` +
        `PARTITION BY content.${idColumn} ` +
        `ORDER BY ${distance} ASC, chunk."chunk_index" ASC` +
        `) AS "rank" ` +
        `FROM ${chunks} chunk ` +
        `INNER JOIN ${documents} document ` +
        `ON document."content_id" = chunk."content_id" ` +
        `AND document."profile" = chunk."profile" ` +
        `INNER JOIN ${table} content ` +
        `ON content.${idColumn} = document."content_id" ` +
        `WHERE ${conditions.join(' AND ')}` +
        `) ranked WHERE "rank" = 1 ` +
        `ORDER BY "score" DESC, "contentId" ASC ` +
        `LIMIT ${this.normalizeLimit(options.limit)}`,
      params,
    );

    return this.mapHits(rows, (row) =>
      row.score == null ? undefined : Number(row.score),
    );
  }

  /** Exponential retry backoff in seconds, capped at fifteen minutes. */
  private retryDelaySeconds(attempts: number): number {
    return Math.min(
      MAX_RETRY_DELAY_SECONDS,
      5 * 2 ** Math.min(Math.max(attempts, 0), 20),
    );
  }

  private errorMessage(error: unknown): string {
    const message =
      error instanceof Error ? error.message : 'Unknown embedding error';

    return message.slice(0, MAX_ERROR_LENGTH);
  }

  private async readCurrentSource(
    queryRunner: QueryRunner,
    contentId: string,
  ): Promise<string | undefined> {
    const table = this.contentTable;
    const idColumn = quoteIdentifier(this.columnName('id'));
    const textColumn = quoteIdentifier(this.columnName('searchText'));
    const rows = await queryRunner.query(
      `SELECT ${textColumn} AS "searchText" FROM ${table} WHERE ${idColumn} = ?`,
      [contentId],
    );

    return rows[0] ? String(rows[0].searchText ?? '') : undefined;
  }

  private async readCurrentJob(
    queryRunner: QueryRunner,
    contentId: string,
  ): Promise<{ revision: number; workerId?: string } | undefined> {
    const rows = await queryRunner.query(
      `SELECT "revision", "locked_by" AS "workerId" ` +
        `FROM ${jobs} WHERE "content_id" = ?`,
      [contentId],
    );
    if (!rows[0]) {
      return undefined;
    }

    return {
      revision: Number(rows[0].revision),
      workerId: rows[0].workerId == null ? undefined : String(rows[0].workerId),
    };
  }
}
