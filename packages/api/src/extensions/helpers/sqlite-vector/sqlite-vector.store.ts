/*
 * Hexabot — Fair Core License (FCL-1.0-ALv2)
 * Copyright (c) 2026 Hexastack.
 * Full terms: see LICENSE.md.
 */

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
} from './sqlite-vector.provisioning';

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

/** Direct SQLite vector persistence and exact cosine search. */
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

  async loadContents(): Promise<SqliteVectorContent[]> {
    await this.assertInfrastructure();
    const table = this.contentTable;
    const idColumn = quoteIdentifier(this.columnName('id'));
    const textColumn = quoteIdentifier(this.columnName('searchText'));
    const statusColumn = quoteIdentifier(this.columnName('status'));
    const rows = await this.dataSource.query(
      `SELECT ${idColumn} AS "id", ${textColumn} AS "searchText", ` +
        `${statusColumn} AS "status" FROM ${table}`,
    );

    return (Array.isArray(rows) ? rows : []).map(
      (row: Record<string, unknown>) => ({
        id: String(row.id),
        searchText: String(row.searchText ?? ''),
        status: Boolean(row.status),
      }),
    );
  }

  async remove(contentId: string): Promise<void> {
    await this.assertInfrastructure();
    await this.dataSource.query(
      `DELETE FROM ${documents} WHERE "content_id" = ?`,
      [contentId],
    );
  }

  async save(
    contentId: string,
    profile: string,
    sourceText: string,
    status: boolean,
    embeddedChunks: SqliteVectorEmbeddedChunk[],
  ): Promise<boolean> {
    await this.assertInfrastructure();
    const queryRunner = this.dataSource.createQueryRunner();
    await queryRunner.connect();
    await queryRunner.startTransaction();
    try {
      const table = this.contentTable;
      const idColumn = quoteIdentifier(this.columnName('id'));
      const textColumn = quoteIdentifier(this.columnName('searchText'));
      const statusColumn = quoteIdentifier(this.columnName('status'));
      const rows = await queryRunner.query(
        `SELECT ${textColumn} AS "searchText", ${statusColumn} AS "status" ` +
          `FROM ${table} WHERE ${idColumn} = ?`,
        [contentId],
      );
      if (
        !rows[0] ||
        String((rows[0] as Record<string, unknown>).searchText ?? '') !==
          sourceText ||
        Boolean((rows[0] as Record<string, unknown>).status) !== status
      ) {
        await queryRunner.commitTransaction();

        return false;
      }

      const now = nowIso();
      await queryRunner.query(
        `DELETE FROM ${documents} WHERE "content_id" = ? AND "profile" = ?`,
        [contentId, profile],
      );
      await queryRunner.query(
        `INSERT INTO ${documents} ` +
          `("content_id", "profile", "source_text", "created_at", "updated_at") ` +
          `VALUES (?, ?, ?, ?, ?)`,
        [contentId, profile, sourceText, now, now],
      );

      for (const chunk of embeddedChunks) {
        await queryRunner.query(
          `INSERT INTO ${chunks} ` +
            `("content_id", "profile", "chunk_index", "chunk_text", "embedding") ` +
            `VALUES (?, ?, ?, ?, vec_f32(?))`,
          [
            contentId,
            profile,
            chunk.index,
            chunk.text,
            JSON.stringify(chunk.embedding),
          ],
        );
      }

      await queryRunner.query(
        `DELETE FROM ${documents} WHERE "content_id" = ? AND "profile" <> ?`,
        [contentId, profile],
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
}
