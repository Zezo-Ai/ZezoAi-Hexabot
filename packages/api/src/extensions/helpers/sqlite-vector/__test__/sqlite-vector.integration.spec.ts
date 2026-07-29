/*
 * Hexabot — Fair Core License (FCL-1.0-ALv2)
 * Copyright (c) 2026 Hexastack.
 * Full terms: see LICENSE.md.
 */

import { randomUUID } from 'node:crypto';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { DataSource } from 'typeorm';

import { ContentTypeOrmEntity } from '@/cms/entities/content-type.entity';
import { ContentOrmEntity } from '@/cms/entities/content.entity';
import { config } from '@/config';

import { isSqliteVectorProvisioned } from '../sqlite-vector.provisioning';
import { SqliteVectorStore } from '../sqlite-vector.store';

const canLoadSqliteVec = (): boolean => {
  try {
    /* eslint-disable @typescript-eslint/no-require-imports */
    const Database = require('better-sqlite3');
    const sqliteVec = require('sqlite-vec');
    /* eslint-enable @typescript-eslint/no-require-imports */
    const probe = new Database(':memory:');
    try {
      sqliteVec.load(probe);
      probe.prepare('SELECT vec_version()').get();
    } finally {
      probe.close();
    }

    return true;
  } catch {
    return false;
  }
};
const describeWithSqliteVec =
  config.database.type !== 'postgres' && canLoadSqliteVec()
    ? describe
    : describe.skip;
const PROFILE = 'profile-a';

describeWithSqliteVec('SQLite sqlite-vector RAG integration', () => {
  jest.setTimeout(30000);

  let directory: string;
  let dataSource: DataSource;
  let store: SqliteVectorStore;
  let contentTypeId: string;

  const insertContent = async (
    id: string,
    searchText: string,
    status = true,
  ): Promise<void> => {
    await dataSource.query(
      `INSERT INTO "contents" ("id", "content_type_id", "title", "status", "searchText") ` +
        `VALUES (?, ?, ?, ?, ?)`,
      [id, contentTypeId, `Title ${id}`, status ? 1 : 0, searchText],
    );
  };
  const saveEmbedding = async (
    contentId: string,
    embedding: number[],
    text: string,
    sourceText: string,
  ): Promise<void> => {
    await store.save(contentId, PROFILE, sourceText, true, [
      { index: 0, text, embedding },
    ]);
  };

  beforeAll(async () => {
    directory = mkdtempSync(join(tmpdir(), 'hexabot-sqlite-vector-'));
    dataSource = await new DataSource({
      type: 'better-sqlite3',
      database: join(directory, 'test.sqlite'),
      entities: [ContentOrmEntity, ContentTypeOrmEntity],
    }).initialize();

    await dataSource.query(
      `CREATE TABLE "content_types" (` +
        `"id" varchar PRIMARY KEY, "name" varchar NOT NULL UNIQUE)`,
    );
    await dataSource.query(
      `CREATE TABLE "contents" (` +
        `"id" varchar PRIMARY KEY, ` +
        `"content_type_id" varchar NOT NULL REFERENCES "content_types"("id") ON DELETE CASCADE, ` +
        `"title" varchar NOT NULL, "status" boolean NOT NULL DEFAULT 1, ` +
        `"properties" text NULL, "searchText" text NOT NULL)`,
    );

    contentTypeId = randomUUID();
    await dataSource.query(
      `INSERT INTO "content_types" ("id", "name") VALUES (?, 'Articles')`,
      [contentTypeId],
    );

    store = new SqliteVectorStore(dataSource);
    await store.assertInfrastructure();
  });

  afterAll(async () => {
    if (dataSource?.isInitialized) {
      await dataSource.destroy();
    }
    rmSync(directory, { force: true, recursive: true });
  });

  beforeEach(async () => {
    await dataSource.query(`DELETE FROM "contents"`);
    await dataSource.query(`DELETE FROM "rag_sqlite_vector_documents"`);
  });

  it('provisions only direct-index tables', async () => {
    const queryRunner = dataSource.createQueryRunner();
    await queryRunner.connect();
    try {
      await expect(isSqliteVectorProvisioned(queryRunner)).resolves.toBe(true);
    } finally {
      await queryRunner.release();
    }

    const objects = await dataSource.query(
      `SELECT "name" FROM "sqlite_master" ` +
        `WHERE "name" LIKE 'rag_sqlite_vector_jobs%' ` +
        `OR "name" LIKE 'contents_enqueue_sqlite_vector_%'`,
    );
    expect(objects).toEqual([]);
  });

  it('loads the corpus for direct reindexing', async () => {
    await insertContent('c1', 'alpha');
    await insertContent('c2', 'beta', false);

    await expect(store.loadContents()).resolves.toEqual([
      { id: 'c1', searchText: 'alpha', status: true },
      { id: 'c2', searchText: 'beta', status: false },
    ]);
  });

  it('ranks by cosine similarity', async () => {
    await insertContent('c1', 'alpha');
    await insertContent('c2', 'beta');
    await saveEmbedding('c1', [1, 0, 0], 'alpha chunk', 'alpha');
    await saveEmbedding('c2', [0, 1, 0], 'beta chunk', 'beta');

    const hits = await store.search([1, 0, 0], PROFILE, { status: true });

    expect(hits).toHaveLength(2);
    expect(hits[0]).toMatchObject({
      contentId: 'c1',
      text: 'alpha chunk',
      contentTypeId,
    });
    expect(hits[0].score).toBeCloseTo(1, 5);
  });

  it('returns the closest chunk per document', async () => {
    await insertContent('c1', 'alpha');
    await store.save('c1', PROFILE, 'alpha', true, [
      { index: 0, text: 'far chunk', embedding: [0, 1, 0] },
      { index: 1, text: 'near chunk', embedding: [1, 0, 0] },
    ]);

    await expect(store.search([1, 0, 0], PROFILE)).resolves.toMatchObject([
      { text: 'near chunk' },
    ]);
  });

  it('ignores stored vectors with a different dimension', async () => {
    await insertContent('c1', 'alpha');
    await insertContent('c2', 'beta');
    await saveEmbedding('c1', [1, 0, 0], 'alpha chunk', 'alpha');
    await saveEmbedding('c2', [1, 0, 0, 0], 'beta chunk', 'beta');

    await expect(store.search([1, 0, 0], PROFILE)).resolves.toMatchObject([
      { contentId: 'c1' },
    ]);
  });

  it('hides stale embeddings', async () => {
    await insertContent('c1', 'alpha');
    await saveEmbedding('c1', [1, 0, 0], 'alpha chunk', 'alpha');
    await dataSource.query(
      `UPDATE "contents" SET "searchText" = ? WHERE "id" = ?`,
      ['rewritten', 'c1'],
    );

    await expect(store.search([1, 0, 0], PROFILE)).resolves.toEqual([]);
  });

  it('removes embeddings directly', async () => {
    await insertContent('c1', 'alpha');
    await saveEmbedding('c1', [1, 0, 0], 'alpha chunk', 'alpha');

    await store.remove('c1');

    await expect(store.search([1, 0, 0], PROFILE)).resolves.toEqual([]);
  });

  it('cascades content deletion to embeddings', async () => {
    await insertContent('c1', 'alpha');
    await saveEmbedding('c1', [1, 0, 0], 'alpha chunk', 'alpha');
    await dataSource.query(`DELETE FROM "contents" WHERE "id" = ?`, ['c1']);

    await expect(
      dataSource.query(
        `SELECT COUNT(*) AS "count" FROM "rag_sqlite_vector_documents"`,
      ),
    ).resolves.toEqual([{ count: 0 }]);
  });
});
