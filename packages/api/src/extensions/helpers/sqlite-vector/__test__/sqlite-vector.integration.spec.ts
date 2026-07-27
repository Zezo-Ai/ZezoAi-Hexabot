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
import { SqliteVectorJob, SqliteVectorStore } from '../sqlite-vector.store';

/**
 * End-to-end coverage of the sqlite-vec path against a real database file.
 * Unlike the pgvector suite this needs no external service, so it runs in
 * normal CI — but it is skipped when the sqlite-vec binary cannot load on the
 * host platform (musl-based images ship no compatible prebuild), which is the
 * same condition under which the helper degrades at runtime, and when the
 * process is configured for PostgreSQL, because the entity column decorators
 * resolve their column types from `DB_TYPE` at import time.
 */
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
const OTHER_PROFILE = 'profile-b';
const WORKER = 'worker-1';

describeWithSqliteVec('SQLite sqlite-vector RAG integration', () => {
  jest.setTimeout(30000);

  let directory: string;
  let dataSource: DataSource;
  let store: SqliteVectorStore;
  let contentTypeId: string;
  // Claiming locks every due job at once, so the ones a call does not consume
  // are held here rather than left locked and unreachable.
  const claimed = new Map<string, SqliteVectorJob>();
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
  const claimOne = async () => {
    const [job] = await store.claimJobs(WORKER, 5);

    return job;
  };
  const saveEmbedding = async (
    contentId: string,
    embedding: number[],
    text: string,
    profile = PROFILE,
  ): Promise<boolean> => {
    if (!claimed.has(contentId)) {
      for (const job of await store.claimJobs(WORKER, 50)) {
        claimed.set(job.contentId, job);
      }
    }
    const job = claimed.get(contentId);
    if (!job) {
      throw new Error(`No job queued for "${contentId}".`);
    }
    claimed.delete(contentId);
    const content = await store.loadContent(contentId);

    return await store.save(job, WORKER, profile, content!.searchText, [
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
    claimed.clear();
    await dataSource.query(`DELETE FROM "contents"`);
    await dataSource.query(`DELETE FROM "rag_sqlite_vector_jobs"`);
    await dataSource.query(`DELETE FROM "rag_sqlite_vector_documents"`);
  });

  it('provisions its tables and triggers idempotently', async () => {
    const queryRunner = dataSource.createQueryRunner();
    await queryRunner.connect();
    try {
      await expect(isSqliteVectorProvisioned(queryRunner)).resolves.toBe(true);
    } finally {
      await queryRunner.release();
    }

    // A second run must be a no-op rather than an error.
    const second = new SqliteVectorStore(dataSource);
    await expect(second.assertInfrastructure()).resolves.toBeUndefined();
  });

  it('enqueues work from the content triggers', async () => {
    await insertContent('c1', 'first body');
    await expect(claimOne()).resolves.toMatchObject({
      contentId: 'c1',
      revision: 1,
    });

    // An edit of the indexed text bumps the revision so an in-flight claim
    // cannot overwrite fresher content.
    await dataSource.query(
      `UPDATE "contents" SET "searchText" = ? WHERE "id" = ?`,
      ['second body', 'c1'],
    );
    await expect(claimOne()).resolves.toMatchObject({
      contentId: 'c1',
      revision: 2,
    });

    // A status flip must schedule work too: the worker decides embed vs remove.
    await dataSource.query(
      `UPDATE "contents" SET "status" = 0 WHERE "id" = ?`,
      ['c1'],
    );
    await expect(claimOne()).resolves.toMatchObject({ revision: 3 });

    // An unrelated column, or a write that changes nothing, must not: the
    // revision stays where the status flip left it.
    await dataSource.query(`UPDATE "contents" SET "title" = ? WHERE "id" = ?`, [
      'Renamed',
      'c1',
    ]);
    await dataSource.query(
      `UPDATE "contents" SET "searchText" = ? WHERE "id" = ?`,
      ['second body', 'c1'],
    );

    const [row] = await dataSource.query(
      `SELECT "revision" FROM "rag_sqlite_vector_jobs" WHERE "content_id" = ?`,
      ['c1'],
    );
    expect(row.revision).toBe(3);
  });

  it('ranks by cosine similarity and returns the best chunk per content', async () => {
    await insertContent('c1', 'alpha');
    await insertContent('c2', 'beta');
    await saveEmbedding('c1', [1, 0, 0], 'alpha chunk');
    await saveEmbedding('c2', [0, 1, 0], 'beta chunk');

    const hits = await store.search([1, 0, 0], PROFILE, { status: true });

    expect(hits).toHaveLength(2);
    expect(hits[0]).toMatchObject({
      contentId: 'c1',
      text: 'alpha chunk',
      contentTypeId,
    });
    expect(hits[0].score).toBeCloseTo(1, 5);
    expect(hits[1].contentId).toBe('c2');
    expect(hits[1].score).toBeCloseTo(0, 5);
  });

  it('returns only the closest chunk of a multi-chunk document', async () => {
    await insertContent('c1', 'alpha');
    const job = await claimOne();
    await store.save(job, WORKER, PROFILE, 'alpha', [
      { index: 0, text: 'far chunk', embedding: [0, 1, 0] },
      { index: 1, text: 'near chunk', embedding: [1, 0, 0] },
    ]);

    const hits = await store.search([1, 0, 0], PROFILE);

    expect(hits).toHaveLength(1);
    expect(hits[0].text).toBe('near chunk');
  });

  it('applies the status and content type filters and the limit', async () => {
    await insertContent('c1', 'alpha');
    await insertContent('c2', 'beta', false);
    await saveEmbedding('c1', [1, 0, 0], 'alpha chunk');
    await saveEmbedding('c2', [1, 0, 0], 'beta chunk');

    await expect(
      store.search([1, 0, 0], PROFILE, { status: true }),
    ).resolves.toHaveLength(1);
    await expect(store.search([1, 0, 0], PROFILE)).resolves.toHaveLength(2);
    await expect(
      store.search([1, 0, 0], PROFILE, { contentTypeId: randomUUID() }),
    ).resolves.toEqual([]);
    await expect(
      store.search([1, 0, 0], PROFILE, { limit: 1 }),
    ).resolves.toHaveLength(1);
  });

  it('hides embeddings whose source text no longer matches the content', async () => {
    await insertContent('c1', 'alpha');
    await saveEmbedding('c1', [1, 0, 0], 'alpha chunk');
    await expect(store.search([1, 0, 0], PROFILE)).resolves.toHaveLength(1);

    await dataSource.query(
      `UPDATE "contents" SET "searchText" = ? WHERE "id" = ?`,
      ['rewritten', 'c1'],
    );

    // The stale document is not served, and the trigger queued a refresh.
    await expect(store.search([1, 0, 0], PROFILE)).resolves.toEqual([]);
    await expect(claimOne()).resolves.toMatchObject({ contentId: 'c1' });
  });

  it('serves the previous profile until the new one is written', async () => {
    await insertContent('c1', 'alpha');
    await saveEmbedding('c1', [1, 0, 0], 'old chunk');
    await expect(store.search([1, 0, 0], PROFILE)).resolves.toHaveLength(1);

    await store.enqueueAll();
    await saveEmbedding('c1', [1, 0, 0], 'new chunk', OTHER_PROFILE);

    await expect(store.search([1, 0, 0], OTHER_PROFILE)).resolves.toMatchObject(
      [{ text: 'new chunk' }],
    );
    // The superseded profile is dropped only after the replacement succeeds.
    await expect(store.search([1, 0, 0], PROFILE)).resolves.toEqual([]);
  });

  it('rejects a save whose claim is stale', async () => {
    await insertContent('c1', 'alpha');
    const job = await claimOne();
    await dataSource.query(
      `UPDATE "contents" SET "searchText" = ? WHERE "id" = ?`,
      ['edited mid-flight', 'c1'],
    );

    await expect(
      store.save(job, WORKER, PROFILE, 'alpha', [
        { index: 0, text: 'stale chunk', embedding: [1, 0, 0] },
      ]),
    ).resolves.toBe(false);
    await expect(store.search([1, 0, 0], PROFILE)).resolves.toEqual([]);
  });

  it('discards inactive content and releases its job', async () => {
    await insertContent('c1', 'alpha', false);
    await saveEmbedding('c1', [1, 0, 0], 'alpha chunk');
    await store.enqueueAll();
    const job = await claimOne();

    await expect(store.discardInactive(job, WORKER)).resolves.toBe(true);

    await expect(store.search([1, 0, 0], PROFILE)).resolves.toEqual([]);
    await expect(claimOne()).resolves.toBeUndefined();
  });

  it('reschedules a failed job with an exponential backoff', async () => {
    await insertContent('c1', 'alpha');
    const job = await claimOne();

    await store.fail(job, WORKER, new Error('provider exploded'));

    const [row] = await dataSource.query(
      `SELECT "attempts", "available_at", "locked_by", "last_error" ` +
        `FROM "rag_sqlite_vector_jobs" WHERE "content_id" = ?`,
      ['c1'],
    );
    expect(row.attempts).toBe(1);
    expect(row.locked_by).toBeNull();
    expect(row.last_error).toBe('provider exploded');
    expect(new Date(row.available_at).getTime()).toBeGreaterThan(Date.now());
    // Still backing off, so it is not claimable yet.
    await expect(claimOne()).resolves.toBeUndefined();

    await store.wakePendingRetries();
    await expect(claimOne()).resolves.toMatchObject({ contentId: 'c1' });
  });

  it('enqueues only content whose index is out of date', async () => {
    await insertContent('c1', 'alpha');
    await insertContent('c2', 'beta');
    await saveEmbedding('c1', [1, 0, 0], 'alpha chunk');
    await saveEmbedding('c2', [0, 1, 0], 'beta chunk');
    await dataSource.query(`DELETE FROM "rag_sqlite_vector_jobs"`);

    await store.enqueueMissing(PROFILE, true);
    await expect(claimOne()).resolves.toBeUndefined();

    // A different profile means every row needs re-embedding.
    await store.enqueueMissing(OTHER_PROFILE, true);
    const jobs = await store.claimJobs(WORKER, 10);
    expect(jobs.map(({ contentId }) => contentId).sort()).toEqual(['c1', 'c2']);
  });

  it('queues removal of inactive content that is still indexed', async () => {
    // The state an operator reaches by turning "index only active content" on
    // after inactive rows were already embedded: a status flip alone cannot
    // produce it, because the content trigger drops the documents itself.
    await insertContent('c1', 'alpha', false);
    await saveEmbedding('c1', [1, 0, 0], 'alpha chunk');
    await dataSource.query(`DELETE FROM "rag_sqlite_vector_jobs"`);

    await store.enqueueMissing(PROFILE, true);

    await expect(claimOne()).resolves.toMatchObject({ contentId: 'c1' });
  });

  it('cascades deletion of a content row to its embeddings and jobs', async () => {
    await insertContent('c1', 'alpha');
    await saveEmbedding('c1', [1, 0, 0], 'alpha chunk');

    await dataSource.query(`DELETE FROM "contents" WHERE "id" = ?`, ['c1']);

    await expect(
      dataSource.query(
        `SELECT COUNT(*) AS "count" FROM "rag_sqlite_vector_chunks"`,
      ),
    ).resolves.toEqual([{ count: 0 }]);
    await expect(
      dataSource.query(
        `SELECT COUNT(*) AS "count" FROM "rag_sqlite_vector_documents"`,
      ),
    ).resolves.toEqual([{ count: 0 }]);
    await expect(
      dataSource.query(
        `SELECT COUNT(*) AS "count" FROM "rag_sqlite_vector_jobs"`,
      ),
    ).resolves.toEqual([{ count: 0 }]);
  });

  it('does not hand the same job to two workers at once', async () => {
    await insertContent('c1', 'alpha');

    const first = await store.claimJobs('worker-a', 5);
    const second = await store.claimJobs('worker-b', 5);

    expect(first).toHaveLength(1);
    expect(second).toEqual([]);
  });
});
