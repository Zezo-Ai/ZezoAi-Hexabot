/*
 * Hexabot — Fair Core License (FCL-1.0-ALv2)
 * Copyright (c) 2026 Hexastack.
 * Full terms: see LICENSE.md.
 */

import { DataSource, QueryRunner } from 'typeorm';

import { SqliteVectorStore } from '../sqlite-vector.store';

const contentMetadata = {
  tableName: 'contents',
  tablePath: 'contents',
  schema: undefined,
  findColumnWithPropertyName: (name: string) => ({ databaseName: name }),
  findRelationWithPropertyPath: () => ({
    joinColumns: [{ databaseName: 'content_type_id' }],
  }),
};
const createDataSource = () => {
  const query = jest.fn().mockResolvedValue([]);
  const queryRunner = {
    connect: jest.fn(),
    startTransaction: jest.fn(),
    commitTransaction: jest.fn(),
    rollbackTransaction: jest.fn(),
    release: jest.fn(),
    query: jest.fn().mockResolvedValue([]),
    connection: { options: { type: 'better-sqlite3' } },
  };
  const dataSource = {
    options: { type: 'better-sqlite3' },
    getMetadata: jest.fn().mockReturnValue(contentMetadata),
    query,
    createQueryRunner: jest.fn().mockReturnValue(queryRunner),
  } as unknown as DataSource;

  return {
    dataSource,
    query,
    queryRunner: queryRunner as unknown as jest.Mocked<QueryRunner>,
  };
};
/** Builds a store that skips provisioning, so the queries are what is tested. */
const createStore = () => {
  const context = createDataSource();
  const store = new SqliteVectorStore(context.dataSource);
  (store as unknown as { infrastructureReady: boolean }).infrastructureReady =
    true;
  (store as unknown as { extensionLoaded: boolean }).extensionLoaded = true;

  return { ...context, store };
};
const ISO = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/;
const job = { contentId: 'c1', revision: 4, attempts: 0 };

describe('SqliteVectorStore', () => {
  describe('claimJobs', () => {
    it('claims due and lease-expired jobs in a single RETURNING statement', async () => {
      const { store, query } = createStore();
      query.mockResolvedValueOnce([
        { contentId: 'c1', revision: '2', attempts: '1' },
      ]);

      await expect(store.claimJobs('worker-1', 2)).resolves.toEqual([
        { contentId: 'c1', revision: 2, attempts: 1 },
      ]);

      const [sql, params] = query.mock.calls[0];
      expect(sql).toContain('UPDATE "rag_sqlite_vector_jobs"');
      expect(sql).toContain('RETURNING "content_id" AS "contentId"');
      expect(sql).toContain('"locked_at" IS NULL AND "available_at" <= ?');
      // PostgreSQL needs FOR UPDATE SKIP LOCKED here; SQLite serializes writers.
      expect(sql).not.toContain('SKIP LOCKED');
      expect(sql).not.toContain('$1');
      expect(params[1]).toBe('worker-1');
      expect(params[5]).toBe(2);
      // The lease cutoff must be in the past relative to the claim timestamp.
      expect(String(params[4]) < String(params[0])).toBe(true);
      params
        .filter((_: unknown, index: number) => index !== 1 && index !== 5)
        .forEach((value: unknown) => expect(String(value)).toMatch(ISO));
    });

    it('floors a fractional limit to at least one row', async () => {
      const { store, query } = createStore();

      await store.claimJobs('worker-1', 0);

      expect(query.mock.calls[0][1][5]).toBe(1);
    });
  });

  describe('fail', () => {
    beforeEach(() => {
      jest.useFakeTimers().setSystemTime(new Date('2026-01-01T00:00:00.000Z'));
    });

    afterEach(() => {
      jest.useRealTimers();
    });

    it('backs off exponentially in JavaScript rather than in SQL', async () => {
      const { store, query } = createStore();

      await store.fail({ ...job, attempts: 3 }, 'worker-1', new Error('boom'));

      const [sql, params] = query.mock.calls[0];
      expect(sql).toContain('"attempts" = "attempts" + 1');
      expect(sql).not.toContain('make_interval');
      // 5 * 2^3 = 40 seconds.
      expect(params[0]).toBe('2026-01-01T00:00:40.000Z');
      expect(params[1]).toBe('boom');
      expect(params.slice(3)).toEqual(['c1', 4, 'worker-1']);
    });

    it('caps the retry delay at fifteen minutes', async () => {
      const { store, query } = createStore();

      await store.fail({ ...job, attempts: 40 }, 'worker-1', new Error('boom'));

      expect(query.mock.calls[0][1][0]).toBe('2026-01-01T00:15:00.000Z');
    });

    it('truncates an oversized provider error', async () => {
      const { store, query } = createStore();

      await store.fail(job, 'worker-1', new Error('x'.repeat(5000)));

      expect(String(query.mock.calls[0][1][1])).toHaveLength(4000);
    });
  });

  describe('search', () => {
    it('scores with sqlite-vec cosine distance and keeps one chunk per content', async () => {
      const { store, query } = createStore();
      query.mockResolvedValueOnce([
        {
          contentId: 'c1',
          title: 'Title',
          text: 'chunk',
          contentTypeId: 'ct1',
          score: 0.75,
        },
      ]);

      await expect(
        store.search([1, 0], 'profile-hash', {
          status: true,
          contentTypeId: 'ct1',
          limit: 5,
        }),
      ).resolves.toEqual([
        {
          contentId: 'c1',
          title: 'Title',
          text: 'chunk',
          contentTypeId: 'ct1',
          score: 0.75,
        },
      ]);

      const [sql, params] = query.mock.calls[0];
      expect(sql).toContain(
        '1 - vec_distance_cosine(chunk."embedding", vec_f32(?))',
      );
      expect(sql).toContain('ROW_NUMBER() OVER');
      expect(sql).toContain('ranked WHERE "rank" = 1');
      expect(sql).toContain('LIMIT 5');
      // The vector is bound twice: once for the score, once for the ordering.
      expect(params).toEqual(['[1,0]', '[1,0]', 'profile-hash', 1, 'ct1']);
    });

    it('omits the status filter when inactive content is included', async () => {
      const { store, query } = createStore();

      await store.search([1, 0], 'profile-hash', {});

      expect(query.mock.calls[0][1]).toEqual([
        '[1,0]',
        '[1,0]',
        'profile-hash',
      ]);
      expect(query.mock.calls[0][0]).toContain('LIMIT 10');
    });

    it('caps an oversized limit', async () => {
      const { store, query } = createStore();

      await store.search([1, 0], 'profile-hash', { limit: 5000 });

      expect(query.mock.calls[0][0]).toContain('LIMIT 50');
    });
  });

  describe('save', () => {
    it('writes chunks as vec_f32 blobs and drops stale profiles afterwards', async () => {
      const { store, queryRunner } = createStore();
      queryRunner.query
        .mockResolvedValueOnce([{ searchText: 'body' }])
        .mockResolvedValueOnce([{ revision: 4, workerId: 'worker-1' }])
        .mockResolvedValue([]);

      await expect(
        store.save(job, 'worker-1', 'profile-hash', 'body', [
          { index: 0, text: 'chunk', embedding: [1, 0] },
        ]),
      ).resolves.toBe(true);

      const statements = queryRunner.query.mock.calls.map(([sql]) => sql);
      expect(statements.some((sql) => sql.includes('FOR UPDATE'))).toBe(false);
      const insert = queryRunner.query.mock.calls.find(([sql]) =>
        sql.includes('"rag_sqlite_vector_chunks"'),
      );
      expect(insert?.[0]).toContain('VALUES (?, ?, ?, ?, vec_f32(?))');
      expect(insert?.[1]).toEqual(['c1', 'profile-hash', 0, 'chunk', '[1,0]']);
      // Old profiles are only removed once the new one is written.
      const staleDelete = statements.findIndex((sql) =>
        sql.includes('"profile" <> ?'),
      );
      expect(staleDelete).toBeGreaterThan(
        statements.findIndex((sql) =>
          sql.includes('"rag_sqlite_vector_chunks"'),
        ),
      );
      expect(queryRunner.commitTransaction).toHaveBeenCalled();
    });

    it('refuses to write when the source changed under the claim', async () => {
      const { store, queryRunner } = createStore();
      queryRunner.query
        .mockResolvedValueOnce([{ searchText: 'edited meanwhile' }])
        .mockResolvedValueOnce([{ revision: 4, workerId: 'worker-1' }]);

      await expect(
        store.save(job, 'worker-1', 'profile-hash', 'body', []),
      ).resolves.toBe(false);

      expect(
        queryRunner.query.mock.calls.some(([sql]) => sql.startsWith('INSERT')),
      ).toBe(false);
      expect(queryRunner.commitTransaction).toHaveBeenCalled();
    });

    it('refuses to write when another worker holds the lease', async () => {
      const { store, queryRunner } = createStore();
      queryRunner.query
        .mockResolvedValueOnce([{ searchText: 'body' }])
        .mockResolvedValueOnce([{ revision: 4, workerId: 'worker-2' }]);

      await expect(
        store.save(job, 'worker-1', 'profile-hash', 'body', []),
      ).resolves.toBe(false);
    });
  });

  describe('enqueueMissing', () => {
    it('binds the profile after the timestamps it follows in the statement', async () => {
      const { store, query } = createStore();

      await store.enqueueMissing('profile-hash', true);

      const [sql, params] = query.mock.calls[0];
      // SQLite has no boolean type: the status column holds 0/1.
      expect(sql).toContain('content."status" = 1');
      expect(sql).toContain('content."status" = 0');
      expect(sql).toContain('ON CONFLICT ("content_id") DO NOTHING');
      expect(params).toEqual([
        expect.stringMatching(ISO),
        expect.stringMatching(ISO),
        'profile-hash',
      ]);
    });

    it('only looks for missing documents when inactive content is indexed too', async () => {
      const { store, query } = createStore();

      await store.enqueueMissing('profile-hash', false);

      expect(query.mock.calls[0][0]).not.toContain('content."status"');
    });
  });

  describe('enqueueAll', () => {
    it('bumps the revision of existing jobs through an upsert', async () => {
      const { store, query } = createStore();

      await store.enqueueAll();

      const [sql] = query.mock.calls[0];
      expect(sql).toContain(
        '"revision" = "rag_sqlite_vector_jobs"."revision" + 1',
      );
      expect(sql).toContain('"available_at" = excluded."available_at"');
      // Without a WHERE clause SQLite cannot parse the upsert after a SELECT.
      expect(sql).toContain('WHERE 1 = 1 ON CONFLICT');
    });
  });
});
