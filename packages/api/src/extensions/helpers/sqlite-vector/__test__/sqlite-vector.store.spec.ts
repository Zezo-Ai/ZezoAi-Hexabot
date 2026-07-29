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
const createStore = () => {
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
  const store = new SqliteVectorStore(dataSource);
  (store as unknown as { infrastructureReady: boolean }).infrastructureReady =
    true;
  (store as unknown as { extensionLoaded: boolean }).extensionLoaded = true;

  return {
    query,
    queryRunner: queryRunner as unknown as jest.Mocked<QueryRunner>,
    store,
  };
};

describe('SqliteVectorStore', () => {
  it('loads the content corpus for direct reindexing', async () => {
    const { query, store } = createStore();
    query.mockResolvedValueOnce([
      { id: 'c1', searchText: 'body', status: 1 },
      { id: 'c2', searchText: null, status: 0 },
    ]);

    await expect(store.loadContents()).resolves.toEqual([
      { id: 'c1', searchText: 'body', status: true },
      { id: 'c2', searchText: '', status: false },
    ]);
  });

  it('removes a content document directly', async () => {
    const { query, store } = createStore();

    await store.remove('c1');

    expect(query).toHaveBeenCalledWith(
      'DELETE FROM "rag_sqlite_vector_documents" WHERE "content_id" = ?',
      ['c1'],
    );
  });

  describe('search', () => {
    it('scores with sqlite-vec cosine distance and keeps one chunk per content', async () => {
      const { query, store } = createStore();
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
      expect(sql).toContain('LIMIT 5');
      expect(params).toEqual(['[1,0]', '[1,0]', 'profile-hash', 1, 'ct1']);
    });
  });

  describe('save', () => {
    it('writes chunks and drops stale profiles afterwards', async () => {
      const { queryRunner, store } = createStore();
      queryRunner.query.mockResolvedValueOnce([{ searchText: 'body' }]);

      await expect(
        store.save('c1', 'profile-hash', 'body', [
          { index: 0, text: 'chunk', embedding: [1, 0] },
        ]),
      ).resolves.toBe(true);

      const statements = queryRunner.query.mock.calls.map(([sql]) => sql);
      const insert = queryRunner.query.mock.calls.find(([sql]) =>
        sql.includes('"rag_sqlite_vector_chunks"'),
      );
      expect(insert?.[0]).toContain('vec_f32(?)');
      expect(insert?.[1]).toEqual(['c1', 'profile-hash', 0, 'chunk', '[1,0]']);
      expect(
        statements.findIndex((sql) => sql.includes('"profile" <> ?')),
      ).toBeGreaterThan(
        statements.findIndex((sql) =>
          sql.includes('"rag_sqlite_vector_chunks"'),
        ),
      );
      expect(queryRunner.commitTransaction).toHaveBeenCalled();
    });

    it('does not overwrite content changed while embedding', async () => {
      const { queryRunner, store } = createStore();
      queryRunner.query.mockResolvedValueOnce([{ searchText: 'updated body' }]);

      await expect(
        store.save('c1', 'profile-hash', 'old body', []),
      ).resolves.toBe(false);

      expect(
        queryRunner.query.mock.calls.some(([sql]) => sql.startsWith('INSERT')),
      ).toBe(false);
    });

    it('rolls back a failed write', async () => {
      const { queryRunner, store } = createStore();
      const error = new Error('write failed');
      queryRunner.query.mockRejectedValueOnce(error);

      await expect(store.save('c1', 'profile-hash', 'body', [])).rejects.toBe(
        error,
      );
      expect(queryRunner.rollbackTransaction).toHaveBeenCalled();
    });
  });
});
