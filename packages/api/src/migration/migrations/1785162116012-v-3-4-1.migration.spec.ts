/*
 * Hexabot — Fair Core License (FCL-1.0-ALv2)
 * Copyright (c) 2026 Hexastack.
 * Full terms: see LICENSE.md.
 */

import { withDefaultContentTypeProperties } from '@hexabot-ai/types';
import { DataSource } from 'typeorm';

import Migration1785162116012_V3_4_1 from './1785162116012-v-3-4-1.migration';

describe('Migration v3.4.1', () => {
  it('adds missing default content type properties', async () => {
    const dataSource = await new DataSource({
      type: 'better-sqlite3',
      database: ':memory:',
    }).initialize();
    const schema = {
      type: 'object',
      properties: {
        title: { type: 'string', title: 'Headline' },
        summary: { type: 'string', title: 'Summary' },
      },
    };

    try {
      await dataSource.query(
        `CREATE TABLE "content_types" ("id" varchar PRIMARY KEY, "schema" text)`,
      );
      await dataSource.query(
        `INSERT INTO "content_types" ("id", "schema") VALUES (?, ?)`,
        ['articles', JSON.stringify(schema)],
      );

      await new Migration1785162116012_V3_4_1().up(
        dataSource.createQueryRunner(),
      );

      const [contentType] = await dataSource.query(
        `SELECT "schema" FROM "content_types" WHERE "id" = ?`,
        ['articles'],
      );
      expect(JSON.parse(contentType.schema)).toEqual({
        ...schema,
        properties: withDefaultContentTypeProperties(schema.properties),
      });
    } finally {
      await dataSource.destroy();
    }
  });
});
