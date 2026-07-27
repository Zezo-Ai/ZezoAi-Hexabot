/*
 * Hexabot — Fair Core License (FCL-1.0-ALv2)
 * Copyright (c) 2026 Hexastack.
 * Full terms: see LICENSE.md.
 */

import { withDefaultContentTypeProperties } from '@hexabot-ai/types';
import { JSONSchema7 } from 'json-schema';
import { MigrationInterface, QueryRunner } from 'typeorm';

export default class Migration1785162116012_V3_4_1
  implements MigrationInterface
{
  name = 'Migration1785162116012_V3_4_1';

  public async up(queryRunner: QueryRunner): Promise<void> {
    const schemaName = (queryRunner.connection.options as { schema?: string })
      .schema;
    const table = [schemaName, 'content_types']
      .filter(Boolean)
      .map((name) => `"${name!.replace(/"/g, '""')}"`)
      .join('.');
    const contentTypes = (await queryRunner.query(
      `SELECT "id", "schema" FROM ${table}`,
    )) as Array<{ id: string; schema: string | JSONSchema7 }>;
    const isPostgres = queryRunner.connection.options.type === 'postgres';

    for (const contentType of contentTypes) {
      const schema = (
        typeof contentType.schema === 'string'
          ? JSON.parse(contentType.schema)
          : contentType.schema
      ) as JSONSchema7;
      const properties = schema.properties ?? {};

      await queryRunner.query(
        `UPDATE ${table} SET "schema" = ${isPostgres ? '$1' : '?'} ` +
          `WHERE "id" = ${isPostgres ? '$2' : '?'}`,
        [
          JSON.stringify({
            ...schema,
            properties: withDefaultContentTypeProperties(properties),
          }),
          contentType.id,
        ],
      );
    }
  }

  public async down(): Promise<void> {
    // Removing these fields could delete pre-existing user definitions.
  }
}
