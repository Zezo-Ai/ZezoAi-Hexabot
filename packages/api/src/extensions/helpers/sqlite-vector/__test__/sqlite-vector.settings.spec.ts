/*
 * Hexabot — Fair Core License (FCL-1.0-ALv2)
 * Copyright (c) 2026 Hexastack.
 * Full terms: see LICENSE.md.
 */

import { vercelAiSdkProviders } from '../../../actions/ai/provider.constants';
import {
  SQLITE_VECTOR_RAG_HELPER_NAME,
  sqliteVectorSettingsSchema,
} from '../sqlite-vector.settings';

describe('sqliteVectorSettingsSchema', () => {
  it('exposes the AI provider list and credential selector metadata', () => {
    const schema = sqliteVectorSettingsSchema.toJSONSchema({
      target: 'draft-07',
    }) as {
      properties?: Record<
        string,
        {
          enum?: string[];
          'ui:widget'?: string;
          'ui:options'?: Record<string, unknown>;
        }
      >;
    };

    expect(schema.properties?.embedding_provider?.enum).toEqual([
      ...vercelAiSdkProviders,
    ]);
    expect(schema.properties?.embedding_api_key).toMatchObject({
      'ui:widget': 'AutoCompleteWidget',
      'ui:options': {
        entity: 'Credential',
        valueKey: 'id',
        labelKey: 'name',
        enableEntityAddButton: true,
      },
    });
    expect(schema.properties?.embedding_base_url).toMatchObject({
      'ui:options': {
        showWhen: {
          field: 'embedding_provider',
          in: ['gateway', 'litellm', 'openai-compatible'],
        },
      },
    });
  });

  it('defaults every field so the setting group can be seeded', () => {
    expect(sqliteVectorSettingsSchema.parse({})).toEqual({
      embedding_provider: 'openai',
      embedding_model: 'text-embedding-3-small',
      embedding_api_key: '',
      embedding_base_url: '',
      embedding_dimensions: 1536,
      chunk_size: 2000,
      chunk_overlap: 200,
      index_only_active_content: true,
    });
  });

  it('rejects an overlap that is not smaller than the chunk size', () => {
    expect(
      sqliteVectorSettingsSchema.safeParse({
        chunk_size: 500,
        chunk_overlap: 500,
      }).success,
    ).toBe(false);
    expect(
      sqliteVectorSettingsSchema.safeParse({
        chunk_size: 500,
        chunk_overlap: 100,
      }).success,
    ).toBe(true);
  });

  it('is registered under the helper name so settings resolve', () => {
    expect(SQLITE_VECTOR_RAG_HELPER_NAME).toBe('sqlite-vector');
  });
});
