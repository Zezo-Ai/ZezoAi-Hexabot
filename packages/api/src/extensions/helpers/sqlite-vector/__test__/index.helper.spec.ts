/*
 * Hexabot — Fair Core License (FCL-1.0-ALv2)
 * Copyright (c) 2026 Hexastack.
 * Full terms: see LICENSE.md.
 */

jest.mock('@ai-sdk/openai', () => ({
  createOpenAI: jest.fn(),
}));
jest.mock('@ai-sdk/openai-compatible', () => ({
  createOpenAICompatible: jest.fn(),
}));
jest.mock('ai', () => ({
  embed: jest.fn(),
  embedMany: jest.fn(),
}));

import { createOpenAI } from '@ai-sdk/openai';
import { createOpenAICompatible } from '@ai-sdk/openai-compatible';
import { embed, embedMany } from 'ai';
import { DataSource } from 'typeorm';

import { RagHelperConfigurationError } from '@/cms/errors/rag.errors';

import SqliteVectorRagHelper from '../index.helper';

const HELPER_SETTINGS = {
  embedding_provider: 'openai',
  embedding_model: 'text-embedding-3-small',
  embedding_api_key: 'credential-id',
  embedding_base_url: '',
  embedding_dimensions: 2,
  chunk_size: 2000,
  chunk_overlap: 200,
};
const settingsWith = (overrides: Record<string, unknown> = {}) => ({
  'sqlite-vector': { ...HELPER_SETTINGS, ...overrides },
});
const createHelper = (
  type: 'better-sqlite3' | 'sqlite' | 'postgres' | 'mongodb' = 'better-sqlite3',
) => {
  const credentialService = {
    findOneValue: jest.fn().mockResolvedValue('secret'),
  };
  const helper = new SqliteVectorRagHelper(
    {
      options: { type },
    } as DataSource,
    credentialService as never,
  );
  const store = {
    assertInfrastructure: jest.fn(),
    search: jest.fn().mockResolvedValue([]),
    loadContents: jest.fn().mockResolvedValue([]),
    remove: jest.fn(),
    save: jest.fn(),
  };
  const logger = {
    log: jest.fn(),
    warn: jest.fn(),
    error: jest.fn(),
    debug: jest.fn(),
    verbose: jest.fn(),
  };
  const settingService = {
    getSettings: jest.fn().mockResolvedValue(settingsWith()),
  };
  (helper as unknown as { store: unknown }).store = store;
  (helper as unknown as { settingService: unknown }).settingService =
    settingService;
  (helper as unknown as { logger: unknown }).logger = logger;

  return { credentialService, helper, logger, settingService, store };
};

describe('SqliteVectorRagHelper', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    (createOpenAI as jest.Mock).mockReturnValue({
      embedding: jest.fn().mockReturnValue({ modelId: 'embedding-model' }),
    });
    (embed as jest.Mock).mockResolvedValue({ embedding: [1, 0] });
  });

  it('is registered on SQLite only, leaving PostgreSQL to the pgvector helper', () => {
    expect(createHelper('better-sqlite3').helper.isAvailable()).toBe(true);
    expect(createHelper('sqlite').helper.isAvailable()).toBe(true);
    expect(createHelper('postgres').helper.isAvailable()).toBe(false);
    expect(createHelper('mongodb').helper.isAvailable()).toBe(false);
  });

  it('embeds the query and performs exact profile-scoped retrieval', async () => {
    const { helper, store } = createHelper();
    store.search.mockResolvedValue([
      {
        contentId: 'c1',
        title: 'Content',
        text: 'best chunk',
        score: 0.8,
      },
    ]);

    await expect(helper.retrieve('semantic query')).resolves.toEqual([
      {
        contentId: 'c1',
        title: 'Content',
        text: 'best chunk',
        score: 0.8,
        source: 'sqlite-vector',
      },
    ]);

    expect(embed).toHaveBeenCalledWith(
      expect.objectContaining({
        value: 'semantic query',
        maxRetries: 0,
        providerOptions: {
          openai: {
            dimensions: 2,
          },
        },
      }),
    );
    expect(createOpenAI).toHaveBeenCalledWith({
      apiKey: 'secret',
      baseURL: undefined,
    });
    expect(store.search).toHaveBeenCalledWith(
      [1, 0],
      expect.stringMatching(/^[a-f0-9]{64}$/),
      {
        status: true,
        contentTypeId: undefined,
        limit: 3,
      },
    );
  });

  it('uses the selected embedding provider', async () => {
    const { helper, settingService } = createHelper();
    settingService.getSettings.mockResolvedValue(
      settingsWith({
        embedding_provider: 'openai-compatible',
        embedding_base_url: 'https://embeddings.example/v1',
      }),
    );
    (createOpenAICompatible as jest.Mock).mockReturnValue({
      embeddingModel: jest.fn().mockReturnValue({
        modelId: 'compatible-embedding-model',
      }),
    });

    await helper.retrieve('semantic query');

    expect(createOpenAICompatible).toHaveBeenCalledWith({
      apiKey: 'secret',
      baseURL: 'https://embeddings.example/v1',
      name: 'openai-compatible',
    });
    expect(createOpenAI).not.toHaveBeenCalled();
  });

  it('fails explicitly when the embedding credential is missing', async () => {
    const { helper, settingService, store } = createHelper();
    settingService.getSettings.mockResolvedValue(
      settingsWith({ embedding_api_key: '' }),
    );

    await expect(helper.retrieve('query')).rejects.toBeInstanceOf(
      RagHelperConfigurationError,
    );
    expect(store.assertInfrastructure).not.toHaveBeenCalled();
  });

  it('fails explicitly when the selected credential is empty', async () => {
    const { credentialService, helper, store } = createHelper();
    credentialService.findOneValue.mockResolvedValue('');

    await expect(helper.retrieve('query')).rejects.toThrow(
      'credential is missing or empty',
    );
    expect(credentialService.findOneValue).toHaveBeenCalledWith(
      'credential-id',
    );
    expect(store.assertInfrastructure).not.toHaveBeenCalled();
  });

  it('accepts an embedding whose size differs from the requested dimension', async () => {
    const { helper, store, logger } = createHelper();
    // "embedding_dimensions" is 2 but the model returns a 3-dim vector: this is
    // a request the provider did not honor, not an error.
    (embed as jest.Mock).mockResolvedValueOnce({ embedding: [1, 2, 3] });

    await expect(helper.retrieve('query')).resolves.toEqual([]);
    expect(store.search).toHaveBeenCalledWith(
      [1, 2, 3],
      expect.any(String),
      expect.objectContaining({ status: true }),
    );
    expect(logger.warn).toHaveBeenCalledWith(
      expect.stringContaining('Embedding dimensions'),
    );
  });

  it('degrades to empty results and warns when the provider fails to embed the query', async () => {
    const { helper, store, logger } = createHelper();
    const providerError = Object.assign(new Error('Forbidden'), {
      statusCode: 403,
    });
    (embed as jest.Mock).mockRejectedValueOnce(providerError);

    await expect(helper.retrieve('query')).resolves.toEqual([]);
    expect(store.search).not.toHaveBeenCalled();
    expect(logger.warn).toHaveBeenCalledWith(
      expect.stringContaining('Unable to embed the RAG query'),
      providerError,
    );
  });

  it('rejects empty and zero vectors', async () => {
    const { helper } = createHelper();
    (embed as jest.Mock).mockResolvedValueOnce({ embedding: [] });
    await expect(helper.retrieve('query')).rejects.toThrow(
      /invalid, empty, or zero vector/,
    );

    (embed as jest.Mock).mockResolvedValueOnce({ embedding: [0, 0] });
    await expect(helper.retrieve('query')).rejects.toThrow('zero vector');
  });

  it('reindexes directly when embedding settings change', async () => {
    const { helper, store } = createHelper();

    await helper.handleSettingsChanged({ label: 'embedding_model' } as never);
    expect(store.loadContents).toHaveBeenCalledTimes(1);

    await helper.handleSettingsChanged({
      label: 'embedding_provider',
    } as never);
    expect(store.loadContents).toHaveBeenCalledTimes(2);

    await helper.handleSettingsChanged({ label: 'embedding_api_key' } as never);
    expect(store.loadContents).toHaveBeenCalledTimes(3);
  });

  it('ignores settings changes on a database it does not support', async () => {
    const { helper, store } = createHelper('postgres');

    await helper.handleSettingsChanged({ label: 'embedding_model' } as never);

    expect(store.loadContents).not.toHaveBeenCalled();
  });

  it('reindexes the corpus directly', async () => {
    const { helper, store } = createHelper();
    store.loadContents.mockResolvedValue([
      { id: 'c1', searchText: 'body', status: true },
    ]);
    (embedMany as jest.Mock).mockResolvedValue({ embeddings: [[1, 0]] });

    await helper.reindex();

    expect(store.loadContents).toHaveBeenCalledTimes(1);
    expect(store.save).toHaveBeenCalledWith(
      'c1',
      expect.stringMatching(/^[a-f0-9]{64}$/),
      'body',
      true,
      [{ index: 0, text: 'body', embedding: [1, 0] }],
    );
  });

  it('re-evaluates the corpus when index_only_active_content is toggled', async () => {
    const { helper, store } = createHelper();

    await helper.handleSettingsChanged({
      label: 'index_only_active_content',
    } as never);

    expect(store.loadContents).toHaveBeenCalledTimes(1);
  });

  it('removes inactive content instead of transmitting it to the provider', async () => {
    const { helper, store } = createHelper();
    const content = {
      id: 'c1',
      searchText: 'draft body',
      status: false,
    };

    await helper.index(content as never);

    expect(store.remove).toHaveBeenCalledWith('c1');
    expect(store.save).not.toHaveBeenCalled();
    expect(embedMany).not.toHaveBeenCalled();
  });

  it('embeds inactive content when index_only_active_content is disabled', async () => {
    const { helper, settingService, store } = createHelper();
    settingService.getSettings.mockResolvedValue(
      settingsWith({ index_only_active_content: false }),
    );
    (embedMany as jest.Mock).mockResolvedValue({ embeddings: [[1, 0]] });
    const content = {
      id: 'c1',
      searchText: 'draft body',
      status: false,
    };

    await helper.index(content as never);

    expect(store.remove).not.toHaveBeenCalled();
    expect(store.save).toHaveBeenCalledWith(
      'c1',
      expect.stringMatching(/^[a-f0-9]{64}$/),
      'draft body',
      false,
      [{ index: 0, text: 'draft body', embedding: [1, 0] }],
    );
  });

  it('serializes direct indexing operations', async () => {
    const { helper } = createHelper();
    let markStarted!: () => void;
    let releaseEmbedding!: (result: { embeddings: number[][] }) => void;
    const started = new Promise<void>((resolve) => {
      markStarted = resolve;
    });
    const blockedEmbedding = new Promise<{ embeddings: number[][] }>(
      (resolve) => {
        releaseEmbedding = resolve;
      },
    );
    (embedMany as jest.Mock)
      .mockImplementationOnce(() => {
        markStarted();

        return blockedEmbedding;
      })
      .mockResolvedValueOnce({ embeddings: [[0, 1]] });

    const first = helper.index({
      id: 'c1',
      searchText: 'first body',
      status: true,
    } as never);
    await started;
    const second = helper.index({
      id: 'c2',
      searchText: 'second body',
      status: true,
    } as never);
    await Promise.resolve();

    expect(embedMany).toHaveBeenCalledTimes(1);

    releaseEmbedding({ embeddings: [[1, 0]] });
    await Promise.all([first, second]);
    expect(embedMany).toHaveBeenCalledTimes(2);
  });

  it('removes deleted content directly', async () => {
    const { helper, store } = createHelper();

    await helper.remove('c1');

    expect(store.remove).toHaveBeenCalledWith('c1');
  });
});
