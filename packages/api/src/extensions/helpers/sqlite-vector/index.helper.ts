/*
 * Hexabot — Fair Core License (FCL-1.0-ALv2)
 * Copyright (c) 2026 Hexastack.
 * Full terms: see LICENSE.md.
 */

import { createHash } from 'node:crypto';

import { createOpenAI } from '@ai-sdk/openai';
import { createOpenAICompatible } from '@ai-sdk/openai-compatible';
import type { ContentFull, Setting } from '@hexabot-ai/types';
import { Injectable } from '@nestjs/common';
import { OnEvent } from '@nestjs/event-emitter';
import { EmbeddingModel, embed, embedMany } from 'ai';
import { Mutex } from 'async-mutex';
import { DataSource } from 'typeorm';
import type z from 'zod';

import { RagHelperConfigurationError } from '@/cms/errors/rag.errors';
import { DEFAULT_RAG_TOP_K, RagHit, RagQueryOptions } from '@/cms/types/rag';
import { BaseRagHelper } from '@/helper/lib/base-rag-helper';
import { CredentialService } from '@/user/services/credential.service';

import { isSqliteDatabase } from './sqlite-vector.provisioning';
import {
  SQLITE_VECTOR_RAG_HELPER_NAME,
  sqliteVectorSettingsSchema,
} from './sqlite-vector.settings';
import {
  SqliteVectorEmbeddedChunk,
  SqliteVectorContent,
  SqliteVectorStore,
} from './sqlite-vector.store';

const EMBEDDING_TIMEOUT_MS = 60000;

type SqliteVectorSettings = z.infer<typeof sqliteVectorSettingsSchema>;

type EmbeddingProviderInitOptions = {
  apiKey?: string;
  baseURL?: string;
};

type EmbeddingProvider = {
  embeddingModel?: (modelId: string) => EmbeddingModel;
  textEmbeddingModel?: (modelId: string) => EmbeddingModel;
  embedding?: (modelId: string) => EmbeddingModel;
};

type EmbeddingProviderFactory = (
  options: EmbeddingProviderInitOptions,
) => unknown;

/**
 * Semantic RAG helper for SQLite deployments, storing its embeddings in the
 * application database through the sqlite-vec extension. It is the SQLite
 * counterpart of the pgvector helper: the two cover different databases, so
 * exactly one of them is ever available on a given deployment.
 *
 * Content lifecycle hooks update the SQLite index directly. A profile hash over
 * the provider, model, dimensions and chunking settings keys every stored
 * vector, so configuration changes rebuild the corpus under a new profile.
 */
@Injectable()
export default class SqliteVectorRagHelper extends BaseRagHelper<
  typeof SQLITE_VECTOR_RAG_HELPER_NAME
> {
  private readonly store: SqliteVectorStore;

  private readonly indexMutex = new Mutex();

  private settingsReindexPromise?: Promise<void>;

  private settingsReindexRequested = false;

  private dimensionMismatchWarned = false;

  constructor(
    private readonly dataSource: DataSource,
    private readonly credentialService: CredentialService,
  ) {
    super(SQLITE_VECTOR_RAG_HELPER_NAME);
    this.store = new SqliteVectorStore(dataSource);
  }

  /** sqlite-vec availability is checked lazily by the store. */
  public override isAvailable(): boolean {
    return isSqliteDatabase(this.dataSource.options.type);
  }

  async retrieve(
    query: string,
    options: RagQueryOptions = {},
  ): Promise<RagHit[]> {
    const trimmed = query?.trim();
    if (!trimmed) {
      return [];
    }

    const settings = await this.getConfiguration();
    await this.store.assertInfrastructure();

    let embedding: number[];
    try {
      embedding = await this.embedQuery(trimmed, settings);
    } catch (error) {
      // Configuration problems (missing credential, unsupported provider,
      // invalid vector) are deterministic and need operator action, so they
      // keep propagating. A transient provider/network failure while embedding
      // the query (e.g. a 403, rate limit, timeout) must not hard-fail
      // retrieval: degrade to no semantic hits so the conversation continues.
      if (error instanceof RagHelperConfigurationError) {
        throw error;
      }
      this.logger.warn(
        'Unable to embed the RAG query; returning no semantic hits for this request.',
        error,
      );

      return [];
    }

    const profile = this.getProfile(settings);
    const limit = options.limit ?? DEFAULT_RAG_TOP_K;
    const hits = await this.store.search(embedding, profile, {
      status: options.includeInactive ? undefined : true,
      contentTypeId: options.contentTypeId,
      limit,
    });

    return hits.map((hit) => ({
      ...hit,
      source: SQLITE_VECTOR_RAG_HELPER_NAME,
    }));
  }

  async index(content: ContentFull): Promise<void> {
    await this.indexMutex.runExclusive(async () => {
      const settings = await this.getParsedSettings();
      if (settings.index_only_active_content && !content.status) {
        await this.store.remove(content.id);

        return;
      }
      const configuration = await this.resolveCredential(settings);
      await this.indexContent(
        content,
        configuration,
        this.getProfile(settings),
      );
    });
  }

  async reindex(): Promise<void> {
    await this.indexMutex.runExclusive(async () => {
      const settings = await this.getParsedSettings();
      const profile = this.getProfile(settings);
      const contents = await this.store.loadContents();
      const embeddable: SqliteVectorContent[] = [];
      for (const content of contents) {
        if (settings.index_only_active_content && !content.status) {
          await this.store.remove(content.id);
        } else {
          embeddable.push(content);
        }
      }
      if (embeddable.length === 0) {
        return;
      }
      const configuration = await this.resolveCredential(settings);
      for (const content of embeddable) {
        await this.indexContent(content, configuration, profile);
      }
    });
  }

  async remove(contentId: string): Promise<void> {
    await this.indexMutex.runExclusive(() => this.store.remove(contentId));
  }

  @OnEvent('hook:sqlite-vector:*')
  async handleSettingsChanged(setting?: Pick<Setting, 'label'>): Promise<void> {
    if (!this.isAvailable() || !setting?.label) {
      return;
    }

    try {
      if (
        [
          'embedding_api_key',
          'embedding_provider',
          'embedding_model',
          'embedding_base_url',
          'embedding_dimensions',
          'chunk_size',
          'chunk_overlap',
          'index_only_active_content',
        ].includes(setting.label)
      ) {
        this.dimensionMismatchWarned = false;
        await this.requestSettingsReindex();
      }
    } catch (error) {
      this.logger.error(
        'Unable to reindex sqlite-vector RAG after a settings change.',
        error,
      );
    }
  }

  private async requestSettingsReindex(): Promise<void> {
    this.settingsReindexRequested = true;
    this.settingsReindexPromise ??= this.runSettingsReindexes().finally(() => {
      this.settingsReindexPromise = undefined;
    });
    await this.settingsReindexPromise;
  }

  private async runSettingsReindexes(): Promise<void> {
    let lastError: unknown;
    while (this.settingsReindexRequested) {
      this.settingsReindexRequested = false;
      try {
        await this.reindex();
        lastError = undefined;
      } catch (error) {
        lastError = error;
      }
    }
    if (lastError) {
      throw lastError;
    }
  }

  private async indexContent(
    content: Pick<SqliteVectorContent, 'id' | 'searchText' | 'status'>,
    settings: SqliteVectorSettings,
    profile: string,
  ): Promise<void> {
    if (settings.index_only_active_content && !content.status) {
      await this.store.remove(content.id);

      return;
    }

    const chunks = this.chunkSearchText(
      content.searchText,
      settings.chunk_size,
      settings.chunk_overlap,
    );
    const embeddings = chunks.length
      ? await this.embedChunks(
          chunks.map(({ text }) => text),
          settings,
        )
      : [];
    const embeddedChunks: SqliteVectorEmbeddedChunk[] = chunks.map(
      (chunk, index) => ({
        ...chunk,
        embedding: embeddings[index],
      }),
    );
    await this.store.save(
      content.id,
      profile,
      content.searchText,
      content.status,
      embeddedChunks,
    );
  }

  private async getConfiguration(): Promise<SqliteVectorSettings> {
    return this.resolveCredential(await this.getParsedSettings());
  }

  private async getParsedSettings(): Promise<SqliteVectorSettings> {
    const result = sqliteVectorSettingsSchema.safeParse(
      await this.getSettings<typeof SQLITE_VECTOR_RAG_HELPER_NAME>(),
    );
    if (!result.success) {
      throw new RagHelperConfigurationError(
        'The sqlite-vector RAG helper settings are missing or invalid.',
      );
    }

    return {
      ...result.data,
      embedding_model: result.data.embedding_model.trim(),
      embedding_base_url: result.data.embedding_base_url.replace(/\/+$/, ''),
    };
  }

  private async resolveCredential(
    settings: SqliteVectorSettings,
  ): Promise<SqliteVectorSettings> {
    const credentialId = settings.embedding_api_key.trim();
    if (!credentialId) {
      throw new RagHelperConfigurationError(
        'The sqlite-vector RAG helper requires an embedding credential.',
      );
    }
    const apiKey = (
      await this.credentialService.findOneValue(credentialId)
    ).trim();
    if (!apiKey) {
      throw new RagHelperConfigurationError(
        'The selected sqlite-vector embedding credential is missing or empty.',
      );
    }

    return {
      ...settings,
      embedding_api_key: apiKey,
    };
  }

  private getProfile(settings: SqliteVectorSettings): string {
    return createHash('sha256')
      .update(
        JSON.stringify({
          provider: settings.embedding_provider,
          baseUrl: settings.embedding_base_url,
          model: settings.embedding_model,
          dimensions: settings.embedding_dimensions,
          chunkSize: settings.chunk_size,
          chunkOverlap: settings.chunk_overlap,
        }),
      )
      .digest('hex');
  }

  private async getEmbeddingModel(
    settings: SqliteVectorSettings,
  ): Promise<EmbeddingModel> {
    const provider = await this.loadEmbeddingProvider(settings);
    const embeddingModel = provider.embeddingModel;
    if (typeof embeddingModel === 'function') {
      return embeddingModel.call(provider, settings.embedding_model);
    }
    const textEmbeddingModel = provider.textEmbeddingModel;
    if (typeof textEmbeddingModel === 'function') {
      return textEmbeddingModel.call(provider, settings.embedding_model);
    }
    const embedding = provider.embedding;
    if (typeof embedding === 'function') {
      return embedding.call(provider, settings.embedding_model);
    }

    throw new RagHelperConfigurationError(
      `Provider "${settings.embedding_provider}" does not expose an embedding model.`,
    );
  }

  private async embedQuery(
    value: string,
    settings: SqliteVectorSettings,
  ): Promise<number[]> {
    const providerOptions = this.getEmbeddingProviderOptions(settings);
    const result = await embed({
      model: await this.getEmbeddingModel(settings),
      value,
      maxRetries: 0,
      abortSignal: AbortSignal.timeout(EMBEDDING_TIMEOUT_MS),
      ...(providerOptions ? { providerOptions } : {}),
    });
    const embedding = this.validateEmbedding(result.embedding);
    this.warnIfRequestedDimensionIgnored(settings, embedding.length);

    return embedding;
  }

  private async embedChunks(
    values: string[],
    settings: SqliteVectorSettings,
  ): Promise<number[][]> {
    const providerOptions = this.getEmbeddingProviderOptions(settings);
    const result = await embedMany({
      model: await this.getEmbeddingModel(settings),
      values,
      maxRetries: 0,
      abortSignal: AbortSignal.timeout(EMBEDDING_TIMEOUT_MS),
      ...(providerOptions ? { providerOptions } : {}),
    });
    if (result.embeddings.length !== values.length) {
      throw new RagHelperConfigurationError(
        `The embedding endpoint returned ${result.embeddings.length} vectors for ${values.length} chunks.`,
      );
    }

    const embeddings = result.embeddings.map((embedding) =>
      this.validateEmbedding(embedding),
    );
    // The stored chunks of one document must share a dimension so cosine search
    // can compare them against a query vector of the same size.
    const dimension = embeddings[0]?.length;
    if (
      dimension !== undefined &&
      embeddings.some((embedding) => embedding.length !== dimension)
    ) {
      throw new RagHelperConfigurationError(
        'The embedding endpoint returned vectors of inconsistent dimensions.',
      );
    }
    if (dimension !== undefined) {
      this.warnIfRequestedDimensionIgnored(settings, dimension);
    }

    return embeddings;
  }

  private getEmbeddingProviderOptions(
    settings: SqliteVectorSettings,
  ): Record<string, { dimensions: number }> | undefined {
    return this.getProviderId(settings.embedding_provider) === 'openai'
      ? {
          openai: {
            dimensions: settings.embedding_dimensions,
          },
        }
      : undefined;
  }

  private async loadEmbeddingProvider(
    settings: SqliteVectorSettings,
  ): Promise<EmbeddingProvider> {
    const provider = settings.embedding_provider;
    const providerId = this.getProviderId(provider);
    const options: EmbeddingProviderInitOptions = {
      apiKey: settings.embedding_api_key,
      baseURL: settings.embedding_base_url || undefined,
    };

    if (providerId === 'openai') {
      return createOpenAI(options);
    }

    if (providerId === 'gateway') {
      const { createGatewayProvider } = await import('@ai-sdk/gateway');

      return createGatewayProvider(options);
    }

    if (providerId === 'litellm' || providerId === 'openai-compatible') {
      if (!options.baseURL) {
        throw new RagHelperConfigurationError(
          `Provider "${provider}" requires an embedding base URL.`,
        );
      }

      return createOpenAICompatible({
        ...options,
        name: providerId,
        baseURL: options.baseURL,
      });
    }

    const normalized = provider.trim().toLowerCase();
    const moduleCandidates = new Set([
      provider,
      normalized,
      providerId,
      `@ai-sdk/${providerId}`,
    ]);
    let lastError: unknown;

    for (const moduleName of moduleCandidates) {
      try {
        const providerModule = await import(moduleName);
        const resolved = this.instantiateEmbeddingProvider(
          providerModule,
          providerId,
          options,
        );
        if (resolved) {
          return resolved;
        }
      } catch (error) {
        lastError = error;
      }
    }

    throw new RagHelperConfigurationError(
      `Unsupported embedding provider "${provider}". Install the matching AI SDK provider package (for example @ai-sdk/${providerId}) and ensure it supports embeddings.` +
        (lastError ? ` Last error: ${(lastError as Error).message}` : ''),
    );
  }

  private instantiateEmbeddingProvider(
    providerModule: Record<string, unknown>,
    provider: string,
    options: EmbeddingProviderInitOptions,
  ): EmbeddingProvider | undefined {
    for (const factory of this.getProviderFactories(providerModule, provider)) {
      try {
        const created = factory(options);
        if (this.isEmbeddingProvider(created)) {
          return created;
        }
      } catch {
        // Try the next matching provider factory.
      }
    }

    const candidates = [
      providerModule[provider],
      providerModule.default,
      ...Object.values(providerModule),
    ];

    return candidates.find((candidate) =>
      this.isEmbeddingProvider(candidate),
    ) as EmbeddingProvider | undefined;
  }

  private getProviderFactories(
    providerModule: Record<string, unknown>,
    provider: string,
  ): EmbeddingProviderFactory[] {
    const pascalName = provider
      .split(/[^a-zA-Z0-9]+/)
      .filter(Boolean)
      .map((part) => part[0].toUpperCase() + part.slice(1))
      .join('');
    const preferredNames = [
      `create${pascalName}`,
      `create${pascalName}Provider`,
      `create${pascalName}AI`,
      'createProvider',
    ];
    const factories: EmbeddingProviderFactory[] = [];
    const seen = new Set<unknown>();

    for (const name of preferredNames) {
      const candidate = providerModule[name];
      if (typeof candidate === 'function' && !seen.has(candidate)) {
        factories.push(candidate as EmbeddingProviderFactory);
        seen.add(candidate);
      }
    }
    for (const [name, candidate] of Object.entries(providerModule)) {
      if (
        typeof candidate === 'function' &&
        name.startsWith('create') &&
        name.toLowerCase().includes(provider) &&
        !seen.has(candidate)
      ) {
        factories.push(candidate as EmbeddingProviderFactory);
        seen.add(candidate);
      }
    }

    return factories;
  }

  private isEmbeddingProvider(
    candidate: unknown,
  ): candidate is EmbeddingProvider {
    if (
      !candidate ||
      (typeof candidate !== 'function' && typeof candidate !== 'object')
    ) {
      return false;
    }
    const provider = candidate as EmbeddingProvider;

    return (
      typeof provider.embeddingModel === 'function' ||
      typeof provider.textEmbeddingModel === 'function' ||
      typeof provider.embedding === 'function'
    );
  }

  private getProviderId(provider: string): string {
    const normalized = provider
      .trim()
      .toLowerCase()
      .replace(/^@ai-sdk\//, '')
      .replace(/^ai-sdk\//, '');
    const aliases: Record<string, string> = {
      claude: 'anthropic',
      gemini: 'google',
      'google-generative-ai': 'google',
      'google-vertex-ai': 'google-vertex',
      'azure-openai': 'azure',
    };

    return aliases[normalized] ?? normalized;
  }

  /**
   * Validates the structure of an embedding vector. The dimension is treated as
   * an output of the model, not a value the operator must match: consistency
   * between the query vector and the stored chunks is guaranteed by the profile
   * hash (which includes provider, model, and requested dimensions), so we only
   * reject empty, non-finite, or all-zero vectors here.
   */
  private validateEmbedding(embedding: number[]): number[] {
    if (
      embedding.length === 0 ||
      embedding.some((value) => !Number.isFinite(value)) ||
      !embedding.some((value) => value !== 0)
    ) {
      throw new RagHelperConfigurationError(
        'The embedding endpoint returned an invalid, empty, or zero vector.',
      );
    }

    return embedding;
  }

  /**
   * The "Embedding dimensions" setting is only a *request*: providers that
   * support dimension reduction (e.g. OpenAI) honor it, others return their
   * model's native size. When a non-zero request is not honored we log once so
   * the discrepancy is visible without failing indexing or retrieval.
   */
  private warnIfRequestedDimensionIgnored(
    settings: SqliteVectorSettings,
    actualDimension: number,
  ): void {
    const requested = settings.embedding_dimensions;
    if (
      this.dimensionMismatchWarned ||
      !requested ||
      requested === actualDimension
    ) {
      return;
    }
    this.dimensionMismatchWarned = true;
    this.logger.warn(
      `The embedding model returned ${actualDimension}-dimensional vectors, ` +
        `but "Embedding dimensions" is set to ${requested}. The requested size ` +
        `is only applied by providers that support dimension reduction; the ` +
        `model's ${actualDimension}-dimensional output is being used.`,
    );
  }
}
