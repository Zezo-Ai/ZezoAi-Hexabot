/*
 * Hexabot — Fair Core License (FCL-1.0-ALv2)
 * Copyright (c) 2026 Hexastack.
 * Full terms: see LICENSE.md.
 */

import type { ContentFull, Setting } from '@hexabot-ai/types';
import { Injectable } from '@nestjs/common';
import { OnEvent } from '@nestjs/event-emitter';
import { Mutex } from 'async-mutex';
import { DataSource } from 'typeorm';
import type z from 'zod';

import { RagHelperConfigurationError } from '@/cms/errors/rag.errors';
import { DEFAULT_RAG_TOP_K, RagHit, RagQueryOptions } from '@/cms/types/rag';
import { BaseRagEmbeddingHelper } from '@/helper/lib/base-rag-embedding-helper';
import { HelperType } from '@/helper/types';
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

type SqliteVectorSettings = z.infer<typeof sqliteVectorSettingsSchema>;

/**
 * Semantic RAG helper for SQLite deployments, storing its embeddings in the
 * application database through the sqlite-vec extension. Database-specific
 * availability ensures it is registered only for SQLite deployments.
 *
 * Content lifecycle hooks update the SQLite index directly. A profile hash over
 * the provider, model, dimensions and chunking settings keys every stored
 * vector, so configuration changes rebuild the corpus under a new profile.
 */
@Injectable()
export default class SqliteVectorRagHelper extends BaseRagEmbeddingHelper<
  typeof SQLITE_VECTOR_RAG_HELPER_NAME
> {
  private readonly store: SqliteVectorStore;

  private readonly indexMutex = new Mutex();

  private settingsReindexPromise?: Promise<void>;

  private settingsReindexRequested = false;

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
      const failures: unknown[] = [];
      for (const content of contents) {
        if (settings.index_only_active_content && !content.status) {
          try {
            await this.store.remove(content.id);
          } catch (error) {
            failures.push(error);
            this.logger.warn(
              `Unable to remove inactive content "${content.id}" from sqlite-vector.`,
              error,
            );
          }
        } else {
          embeddable.push(content);
        }
      }
      if (embeddable.length > 0) {
        const configuration = await this.resolveCredential(settings);
        for (const content of embeddable) {
          try {
            await this.indexContent(content, configuration, profile);
          } catch (error) {
            failures.push(error);
            this.logger.warn(
              `Unable to index content "${content.id}" in sqlite-vector.`,
              error,
            );
          }
        }
      }
      if (failures.length > 0) {
        throw new AggregateError(
          failures,
          `Unable to reindex ${failures.length} sqlite-vector content item(s).`,
        );
      }
    });
  }

  async remove(contentId: string): Promise<void> {
    await this.indexMutex.runExclusive(() => this.store.remove(contentId));
  }

  @OnEvent('hook:sqlite-vector:*')
  async handleSettingsChanged(setting?: Pick<Setting, 'label'>): Promise<void> {
    if (!this.isAvailable() || !setting?.label || !(await this.isSelected())) {
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
    this.settingsReindexPromise ??= this.runSettingsReindexes();
    await this.settingsReindexPromise;
  }

  private async runSettingsReindexes(): Promise<void> {
    let lastError: unknown;
    try {
      while (this.settingsReindexRequested) {
        this.settingsReindexRequested = false;
        try {
          await this.reindex();
          lastError = undefined;
        } catch (error) {
          lastError = error;
        }
      }
    } finally {
      this.settingsReindexPromise = undefined;
    }
    if (lastError) {
      throw lastError;
    }
  }

  private async isSelected(): Promise<boolean> {
    try {
      const helper = await this.helperService.getDefaultHelper(HelperType.RAG);

      return helper.getName() === SQLITE_VECTOR_RAG_HELPER_NAME;
    } catch {
      return false;
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
}
