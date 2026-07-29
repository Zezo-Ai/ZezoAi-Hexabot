/*
 * Hexabot — Fair Core License (FCL-1.0-ALv2)
 * Copyright (c) 2026 Hexastack.
 * Full terms: see LICENSE.md.
 */

import { randomUUID } from 'node:crypto';

import type { ContentFull, Setting } from '@hexabot-ai/types';
import {
  Injectable,
  OnApplicationBootstrap,
  OnApplicationShutdown,
} from '@nestjs/common';
import { OnEvent } from '@nestjs/event-emitter';
import { DataSource } from 'typeorm';

import {
  RagHelperConfigurationError,
  RagHelperUnavailableError,
} from '@/cms/errors/rag.errors';
import { DEFAULT_RAG_TOP_K, RagHit, RagQueryOptions } from '@/cms/types/rag';
import {
  BaseRagEmbeddingHelper,
  RagEmbeddingSettings,
} from '@/helper/lib/base-rag-embedding-helper';
import { HelperType } from '@/helper/types';
import { CredentialService } from '@/user/services/credential.service';

import {
  PGVECTOR_RAG_HELPER_NAME,
  pgvectorSettingsSchema,
} from './pgvector.settings';
import {
  PgvectorEmbeddedChunk,
  PgvectorJob,
  PgvectorStore,
} from './pgvector.store';

const WORKER_INTERVAL_MS = 2000;
const RECONCILIATION_INTERVAL_MS = 60000;
const WORKER_CONCURRENCY = 2;

type PgvectorSettings = RagEmbeddingSettings & {
  index_only_active_content: boolean;
};

@Injectable()
export default class PgvectorRagHelper
  extends BaseRagEmbeddingHelper<typeof PGVECTOR_RAG_HELPER_NAME>
  implements OnApplicationBootstrap, OnApplicationShutdown
{
  private readonly store: PgvectorStore;

  private readonly workerId = randomUUID();

  private workerTimer?: NodeJS.Timeout;

  private processing = false;

  private wakeScheduled = false;

  private lastReconciliationAt = 0;

  private infrastructureWarningLogged = false;

  constructor(
    private readonly dataSource: DataSource,
    private readonly credentialService: CredentialService,
  ) {
    super(PGVECTOR_RAG_HELPER_NAME);
    this.store = new PgvectorStore(dataSource);
  }

  public override isAvailable(): boolean {
    return this.dataSource.options.type === 'postgres';
  }

  async onApplicationBootstrap(): Promise<void> {
    if (!this.isAvailable()) {
      return;
    }

    this.workerTimer = setInterval(() => this.wakeWorker(), WORKER_INTERVAL_MS);
    this.workerTimer.unref();
    this.wakeWorker();
  }

  onApplicationShutdown(): void {
    if (this.workerTimer) {
      clearInterval(this.workerTimer);
      this.workerTimer = undefined;
    }
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
      // retrieval: degrade to no semantic hits so the conversation continues,
      // mirroring the durable indexing path's "warn and move on" behavior.
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
      source: PGVECTOR_RAG_HELPER_NAME,
    }));
  }

  /**
   * Content triggers already enqueue durable work in the same transaction.
   * This lifecycle hook only reduces the worker's wake-up latency.
   */
  async index(_content: ContentFull): Promise<void> {
    this.wakeWorker();
  }

  /**
   * Enqueues a bounded job per live content row. Existing embeddings remain
   * usable until their replacement profile succeeds.
   */
  async reindex(): Promise<void> {
    await this.store.enqueueAll();
    this.wakeWorker();
  }

  @OnEvent('hook:pgvector:*')
  async handleSettingsChanged(setting?: Pick<Setting, 'label'>): Promise<void> {
    if (!this.isAvailable() || !setting?.label) {
      return;
    }

    try {
      if (setting.label === 'embedding_api_key') {
        await this.store.wakePendingRetries();
      } else if (
        [
          'embedding_provider',
          'embedding_model',
          'embedding_base_url',
          'embedding_dimensions',
          'chunk_size',
          'chunk_overlap',
          // Re-evaluate every row: the worker embeds active content and drops
          // inactive content, so toggling in either direction converges the
          // index (purging inactive rows when enabled, backfilling them when
          // disabled).
          'index_only_active_content',
        ].includes(setting.label)
      ) {
        // The model/provider/dimension request may have changed; re-evaluate
        // the "requested dimension not honored" warning on the next embed.
        this.dimensionMismatchWarned = false;
        await this.store.enqueueAll();
      }
      this.wakeWorker();
    } catch (error) {
      this.logger.error(
        'Unable to schedule pgvector RAG work after a settings change.',
        error,
      );
    }
  }

  private wakeWorker(): void {
    if (this.wakeScheduled || this.processing || !this.isAvailable()) {
      return;
    }

    this.wakeScheduled = true;
    queueMicrotask(() => {
      this.wakeScheduled = false;
      void this.processJobs();
    });
  }

  private async processJobs(): Promise<void> {
    if (this.processing || !(await this.isSelected())) {
      return;
    }

    this.processing = true;
    let claimedJobs = 0;
    try {
      const settings = await this.getConfiguration();
      const profile = this.getProfile(settings);
      if (
        Date.now() - this.lastReconciliationAt >=
        RECONCILIATION_INTERVAL_MS
      ) {
        await this.store.enqueueMissing(
          profile,
          settings.index_only_active_content,
        );
        this.lastReconciliationAt = Date.now();
      }

      const jobs = await this.store.claimJobs(
        this.workerId,
        WORKER_CONCURRENCY,
      );
      claimedJobs = jobs.length;
      await Promise.all(
        jobs.map((job) => this.processJob(job, settings, profile)),
      );
      this.infrastructureWarningLogged = false;
    } catch (error) {
      if (error instanceof RagHelperConfigurationError) {
        return;
      }

      if (
        error instanceof RagHelperUnavailableError &&
        this.infrastructureWarningLogged
      ) {
        return;
      }
      this.infrastructureWarningLogged =
        error instanceof RagHelperUnavailableError;
      this.logger.error('Unable to process the pgvector RAG queue.', error);
    } finally {
      this.processing = false;
      if (claimedJobs === WORKER_CONCURRENCY) {
        this.wakeWorker();
      }
    }
  }

  private async processJob(
    job: PgvectorJob,
    settings: PgvectorSettings,
    profile: string,
  ): Promise<void> {
    try {
      const content = await this.store.loadContent(job.contentId);
      if (!content) {
        return;
      }

      // Never embed (i.e. transmit to the external provider) inactive content
      // when the operator opted to index only active content. Drop any existing
      // embeddings and clear the job instead.
      if (settings.index_only_active_content && !content.status) {
        await this.store.discardInactive(job, this.workerId);

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
      const embeddedChunks: PgvectorEmbeddedChunk[] = chunks.map(
        (chunk, index) => ({
          ...chunk,
          embedding: embeddings[index],
        }),
      );
      await this.store.save(
        job,
        this.workerId,
        profile,
        content.searchText,
        embeddedChunks,
      );
    } catch (error) {
      await this.store.fail(job, this.workerId, error);
      this.logger.warn(
        `Unable to embed content "${job.contentId}"; the durable RAG job will be retried.`,
        error,
      );
    }
  }

  private async isSelected(): Promise<boolean> {
    try {
      const helper = await this.helperService.getDefaultHelper(HelperType.RAG);

      return helper.getName() === PGVECTOR_RAG_HELPER_NAME;
    } catch {
      return false;
    }
  }

  private async getConfiguration(): Promise<PgvectorSettings> {
    const result = pgvectorSettingsSchema.safeParse(
      await this.getSettings<typeof PGVECTOR_RAG_HELPER_NAME>(),
    );
    if (!result.success) {
      throw new RagHelperConfigurationError(
        'The pgvector RAG helper settings are missing or invalid.',
      );
    }

    const settings = result.data;
    const credentialId = settings.embedding_api_key.trim();
    if (!credentialId) {
      throw new RagHelperConfigurationError(
        'The pgvector RAG helper requires an embedding credential.',
      );
    }
    const apiKey = (
      await this.credentialService.findOneValue(credentialId)
    ).trim();
    if (!apiKey) {
      throw new RagHelperConfigurationError(
        'The selected pgvector embedding credential is missing or empty.',
      );
    }

    return {
      ...settings,
      embedding_provider: settings.embedding_provider.trim(),
      embedding_api_key: apiKey,
      embedding_model: settings.embedding_model.trim(),
      embedding_base_url: settings.embedding_base_url.replace(/\/+$/, ''),
    };
  }
}
