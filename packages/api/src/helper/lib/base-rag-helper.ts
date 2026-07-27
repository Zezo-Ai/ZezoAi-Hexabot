/*
 * Hexabot — Fair Core License (FCL-1.0-ALv2)
 * Copyright (c) 2026 Hexastack.
 * Full terms: see LICENSE.md.
 */

import type { ContentFull } from '@hexabot-ai/types';

import type { RagHit, RagQueryOptions } from '@/cms/types/rag';

import { HelperName, HelperType } from '../types';

import BaseHelper from './base-helper';

/** One window of content text, ready to be embedded and stored. */
export interface RagChunk {
  index: number;
  text: string;
}

/**
 * Base class for Retrieval-Augmented Generation (RAG) helpers.
 *
 * The read contract (`retrieve`) is required — it is the only thing a
 * retrieval-only helper (e.g. the built-in DB-native full-text helper) needs
 * to implement, since its search corpus is the live database.
 *
 * The write hooks (`index`, `remove`, `reindex`) are optional. They are useful
 * for helpers that maintain an external index. CMS lifecycle forwarding is
 * deliberately best-effort: helpers that require durable consistency should
 * implement database change capture, an outbox, or an equivalent reconciliation
 * mechanism and use these hooks only to reduce indexing latency.
 */
export abstract class BaseRagHelper<
  N extends HelperName = HelperName,
> extends BaseHelper<N> {
  protected readonly type: HelperType = HelperType.RAG;

  constructor(name: N) {
    super(name);
  }

  /**
   * Retrieves the most relevant content for a query.
   *
   * @param query - The natural language query.
   * @param options - Optional retrieval filters (limit, content type, inactive).
   * @returns A ranked list of RAG hits.
   */
  abstract retrieve(
    query: string,
    options?: RagQueryOptions,
  ): Promise<RagHit[]>;

  /**
   * Optional best-effort notification to index a content item.
   */
  index?(content: ContentFull): Promise<void>;

  /**
   * Optional best-effort notification to remove a content item.
   */
  remove?(contentId: string): Promise<void>;

  /**
   * Optional: reconcile or rebuild the helper from the source database.
   */
  reindex?(): Promise<void>;

  /**
   * Splits canonical content text into deterministic, overlapping character
   * windows. Newline boundaries are preferred in the latter half of a window;
   * oversized lines are hard-split at the configured size.
   *
   * Lives here so every embedding-backed helper chunks identically: the window
   * boundaries are baked into each stored vector, so two helpers that disagreed
   * about them would produce indexes that cannot be compared or migrated
   * between. Determinism matters for the same reason — re-chunking unchanged
   * text must yield byte-identical windows, or every reconciliation pass would
   * consider the whole corpus stale.
   *
   * @throws RangeError when the chunk size or overlap is not usable.
   */
  protected chunkSearchText(
    source: string,
    chunkSize: number,
    overlap: number,
  ): RagChunk[] {
    if (!Number.isInteger(chunkSize) || chunkSize <= 0) {
      throw new RangeError('Chunk size must be a positive integer.');
    }
    if (!Number.isInteger(overlap) || overlap < 0 || overlap >= chunkSize) {
      throw new RangeError(
        'Chunk overlap must be a non-negative integer smaller than chunk size.',
      );
    }

    const text = source.replace(/\r\n?/g, '\n');
    const chunks: RagChunk[] = [];
    let start = 0;

    while (start < text.length) {
      const hardEnd = Math.min(start + chunkSize, text.length);
      let end = hardEnd;

      if (hardEnd < text.length) {
        const minimumBoundary = start + Math.floor(chunkSize / 2);
        const paragraphBoundary = text.lastIndexOf('\n\n', hardEnd - 1);
        const lineBoundary = text.lastIndexOf('\n', hardEnd - 1);
        if (paragraphBoundary >= minimumBoundary) {
          end = paragraphBoundary + 2;
        } else if (lineBoundary >= minimumBoundary) {
          end = lineBoundary + 1;
        }
      }

      const value = text.slice(start, end).trim();
      if (value) {
        chunks.push({
          index: chunks.length,
          text: value,
        });
      }

      if (end >= text.length) {
        break;
      }

      const nextStart = Math.max(0, end - overlap);
      start = nextStart > start ? nextStart : end;
    }

    return chunks;
  }
}

export default BaseRagHelper;
