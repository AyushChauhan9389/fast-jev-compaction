import { JevHttpError } from './request.js';
import type { JevAsker } from './types.js';

/**
 * A cancellable wait. Injected because the Claude Code hook runtime has no
 * `setTimeout`, only `$.clock.sleep`; Node callers can use `nodeSleep`.
 */
export type Sleep = (ms: number, options: { signal: AbortSignal }) => Promise<void>;

export interface ResilienceOptions {
  /** Longest wait for one request. */
  timeoutMs: number;
  /** Extra attempts after a rate limit (429) or a server error (5xx). */
  retries: number;
  /** Wait before the first retry; doubles on each further one. */
  retryDelayMs: number;
  sleep: Sleep;
}

export const DEFAULT_RESILIENCE = { timeoutMs: 15_000, retries: 1, retryDelayMs: 500 };

export class JevTimeoutError extends Error {
  constructor(ms: number) {
    super(`Jev request timed out after ${ms}ms`);
  }
}

async function withTimeout<T>(work: () => Promise<T>, ms: number, sleep: Sleep): Promise<T> {
  const timer = new AbortController();
  const deadline = new Promise<never>((_, reject) => {
    // An aborted sleep means the work settled first: nothing to report.
    sleep(ms, { signal: timer.signal }).then(
      () => reject(new JevTimeoutError(ms)),
      () => {},
    );
  });
  try {
    return await Promise.race([work(), deadline]);
  } finally {
    timer.abort();
  }
}

/**
 * Wraps an asker with a per-request timeout and retries for rate limits and
 * server errors. A timeout is not retried: a second wait as long as the first
 * costs more than falling back.
 */
export function resilientAsker(asker: JevAsker, options: ResilienceOptions): JevAsker {
  return {
    async ask(state, questions) {
      for (let attempt = 0; ; attempt++) {
        try {
          return await withTimeout(() => asker.ask(state, questions), options.timeoutMs, options.sleep);
        } catch (error) {
          const retryable = error instanceof JevHttpError && error.retryable;
          if (!retryable || attempt >= options.retries) throw error;
          await options.sleep(options.retryDelayMs * 2 ** attempt, {
            signal: new AbortController().signal,
          });
        }
      }
    },
  };
}
