import { JevClient, type JevClientOptions } from './client.js';
import { compact } from './compact.js';
import { DEFAULT_RESILIENCE, resilientAsker, type ResilienceOptions, type Sleep } from './resilient.js';
import type { CompactOptions, CompactResult, Message } from './types.js';

export type CompactMessagesOptions = CompactOptions &
  JevClientOptions &
  Partial<Omit<ResilienceOptions, 'sleep'>>;

/** `setTimeout`-based `Sleep` for Node. */
export const nodeSleep: Sleep = (ms, { signal }) =>
  new Promise((resolve, reject) => {
    const timer = setTimeout(resolve, ms);
    signal.addEventListener(
      'abort',
      () => {
        clearTimeout(timer);
        reject(new Error('aborted'));
      },
      { once: true },
    );
  });

/**
 * `compact` with a `JevClient` built from the options (key from
 * `TYPESAFE_API_KEY` by default), with a per-request timeout and retries.
 */
export function compactMessages(
  messages: readonly Message[],
  options: CompactMessagesOptions = {},
): Promise<CompactResult> {
  const asker = resilientAsker(new JevClient(options), {
    timeoutMs: options.timeoutMs ?? DEFAULT_RESILIENCE.timeoutMs,
    retries: options.retries ?? DEFAULT_RESILIENCE.retries,
    retryDelayMs: options.retryDelayMs ?? DEFAULT_RESILIENCE.retryDelayMs,
    sleep: nodeSleep,
  });
  return compact(messages, asker, options);
}
