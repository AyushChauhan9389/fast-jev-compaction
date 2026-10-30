import type {
  On,
  PluginOptions,
  Register,
  SessionCompactTrigger,
  SessionMessage,
  ToolResultSummary,
  ToolUseSummary,
  TurnCompleteInput,
} from 'claude-code';

import { compact, reductionRatio, resolveOptions } from '../src/compact.js';
import { buildJevRequest, DEFAULT_MODEL, parseJevResponse } from '../src/request.js';
import { DEFAULT_RESILIENCE, resilientAsker, type Sleep } from '../src/resilient.js';
import type {
  CompactOptions,
  CompactResult,
  JevAsker,
  Message,
  ToolResult,
  ToolUse,
} from '../src/types.js';

const HOOK_DEFAULTS = {
  compactAtPercent: 60,
  minReductionRatio: 0.25,
  model: DEFAULT_MODEL,
  requestTimeoutMs: DEFAULT_RESILIENCE.timeoutMs,
};

/**
 * Percentage points the context must grow after a skipped `turn.complete`
 * compaction before the next try, so a conversation with little left to
 * remove is not re-scored after every turn.
 */
const RETRY_AFTER_SKIP_PERCENT = 10;

export type HookFetchInit = {
  method?: string;
  headers?: Record<string, string>;
  body?: string;
};

export type HookFetchResponse = {
  status: number;
  ok: boolean;
  text: string;
};

/** The shape of `$.http.fetch`, so the hook can be driven without an engine. */
export type HookFetch = (url: string, init?: HookFetchInit) => Promise<HookFetchResponse>;

export type HookConfig = CompactOptions & {
  apiKey?: string;
  compactAtPercent: number;
  minReductionRatio: number;
  model: string;
  requestTimeoutMs: number;
  baseUrl?: string;
};

function optionNumber(options: PluginOptions, key: string, fallback: number): number {
  const value = options[key];
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function optionString(options: PluginOptions, key: string): string | undefined {
  const value = options[key];
  return typeof value === 'string' && value.length > 0 ? value : undefined;
}

/** Reads the plugin's `userConfig` values; anything missing takes the defaults. */
export function resolveHookConfig(options: PluginOptions): HookConfig {
  const numbers: Partial<Omit<CompactOptions, 'goal'>> = {};
  for (const key of [
    'keepThreshold',
    'preserveRecentMessages',
    'maxStateTokens',
    'maxRequestTokens',
    'truncateHeadChars',
    'minSavingsChars',
    'maxRequests',
  ] as const) {
    const value = options[key];
    if (typeof value === 'number' && Number.isFinite(value)) numbers[key] = value;
  }
  const config: HookConfig = {
    ...numbers,
    compactAtPercent: optionNumber(options, 'compactAtPercent', HOOK_DEFAULTS.compactAtPercent),
    minReductionRatio: optionNumber(
      options,
      'minReductionRatio',
      HOOK_DEFAULTS.minReductionRatio,
    ),
    model: optionString(options, 'model') ?? HOOK_DEFAULTS.model,
    requestTimeoutMs: optionNumber(options, 'requestTimeoutMs', HOOK_DEFAULTS.requestTimeoutMs),
  };
  const apiKey = optionString(options, 'apiKey');
  if (apiKey) config.apiKey = apiKey;
  const goal = optionString(options, 'goal');
  if (goal) config.goal = goal;
  const baseUrl = optionString(options, 'baseUrl');
  if (baseUrl) config.baseUrl = baseUrl;
  return config;
}

/** A `JevAsker` over the engine's `$.http.fetch`. */
export function jevAsker(
  fetchFn: HookFetch,
  apiKey: string,
  model: string,
  baseUrl?: string,
): JevAsker {
  return {
    async ask(state, questions) {
      const request = buildJevRequest({ apiKey, model, baseUrl }, state, questions);
      const response = await fetchFn(request.url, {
        method: request.method,
        headers: request.headers,
        body: request.body,
      });
      return parseJevResponse(response.status, response.ok, response.text);
    },
  };
}

function toolUseSummary(tool: ToolUse): ToolUseSummary {
  const summary: ToolUseSummary = {
    tool_use_id: tool.tool_use_id,
    tool: tool.tool,
    input: tool.input,
  };
  if (tool.text !== undefined) summary.text = tool.text;
  if (tool.isError) summary.isError = true;
  return summary;
}

function toolResultSummary(result: ToolResult): ToolResultSummary {
  return {
    tool_use_id: result.tool_use_id,
    text: result.text,
    isError: result.isError ?? false,
  };
}

/**
 * Maps the library's output back onto session messages. Unchanged tool uses
 * and tool results are the engine's own objects. Every message goes back
 * without its `handle`: with one, the engine keeps the original record, whose
 * parentUuid chain runs back behind the compact boundary, so `--resume`
 * reloads the pre-compaction history (anthropics/claude-code#95328).
 * Handle-less messages are written fresh after the boundary, at the cost of
 * the engine's hidden parts of them (reasoning, images); text and tool pairs
 * stay.
 */
export function toSessionMessages(
  input: readonly SessionMessage[],
  output: readonly Message[],
): SessionMessage[] {
  const uses = new Map<ToolUse, ToolUseSummary>();
  const results = new Map<ToolResult, ToolResultSummary>();
  for (const message of input) {
    for (const tool of message.toolUses) uses.set(tool, tool);
    for (const result of message.toolResults ?? []) results.set(result, result);
  }
  return output.map((message) => {
    const rebuilt: SessionMessage = {
      role: message.role,
      text: message.text,
      toolUses: message.toolUses.map((tool) => uses.get(tool) ?? toolUseSummary(tool)),
    };
    if (message.toolResults && message.toolResults.length > 0) {
      rebuilt.toolResults = message.toolResults.map(
        (result) => results.get(result) ?? toolResultSummary(result),
      );
    }
    return rebuilt;
  });
}

export type SessionCompaction = {
  result: CompactResult;
  messages: SessionMessage[];
};

/**
 * Runs the library over a session transcript; throws when the key is missing
 * or every Jev request fails. With `sleep`, each request gets
 * `requestTimeoutMs` and a retry on rate limits and server errors.
 */
export async function compactSession(
  messages: readonly SessionMessage[],
  config: HookConfig,
  fetchFn: HookFetch,
  sleep?: Sleep,
): Promise<SessionCompaction> {
  if (!config.apiKey) throw new Error('TYPESAFE_API_KEY is not configured');
  const asker = jevAsker(fetchFn, config.apiKey, config.model, config.baseUrl);
  const result = await compact(
    messages,
    sleep
      ? resilientAsker(asker, { ...DEFAULT_RESILIENCE, timeoutMs: config.requestTimeoutMs, sleep })
      : asker,
    config,
  );
  return { result, messages: toSessionMessages(messages, result.messages) };
}

function percent(ratio: number): string {
  return `${Math.round(ratio * 100)}%`;
}

export function summarize(result: CompactResult): string {
  const { stats } = result;
  const parts = [
    stats.kept > 0 ? `${stats.kept} kept` : '',
    stats.resultsDropped > 0 ? `${stats.resultsDropped} results truncated` : '',
    stats.callsDropped > 0 ? `${stats.callsDropped} call_dropped` : '',
    stats.pinned > 0 ? `${stats.pinned} pinned` : '',
    stats.skipped > 0 ? `${stats.skipped} not asked` : '',
    stats.unanswered > 0 ? `${stats.unanswered} unanswered` : '',
  ].filter(Boolean);
  return `${percent(reductionRatio(result))} reduction; ${
    parts.join(', ') || 'no tool calls'
  }; state ~${stats.stateTokens} tokens (${stats.stateStage}) in ${stats.requests} request(s)${
    stats.failedRequests > 0 ? `, ${stats.failedRequests} failed (${stats.errors[0]})` : ''
  }`;
}

const UI_LOG_MAX_CHARS = 4096;

export function decisionLog(result: CompactResult): string {
  return result.decisions
    .filter((d) => d.reason !== 'pinned' && d.reason !== 'skipped')
    .map((d) =>
      d.reason === 'unanswered'
        ? `${d.id}:${d.tool}:unanswered`
        : `${d.id}:${d.tool}:${d.action}/call=${d.keepCall.toFixed(2)}/result=${d.keepResult.toFixed(2)}`,
    )
    .join(' ');
}

export function decisionLogLines(
  result: CompactResult,
  maxChars: number = UI_LOG_MAX_CHARS,
): string[] {
  const entries = decisionLog(result).split(' ').filter(Boolean);
  if (entries.length === 0) return ['decisions: (none)'];
  const chunks: string[] = [];
  let current = '';
  for (const entry of entries) {
    const next = current ? `${current} ${entry}` : entry;
    if (current && next.length > maxChars - 24) {
      chunks.push(current);
      current = entry;
    } else current = next;
  }
  chunks.push(current);
  return chunks.map((chunk, index) =>
    chunks.length === 1
      ? `decisions: ${chunk}`
      : `decisions (${index + 1}/${chunks.length}): ${chunk}`,
  );
}

async function getApiKey(
  $: {
    env: { get: (name: string) => Promise<string | undefined> };
    settings: { read: () => Promise<Readonly<Record<string, unknown>>> };
  },
  config: HookConfig,
): Promise<string | undefined> {
  if (config.apiKey) return config.apiKey;
  const fromEnv = await $.env.get('TYPESAFE_API_KEY');
  if (fromEnv) return fromEnv;
  const settings = await $.settings.read();
  const env = settings['env'];
  if (env && typeof env === 'object') {
    const value = (env as Record<string, unknown>)['TYPESAFE_API_KEY'];
    if (typeof value === 'string' && value) return value;
  }
  return undefined;
}

function notify(
  $: {
    ui: {
      log: (text: string) => void;
      toast: (text: string, options?: { timeoutMs?: number }) => void;
    };
  },
  text: string,
): void {
  $.ui.log(text);
  $.ui.toast(text, { timeoutMs: 15_000 });
}

/**
 * Whether a compaction Jev could not do goes on to Claude Code's built-in
 * summary. `auto` (the context is full) and `manual` (`/compact`) must
 * compact. `precompute` runs ahead of time and `plugin` is this plugin's own
 * early request: for those, a slow built-in summary is not worth it, and the
 * conversation stays as it is.
 */
export function fallsBackToBuiltin(trigger: SessionCompactTrigger): boolean {
  return trigger === 'auto' || trigger === 'manual';
}

export const register: Register = (on: On, options: PluginOptions) => {
  const configured = resolveHookConfig(options);
  let compacting = false;
  let retryAtPercent = 0;

  on('session.compact', async ($, event, next) => {
    // A subagent's or fork's own transcript: leave it to the built-in path.
    if (event.agentId) return next(event);
    const giveUp = (why: string) => {
      if (fallsBackToBuiltin(event.trigger)) {
        notify($, `fallback to built-in summary (${why})`);
        return next(event);
      }
      $.ui.log(`not compacted on ${event.trigger} (${why})`);
      return { skip: `fast-jev-compaction: ${why.slice(0, 200)}` };
    };
    try {
      const config = { ...configured, apiKey: await getApiKey($, configured) };
      const { result, messages } = await compactSession(
        event.messages,
        config,
        async (url, init) => {
          const response = await $.http.fetch(url, init);
          return { status: response.status, ok: response.ok, text: response.text };
        },
        (ms, sleepOptions) => $.clock.sleep(ms, sleepOptions),
      );
      for (const line of decisionLogLines(result)) $.ui.log(line);
      if (reductionRatio(result) < config.minReductionRatio) {
        return giveUp(`below ${percent(config.minReductionRatio)} minimum: ${summarize(result)}`);
      }
      const text = `kept ${messages.length}/${event.messages.length} messages, no summary (${summarize(result)})`;
      // A precompute is installed later, if at all: log it, do not toast it.
      if (event.trigger === 'precompute') $.ui.log(`precomputed: ${text}`);
      else notify($, text);
      return { messages };
    } catch (error) {
      return giveUp(error instanceof Error ? error.message : String(error));
    }
  });

  on('turn.complete', async ($, event: TurnCompleteInput, next) => {
    if (event.agentId || event.reason !== 'answer' || compacting) return next(event);
    compacting = true;
    try {
      const used = (await $.session.usage()).context.percent ?? 0;
      if (used < configured.compactAtPercent) retryAtPercent = 0;
      if (used >= Math.max(configured.compactAtPercent, retryAtPercent)) {
        const outcome = await $.session.compact();
        retryAtPercent = outcome.skip === undefined ? 0 : used + RETRY_AFTER_SKIP_PERCENT;
      }
    } catch (error) {
      $.ui.log(
        `auto-compact skipped (${error instanceof Error ? error.message : String(error)})`,
      );
    } finally {
      compacting = false;
    }
    return next(event);
  });
};

export { resolveOptions };
