import { JevHttpError, noulAnswer } from './request.js';
import { collectToolCalls, estimateTokens, fitState, safeCut } from './state.js';
import type {
  CallAnswer,
  CallDecision,
  CompactOptions,
  CompactResult,
  CompactionState,
  JevAsker,
  JevQuestions,
  Message,
  ResolvedCompactOptions,
  ToolCall,
  ToolUse,
} from './types.js';

export const DEFAULT_OPTIONS: ResolvedCompactOptions = {
  goal: '',
  keepThreshold: 0.5,
  preserveRecentMessages: 6,
  maxStateTokens: 25_000,
  maxRequestTokens: 30_000,
  truncateHeadChars: 300,
  minSavingsChars: 500,
  maxRequests: 1,
};

/** How far the state budget shrinks when Jev says a request was too large. */
const TOO_LARGE_SHRINK = 0.8;

/** Tokens the request envelope (`model`, key names) adds around state and questions. */
const REQUEST_OVERHEAD_TOKENS = 20;

function finite(value: number | undefined, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

export function resolveOptions(options: CompactOptions = {}): ResolvedCompactOptions {
  return {
    goal: options.goal ?? DEFAULT_OPTIONS.goal,
    keepThreshold: finite(options.keepThreshold, DEFAULT_OPTIONS.keepThreshold),
    preserveRecentMessages: Math.max(
      0,
      Math.floor(
        finite(options.preserveRecentMessages, DEFAULT_OPTIONS.preserveRecentMessages),
      ),
    ),
    maxStateTokens: Math.max(1, finite(options.maxStateTokens, DEFAULT_OPTIONS.maxStateTokens)),
    maxRequestTokens: Math.max(
      1,
      finite(options.maxRequestTokens, DEFAULT_OPTIONS.maxRequestTokens),
    ),
    truncateHeadChars: Math.max(
      0,
      Math.floor(finite(options.truncateHeadChars, DEFAULT_OPTIONS.truncateHeadChars)),
    ),
    minSavingsChars: Math.max(0, finite(options.minSavingsChars, DEFAULT_OPTIONS.minSavingsChars)),
    maxRequests: Math.max(
      1,
      Math.floor(finite(options.maxRequests, DEFAULT_OPTIONS.maxRequests)),
    ),
  };
}

/** Characters dropping the call would free: its result and its input. */
export function callSavings(call: Pick<ToolCall, 'resultChars' | 'inputChars'>): number {
  return call.resultChars + call.inputChars;
}

/**
 * The calls worth asking Jev about, in request-sized batches. Pinned calls and
 * calls that would free less than `minSavingsChars` are left out; the rest go
 * largest first, so when `maxRequests` cuts the list it cuts the smallest.
 * Each batch lists its calls in transcript order.
 */
export function selectBatches(
  calls: readonly ToolCall[],
  stateTokens: number,
  options: Pick<ResolvedCompactOptions, 'maxRequestTokens' | 'minSavingsChars' | 'maxRequests'>,
): ToolCall[][] {
  const order = new Map(calls.map((call, index) => [call, index]));
  const ranked = calls
    .filter((call) => !call.pinned && callSavings(call) >= options.minSavingsChars)
    .sort((a, b) => callSavings(b) - callSavings(a));
  return batchCalls(ranked, stateTokens, options)
    .slice(0, options.maxRequests)
    .map((batch) => batch.sort((a, b) => order.get(a)! - order.get(b)!));
}

/** The two `noul` questions asked about one call: keep the call, keep its result. */
export function questionsFor(call: ToolCall): JevQuestions {
  return {
    [`call_${call.id}`]: {
      type: 'noul',
      instructions: `Tool call ${call.id} (${call.tool}) should stay in the history: knowing this call was made, with its input, still matters for what the assistant does next`,
    },
    [`result_${call.id}`]: {
      type: 'noul',
      instructions: `The full output of tool call ${call.id} (${call.tool}, ${call.resultChars} chars) should stay in the history verbatim: the assistant still needs its contents and re-running the tool would not do`,
    },
  };
}

/** Jev accepts at most 128 questions per request; each call asks two. */
const MAX_CALLS_PER_REQUEST = 64;

/**
 * Splits the candidate calls into batches whose questions, together with the
 * (always complete) state, fit one request.
 */
export function batchCalls(
  calls: readonly ToolCall[],
  stateTokens: number,
  options: Pick<ResolvedCompactOptions, 'maxRequestTokens'>,
): ToolCall[][] {
  const budget = options.maxRequestTokens - stateTokens - REQUEST_OVERHEAD_TOKENS;
  const batches: ToolCall[][] = [];
  let current: ToolCall[] = [];
  let currentTokens = 0;
  for (const call of calls) {
    const tokens = estimateTokens(JSON.stringify(questionsFor(call)));
    if (
      current.length > 0 &&
      (currentTokens + tokens > budget || current.length >= MAX_CALLS_PER_REQUEST)
    ) {
      batches.push(current);
      current = [];
      currentTokens = 0;
    }
    if (current.length === 0 && tokens > budget) {
      throw new Error(
        `state leaves no room for questions (~${stateTokens} of ${options.maxRequestTokens} tokens)`,
      );
    }
    current.push(call);
    currentTokens += tokens;
  }
  if (current.length > 0) batches.push(current);
  return batches;
}

export function decideCall(
  call: Pick<ToolCall, 'id' | 'tool' | 'pinned'>,
  answer: CallAnswer,
  options: Pick<ResolvedCompactOptions, 'keepThreshold'>,
): CallDecision {
  const base = { id: call.id, tool: call.tool, ...answer };
  if (call.pinned) return { ...base, action: 'keep', reason: 'pinned' };
  if (answer.keepResult >= options.keepThreshold) {
    return { ...base, action: 'keep', reason: 'kept' };
  }
  if (answer.keepCall >= options.keepThreshold) {
    return { ...base, action: 'drop_result', reason: 'result_dropped' };
  }
  return { ...base, action: 'drop_call', reason: 'call_dropped' };
}

async function askBatch(
  asker: JevAsker,
  state: CompactionState,
  batch: readonly ToolCall[],
): Promise<Map<string, CallAnswer>> {
  const questions: JevQuestions = Object.assign({}, ...batch.map(questionsFor));
  const { answers } = await asker.ask(state, questions);
  return new Map(
    batch.map((call) => [
      call.id,
      {
        keepCall: noulAnswer(answers, `call_${call.id}`),
        keepResult: noulAnswer(answers, `result_${call.id}`),
      },
    ]),
  );
}

function truncatedResultText(text: string, isError: boolean, headChars: number): string {
  if (text.length <= headChars + 120) return text;
  const cut = safeCut(text, headChars);
  const head = cut > 0 ? `${text.slice(0, cut)}\n` : '';
  return `${head}[fast-jev-compaction truncated ${text.length - cut} chars of this tool result${
    isError ? ' (error)' : ''
  }; re-run the tool if needed]`;
}

/**
 * Rebuilds the conversation from the decisions. A dropped call disappears
 * together with its result; a dropped result keeps a bounded head and note.
 * Messages that lose all their content are removed; untouched messages are
 * returned as the same objects they came in as.
 */
export function applyDecisions(
  messages: readonly Message[],
  decisions: readonly CallDecision[],
  calls: readonly ToolCall[],
  headChars: number,
): Message[] {
  const byId = new Map(calls.map((call) => [call.id, call]));
  const actions = new Map<string, CallDecision['action']>();
  for (const decision of decisions) {
    const call = byId.get(decision.id);
    if (call && decision.action !== 'keep') actions.set(call.tool_use_id, decision.action);
  }
  const kept: Message[] = [];
  for (const message of messages) {
    const touched =
      message.toolUses.some((tool) => actions.has(tool.tool_use_id)) ||
      (message.toolResults ?? []).some((result) => actions.has(result.tool_use_id));
    if (!touched) {
      kept.push(message);
      continue;
    }
    const toolUses = message.toolUses
      .filter((tool) => actions.get(tool.tool_use_id) !== 'drop_call')
      .map((tool) => {
        if (actions.get(tool.tool_use_id) !== 'drop_result') return tool;
        const text = truncatedResultText(
          tool.text ?? '',
          tool.isError ?? false,
          headChars,
        );
        if ((tool.text ?? '') === text) return tool;
        const copy: ToolUse = {
          tool_use_id: tool.tool_use_id,
          tool: tool.tool,
          input: tool.input,
          text,
        };
        if (tool.isError) copy.isError = true;
        return copy;
      });
    const toolResults = (message.toolResults ?? [])
      .filter((result) => actions.get(result.tool_use_id) !== 'drop_call')
      .map((result) => {
        if (actions.get(result.tool_use_id) !== 'drop_result') return result;
        const text = truncatedResultText(result.text, result.isError ?? false, headChars);
        return text === result.text
          ? result
          : {
              tool_use_id: result.tool_use_id,
              text,
              isError: result.isError,
            };
      });
    if (
      !message.toolUses.some(
        (tool) => actions.get(tool.tool_use_id) === 'drop_call',
      ) &&
      !(message.toolResults ?? []).some(
        (result) => actions.get(result.tool_use_id) === 'drop_call',
      ) &&
      toolUses.every((tool, index) => tool === message.toolUses[index]) &&
      toolResults.every(
        (result, index) => result === message.toolResults?.[index],
      )
    ) {
      kept.push(message);
      continue;
    }
    if (message.text.trim().length === 0 && toolUses.length === 0 && toolResults.length === 0) {
      continue;
    }
    const rebuilt: Message = { role: message.role, text: message.text, toolUses };
    if (toolResults.length > 0) rebuilt.toolResults = toolResults;
    kept.push(rebuilt);
  }
  return kept;
}

/** Characters of text, tool input and tool output a message holds. */
export function messageChars(message: Message): number {
  let total = message.text.length;
  for (const tool of message.toolUses) {
    try {
      total += JSON.stringify(tool.input).length;
    } catch {
      total += 20;
    }
  }
  for (const result of message.toolResults ?? []) total += result.text.length;
  return total;
}

export function reductionRatio(result: Pick<CompactResult, 'stats'>): number {
  const { charsBefore, charsAfter } = result.stats;
  return charsBefore === 0 ? 0 : (charsBefore - charsAfter) / charsBefore;
}

function count(decisions: readonly CallDecision[], reason: CallDecision['reason']): number {
  return decisions.filter((decision) => decision.reason === reason).length;
}

const KEEP: CallAnswer = { keepCall: 1, keepResult: 1 };

type FailedBatch = { batch: ToolCall[]; error: unknown };

/** Asks every batch concurrently; answers land in `answers`, failures come back. */
async function askAll(
  asker: JevAsker,
  state: CompactionState,
  batches: readonly ToolCall[][],
  answers: Map<string, CallAnswer>,
): Promise<FailedBatch[]> {
  const settled = await Promise.allSettled(batches.map((batch) => askBatch(asker, state, batch)));
  const failed: FailedBatch[] = [];
  settled.forEach((outcome, index) => {
    if (outcome.status === 'fulfilled') {
      for (const [id, answer] of outcome.value) answers.set(id, answer);
    } else failed.push({ batch: batches[index]!, error: outcome.reason });
  });
  return failed;
}

function tryFitState(...args: Parameters<typeof fitState>): ReturnType<typeof fitState> | undefined {
  try {
    return fitState(...args);
  } catch {
    return undefined;
  }
}

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/**
 * Compacts a transcript by asking Jev about the tool calls worth removing:
 * outside the pinned first and newest messages, freeing at least
 * `minSavingsChars`, largest first, in at most `maxRequests` requests. For
 * each, Jev says whether the call and whether its result must stay. The whole
 * history (results omitted, fitted into `maxStateTokens`) is sent as state
 * with every request.
 *
 * Calls not asked about, or whose request failed, are kept. A request Jev
 * rejects as too large is retried once against a smaller state. Throws when
 * every request fails or the history cannot be fitted; the caller decides
 * whether to fall back.
 */
export async function compact(
  messages: readonly Message[],
  asker: JevAsker,
  options: CompactOptions = {},
): Promise<CompactResult> {
  const started = Date.now();
  const resolved = resolveOptions(options);
  const calls = collectToolCalls(messages, resolved.preserveRecentMessages);
  const charsBefore = messages.reduce((sum, message) => sum + messageChars(message), 0);

  let fitted: { tokens: number; stage: string } = { tokens: 0, stage: '' };
  let batches: ToolCall[][] = [];
  let requests = 0;
  let failed: FailedBatch[] = [];
  const answers = new Map<string, CallAnswer>();
  const worthAsking = calls.some(
    (call) => !call.pinned && callSavings(call) >= resolved.minSavingsChars,
  );
  if (worthAsking) {
    const state = fitState(messages, calls, resolved);
    fitted = state;
    batches = selectBatches(calls, state.tokens, resolved);
    requests = batches.length;
    failed = await askAll(asker, state.state, batches, answers);

    const tooLarge = failed.filter(
      ({ error }) => error instanceof JevHttpError && error.tooLarge,
    );
    // If the state cannot shrink that far, those requests stay failed.
    const smaller =
      tooLarge.length > 0
        ? tryFitState(messages, calls, {
            ...resolved,
            maxStateTokens: Math.floor(state.tokens * TOO_LARGE_SHRINK),
          })
        : undefined;
    if (smaller) {
      fitted = smaller;
      requests += tooLarge.length;
      const retried = await askAll(
        asker,
        smaller.state,
        tooLarge.map(({ batch }) => batch),
        answers,
      );
      failed = [...failed.filter((entry) => !tooLarge.includes(entry)), ...retried];
    }
    if (failed.length === batches.length) throw failed[0]!.error;
  }

  const asked = new Set(batches.flat().map((call) => call.id));
  const decisions = calls.map((call): CallDecision => {
    const answer = answers.get(call.id);
    if (call.pinned || answer) return decideCall(call, answer ?? KEEP, resolved);
    return {
      id: call.id,
      tool: call.tool,
      ...KEEP,
      action: 'keep',
      reason: asked.has(call.id) ? 'unanswered' : 'skipped',
    };
  });
  const kept = applyDecisions(
    messages,
    decisions,
    calls,
    resolved.truncateHeadChars,
  );
  return {
    messages: kept,
    decisions,
    stats: {
      messagesBefore: messages.length,
      messagesAfter: kept.length,
      charsBefore,
      charsAfter: kept.reduce((sum, message) => sum + messageChars(message), 0),
      calls: calls.length,
      kept: count(decisions, 'kept'),
      resultsDropped: count(decisions, 'result_dropped'),
      callsDropped: count(decisions, 'call_dropped'),
      pinned: count(decisions, 'pinned'),
      skipped: count(decisions, 'skipped'),
      unanswered: count(decisions, 'unanswered'),
      stateTokens: fitted.tokens,
      stateStage: fitted.stage,
      requests,
      failedRequests: failed.length,
      errors: failed.map(({ error }) => errorText(error)),
      ms: Date.now() - started,
    },
  };
}
