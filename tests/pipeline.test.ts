import { describe, expect, it } from 'vitest';
import {
  applyDecisions,
  buildJevRequest,
  collectToolCalls,
  compact,
  decideCall,
  estimateTokens,
  fitState,
  JevHttpError,
  JevTimeoutError,
  questionsFor,
  resilientAsker,
  resolveOptions,
  selectBatches,
  truncate,
  type JevAsker,
  type JevQuestions,
  type Message,
  type Sleep,
} from '../src/index.js';

function message(role: Message['role'], text: string, extra: Partial<Message> = {}): Message {
  return { role, text, toolUses: [], ...extra };
}

function call(id: string, tool: string, input: Record<string, unknown>, text: string): Message {
  return message('assistant', '', { toolUses: [{ tool_use_id: id, tool, input, text }] });
}

function result(id: string, text: string, isError = false): Message {
  return message('user', '', { toolResults: [{ tool_use_id: id, text, isError }] });
}

type Seen = { state: unknown; questions: string[] };

function fakeJev(answer: (name: string) => number, seen: Seen[] = []): JevAsker {
  return {
    async ask(state, questions: JevQuestions) {
      seen.push({ state, questions: Object.keys(questions) });
      return {
        answers: Object.fromEntries(
          Object.keys(questions).map((key) => [key, { type: 'noul' as const, noul: answer(key) }]),
        ),
      };
    },
  };
}

/** A Jev that throws `fail`'s error for the requests it picks, answers 0 otherwise. */
function failing(fail: (questions: string[]) => Error | undefined, seen: Seen[] = []): JevAsker {
  const inner = fakeJev(() => 0, seen);
  return {
    async ask(state, questions) {
      const error = fail(Object.keys(questions));
      if (!error) return inner.ask(state, questions);
      seen.push({ state, questions: Object.keys(questions) });
      throw error;
    },
  };
}

const big = 'x'.repeat(5000);
const mid = 'y'.repeat(2000);

/** t1: 2000 chars, t2: 2 chars, t3: 5000 chars; the last message is pinned. */
function sized(extra: Message[] = []): Message[] {
  return [
    message('user', 'Fix it.'),
    ...extra,
    call('a', 'Read', { file_path: 'a' }, mid),
    result('a', mid),
    call('b', 'Bash', { command: 'ls' }, 'ok'),
    result('b', 'ok'),
    call('c', 'Read', { file_path: 'c' }, big),
    result('c', big),
    message('user', 'go on'),
  ];
}

describe('selection', () => {
  it('leaves out calls below minSavingsChars and pinned calls', () => {
    const calls = collectToolCalls(sized(), 1);
    const batches = selectBatches(calls, 1000, resolveOptions({}));
    expect(batches.flat().map((c) => c.tool_use_id)).toEqual(['a', 'c']);
  });

  it('cuts the smallest calls when maxRequests is reached', () => {
    const calls = collectToolCalls(sized(), 1);
    const largest = calls.find((c) => c.tool_use_id === 'c')!;
    const roomForOne = 20 + estimateTokens(JSON.stringify(questionsFor(largest))) + 1;
    const batches = selectBatches(calls, 0, {
      maxRequestTokens: roomForOne,
      minSavingsChars: 500,
      maxRequests: 1,
    });
    expect(batches.map((b) => b.map((c) => c.tool_use_id))).toEqual([['c']]);
  });

  it('lists a batch in transcript order although it was filled largest first', () => {
    const calls = collectToolCalls(sized(), 1);
    const [batch] = selectBatches(calls, 0, resolveOptions({}));
    expect(batch!.map((c) => c.id)).toEqual(['t1', 't3']);
  });

  it('keeps unasked calls as skipped and never asks Jev about them', async () => {
    const seen: Seen[] = [];
    const output = await compact(sized(), fakeJev(() => 0, seen), { preserveRecentMessages: 1 });
    expect(seen).toHaveLength(1);
    expect(seen[0]!.questions.sort()).toEqual(['call_t1', 'call_t3', 'result_t1', 'result_t3']);
    expect(output.decisions.map((d) => d.reason)).toEqual(['call_dropped', 'skipped', 'call_dropped']);
    expect(output.stats).toMatchObject({ skipped: 1, callsDropped: 2, requests: 1, failedRequests: 0 });
  });

  it('makes no request when nothing is worth asking about', async () => {
    const seen: Seen[] = [];
    const output = await compact(sized(), fakeJev(() => 0, seen), {
      preserveRecentMessages: 1,
      minSavingsChars: 1e6,
    });
    expect(seen).toHaveLength(0);
    expect(output.stats).toMatchObject({ requests: 0, skipped: 3, stateStage: '' });
    expect(output.messages).toEqual(sized());
  });
});

describe('partial failures', () => {
  it('keeps the calls of a failed request and uses the answers of the others', async () => {
    const calls = collectToolCalls(sized(), 1);
    const stateTokens = fitState(sized(), calls, { maxStateTokens: 25_000, preserveRecentMessages: 1, goal: '' }).tokens;
    const oneCall = estimateTokens(JSON.stringify(questionsFor(calls[2]!)));
    const output = await compact(
      sized(),
      failing((q) => (q.includes('call_t1') ? new JevHttpError(503, 'busy') : undefined)),
      // Room for one call's questions per request: t1 and t3 go separately.
      { preserveRecentMessages: 1, maxRequests: 10, maxRequestTokens: stateTokens + 20 + oneCall + 1 },
    );
    expect(output.stats.requests).toBe(2);
    expect(output.decisions.map((d) => d.reason)).toEqual(['unanswered', 'skipped', 'call_dropped']);
    expect(output.stats).toMatchObject({ failedRequests: 1, unanswered: 1, callsDropped: 1 });
    expect(output.stats.errors[0]).toMatch(/503/);
  });

  it('throws when every request fails, so the caller can fall back', async () => {
    await expect(
      compact(sized(), failing(() => new JevHttpError(500, 'down')), { preserveRecentMessages: 1 }),
    ).rejects.toThrow(/500/);
  });

  it('retries a request Jev rejects as too large against a smaller state', async () => {
    const seen: Seen[] = [];
    let first = true;
    const asker = failing(() => {
      if (!first) return undefined;
      first = false;
      return new JevHttpError(400, '{"detail":{"error_type":"max_tokens_exceeded"}}');
    }, seen);
    // A long old message gives the state room to shrink.
    const output = await compact(sized([message('assistant', 'z '.repeat(3000))]), asker, {
      preserveRecentMessages: 1,
    });
    expect(seen).toHaveLength(2);
    expect(JSON.stringify(seen[1]!.state).length).toBeLessThan(JSON.stringify(seen[0]!.state).length);
    expect(output.stats).toMatchObject({ requests: 2, failedRequests: 0, callsDropped: 2 });
  });
});

describe('resilientAsker', () => {
  /** Resolves on the next microtask unless already aborted. */
  const instant: Sleep = (_ms, { signal }) =>
    new Promise((resolve, reject) => {
      if (signal.aborted) reject(new Error('aborted'));
      else queueMicrotask(resolve);
    });
  /** Never resolves: a timeout that does not fire. */
  const never: Sleep = () => new Promise(() => {});
  const answer = { answers: { q: { type: 'noul' as const, noul: 1 } } };

  it('retries a rate limit, then succeeds', async () => {
    let calls = 0;
    const asker = resilientAsker(
      {
        async ask() {
          calls += 1;
          if (calls === 1) throw new JevHttpError(429, 'slow down');
          return answer;
        },
      },
      // The 1000ms timeout never fires; the 1ms retry delay passes at once.
      { timeoutMs: 1000, retries: 1, retryDelayMs: 1, sleep: (ms, o) => (ms === 1000 ? never(ms, o) : instant(ms, o)) },
    );
    await expect(asker.ask('s', {})).resolves.toEqual(answer);
    expect(calls).toBe(2);
  });

  it('gives up after the last retry', async () => {
    let calls = 0;
    const asker = resilientAsker(
      {
        async ask() {
          calls += 1;
          throw new JevHttpError(503, 'down');
        },
      },
      { timeoutMs: 1000, retries: 2, retryDelayMs: 1, sleep: (ms, o) => (ms === 1000 ? never(ms, o) : instant(ms, o)) },
    );
    await expect(asker.ask('s', {})).rejects.toThrow(/503/);
    expect(calls).toBe(3);
  });

  it('does not retry a final error', async () => {
    let calls = 0;
    const asker = resilientAsker(
      {
        async ask() {
          calls += 1;
          throw new JevHttpError(400, 'bad');
        },
      },
      { timeoutMs: 1000, retries: 3, retryDelayMs: 1, sleep: never },
    );
    await expect(asker.ask('s', {})).rejects.toThrow(/400/);
    expect(calls).toBe(1);
  });

  it('fails a request that outlives the timeout, without retrying it', async () => {
    let calls = 0;
    const asker = resilientAsker(
      {
        ask: () => {
          calls += 1;
          return new Promise(() => {});
        },
      },
      { timeoutMs: 5, retries: 1, retryDelayMs: 1, sleep: instant },
    );
    await expect(asker.ask('s', {})).rejects.toBeInstanceOf(JevTimeoutError);
    expect(calls).toBe(1);
  });
});

describe('unicode safety', () => {
  const lone = /[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/;

  it('never cuts an emoji in half', () => {
    const text = 'ab\u{1F525}cd'.repeat(10);
    expect(truncate(text, 4)).toBe('ab…');
    for (let limit = 1; limit < 30; limit++) expect(lone.test(truncate(text, limit))).toBe(false);
  });

  it('replaces lone surrogates in the request body and leaves pairs alone', () => {
    const request = buildJevRequest({ apiKey: 'k' }, { text: 'x\uD83D y \u{1F525}' }, {});
    expect(JSON.parse(request.body).state.text).toBe('x� y \u{1F525}');
  });

  it('truncates a dropped result on a code point boundary', () => {
    // 'a' then pairs: unit 300 falls between the halves of the 150th emoji.
    const text = `a${'\u{1F525}'.repeat(400)}`;
    const messages = [message('user', 'go'), call('e', 'Read', {}, text), result('e', text), message('user', 'next')];
    const calls = collectToolCalls(messages, 1);
    const decisions = calls.map((c) => decideCall(c, { keepCall: 1, keepResult: 0 }, { keepThreshold: 0.5 }));
    const kept = applyDecisions(messages, decisions, calls, 300)[2]!.toolResults![0]!.text;
    expect(kept.startsWith(`a${'\u{1F525}'.repeat(149)}\n[fast-jev-compaction truncated`)).toBe(true);
    expect(lone.test(kept)).toBe(false);
  });
});
