import { describe, expect, it } from 'vitest';
import {
  compactSession,
  decisionLog,
  decisionLogLines,
  fallsBackToBuiltin,
  register,
  resolveHookConfig,
  summarize,
  toSessionMessages,
} from '../hooks/fast-jev.ts';
import { applyDecisions, collectToolCalls, decideCall, type Message } from '../src/index.js';

type SessionMessage = Message & { handle?: string };

function message(role: Message['role'], text: string, extra: Partial<SessionMessage> = {}): SessionMessage {
  return { role, text, toolUses: [], ...extra };
}

function call(id: string, tool: string, input: Record<string, unknown>, text: string): SessionMessage {
  return message('assistant', '', {
    toolUses: [{ tool_use_id: id, tool, input, text }],
    handle: `h-${id}`,
  });
}

function result(id: string, text: string, isError = false): SessionMessage {
  return message('user', '', { toolResults: [{ tool_use_id: id, text, isError }], handle: `r-${id}` });
}

const fileA = 'export const a = 1;\n'.repeat(50);

function transcript(): SessionMessage[] {
  return [
    message('user', 'Fix the failing test.', { handle: 'h-0' }),
    call('tool-1', 'Read', { file_path: 'src/a.ts' }, fileA),
    result('tool-1', fileA),
    call('tool-2', 'Bash', { command: 'npm test' }, 'FAIL'),
    result('tool-2', 'FAIL b.test.ts: expected 2 to be 3', true),
    message('assistant', 'Fixing now.', { handle: 'h-5' }),
    message('user', 'go ahead', { handle: 'h-6' }),
  ];
}

function jevFetch(answer: (name: string) => number, bodies: string[] = [], urls: string[] = []) {
  return async (url: string, init?: { body?: string }) => {
    urls.push(url);
    bodies.push(init?.body ?? '');
    const { questions } = JSON.parse(init?.body ?? '{}') as { questions: Record<string, unknown> };
    const answers = Object.fromEntries(
      Object.keys(questions).map((key) => [key, { type: 'noul', noul: answer(key) }]),
    );
    return { status: 200, ok: true, text: JSON.stringify({ answers }) };
  };
}

describe('hook config', () => {
  it('reads userConfig values and falls back to defaults', () => {
    expect(resolveHookConfig({})).toEqual({
      compactAtPercent: 60,
      minReductionRatio: 0.25,
      model: 'jev-latest',
      requestTimeoutMs: 15_000,
    });
    expect(
      resolveHookConfig({ apiKey: 'k', keepThreshold: 0.3, maxStateTokens: 1000, model: 'jev-x', goal: 'g', compactAtPercent: 'no' }),
    ).toEqual({
      apiKey: 'k',
      keepThreshold: 0.3,
      maxStateTokens: 1000,
      model: 'jev-x',
      goal: 'g',
      compactAtPercent: 60,
      minReductionRatio: 0.25,
      requestTimeoutMs: 15_000,
    });
  });
});

describe('session message mapping', () => {
  it('returns every message without its handle, keeping the content', () => {
    const messages = transcript();
    const calls = collectToolCalls(messages, 0);
    const decisions = [
      decideCall(calls[0]!, { keepCall: 0.9, keepResult: 0.1 }, { keepThreshold: 0.5 }),
      decideCall(calls[1]!, { keepCall: 0.9, keepResult: 0.9 }, { keepThreshold: 0.5 }),
    ];
    messages[1]!.toolUses[0]!.text = 'x'.repeat(2000);
    messages[2]!.toolResults![0]!.text = 'x'.repeat(2000);
    const out = toSessionMessages(messages, applyDecisions(messages, decisions, calls, 300));
    expect(out).toHaveLength(messages.length);
    expect(out.every((m) => !("handle" in m))).toBe(true);
    expect(out[0]).toEqual({ role: 'user', text: 'Fix the failing test.', toolUses: [] });
    expect(out[1]?.toolUses[0]?.text).toMatch(
      new RegExp(`^${'x'.repeat(300)}\\n\\[fast-jev-compaction truncated 1700 chars`),
    );
    expect(out[2]?.toolResults?.[0]?.text).toMatch(
      new RegExp(`^${'x'.repeat(300)}\\n\\[fast-jev-compaction truncated 1700 chars`),
    );
    expect(out[2]?.toolResults?.[0]).toMatchObject({ tool_use_id: 'tool-1', isError: false });
    expect(out[3]?.toolUses[0]).toBe(messages[3]!.toolUses[0]);
    expect(out[4]?.toolResults?.[0]).toBe(messages[4]!.toolResults![0]);
  });

  it('keeps short dropped results as the engine objects', () => {
    const messages = transcript();
    messages[1]!.toolUses[0]!.text = 'y'.repeat(100);
    messages[2]!.toolResults![0]!.text = 'y'.repeat(100);
    const calls = collectToolCalls(messages, 0);
    const decisions = [
      decideCall(calls[0]!, { keepCall: 0.9, keepResult: 0.1 }, { keepThreshold: 0.5 }),
      decideCall(calls[1]!, { keepCall: 0.9, keepResult: 0.9 }, { keepThreshold: 0.5 }),
    ];
    const out = toSessionMessages(messages, applyDecisions(messages, decisions, calls, 300));
    expect(out[1]?.toolUses[0]).toBe(messages[1]!.toolUses[0]);
    expect(out[2]?.toolResults?.[0]).toBe(messages[2]!.toolResults![0]);
  });
});

describe('compactSession', () => {
  it('runs the library over the engine fetch and reports the outcome', async () => {
    const bodies: string[] = [];
    const urls: string[] = [];
    const config = {
      ...resolveHookConfig({ preserveRecentMessages: 1, baseUrl: 'https://jev.example/v1' }),
      apiKey: 'k',
      model: 'jev-x',
    };
    const { result: output, messages } = await compactSession(
      transcript(),
      config,
      jevFetch((name) => (name === 'call_t2' || name === 'result_t2' ? 0.9 : 0.1), bodies, urls),
    );
    expect(urls).toEqual(['https://jev.example/v1']);
    expect(bodies).toHaveLength(1);
    expect(JSON.parse(bodies[0]!).model).toBe('jev-x');
    // t2's result is a few characters: not worth a question, kept as is.
    expect(JSON.parse(bodies[0]!).questions).toHaveProperty('call_t1');
    expect(JSON.parse(bodies[0]!).questions).not.toHaveProperty('call_t2');
    expect(output.decisions.map((d) => [d.action, d.reason])).toEqual([
      ['drop_call', 'call_dropped'],
      ['keep', 'skipped'],
    ]);
    expect(messages.map((m) => m.text)).toEqual(['Fix the failing test.', '', '', 'Fixing now.', 'go ahead']);
    expect(messages.every((m) => m.handle === undefined)).toBe(true);
    expect(summarize(output)).toMatch(/^\d+% reduction; 1 call_dropped, 1 too small to ask; state ~\d+ tokens \(full\) in 1 request\(s\)$/);
    expect(decisionLog(output)).toBe('t1:Read:drop_call/call=0.10/result=0.10');
    expect(decisionLogLines(output)).toEqual([`decisions: ${decisionLog(output)}`]);
  });

  it('splits a long decision log into ui.log lines under the host limit', async () => {
    const config = { ...resolveHookConfig({ preserveRecentMessages: 1, minSavingsChars: 0 }), apiKey: 'k' };
    const { result: output } = await compactSession(transcript(), config, jevFetch(() => 0.1));
    const lines = decisionLogLines(output, 60);
    expect(lines).toEqual([
      'decisions (1/2): t1:Read:drop_call/call=0.10/result=0.10',
      'decisions (2/2): t2:Bash:drop_call/call=0.10/result=0.10',
    ]);
    expect(lines.every((line) => line.length <= 60)).toBe(true);
    expect(decisionLogLines({ ...output, decisions: [] })).toEqual(['decisions: (none)']);
  });

  it('throws on a missing key and on failed requests so the hook falls back', async () => {
    const config = resolveHookConfig({ preserveRecentMessages: 1 });
    await expect(compactSession(transcript(), config, jevFetch(() => 0))).rejects.toThrow(/TYPESAFE_API_KEY/);
    await expect(
      compactSession(transcript(), { ...config, apiKey: 'k' }, async () => ({ status: 500, ok: false, text: 'x' })),
    ).rejects.toThrow(/500/);
  });
});

describe('trigger policy', () => {
  it('falls back to the built-in summary only when Claude Code must compact', () => {
    expect(['auto', 'manual', 'plugin', 'precompute'].map((t) => fallsBackToBuiltin(t as never))).toEqual([
      true,
      true,
      false,
      false,
    ]);
  });

  type Handler = (...args: any[]) => Promise<any>;

  /** Registers the hook against a fake engine and returns its handlers and what they did. */
  function engine(fetchOk: boolean, usage: number[] = [], compactOutcome: object = { skip: 'x' }) {
    const handlers: Record<string, Handler> = {};
    register(((name: string, handler: Handler) => {
      handlers[name] = handler;
    }) as never, { preserveRecentMessages: 1 });
    const seen = { logs: [] as string[], toasts: [] as string[], fetches: 0, compacts: 0 };
    const $ = {
      env: { get: async () => 'k' },
      settings: { read: async () => ({}) },
      http: {
        fetch: async (url: string, init?: { body?: string }) => {
          seen.fetches += 1;
          return fetchOk
            ? { ...(await jevFetch(() => 0)(url, init)), headers: {} }
            : { status: 500, ok: false, text: 'down', headers: {} };
        },
      },
      // Retry delays pass at once; the 15s request timeout never fires.
      clock: { sleep: (ms: number) => (ms >= 15_000 ? new Promise(() => {}) : Promise.resolve()) },
      ui: { log: (t: string) => seen.logs.push(t), toast: (t: string) => seen.toasts.push(t) },
      session: {
        usage: async () => ({ context: { percent: usage.shift() } }),
        compact: async () => {
          seen.compacts += 1;
          return compactOutcome;
        },
      },
    };
    const next = async () => ({ fromBuiltin: true });
    return { handlers, seen, $, next };
  }

  it('skips a failed precompute instead of running the built-in summary', async () => {
    const { handlers, seen, $, next } = engine(false);
    const out = await handlers['session.compact']!($, { trigger: 'precompute', messages: transcript() }, next);
    expect(out).toEqual({ skip: expect.stringMatching(/500/) });
    expect(seen.fetches).toBe(2); // one retry on the 500
    expect(seen.toasts).toEqual([]);
  });

  it('falls back on a failed auto compaction', async () => {
    const { handlers, $, next } = engine(false);
    const out = await handlers['session.compact']!($, { trigger: 'auto', messages: transcript() }, next);
    expect(out).toEqual({ fromBuiltin: true });
  });

  it('precomputes quietly and hands the compacted messages back', async () => {
    const { handlers, seen, $, next } = engine(true);
    const out = await handlers['session.compact']!($, { trigger: 'precompute', messages: transcript() }, next);
    expect(out.messages.length).toBeLessThan(transcript().length);
    expect(seen.toasts).toEqual([]);
    expect(seen.logs.some((l) => l.startsWith('precomputed: kept'))).toBe(true);
  });

  it("leaves a subagent's compaction to the built-in path", async () => {
    const { handlers, seen, $, next } = engine(true);
    const out = await handlers['session.compact']!($, { trigger: 'auto', agentId: 'a1', messages: transcript() }, next);
    expect(out).toEqual({ fromBuiltin: true });
    expect(seen.fetches).toBe(0);
  });

  it('waits for the context to grow before retrying a skipped turn.complete compaction', async () => {
    const { handlers, seen, $, next } = engine(true, [65, 70, 75, 50, 61]);
    const turn = { reason: 'answer' };
    for (let i = 0; i < 5; i++) await handlers['turn.complete']!($, turn, next);
    // 65: tries, skipped -> next try at 75. 70: waits. 75: tries. 50: resets. 61: tries.
    expect(seen.compacts).toBe(3);
    await handlers['turn.complete']!($, { reason: 'aborted' }, next);
    await handlers['turn.complete']!($, { reason: 'answer', agentId: 'a1' }, next);
    expect(seen.compacts).toBe(3);
  });
});
