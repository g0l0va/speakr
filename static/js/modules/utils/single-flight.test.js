import { describe, expect, it, vi } from 'vitest';

import { guardSingleFlight } from './single-flight.js';

// Simple deferred helper so tests can control exactly when an in-flight
// call resolves, to assert on state *while* it's still pending.
function deferred() {
    let resolve, reject;
    const promise = new Promise((res, rej) => { resolve = res; reject = rej; });
    return { promise, resolve, reject };
}

describe('guardSingleFlight', () => {
    it('runs the wrapped function and releases the flag on success', async () => {
        const flag = { value: false };
        const fn = vi.fn().mockResolvedValue('ok');
        const guarded = guardSingleFlight(flag, fn);

        expect(flag.value).toBe(false);
        const result = await guarded('a');
        expect(result).toBe('ok');
        expect(fn).toHaveBeenCalledWith('a');
        expect(flag.value).toBe(false);
    });

    it('is a no-op if called again while already in flight', async () => {
        const flag = { value: false };
        const d = deferred();
        const fn = vi.fn().mockReturnValue(d.promise);
        const guarded = guardSingleFlight(flag, fn);

        const first = guarded();
        expect(flag.value).toBe(true);

        // A second call while the first is still pending (e.g. a double
        // click before the button's :disabled binding re-renders).
        const second = guarded();
        expect(fn).toHaveBeenCalledOnce(); // second call never invoked fn

        d.resolve('done');
        await first;
        await second;
        expect(flag.value).toBe(false);
    });

    it('releases the flag even if the wrapped function throws', async () => {
        const flag = { value: false };
        const fn = vi.fn().mockRejectedValue(new Error('boom'));
        const guarded = guardSingleFlight(flag, fn);

        await expect(guarded()).rejects.toThrow('boom');
        expect(flag.value).toBe(false);

        // Flag correctly released -- a subsequent call is allowed to run.
        const fn2 = vi.fn().mockResolvedValue('recovered');
        const guarded2 = guardSingleFlight(flag, fn2);
        await expect(guarded2()).resolves.toBe('recovered');
    });

    it('does not deadlock when two guarded actions delegate to each other\'s UNGUARDED implementations', async () => {
        // Mirrors composables/speakers.js: saveSpeakerNames and
        // saveTranscriptEdits each fall back to the OTHER's inner
        // (unguarded) implementation depending on a condition, and only the
        // exported wrappers share one guard. Calling either public entry
        // point must always actually run something -- never silently no-op
        // because the shared flag was already set by itself.
        const flag = { value: false };
        let useTranscriptPath = true;

        const saveSpeakerNamesInner = vi.fn(async () => {
            if (useTranscriptPath) {
                return saveTranscriptEditsInner();
            }
            return 'names-saved';
        });
        const saveTranscriptEditsInner = vi.fn(async () => {
            if (!useTranscriptPath) {
                return saveSpeakerNamesInner();
            }
            return 'transcript-saved';
        });

        const saveSpeakerNames = guardSingleFlight(flag, saveSpeakerNamesInner);
        const saveTranscriptEdits = guardSingleFlight(flag, saveTranscriptEditsInner);

        await expect(saveSpeakerNames()).resolves.toBe('transcript-saved');
        expect(flag.value).toBe(false);

        useTranscriptPath = false;
        await expect(saveTranscriptEdits()).resolves.toBe('names-saved');
        expect(flag.value).toBe(false);
    });

    it('forwards all arguments and the resolved value through', async () => {
        const flag = { value: false };
        const fn = vi.fn(async (a, b) => a + b);
        const guarded = guardSingleFlight(flag, fn);

        await expect(guarded(2, 3)).resolves.toBe(5);
        expect(fn).toHaveBeenCalledWith(2, 3);
    });
});
