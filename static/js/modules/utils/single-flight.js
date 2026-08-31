/**
 * Single-flight guard for an async action bound to a Vue-style ref.
 *
 * Wraps an async function so that while it's already running (flagRef.value
 * is true), a second call is a no-op instead of firing a second overlapping
 * request -- e.g. a double-click or a slow response arriving before the UI
 * disables the button. The flag is always released in a `finally`, even if
 * the wrapped function throws, so a failed request doesn't permanently wedge
 * the action.
 *
 * The guard must be applied at the ONE externally-invoked entry point, not
 * inside every function that might call another as an internal fallback: if
 * two functions delegate to each other (A calls B when condition X, B calls
 * A otherwise) and each independently checks/sets the SAME flag, whichever
 * one runs first sets the flag, and the other's own guard then sees it
 * already set and silently no-ops -- breaking the delegation instead of
 * protecting it. Wrap the public entry point(s) around the *unguarded*
 * implementations instead (see composables/speakers.js for a concrete case:
 * saveSpeakerNames / saveTranscriptEdits delegate to each other's Inner
 * implementations, and only the exported wrappers are guarded).
 */

/**
 * @param {{value: boolean}} flagRef - a ref-like object; only `.value` is used.
 * @param {(...args: any[]) => Promise<any>} fn - the async function to guard.
 * @returns {(...args: any[]) => Promise<any>} a wrapped function that is a
 *   no-op (resolves to undefined) if called while already in flight.
 */
export function guardSingleFlight(flagRef, fn) {
    return async (...args) => {
        if (flagRef.value) return undefined;
        flagRef.value = true;
        try {
            return await fn(...args);
        } finally {
            flagRef.value = false;
        }
    };
}
