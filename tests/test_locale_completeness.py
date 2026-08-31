#!/usr/bin/env python3
"""
Locale key-coverage check.

static/locales/en.json is the source of truth for every UI string key. The
i18n runtime (static/js/i18n.js) gracefully falls back to English for a
locale that's missing a key, but a *silent* fallback is easy to forget about
during review -- this test makes a missing key a visible, targeted CI
failure instead, naming exactly which locale and which key path is short.

There was no existing test covering this before the SpeakerEmbeddingSample
review UI added ~31 new speakersManagement.* keys; this is a general-purpose
check, not specific to that feature, so any future key addition is covered
too.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LOCALES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static", "locales")
REFERENCE_LOCALE = "en"


def _load_locale(name):
    path = os.path.join(LOCALES_DIR, f"{name}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _all_locale_names():
    names = []
    for filename in os.listdir(LOCALES_DIR):
        if filename.endswith(".json"):
            names.append(filename[: -len(".json")])
    return sorted(names)


def _collect_key_paths(obj, prefix=""):
    """Flatten a nested dict into a set of dotted key paths."""
    paths = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                paths |= _collect_key_paths(value, full_key)
            else:
                paths.add(full_key)
    return paths


def test_every_locale_file_is_valid_json():
    for name in _all_locale_names():
        try:
            _load_locale(name)
        except json.JSONDecodeError as e:
            raise AssertionError(f"static/locales/{name}.json is not valid JSON: {e}")


def test_reference_locale_exists():
    assert REFERENCE_LOCALE in _all_locale_names(), (
        f"Reference locale '{REFERENCE_LOCALE}.json' is missing from {LOCALES_DIR}"
    )


def test_all_locales_cover_every_english_key():
    """Every key path present in en.json must exist in every other locale.

    This does not require translated *values* (a locale is free to
    temporarily reuse the English string as a placeholder), only that the
    key path itself exists -- so window.i18n.t() resolves it from that
    locale directly instead of silently falling back to English.
    """
    reference = _collect_key_paths(_load_locale(REFERENCE_LOCALE))
    failures = []

    for name in _all_locale_names():
        if name == REFERENCE_LOCALE:
            continue
        locale_keys = _collect_key_paths(_load_locale(name))
        missing = sorted(reference - locale_keys)
        if missing:
            failures.append((name, missing))

    if failures:
        details = "\n".join(
            f"  {name}.json missing {len(missing)} key(s): {missing[:10]}"
            f"{' ...' if len(missing) > 10 else ''}"
            for name, missing in failures
        )
        raise AssertionError(f"Locale files missing keys present in {REFERENCE_LOCALE}.json:\n{details}")


def test_no_locale_has_orphaned_keys_not_in_english():
    """A key that exists in a translated locale but not in en.json is
    usually a typo/rename that fell out of sync -- surface it instead of
    letting it silently become dead weight."""
    reference = _collect_key_paths(_load_locale(REFERENCE_LOCALE))
    failures = []

    for name in _all_locale_names():
        if name == REFERENCE_LOCALE:
            continue
        locale_keys = _collect_key_paths(_load_locale(name))
        orphaned = sorted(locale_keys - reference)
        if orphaned:
            failures.append((name, orphaned))

    if failures:
        details = "\n".join(
            f"  {name}.json has {len(orphaned)} key(s) not in {REFERENCE_LOCALE}.json: {orphaned[:10]}"
            f"{' ...' if len(orphaned) > 10 else ''}"
            for name, orphaned in failures
        )
        raise AssertionError(f"Locale files have orphaned keys:\n{details}")


def test_speakers_management_sample_review_keys_present_in_every_locale():
    """Targeted check for this increment's new keys specifically, so a
    partial/interrupted translation merge fails loudly and by name."""
    new_keys = {
        "failedToLoadSamples", "loadingSamples", "noSamplesForFilter", "noSamplesYet",
        "reviewSamples", "sampleActionConfirm", "sampleActionDelete", "sampleActionReject",
        "sampleColumnActions", "sampleColumnCreated", "sampleColumnModel", "sampleColumnRecording",
        "sampleColumnReviewed", "sampleColumnSimilarity", "sampleColumnSource", "sampleColumnStatus",
        "sampleConfirmDelete", "sampleConfirmReject", "sampleConfirmed", "sampleDeleted",
        "sampleFilterAll", "sampleFilterConfirmed", "sampleFilterPending", "sampleFilterRejected",
        "sampleRejected", "sampleSourceAutomatic", "sampleSourceImport", "sampleSourceManual",
        "samplesConfirmed", "samplesPending", "samplesRejected",
    }

    for name in _all_locale_names():
        sm = _load_locale(name).get("speakersManagement", {})
        missing = new_keys - set(sm.keys())
        assert not missing, f"{name}.json's speakersManagement is missing: {sorted(missing)}"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
