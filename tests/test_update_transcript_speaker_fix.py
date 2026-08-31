#!/usr/bin/env python3
"""
Regression tests for the /update_transcript "Save Names" bug.

Bug: a combined edit (reassigning one segment's speaker label, e.g.
SPEAKER_04 -> SPEAKER_02 via the per-line "change speaker" control) plus
naming (SPEAKER_00..03) always routes through /update_transcript, not
/update_speakers, because editedTranscriptData is set as soon as any
per-line edit is staged. /update_transcript never confirmed voice
embeddings or created snippets, and had no protection against a duplicate
submission (double-click, retried request) re-bumping use_count or
re-creating samples/snippets/summary jobs.

Covers:
  - Combined edit+naming: /update_transcript now confirms voice samples
    (record_manual_confirmation) and creates snippets, same as
    /update_speakers -- the actual fix.
  - The resulting SpeakerEmbeddingSample is 'manual'/'confirmed' and the
    speaker's average_embedding is rebuilt from it.
  - No-op resubmit: an identical second submission does not duplicate
    use_count / confirmed samples / snippets / a summary job, but still
    responds successfully (200, success: true).
  - A genuinely different follow-up edit is still applied even after an
    earlier no-op on the same recording (the no-op detector isn't sticky).

Pattern follows tests/test_cov_recordings_write.py (session login, no CSRF).
"""

import json
import os
import sys
import uuid
from unittest.mock import patch, MagicMock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.app import app, db
from src.models import User, Recording, Speaker, SpeakerEmbeddingSample

app.config["WTF_CSRF_ENABLED"] = False


def _vec(seed, dim=256):
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _mk_user(prefix="ut"):
    suffix = uuid.uuid4().hex[:8]
    user = User(username=f"{prefix}_{suffix}", email=f"{prefix}_{suffix}@local.test", password="x")
    db.session.add(user)
    db.session.commit()
    return user


def _login(client, user):
    with client.session_transaction() as sess:
        sess["_user_id"] = str(user.id)
        sess["_fresh"] = True


def _mk_recording(user, transcription, speaker_embeddings):
    rec = Recording(
        user_id=user.id,
        title=f"rec_{uuid.uuid4().hex[:8]}",
        status="COMPLETED",
        transcription=json.dumps(transcription),
        speaker_embeddings=speaker_embeddings,
    )
    db.session.add(rec)
    db.session.commit()
    return rec


def _mk_speaker(user, name):
    speaker = Speaker(name=name, user_id=user.id)
    db.session.add(speaker)
    db.session.commit()
    return speaker


def _cleanup(*objects):
    for obj in reversed(objects):
        try:
            db.session.delete(obj)
        except Exception:
            db.session.rollback()
            try:
                db.session.merge(obj)
            except Exception:
                pass
    db.session.commit()


# Original diarized transcript: 5 generic speaker labels.
ORIGINAL_TRANSCRIPT = [
    {"speaker": "SPEAKER_00", "sentence": "Hello everyone"},
    {"speaker": "SPEAKER_01", "sentence": "Hi there"},
    {"speaker": "SPEAKER_02", "sentence": "Yes, agreed"},
    {"speaker": "SPEAKER_03", "sentence": "No, disagreed"},
    {"speaker": "SPEAKER_04", "sentence": "This segment gets reassigned"},
]

# After the user's per-line edit: the last segment (originally SPEAKER_04)
# is relabeled SPEAKER_02 -- the exact bug-report scenario. The frontend
# stages this client-side (changeSpeaker) before the whole payload is sent.
EDITED_TRANSCRIPT = [
    {"speaker": "SPEAKER_00", "sentence": "Hello everyone"},
    {"speaker": "SPEAKER_01", "sentence": "Hi there"},
    {"speaker": "SPEAKER_02", "sentence": "Yes, agreed"},
    {"speaker": "SPEAKER_03", "sentence": "No, disagreed"},
    {"speaker": "SPEAKER_02", "sentence": "This segment gets reassigned"},
]

SPEAKER_MAP = {
    "SPEAKER_00": {"name": "Alice", "isMe": False},
    "SPEAKER_01": {"name": "Bob", "isMe": False},
    "SPEAKER_02": {"name": "Carol", "isMe": False},
    "SPEAKER_03": {"name": "Dave", "isMe": False},
}

EMBEDDINGS = {
    "SPEAKER_00": _vec(200),
    "SPEAKER_01": _vec(201),
    "SPEAKER_02": _vec(202),
    "SPEAKER_03": _vec(203),
    "SPEAKER_04": _vec(204),
}


def test_combined_edit_and_naming_confirms_voice_samples_and_creates_snippets():
    with app.app_context():
        user = _mk_user("combo1")
        alice = _mk_speaker(user, "Alice")
        bob = _mk_speaker(user, "Bob")
        carol = _mk_speaker(user, "Carol")
        dave = _mk_speaker(user, "Dave")
        rec = _mk_recording(user, ORIGINAL_TRANSCRIPT, EMBEDDINGS)
        client = app.test_client()
        _login(client, user)
        try:
            mock_snippets = MagicMock(return_value=4)
            with patch("src.api.recordings.create_speaker_snippets", mock_snippets):
                resp = client.post(
                    f"/recording/{rec.id}/update_transcript",
                    json={
                        "transcript_data": json.loads(json.dumps(EDITED_TRANSCRIPT)),
                        "speaker_map": SPEAKER_MAP,
                        "regenerate_summary": False,
                    },
                )
            assert resp.status_code == 200, resp.get_data(as_text=True)
            assert resp.get_json()["success"] is True

            db.session.refresh(rec)
            segments = json.loads(rec.transcription)
            # The reassigned segment now carries Carol's name (SPEAKER_02),
            # exactly like the segment that was already SPEAKER_02.
            assert segments[2]["speaker"] == "Carol"
            assert segments[4]["speaker"] == "Carol"

            # The actual bug fix: /update_transcript must confirm voice
            # samples exactly like /update_speakers already does.
            for speaker in (alice, bob, carol, dave):
                confirmed = SpeakerEmbeddingSample.query.filter_by(
                    speaker_id=speaker.id, review_status="confirmed", source="manual"
                ).all()
                assert len(confirmed) == 1, f"Expected 1 confirmed sample for {speaker.name}, got {len(confirmed)}"
                db.session.refresh(speaker)
                assert speaker.average_embedding is not None

            mock_snippets.assert_called_once()
        finally:
            _cleanup(rec, alice, bob, carol, dave)


def test_identical_resubmit_is_a_noop_but_still_succeeds():
    with app.app_context():
        user = _mk_user("noop1")
        alice = _mk_speaker(user, "Alice")
        bob = _mk_speaker(user, "Bob")
        carol = _mk_speaker(user, "Carol")
        dave = _mk_speaker(user, "Dave")
        rec = _mk_recording(user, ORIGINAL_TRANSCRIPT, EMBEDDINGS)
        client = app.test_client()
        _login(client, user)
        try:
            mock_snippets = MagicMock(return_value=4)
            mock_enqueue = MagicMock(return_value="job-1")
            with patch("src.api.recordings.create_speaker_snippets", mock_snippets), \
                 patch("src.services.job_queue.job_queue.enqueue", mock_enqueue):
                first = client.post(
                    f"/recording/{rec.id}/update_transcript",
                    json={
                        "transcript_data": json.loads(json.dumps(EDITED_TRANSCRIPT)),
                        "speaker_map": SPEAKER_MAP,
                        "regenerate_summary": True,
                    },
                )
                assert first.status_code == 200, first.get_data(as_text=True)
                assert first.get_json()["summary_queued"] is True

                db.session.refresh(alice)
                use_count_after_first = alice.use_count
                confirmed_after_first = SpeakerEmbeddingSample.query.filter_by(
                    speaker_id=alice.id, review_status="confirmed"
                ).count()
                mock_snippets.assert_called_once()
                mock_enqueue.assert_called_once()

                # Second, IDENTICAL submission -- e.g. a duplicate network
                # request from before the frontend single-flight guard existed.
                second = client.post(
                    f"/recording/{rec.id}/update_transcript",
                    json={
                        "transcript_data": json.loads(json.dumps(EDITED_TRANSCRIPT)),
                        "speaker_map": SPEAKER_MAP,
                        "regenerate_summary": True,
                    },
                )
            assert second.status_code == 200, second.get_data(as_text=True)
            assert second.get_json()["success"] is True
            assert second.get_json()["summary_queued"] is False, "No-op resubmit must not queue a second summary job"

            db.session.refresh(alice)
            assert alice.use_count == use_count_after_first, "No-op resubmit must not re-bump use_count"

            confirmed_after_second = SpeakerEmbeddingSample.query.filter_by(
                speaker_id=alice.id, review_status="confirmed"
            ).count()
            assert confirmed_after_second == confirmed_after_first, "No-op resubmit must not duplicate confirmed samples"

            mock_snippets.assert_called_once()  # still only the first call
            mock_enqueue.assert_called_once()   # still only the first call
        finally:
            _cleanup(rec, alice, bob, carol, dave)


def test_different_resubmit_after_identical_one_is_treated_as_a_real_edit():
    """Sanity check: the no-op detector must not get 'stuck' -- a genuine
    follow-up edit on the same recording must still be applied."""
    with app.app_context():
        user = _mk_user("real1")
        alice = _mk_speaker(user, "Alice")
        bob = _mk_speaker(user, "Bob")
        carol = _mk_speaker(user, "Carol")
        dave = _mk_speaker(user, "Dave")
        rec = _mk_recording(user, ORIGINAL_TRANSCRIPT, EMBEDDINGS)
        client = app.test_client()
        _login(client, user)
        try:
            mock_snippets = MagicMock(return_value=4)
            with patch("src.api.recordings.create_speaker_snippets", mock_snippets):
                first = client.post(
                    f"/recording/{rec.id}/update_transcript",
                    json={
                        "transcript_data": json.loads(json.dumps(EDITED_TRANSCRIPT)),
                        "speaker_map": SPEAKER_MAP,
                        "regenerate_summary": False,
                    },
                )
                assert first.status_code == 200

                # A genuinely different follow-up edit: rename Dave -> Erin.
                different_map = dict(SPEAKER_MAP)
                different_map["SPEAKER_03"] = {"name": "Erin", "isMe": False}
                second = client.post(
                    f"/recording/{rec.id}/update_transcript",
                    json={
                        "transcript_data": json.loads(json.dumps(EDITED_TRANSCRIPT)),
                        "speaker_map": different_map,
                        "regenerate_summary": False,
                    },
                )
            assert second.status_code == 200, second.get_data(as_text=True)
            db.session.refresh(rec)
            segments = json.loads(rec.transcription)
            assert segments[3]["speaker"] == "Erin", "A real follow-up edit must still be applied"
        finally:
            _cleanup(rec, alice, bob, carol, dave)


def test_row_lock_does_not_block_a_normal_single_request():
    """Sanity check: with_for_update() must not itself break the plain,
    single-request happy path (e.g. if SQLite rejected FOR UPDATE outright)."""
    with app.app_context():
        user = _mk_user("lock1")
        alice = _mk_speaker(user, "Alice")
        rec = _mk_recording(user, ORIGINAL_TRANSCRIPT, EMBEDDINGS)
        client = app.test_client()
        _login(client, user)
        try:
            with patch("src.api.recordings.create_speaker_snippets", MagicMock(return_value=0)):
                resp = client.post(
                    f"/recording/{rec.id}/update_transcript",
                    json={
                        "transcript_data": json.loads(json.dumps(ORIGINAL_TRANSCRIPT)),
                        "speaker_map": {"SPEAKER_00": {"name": "Alice", "isMe": False}},
                        "regenerate_summary": False,
                    },
                )
            assert resp.status_code == 200, resp.get_data(as_text=True)
            assert resp.get_json()["success"] is True
        finally:
            _cleanup(rec, alice)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
