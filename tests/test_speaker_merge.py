#!/usr/bin/env python3
"""
Test suite for speaker_merge.py's integration with the SpeakerEmbeddingSample
subsystem (src/services/speaker_embedding_samples.py).

Before this integration, merging speakers silently lost data: source
speakers were deleted with `ondelete='CASCADE'` on
speaker_embedding_sample.speaker_id, so every individual voice sample a
source held vanished with it, and the target's average_embedding was
recomputed via the old weighted-average blend of raw blobs rather than from
confirmed samples.

Covers:
  - Confirmed/pending/rejected samples are re-parented (not lost) on merge
  - Recording/label/similarity/model metadata survives the transfer unchanged
  - A source speaker is actually deleted, but its samples are not
  - Duplicate pending samples for the same (recording, label) are resolved
    deterministically (newer stays pending, older is rejected -- not
    silently deleted) in both possible timestamp orderings
  - Confirmed samples for the same (recording, label) are NOT deduped
  - The target's profile is rebuilt from confirmed samples after merge
  - The legacy weighted-average blend still runs as a fallback when neither
    the target nor any source has a single confirmed sample anywhere
  - Ownership (user_id) is preserved on every transferred sample

Pattern follows tests/test_speaker_embedding_samples.py -- standalone, no
pytest fixtures.
"""

import sys
import os
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.app import app, db
from src.models import User, Speaker, SpeakerEmbeddingSample, Recording
from src.services.speaker_embedding_matcher import serialize_embedding
from src.services.speaker_embedding_samples import create_sample
from src.services.speaker_merge import merge_speakers


def _vec(seed, dim=256):
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


VEC_A = _vec(101)
VEC_B = _vec(102)
VEC_C = _vec(103)


def _get_or_create_user(suffix):
    username = f"merge_test_user_{suffix}"
    user = User.query.filter_by(username=username).first()
    if not user:
        user = User(username=username, email=f"{username}@local.test", name="Test User")
        db.session.add(user)
        db.session.commit()
    return user


def _create_speaker(user, name, average_embedding_vec=None, embedding_count=None):
    speaker = Speaker(name=name, user_id=user.id)
    if average_embedding_vec is not None:
        speaker.average_embedding = serialize_embedding(np.array(average_embedding_vec, dtype=np.float32))
        speaker.embedding_count = embedding_count if embedding_count is not None else 1
    db.session.add(speaker)
    db.session.commit()
    return speaker


def _create_recording(user):
    rec = Recording(user_id=user.id, title="Test Recording", status="COMPLETED")
    db.session.add(rec)
    db.session.commit()
    return rec


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


def _confirmed_samples(speaker_id):
    return SpeakerEmbeddingSample.query.filter_by(
        speaker_id=speaker_id, review_status="confirmed"
    ).all()


# =========================================================================
# Transfer: confirmed/pending/rejected samples survive the merge
# =========================================================================

def test_merge_transfers_confirmed_samples_and_rebuilds_target():
    with app.app_context():
        user = _get_or_create_user("xfer1")
        target = _create_speaker(user, "Target")
        source = _create_speaker(user, "Source")
        target_sample = create_sample(target, VEC_A, user_id=user.id, source="manual", review_status="confirmed")
        source_sample = create_sample(source, VEC_B, user_id=user.id, source="manual", review_status="confirmed")
        try:
            merge_speakers(target.id, [source.id], user.id)

            # Source speaker is gone...
            assert db.session.get(Speaker, source.id) is None

            # ...but its sample is not: re-parented onto the target.
            db.session.refresh(source_sample)
            assert source_sample.speaker_id == target.id
            assert source_sample.user_id == user.id, "Ownership must survive the transfer"

            db.session.refresh(target_sample)
            assert target_sample.speaker_id == target.id

            confirmed = _confirmed_samples(target.id)
            assert len(confirmed) == 2

            db.session.refresh(target)
            expected = np.mean(np.stack([np.array(VEC_A), np.array(VEC_B)]), axis=0)
            actual = np.frombuffer(target.average_embedding, dtype=np.float32)
            assert np.allclose(actual, expected, atol=1e-4), "Target profile must rebuild from confirmed samples"
            assert target.embedding_count == 2
        finally:
            _cleanup(target)


def test_merge_preserves_metadata_on_pending_and_rejected_samples():
    with app.app_context():
        user = _get_or_create_user("xfer2")
        target = _create_speaker(user, "Target")
        source = _create_speaker(user, "Source")
        rec = _create_recording(user)

        pending = create_sample(
            source, VEC_A, user_id=user.id, recording_id=rec.id,
            source_speaker_label="SPEAKER_00", source="automatic",
            review_status="pending", similarity_to_profile=0.83, model_name="test-model-v1",
        )
        rejected = create_sample(
            source, VEC_B, user_id=user.id, recording_id=rec.id,
            source_speaker_label="SPEAKER_01", source="automatic",
            review_status="rejected", similarity_to_profile=0.41,
        )
        try:
            merge_speakers(target.id, [source.id], user.id)

            db.session.refresh(pending)
            assert pending.speaker_id == target.id
            assert pending.review_status == "pending"
            assert pending.recording_id == rec.id
            assert pending.source_speaker_label == "SPEAKER_00"
            assert pending.similarity_to_profile == 0.83
            assert pending.model_name == "test-model-v1"
            assert pending.user_id == user.id

            db.session.refresh(rejected)
            assert rejected.speaker_id == target.id
            assert rejected.review_status == "rejected"
            assert rejected.recording_id == rec.id
            assert rejected.source_speaker_label == "SPEAKER_01"
            assert rejected.similarity_to_profile == 0.41
        finally:
            _cleanup(rec, target)


# =========================================================================
# Duplicate pending resolution: deterministic, both timestamp orderings
# =========================================================================

def test_merge_dedups_pending_keeping_newer_when_source_is_newer():
    with app.app_context():
        user = _get_or_create_user("dedup1")
        target = _create_speaker(user, "Target")
        source = _create_speaker(user, "Source")
        rec = _create_recording(user)

        target_pending = create_sample(
            target, VEC_A, user_id=user.id, recording_id=rec.id,
            source_speaker_label="SPEAKER_00", source="automatic", review_status="pending",
        )
        target_pending.created_at = datetime(2026, 1, 1, 0, 0, 0)
        source_pending = create_sample(
            source, VEC_B, user_id=user.id, recording_id=rec.id,
            source_speaker_label="SPEAKER_00", source="automatic", review_status="pending",
        )
        source_pending.created_at = datetime(2026, 1, 2, 0, 0, 0)  # newer
        db.session.commit()

        try:
            merge_speakers(target.id, [source.id], user.id)

            db.session.refresh(target_pending)
            db.session.refresh(source_pending)

            assert target_pending.review_status == "rejected", "Older sample must be superseded, not left pending"
            assert target_pending.speaker_id == target.id, "Already the target's own row -- untouched otherwise"

            assert source_pending.review_status == "pending", "Newer sample must win and stay pending"
            assert source_pending.speaker_id == target.id, "Winner must still be re-parented onto target"

            still_pending = SpeakerEmbeddingSample.query.filter_by(
                speaker_id=target.id, recording_id=rec.id,
                source_speaker_label="SPEAKER_00", review_status="pending",
            ).all()
            assert len(still_pending) == 1, "Partial unique index must hold: at most one pending survivor"
            assert still_pending[0].id == source_pending.id
        finally:
            _cleanup(rec, target)


def test_merge_dedups_pending_keeping_newer_when_target_is_newer():
    with app.app_context():
        user = _get_or_create_user("dedup2")
        target = _create_speaker(user, "Target")
        source = _create_speaker(user, "Source")
        rec = _create_recording(user)

        target_pending = create_sample(
            target, VEC_A, user_id=user.id, recording_id=rec.id,
            source_speaker_label="SPEAKER_00", source="automatic", review_status="pending",
        )
        target_pending.created_at = datetime(2026, 1, 2, 0, 0, 0)  # newer
        source_pending = create_sample(
            source, VEC_B, user_id=user.id, recording_id=rec.id,
            source_speaker_label="SPEAKER_00", source="automatic", review_status="pending",
        )
        source_pending.created_at = datetime(2026, 1, 1, 0, 0, 0)
        db.session.commit()

        try:
            merge_speakers(target.id, [source.id], user.id)

            db.session.refresh(target_pending)
            db.session.refresh(source_pending)

            assert target_pending.review_status == "pending", "Newer (target's own) sample must stay pending"
            assert source_pending.review_status == "rejected", "Older, transferred sample must be superseded"
            assert source_pending.speaker_id == target.id, "Loser must still be re-parented, never left on the deleted source"

            still_pending = SpeakerEmbeddingSample.query.filter_by(
                speaker_id=target.id, recording_id=rec.id,
                source_speaker_label="SPEAKER_00", review_status="pending",
            ).all()
            assert len(still_pending) == 1
            assert still_pending[0].id == target_pending.id
        finally:
            _cleanup(rec, target)


def test_merge_does_not_dedup_confirmed_samples_with_same_recording_and_label():
    with app.app_context():
        user = _get_or_create_user("dedup3")
        target = _create_speaker(user, "Target")
        source = _create_speaker(user, "Source")
        rec = _create_recording(user)

        create_sample(
            target, VEC_A, user_id=user.id, recording_id=rec.id,
            source_speaker_label="SPEAKER_00", source="manual", review_status="confirmed",
        )
        create_sample(
            source, VEC_B, user_id=user.id, recording_id=rec.id,
            source_speaker_label="SPEAKER_00", source="manual", review_status="confirmed",
        )
        try:
            merge_speakers(target.id, [source.id], user.id)
            confirmed = SpeakerEmbeddingSample.query.filter_by(
                speaker_id=target.id, recording_id=rec.id,
                source_speaker_label="SPEAKER_00", review_status="confirmed",
            ).all()
            assert len(confirmed) == 2, "Confirmed samples are never deduped by the pending-only unique index"
        finally:
            _cleanup(rec, target)


# =========================================================================
# Legacy fallback: no confirmed samples anywhere
# =========================================================================

def test_merge_falls_back_to_legacy_blend_with_no_confirmed_samples_anywhere():
    with app.app_context():
        user = _get_or_create_user("legacy1")
        target = _create_speaker(user, "Target", average_embedding_vec=VEC_A, embedding_count=3)
        source = _create_speaker(user, "Source", average_embedding_vec=VEC_B, embedding_count=1)
        try:
            assert SpeakerEmbeddingSample.query.filter_by(speaker_id=target.id).count() == 0
            assert SpeakerEmbeddingSample.query.filter_by(speaker_id=source.id).count() == 0

            merge_speakers(target.id, [source.id], user.id)

            db.session.refresh(target)
            # Weighted average: target weight 3, source weight 1.
            expected = np.average([np.array(VEC_A), np.array(VEC_B)], axis=0, weights=[3, 1])
            actual = np.frombuffer(target.average_embedding, dtype=np.float32)
            assert np.allclose(actual, expected, atol=1e-4), "Must fall back to the legacy weighted blend"
            assert target.embedding_count == 4
        finally:
            _cleanup(target)


def test_merge_prefers_confirmed_samples_over_legacy_blend_when_any_exist():
    """As soon as even one confirmed sample exists anywhere in the merge,
    the rebuild-from-samples path takes over -- the legacy blend must not
    also run and overwrite it."""
    with app.app_context():
        user = _get_or_create_user("legacy2")
        target = _create_speaker(user, "Target", average_embedding_vec=VEC_A, embedding_count=1)
        source = _create_speaker(user, "Source")
        create_sample(source, VEC_B, user_id=user.id, source="manual", review_status="confirmed")
        try:
            merge_speakers(target.id, [source.id], user.id)
            db.session.refresh(target)
            actual = np.frombuffer(target.average_embedding, dtype=np.float32)
            expected = np.array(VEC_B, dtype=np.float32)
            assert np.allclose(actual, expected, atol=1e-5), (
                "Rebuild-from-confirmed-samples must win outright, not blend with the legacy VEC_A"
            )
        finally:
            _cleanup(target)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
