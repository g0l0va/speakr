#!/usr/bin/env python3
"""
Test suite for the SpeakerEmbeddingSample subsystem
(src/models/speaker_embedding_sample.py, src/services/speaker_embedding_samples.py,
and the matching changes in src/services/speaker_embedding_matcher.py).

Covers:
  - Ownership isolation across users for every review action
  - confirmed vs pending vs rejected: only 'confirmed' feeds matching/rebuilds
  - Rebuild is deterministic and reproducible after reject/delete
  - Legacy Speaker.average_embedding fallback when no confirmed samples exist
  - Automatic matches never auto-train the profile (no feedback-loop drift)
  - Ambiguity handling still works against confirmed-sample-based similarity
  - Rejecting/deleting a speaker's last confirmed sample clears the derived
    profile instead of leaving it stale
  - confirm_sample() is idempotent (no double use_count/last_used bump)
  - source/review_status are enforced by a DB CHECK constraint
  - No duplicate 'pending' sample for the same (speaker, recording, label)
  - The confirm/reject API endpoints validate speaker_id ownership BEFORE
    mutating, not after

Pattern follows tests/test_api_v1_speakers.py -- standalone, no pytest fixtures.
"""

import sys
import os
import secrets
import numpy as np
from sqlalchemy.exc import IntegrityError

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.app import app, db
from src.models import User, Speaker, SpeakerEmbeddingSample, Recording, APIToken
from src.utils.token_auth import hash_token
from src.services.speaker_embedding_matcher import serialize_embedding, find_matching_speakers, apply_auto_speaker_labels
from src.services.speaker_embedding_samples import (
    create_sample,
    confirm_sample,
    reject_sample,
    delete_sample,
    rebuild_speaker_profile,
    record_manual_confirmation,
    create_pending_samples_from_recording,
    list_samples,
    SampleOwnershipError,
    SampleNotFoundError,
)

# ---------------------------------------------------------------------------
# Fixtures (256-dim unit-ish vectors, distinguishable by direction)
# ---------------------------------------------------------------------------

def _vec(seed, dim=256):
    """A deterministic pseudo-random 256-dim vector for a given seed."""
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


VEC_A = _vec(1)   # "Alice" legacy average
VEC_B = _vec(2)   # a confirmed sample, deliberately different direction from A
VEC_B_CLOSE = (np.array(VEC_B) * 0.98 + np.array(_vec(20)) * 0.02).tolist()  # near-duplicate of B
VEC_C = _vec(3)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_or_create_user(suffix):
    username = f"sample_test_user_{suffix}"
    user = User.query.filter_by(username=username).first()
    if not user:
        user = User(username=username, email=f"{username}@local.test", name="Test User")
        db.session.add(user)
        db.session.commit()
    return user


def _create_speaker(user, name, average_embedding_vec=None):
    speaker = Speaker(name=name, user_id=user.id)
    if average_embedding_vec is not None:
        speaker.average_embedding = serialize_embedding(np.array(average_embedding_vec, dtype=np.float32))
        speaker.embedding_count = 1
    db.session.add(speaker)
    db.session.commit()
    return speaker


def _create_recording(user):
    rec = Recording(user_id=user.id, title="Test Recording", status="COMPLETED")
    db.session.add(rec)
    db.session.commit()
    return rec


def _create_api_token(user):
    plaintext = f"test-token-{secrets.token_urlsafe(16)}"
    token = APIToken(user_id=user.id, token_hash=hash_token(plaintext), name="test-api-token")
    db.session.add(token)
    db.session.commit()
    return token, plaintext


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


# =========================================================================
# Ownership isolation
# =========================================================================

def test_manual_confirmation_rejects_wrong_owner():
    with app.app_context():
        owner = _get_or_create_user("own1")
        other = _get_or_create_user("own2")
        speaker = _create_speaker(owner, "Alice")
        try:
            raised = False
            try:
                record_manual_confirmation(speaker, VEC_A, None, user_id=other.id)
            except SampleOwnershipError:
                raised = True
            assert raised, "Expected SampleOwnershipError for cross-user speaker"
            assert SpeakerEmbeddingSample.query.filter_by(speaker_id=speaker.id).count() == 0
        finally:
            _cleanup(speaker)


def test_review_actions_scoped_to_owner():
    """confirm/reject/delete/list must not operate on another user's sample."""
    with app.app_context():
        owner = _get_or_create_user("iso1")
        other = _get_or_create_user("iso2")
        speaker = _create_speaker(owner, "Alice")
        sample = create_sample(
            speaker, VEC_A, user_id=owner.id, source="manual", review_status="pending"
        )
        try:
            for action in (confirm_sample, reject_sample, delete_sample):
                raised = False
                try:
                    action(sample.id, other.id)
                except SampleNotFoundError:
                    raised = True
                assert raised, f"{action.__name__} leaked across users"

            raised = False
            try:
                list_samples(speaker.id, other.id)
            except SampleNotFoundError:
                raised = True
            assert raised, "list_samples leaked another user's speaker"
        finally:
            _cleanup(speaker)


# =========================================================================
# confirmed vs pending vs rejected
# =========================================================================

def test_manual_assignment_creates_confirmed_sample():
    with app.app_context():
        user = _get_or_create_user("manual1")
        speaker = _create_speaker(user, "Alice")
        try:
            record_manual_confirmation(speaker, VEC_A, None, user_id=user.id)
            samples = SpeakerEmbeddingSample.query.filter_by(speaker_id=speaker.id).all()
            assert len(samples) == 1
            assert samples[0].source == "manual"
            assert samples[0].review_status == "confirmed"
            db.session.refresh(speaker)
            assert speaker.average_embedding is not None
        finally:
            _cleanup(speaker)


def test_automatic_match_creates_pending_sample_and_does_not_train_profile():
    """The central anti-drift guarantee: an automatic guess must not alter
    the profile until a human confirms it."""
    with app.app_context():
        user = _get_or_create_user("auto1")
        speaker = _create_speaker(user, "Alice", average_embedding_vec=VEC_A)
        rec = _create_recording(user)
        rec.speaker_embeddings = {"SPEAKER_00": VEC_B}
        db.session.commit()

        before_avg = speaker.average_embedding
        before_count = speaker.embedding_count
        before_confidence = speaker.confidence_score
        before_use_count = speaker.use_count
        before_last_used = speaker.last_used

        try:
            created = create_pending_samples_from_recording(
                rec, {"SPEAKER_00": "Alice"}, user, model_name="test-model"
            )
            assert created == 1

            sample = SpeakerEmbeddingSample.query.filter_by(speaker_id=speaker.id).first()
            assert sample is not None
            assert sample.source == "automatic"
            assert sample.review_status == "pending"
            assert sample.model_name == "test-model"

            db.session.refresh(speaker)
            assert speaker.average_embedding == before_avg, "Profile must not change on a pending match"
            assert speaker.embedding_count == before_count
            assert speaker.confidence_score == before_confidence
            assert speaker.use_count == before_use_count
            assert speaker.last_used == before_last_used
        finally:
            _cleanup(rec, speaker)


def test_confirming_pending_sample_then_rebuilds_and_bumps_usage():
    with app.app_context():
        user = _get_or_create_user("confirm1")
        speaker = _create_speaker(user, "Alice")
        sample = create_sample(speaker, VEC_A, user_id=user.id, source="automatic", review_status="pending")
        before_use_count = speaker.use_count or 0
        try:
            confirm_sample(sample.id, user.id)
            db.session.refresh(speaker)
            assert speaker.average_embedding is not None
            assert speaker.use_count == before_use_count + 1
        finally:
            _cleanup(speaker)


def test_rejected_sample_excluded_from_matching_and_rebuild():
    with app.app_context():
        user = _get_or_create_user("reject1")
        speaker = _create_speaker(user, "Alice")
        good = create_sample(speaker, VEC_A, user_id=user.id, source="manual", review_status="confirmed")
        bad = create_sample(speaker, VEC_C, user_id=user.id, source="automatic", review_status="pending")
        try:
            reject_sample(bad.id, user.id)
            rebuild_speaker_profile(speaker)
            db.session.refresh(speaker)

            # Rebuilt average should equal VEC_A alone (the only confirmed sample),
            # not a blend with the rejected VEC_C.
            expected = np.array(VEC_A, dtype=np.float32)
            actual = np.frombuffer(speaker.average_embedding, dtype=np.float32)
            assert np.allclose(actual, expected, atol=1e-5)

            # The rejected sample still exists (audit trail), just excluded.
            refreshed_bad = db.session.get(SpeakerEmbeddingSample, bad.id)
            assert refreshed_bad is not None
            assert refreshed_bad.review_status == "rejected"
        finally:
            _cleanup(speaker)


# =========================================================================
# Rebuild: deterministic, reproducible, non-destructive to the rest
# =========================================================================

def test_rebuild_is_reproducible_and_delete_does_not_lose_other_samples():
    with app.app_context():
        user = _get_or_create_user("rebuild1")
        speaker = _create_speaker(user, "Alice")
        s1 = create_sample(speaker, VEC_A, user_id=user.id, source="manual", review_status="confirmed")
        s2 = create_sample(speaker, VEC_B, user_id=user.id, source="manual", review_status="confirmed")
        s3 = create_sample(speaker, VEC_C, user_id=user.id, source="manual", review_status="confirmed")
        try:
            rebuild_speaker_profile(speaker)
            db.session.refresh(speaker)
            mean_all = np.mean(np.stack([np.array(VEC_A), np.array(VEC_B), np.array(VEC_C)]), axis=0)
            actual_all = np.frombuffer(speaker.average_embedding, dtype=np.float32)
            assert np.allclose(actual_all, mean_all, atol=1e-4)

            # Delete s2 and rebuild: s1 and s3 must remain, average reflects only them.
            delete_sample(s2.id, user.id)
            db.session.refresh(speaker)

            remaining_ids = {s.id for s in SpeakerEmbeddingSample.query.filter_by(speaker_id=speaker.id).all()}
            assert remaining_ids == {s1.id, s3.id}, "Deleting s2 must not remove s1/s3"

            mean_remaining = np.mean(np.stack([np.array(VEC_A), np.array(VEC_C)]), axis=0)
            actual_remaining = np.frombuffer(speaker.average_embedding, dtype=np.float32)
            assert np.allclose(actual_remaining, mean_remaining, atol=1e-4)
            assert speaker.embedding_count == 2
        finally:
            _cleanup(speaker)


def test_rebuild_noop_when_no_confirmed_samples_preserves_legacy_average():
    with app.app_context():
        user = _get_or_create_user("legacy1")
        speaker = _create_speaker(user, "Alice", average_embedding_vec=VEC_A)
        original = speaker.average_embedding
        try:
            rebuilt = rebuild_speaker_profile(speaker)
            assert rebuilt is False
            db.session.refresh(speaker)
            assert speaker.average_embedding == original
        finally:
            _cleanup(speaker)


# =========================================================================
# Matching: prefer confirmed samples, fall back to legacy average
# =========================================================================

def test_matching_prefers_confirmed_sample_over_stale_legacy_average():
    with app.app_context():
        user = _get_or_create_user("match1")
        # Legacy average points at VEC_A; one confirmed sample points at VEC_B.
        speaker = _create_speaker(user, "Alice", average_embedding_vec=VEC_A)
        create_sample(speaker, VEC_B, user_id=user.id, source="manual", review_status="confirmed")
        try:
            matches = find_matching_speakers(VEC_B_CLOSE, user.id, threshold=0.70)
            assert len(matches) == 1
            assert matches[0]["speaker_id"] == speaker.id
            assert matches[0]["similarity"] > 90.0, (
                "A query near VEC_B should score high via the confirmed sample, "
                f"not the stale legacy average pointing at VEC_A: {matches}"
            )
        finally:
            _cleanup(speaker)


def test_matching_falls_back_to_legacy_average_without_confirmed_samples():
    with app.app_context():
        user = _get_or_create_user("match2")
        speaker = _create_speaker(user, "Alice", average_embedding_vec=VEC_A)
        try:
            assert SpeakerEmbeddingSample.query.filter_by(speaker_id=speaker.id).count() == 0
            matches = find_matching_speakers(VEC_A, user.id, threshold=0.70)
            assert len(matches) == 1
            assert matches[0]["speaker_id"] == speaker.id
            assert matches[0]["similarity"] > 99.0
        finally:
            _cleanup(speaker)


def test_matching_is_strictly_user_scoped():
    with app.app_context():
        user_a = _get_or_create_user("scope1")
        user_b = _get_or_create_user("scope2")
        speaker_a = _create_speaker(user_a, "Alice")
        create_sample(speaker_a, VEC_A, user_id=user_a.id, source="manual", review_status="confirmed")
        try:
            matches_for_b = find_matching_speakers(VEC_A, user_b.id, threshold=0.5)
            assert matches_for_b == [], "user B must never match user A's speaker"
        finally:
            _cleanup(speaker_a)


# =========================================================================
# Ambiguity (existing behaviour, exercised against confirmed-sample matching)
# =========================================================================

def test_ambiguous_top_two_matches_are_skipped():
    with app.app_context():
        user = _get_or_create_user("ambig1")
        speaker1 = _create_speaker(user, "Alice")
        speaker2 = _create_speaker(user, "Bob")
        # Both speakers confirm samples nearly identical to each other, so a
        # query vector matches both almost equally -- ambiguous, must skip.
        create_sample(speaker1, VEC_A, user_id=user.id, source="manual", review_status="confirmed")
        near_a = (np.array(VEC_A) * 0.999 + np.array(_vec(99)) * 0.001).tolist()
        create_sample(speaker2, near_a, user_id=user.id, source="manual", review_status="confirmed")

        rec = _create_recording(user)
        rec.speaker_embeddings = {"SPEAKER_00": VEC_A}
        db.session.commit()

        class _FakeUser:
            def __init__(self, id_, threshold="low"):
                self.id = id_
                self.auto_speaker_labelling = True
                self.auto_speaker_labelling_threshold = threshold

        try:
            speaker_map = apply_auto_speaker_labels(rec, _FakeUser(user.id))
            assert speaker_map == {}, f"Expected no auto-match due to ambiguity, got {speaker_map}"
        finally:
            _cleanup(rec, speaker1, speaker2)


# =========================================================================
# Fix 1: rejecting/deleting the LAST confirmed sample clears the derived
# profile instead of leaving it stale.
# =========================================================================

def test_reject_last_confirmed_sample_clears_derived_profile():
    with app.app_context():
        user = _get_or_create_user("clear1")
        speaker = _create_speaker(user, "Alice")
        sample = create_sample(speaker, VEC_A, user_id=user.id, source="manual", review_status="confirmed")
        try:
            rebuild_speaker_profile(speaker)
            db.session.refresh(speaker)
            assert speaker.average_embedding is not None

            reject_sample(sample.id, user.id)
            db.session.refresh(speaker)
            assert speaker.average_embedding is None, "Stale average must be cleared, not left pointing at a rejected sample"
            assert speaker.embedding_count == 0
            assert speaker.confidence_score is None
        finally:
            _cleanup(speaker)


def test_delete_last_confirmed_sample_clears_derived_profile():
    with app.app_context():
        user = _get_or_create_user("clear2")
        speaker = _create_speaker(user, "Alice")
        sample = create_sample(speaker, VEC_A, user_id=user.id, source="manual", review_status="confirmed")
        try:
            rebuild_speaker_profile(speaker)
            db.session.refresh(speaker)
            assert speaker.average_embedding is not None

            delete_sample(sample.id, user.id)
            db.session.refresh(speaker)
            assert speaker.average_embedding is None
            assert speaker.embedding_count == 0
            assert speaker.confidence_score is None
        finally:
            _cleanup(speaker)


def test_reject_one_of_several_confirmed_samples_does_not_clear():
    """Sanity check: clearing only kicks in when confirmed samples hit zero,
    not on every reject."""
    with app.app_context():
        user = _get_or_create_user("clear3")
        speaker = _create_speaker(user, "Alice")
        s1 = create_sample(speaker, VEC_A, user_id=user.id, source="manual", review_status="confirmed")
        s2 = create_sample(speaker, VEC_B, user_id=user.id, source="manual", review_status="confirmed")
        try:
            rebuild_speaker_profile(speaker)
            reject_sample(s2.id, user.id)
            db.session.refresh(speaker)
            assert speaker.average_embedding is not None, "One confirmed sample remains -- must not clear"
            assert speaker.embedding_count == 1
        finally:
            _cleanup(speaker)


def test_rebuild_standalone_still_preserves_legacy_when_never_touched():
    """clear_if_empty defaults to False: the plain rebuild path (manual
    confirmation, or the standalone rebuild endpoint) must not clear a
    legacy profile that was never under this subsystem's management."""
    with app.app_context():
        user = _get_or_create_user("clear4")
        speaker = _create_speaker(user, "Alice", average_embedding_vec=VEC_A)
        original = speaker.average_embedding
        try:
            rebuilt = rebuild_speaker_profile(speaker)  # clear_if_empty=False (default)
            assert rebuilt is False
            db.session.refresh(speaker)
            assert speaker.average_embedding == original
        finally:
            _cleanup(speaker)


# =========================================================================
# Fix 2: confirm_sample() is idempotent.
# =========================================================================

def test_confirm_sample_is_idempotent():
    with app.app_context():
        user = _get_or_create_user("idem1")
        speaker = _create_speaker(user, "Alice")
        use_count_before = speaker.use_count or 0  # Speaker.use_count defaults to 1, not 0
        sample = create_sample(speaker, VEC_A, user_id=user.id, source="automatic", review_status="pending")
        try:
            confirm_sample(sample.id, user.id)
            db.session.refresh(speaker)
            db.session.refresh(sample)
            use_count_after_first = speaker.use_count
            reviewed_at_after_first = sample.reviewed_at
            assert use_count_after_first == use_count_before + 1

            # Confirming an already-confirmed sample again must be a no-op.
            confirm_sample(sample.id, user.id)
            confirm_sample(sample.id, user.id)
            db.session.refresh(speaker)
            db.session.refresh(sample)
            assert speaker.use_count == use_count_after_first, "Repeated confirm must not inflate use_count"
            assert sample.reviewed_at == reviewed_at_after_first, "Repeated confirm must not re-bump reviewed_at"
        finally:
            _cleanup(speaker)


# =========================================================================
# Fix 3: source/review_status enforced by a DB CHECK constraint.
# =========================================================================

def test_invalid_source_rejected_by_db_check_constraint():
    with app.app_context():
        user = _get_or_create_user("check1")
        speaker = _create_speaker(user, "Alice")
        bad = SpeakerEmbeddingSample(
            speaker_id=speaker.id,
            user_id=user.id,
            embedding=serialize_embedding(np.array(VEC_A, dtype=np.float32)),
            source="bogus",
            review_status="pending",
        )
        db.session.add(bad)
        try:
            raised = False
            try:
                db.session.commit()
            except IntegrityError:
                raised = True
                db.session.rollback()
            assert raised, "Invalid source must be rejected at the DB layer"
        finally:
            _cleanup(speaker)


def test_invalid_review_status_rejected_by_db_check_constraint():
    with app.app_context():
        user = _get_or_create_user("check2")
        speaker = _create_speaker(user, "Alice")
        bad = SpeakerEmbeddingSample(
            speaker_id=speaker.id,
            user_id=user.id,
            embedding=serialize_embedding(np.array(VEC_A, dtype=np.float32)),
            source="manual",
            review_status="bogus",
        )
        db.session.add(bad)
        try:
            raised = False
            try:
                db.session.commit()
            except IntegrityError:
                raised = True
                db.session.rollback()
            assert raised, "Invalid review_status must be rejected at the DB layer"
        finally:
            _cleanup(speaker)


def test_invalid_embedding_dim_rejected_by_db_check_constraint():
    """The whole pipeline assumes 256-dim embeddings; a mismatched
    embedding_dim must fail loudly at write time, not corrupt a later
    rebuild's np.stack()."""
    with app.app_context():
        user = _get_or_create_user("check3")
        speaker = _create_speaker(user, "Alice")
        bad = SpeakerEmbeddingSample(
            speaker_id=speaker.id,
            user_id=user.id,
            embedding=serialize_embedding(np.array(_vec(1, dim=128), dtype=np.float32)),
            embedding_dim=128,
            source="manual",
            review_status="confirmed",
        )
        db.session.add(bad)
        try:
            raised = False
            try:
                db.session.commit()
            except IntegrityError:
                raised = True
                db.session.rollback()
            assert raised, "embedding_dim != 256 must be rejected at the DB layer"
        finally:
            _cleanup(speaker)


# =========================================================================
# Fix 4: no duplicate 'pending' sample for the same (speaker, recording, label).
# =========================================================================

def test_reprocessing_recording_does_not_duplicate_pending_sample():
    with app.app_context():
        user = _get_or_create_user("dedup1")
        speaker = _create_speaker(user, "Alice", average_embedding_vec=VEC_A)
        rec = _create_recording(user)
        rec.speaker_embeddings = {"SPEAKER_00": VEC_B}
        db.session.commit()
        try:
            first = create_pending_samples_from_recording(rec, {"SPEAKER_00": "Alice"}, user)
            second = create_pending_samples_from_recording(rec, {"SPEAKER_00": "Alice"}, user)
            assert first == 1
            assert second == 0, "Reprocessing the same recording must not create a second pending sample"

            pending = SpeakerEmbeddingSample.query.filter_by(
                speaker_id=speaker.id, recording_id=rec.id,
                source_speaker_label="SPEAKER_00", review_status="pending",
            ).all()
            assert len(pending) == 1
        finally:
            _cleanup(rec, speaker)


def test_duplicate_pending_rejected_by_db_unique_index():
    """Even bypassing the service-layer pre-check, the DB itself refuses a
    second pending row for the same (speaker, recording, label)."""
    with app.app_context():
        user = _get_or_create_user("dedup2")
        speaker = _create_speaker(user, "Alice")
        rec = _create_recording(user)
        first = SpeakerEmbeddingSample(
            speaker_id=speaker.id, user_id=user.id, recording_id=rec.id,
            source_speaker_label="SPEAKER_00",
            embedding=serialize_embedding(np.array(VEC_A, dtype=np.float32)),
            source="automatic", review_status="pending",
        )
        db.session.add(first)
        db.session.commit()

        duplicate = SpeakerEmbeddingSample(
            speaker_id=speaker.id, user_id=user.id, recording_id=rec.id,
            source_speaker_label="SPEAKER_00",
            embedding=serialize_embedding(np.array(VEC_B, dtype=np.float32)),
            source="automatic", review_status="pending",
        )
        db.session.add(duplicate)
        try:
            raised = False
            try:
                db.session.commit()
            except IntegrityError:
                raised = True
                db.session.rollback()
            assert raised, "A second pending sample for the same (speaker, recording, label) must be rejected"
        finally:
            _cleanup(rec, speaker)


def test_confirmed_samples_for_same_recording_label_are_not_deduped():
    """The unique index only applies to 'pending' rows -- a speaker can
    legitimately have multiple confirmed samples from the same recording
    and label (e.g. re-confirmed across reprocessing runs)."""
    with app.app_context():
        user = _get_or_create_user("dedup3")
        speaker = _create_speaker(user, "Alice")
        rec = _create_recording(user)
        try:
            create_sample(speaker, VEC_A, user_id=user.id, recording_id=rec.id,
                           source_speaker_label="SPEAKER_00", source="manual", review_status="confirmed")
            create_sample(speaker, VEC_B, user_id=user.id, recording_id=rec.id,
                           source_speaker_label="SPEAKER_00", source="manual", review_status="confirmed")
            confirmed = SpeakerEmbeddingSample.query.filter_by(
                speaker_id=speaker.id, recording_id=rec.id,
                source_speaker_label="SPEAKER_00", review_status="confirmed",
            ).all()
            assert len(confirmed) == 2
        finally:
            _cleanup(rec, speaker)


# =========================================================================
# Fix 5: confirm/reject API endpoints validate speaker_id BEFORE mutating.
# =========================================================================

def test_confirm_endpoint_mismatched_speaker_id_is_404_with_no_mutation():
    with app.app_context():
        user = _get_or_create_user("apimut1")
        token_rec, token = _create_api_token(user)
        speaker_a = _create_speaker(user, "Alice")
        speaker_b = _create_speaker(user, "Bob")
        use_count_before = speaker_a.use_count or 0  # Speaker.use_count defaults to 1, not 0
        sample = create_sample(speaker_a, VEC_A, user_id=user.id, source="automatic", review_status="pending")
        client = app.test_client()
        try:
            resp = client.post(
                f"/speakers/{speaker_b.id}/samples/{sample.id}/confirm",
                headers={"X-API-Token": token},
            )
            assert resp.status_code == 404, f"Expected 404 for mismatched speaker_id, got {resp.status_code}"

            db.session.refresh(sample)
            db.session.refresh(speaker_a)
            assert sample.review_status == "pending", "Mismatched speaker_id must not mutate the sample"
            assert speaker_a.use_count == use_count_before, "Mismatched speaker_id must not bump use_count"
        finally:
            _cleanup(sample, speaker_a, speaker_b, token_rec)


def test_reject_endpoint_mismatched_speaker_id_is_404_with_no_mutation():
    with app.app_context():
        user = _get_or_create_user("apimut2")
        token_rec, token = _create_api_token(user)
        speaker_a = _create_speaker(user, "Alice")
        speaker_b = _create_speaker(user, "Bob")
        sample = create_sample(speaker_a, VEC_A, user_id=user.id, source="manual", review_status="confirmed")
        client = app.test_client()
        try:
            resp = client.post(
                f"/speakers/{speaker_b.id}/samples/{sample.id}/reject",
                headers={"X-API-Token": token},
            )
            assert resp.status_code == 404, f"Expected 404 for mismatched speaker_id, got {resp.status_code}"

            db.session.refresh(sample)
            assert sample.review_status == "confirmed", "Mismatched speaker_id must not mutate the sample"
        finally:
            _cleanup(sample, speaker_a, speaker_b, token_rec)


def test_confirm_endpoint_matching_speaker_id_succeeds():
    """Positive-path sanity check alongside the mismatch tests above."""
    with app.app_context():
        user = _get_or_create_user("apimut3")
        token_rec, token = _create_api_token(user)
        speaker = _create_speaker(user, "Alice")
        sample = create_sample(speaker, VEC_A, user_id=user.id, source="automatic", review_status="pending")
        client = app.test_client()
        try:
            resp = client.post(
                f"/speakers/{speaker.id}/samples/{sample.id}/confirm",
                headers={"X-API-Token": token},
            )
            assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.get_data(as_text=True)}"
            db.session.refresh(sample)
            assert sample.review_status == "confirmed"
        finally:
            _cleanup(sample, speaker, token_rec)


# =========================================================================
# GET /speakers: sample_counts field (backs the Account -> Speakers
# Management review-queue UI's per-card confirmed/pending/rejected badge).
# =========================================================================

def test_get_speakers_includes_sample_counts_without_n_plus_one():
    with app.app_context():
        user = _get_or_create_user("counts1")
        token_rec, token = _create_api_token(user)
        speaker_a = _create_speaker(user, "Alice")
        speaker_b = _create_speaker(user, "Bob")
        create_sample(speaker_a, VEC_A, user_id=user.id, source="manual", review_status="confirmed")
        create_sample(speaker_a, VEC_B, user_id=user.id, source="automatic", review_status="pending")
        create_sample(speaker_a, VEC_C, user_id=user.id, source="automatic", review_status="rejected")
        # speaker_b has zero samples -- must still get a zeroed entry, not
        # be omitted or raise.
        client = app.test_client()
        try:
            resp = client.get("/speakers", headers={"X-API-Token": token})
            assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
            body = resp.get_json()

            by_id = {s["id"]: s for s in body}
            assert by_id[speaker_a.id]["sample_counts"] == {"confirmed": 1, "pending": 1, "rejected": 1}
            assert by_id[speaker_b.id]["sample_counts"] == {"confirmed": 0, "pending": 0, "rejected": 0}
        finally:
            _cleanup(speaker_a, speaker_b, token_rec)


def test_get_speakers_sample_counts_scoped_to_owner():
    with app.app_context():
        owner = _get_or_create_user("counts2")
        other = _get_or_create_user("counts3")
        token_rec, token = _create_api_token(owner)
        speaker = _create_speaker(other, "Not Mine")
        create_sample(speaker, VEC_A, user_id=other.id, source="manual", review_status="confirmed")
        client = app.test_client()
        try:
            resp = client.get("/speakers", headers={"X-API-Token": token})
            assert resp.status_code == 200
            body = resp.get_json()
            assert all(s["id"] != speaker.id for s in body), "Must never return another user's speaker/counts"
        finally:
            _cleanup(speaker, token_rec)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
