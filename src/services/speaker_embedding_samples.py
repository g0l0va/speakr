"""
Speaker Embedding Sample Service.

Companion service to speaker_embedding_matcher.py. That module owns the
single blended Speaker.average_embedding; this one owns the individual
samples that justify it, so the blend is reviewable and reversible instead
of a one-way average.

Design rules this module exists to enforce:

- A MANUAL name assignment (a human looked at a recording and typed a name)
  creates a 'confirmed' sample immediately -- see record_manual_confirmation().
- An AUTOMATIC voice match (the matcher's best guess) creates a 'pending'
  sample and must NEVER touch Speaker.average_embedding on its own -- see
  create_pending_samples_from_recording(). Doing otherwise lets a wrong
  guess retrain the profile, which makes the next guess more likely to be
  wrong too (the feedback-loop drift this module is built to avoid).
- Only 'confirmed' samples are ever averaged into a profile. rebuild_speaker_profile()
  recomputes Speaker.average_embedding/embedding_count/confidence_score
  deterministically from *all* of a speaker's currently-confirmed samples,
  so confirming, rejecting, or deleting one sample and rebuilding is
  reproducible and never loses any other sample's contribution.
- A speaker with zero confirmed samples (nothing has gone through this
  subsystem yet) keeps whatever legacy Speaker.average_embedding it already
  had -- rebuild_speaker_profile() is a no-op in that case. Pre-existing
  profiles are also backfilled as a single confirmed 'import' sample at
  startup (see init_db.py), so old data keeps working and stays rebuildable.
"""

import logging
from datetime import datetime

import numpy as np

from src.database import db
from src.models import Speaker, SpeakerEmbeddingSample, SAMPLE_SOURCES, SAMPLE_REVIEW_STATUSES
from src.services.speaker_embedding_matcher import (
    serialize_embedding,
    deserialize_embedding,
    calculate_similarity,
)

logger = logging.getLogger(__name__)


class SampleOwnershipError(ValueError):
    """Raised when a sample is accessed by a user_id that does not own it."""


class SampleNotFoundError(ValueError):
    """Raised when a sample_id does not exist (or does not belong to the speaker)."""


def _validate_choice(value, allowed, field_name):
    if value not in allowed:
        raise ValueError(f"Invalid {field_name} '{value}'; must be one of {allowed}")


def create_sample(
    speaker,
    embedding,
    *,
    user_id=None,
    recording_id=None,
    source_speaker_label=None,
    source='manual',
    review_status='pending',
    similarity_to_profile=None,
    quality_metadata=None,
    model_name=None,
    model_version=None,
    commit=True,
):
    """
    Create a single SpeakerEmbeddingSample row for *speaker*.

    Args:
        speaker: Speaker model instance this sample belongs to.
        embedding: 256-dim array/list of floats.
        user_id: expected owner of *speaker*; if given, must match
                 speaker.user_id (defense in depth -- callers should already
                 have checked this, but a mismatch here is a bug worth
                 raising loudly rather than silently cross-linking data).
        commit: whether to commit immediately (False lets a caller batch
                this with other writes, e.g. a profile rebuild, in one
                transaction).

    Returns:
        SpeakerEmbeddingSample: the created (session-added) row.
    """
    _validate_choice(source, SAMPLE_SOURCES, 'source')
    _validate_choice(review_status, SAMPLE_REVIEW_STATUSES, 'review_status')

    if user_id is not None and speaker.user_id != user_id:
        raise SampleOwnershipError(
            f"Speaker {speaker.id} belongs to user {speaker.user_id}, not {user_id}"
        )

    emb_array = np.array(embedding, dtype=np.float32)

    sample = SpeakerEmbeddingSample(
        speaker_id=speaker.id,
        user_id=speaker.user_id,
        recording_id=recording_id,
        source_speaker_label=source_speaker_label,
        embedding=serialize_embedding(emb_array),
        embedding_dim=int(emb_array.shape[0]),
        source=source,
        review_status=review_status,
        similarity_to_profile=similarity_to_profile,
        quality_metadata=quality_metadata,
        model_name=model_name,
        model_version=model_version,
    )
    db.session.add(sample)

    if commit:
        db.session.commit()

    return sample


def get_sample_for_user(sample_id, user_id):
    sample = db.session.get(SpeakerEmbeddingSample, sample_id)
    if not sample or sample.user_id != user_id:
        # Same error for "doesn't exist" and "belongs to someone else" --
        # a 404-shaped failure, not a signal that leaks which sample_ids
        # exist for other users.
        raise SampleNotFoundError(f"Sample {sample_id} not found for this user")
    return sample


def get_confirmed_samples(speaker_id):
    """Return a speaker's confirmed SpeakerEmbeddingSample rows, oldest first."""
    return (
        SpeakerEmbeddingSample.query
        .filter_by(speaker_id=speaker_id, review_status='confirmed')
        .order_by(SpeakerEmbeddingSample.created_at.asc())
        .all()
    )


def get_confirmed_sample_embeddings(speaker_id):
    """Return the deserialized embeddings of a speaker's confirmed samples."""
    return [deserialize_embedding(s.embedding) for s in get_confirmed_samples(speaker_id)]


def calculate_confidence_from_samples(embeddings, mean_embedding=None):
    """
    Deterministic confidence score (0-1) from a set of embeddings.

    Same shape as the legacy calculate_confidence() (sample-count factor x
    consistency factor), but computed directly from the sample set instead
    of a truncated similarity-history log, so it is reproducible from
    confirmed samples alone -- rebuilding after a reject/delete recomputes
    this exactly, rather than replaying an incremental history.
    """
    if not embeddings:
        return 0.0
    if len(embeddings) == 1:
        return 0.5

    mean = mean_embedding if mean_embedding is not None else np.mean(np.stack(embeddings), axis=0)
    similarities = [calculate_similarity(e, mean) for e in embeddings]
    avg_similarity = sum(similarities) / len(similarities)

    sample_factor = min(1.0, len(embeddings) / 5.0)
    confidence = avg_similarity * sample_factor
    return min(1.0, max(0.0, confidence))


def rebuild_speaker_profile(speaker, commit=True, clear_if_empty=False):
    """
    Recompute Speaker.average_embedding/embedding_count/confidence_score
    from *all* of the speaker's currently-confirmed samples.

    Args:
        clear_if_empty: what to do when the speaker has zero confirmed
            samples. Default False leaves existing fields untouched -- this
            is what makes legacy, pre-migration profiles safe: a speaker
            that has never had a sample reviewed under this system keeps
            whatever average_embedding it already had. Pass True from a
            caller that just rejected/deleted a sample: a speaker can only
            reach zero confirmed samples that way if it had at least one
            moments ago (you need an existing sample_id to reject/delete),
            so it IS under this subsystem's management, and leaving a stale
            average/confidence pointing at data that no longer counts would
            be wrong -- it must be cleared instead.

    Returns:
        bool: True if the profile was changed (recomputed or cleared),
              False if left untouched.
    """
    embeddings = get_confirmed_sample_embeddings(speaker.id)

    if not embeddings:
        if not clear_if_empty:
            return False
        changed = (
            speaker.average_embedding is not None
            or (speaker.embedding_count or 0) != 0
            or speaker.confidence_score is not None
        )
        speaker.average_embedding = None
        speaker.embedding_count = 0
        speaker.confidence_score = None
        if commit:
            db.session.commit()
        return changed

    mean_embedding = np.mean(np.stack(embeddings), axis=0)

    speaker.average_embedding = serialize_embedding(mean_embedding)
    speaker.embedding_count = len(embeddings)
    speaker.confidence_score = calculate_confidence_from_samples(embeddings, mean_embedding)

    if commit:
        db.session.commit()

    return True


def record_manual_confirmation(
    speaker,
    embedding,
    recording_id,
    *,
    user_id,
    source_speaker_label=None,
    model_name=None,
    model_version=None,
):
    """
    Record a manual name assignment as a confirmed voice sample and rebuild
    the speaker's profile from confirmed samples.

    Drop-in replacement for speaker_embedding_matcher.update_speaker_embedding()
    at manual-assignment call sites: same positional signature, and returns
    the same kind of value (similarity of the new sample to the speaker's
    *previous* profile, or None if this is the speaker's first sample) so
    existing logging/response code that only reads that return value for a
    similarity percentage keeps working unchanged.

    Args:
        user_id: expected owner of *speaker* (raises SampleOwnershipError on
                 mismatch -- manual assignment is always user-scoped).
    """
    previous_avg = speaker.average_embedding
    similarity_to_previous = None
    if previous_avg is not None:
        try:
            similarity_to_previous = calculate_similarity(
                embedding, deserialize_embedding(previous_avg)
            )
        except Exception:
            similarity_to_previous = None

    create_sample(
        speaker,
        embedding,
        user_id=user_id,
        recording_id=recording_id,
        source_speaker_label=source_speaker_label,
        source='manual',
        review_status='confirmed',
        similarity_to_profile=similarity_to_previous,
        model_name=model_name,
        model_version=model_version,
        commit=False,
    )

    rebuild_speaker_profile(speaker, commit=False)
    db.session.commit()

    return similarity_to_previous


def create_pending_samples_from_recording(recording, speaker_map, user, model_name=None, model_version=None):
    """
    Record automatic voice matches from a recording as PENDING samples.

    This is the automatic-labelling counterpart of record_manual_confirmation(),
    and deliberately does much less: it never calls rebuild_speaker_profile()
    and never touches Speaker.average_embedding/embedding_count/
    confidence_score/use_count/last_used. An unreviewed guess must not be
    able to influence future matching -- that would let a single wrong
    auto-match retrain the profile, making the next guess more likely to be
    wrong too. The sample sits as 'pending' until a human confirms or
    rejects it via confirm_sample()/reject_sample().

    Args:
        recording: Recording instance with speaker_embeddings.
        speaker_map: {SPEAKER_XX: speaker_name} mapping that was applied
                     (same shape apply_auto_speaker_labels returns).
        user: owning User instance.

    Returns:
        int: number of pending samples created.
    """
    if not speaker_map or not recording.speaker_embeddings:
        return 0

    embeddings = recording.speaker_embeddings
    created = 0

    for speaker_label, speaker_name in speaker_map.items():
        if speaker_label not in embeddings:
            continue

        embedding_data = embeddings[speaker_label]
        if not embedding_data or not isinstance(embedding_data, list):
            continue

        speaker = Speaker.query.filter_by(user_id=user.id, name=speaker_name).first()
        if not speaker:
            continue

        # Idempotent: re-processing the same recording (retry, reprocess)
        # must not pile up a second pending sample for the same guess. Also
        # backed by a DB-level partial unique index (see
        # SpeakerEmbeddingSample.__table_args__) as a belt-and-suspenders
        # check against races between concurrent processing runs.
        existing_pending = SpeakerEmbeddingSample.query.filter_by(
            speaker_id=speaker.id,
            recording_id=recording.id,
            source_speaker_label=speaker_label,
            review_status='pending',
        ).first()
        if existing_pending:
            continue

        similarity = None
        if speaker.average_embedding is not None:
            try:
                similarity = calculate_similarity(
                    embedding_data, deserialize_embedding(speaker.average_embedding)
                )
            except Exception:
                similarity = None

        try:
            create_sample(
                speaker,
                embedding_data,
                user_id=user.id,
                recording_id=recording.id,
                source_speaker_label=speaker_label,
                source='automatic',
                review_status='pending',
                similarity_to_profile=similarity,
                model_name=model_name,
                model_version=model_version,
                commit=False,
            )
            created += 1
        except Exception as e:
            logger.warning(f"Failed to record pending voice sample for speaker {speaker.id}: {e}")

    if created > 0:
        db.session.commit()

    return created


def confirm_sample(sample_id, user_id, rebuild=True):
    """
    Confirm a pending (or previously-rejected) sample and rebuild the
    speaker's profile from confirmed samples.

    Idempotent: confirming a sample that is already 'confirmed' is a no-op
    on use_count/last_used/reviewed_at (and skips the rebuild by default,
    since nothing changed) -- a retried request or a double click must not
    inflate use_count or repeatedly bump last_used. Confirming is the point
    at which an automatic guess becomes a genuine "this identity was used"
    signal, so use_count/last_used only advance on an actual pending/
    rejected -> confirmed transition.
    """
    sample = get_sample_for_user(sample_id, user_id)
    speaker = sample.speaker

    if sample.review_status == 'confirmed':
        return sample

    sample.review_status = 'confirmed'
    sample.reviewed_at = datetime.utcnow()
    speaker.use_count = (speaker.use_count or 0) + 1
    speaker.last_used = datetime.utcnow()

    if rebuild:
        rebuild_speaker_profile(speaker, commit=False)

    db.session.commit()
    return sample


def reject_sample(sample_id, user_id, rebuild=True):
    """
    Reject a sample (pending or confirmed) and rebuild from what remains.

    clear_if_empty=True: if this was the speaker's last confirmed sample,
    the derived average/confidence must be cleared rather than left
    pointing at a sample that no longer counts (see rebuild_speaker_profile).
    """
    sample = get_sample_for_user(sample_id, user_id)
    sample.review_status = 'rejected'
    sample.reviewed_at = datetime.utcnow()

    if rebuild:
        rebuild_speaker_profile(sample.speaker, commit=False, clear_if_empty=True)

    db.session.commit()
    return sample


def delete_sample(sample_id, user_id, rebuild=True):
    """
    Hard-delete a sample and rebuild the profile from what remains.

    clear_if_empty=True: see reject_sample() -- deleting a speaker's last
    confirmed sample must clear the derived profile, not leave it stale.
    """
    sample = get_sample_for_user(sample_id, user_id)
    speaker = sample.speaker

    db.session.delete(sample)

    if rebuild:
        # Flush the delete before rebuilding so the query behind
        # rebuild_speaker_profile() doesn't see the row being removed.
        db.session.flush()
        rebuild_speaker_profile(speaker, commit=False, clear_if_empty=True)

    db.session.commit()
    return True


def list_samples(speaker_id, user_id, review_status=None):
    """List a speaker's samples, newest first, scoped to *user_id*."""
    speaker = Speaker.query.filter_by(id=speaker_id, user_id=user_id).first()
    if not speaker:
        raise SampleNotFoundError(f"Speaker {speaker_id} not found for this user")

    query = SpeakerEmbeddingSample.query.filter_by(speaker_id=speaker_id)
    if review_status is not None:
        _validate_choice(review_status, SAMPLE_REVIEW_STATUSES, 'review_status')
        query = query.filter_by(review_status=review_status)

    return query.order_by(SpeakerEmbeddingSample.created_at.desc()).all()


def get_sample_counts_by_speaker(speaker_ids):
    """
    Grouped confirmed/pending/rejected sample counts for a list of speakers.

    One query for the whole list, not one per speaker -- callers rendering
    a speaker list/grid (e.g. GET /speakers) should use this instead of
    counting per speaker to avoid an N+1.

    Args:
        speaker_ids: list of speaker ids to count for. Caller is responsible
            for having already scoped this list to the requesting user.

    Returns:
        dict: {speaker_id: {'confirmed': n, 'pending': n, 'rejected': n}},
              with a zeroed entry for every id in speaker_ids even if it has
              no samples at all.
    """
    from sqlalchemy import func

    counts = {sid: {'confirmed': 0, 'pending': 0, 'rejected': 0} for sid in speaker_ids}
    if not speaker_ids:
        return counts

    rows = (
        db.session.query(
            SpeakerEmbeddingSample.speaker_id,
            SpeakerEmbeddingSample.review_status,
            func.count(SpeakerEmbeddingSample.id),
        )
        .filter(SpeakerEmbeddingSample.speaker_id.in_(speaker_ids))
        .group_by(SpeakerEmbeddingSample.speaker_id, SpeakerEmbeddingSample.review_status)
        .all()
    )

    for speaker_id, review_status, count in rows:
        counts.setdefault(speaker_id, {'confirmed': 0, 'pending': 0, 'rejected': 0})[review_status] = count

    return counts
