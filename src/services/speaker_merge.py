"""
Speaker Merge Service.

This service handles merging multiple speaker profiles into one.
Useful when users accidentally create duplicate speakers for the same person.

When speakers are merged:
- Individual SpeakerEmbeddingSample rows are re-parented from every source
  onto the target (never cascade-deleted -- see _transfer_embedding_samples).
- The target's average_embedding/embedding_count/confidence_score are then
  rebuilt from ITS confirmed samples alone (now including every confirmed
  sample re-parented from the sources). Only when that rebuild finds zero
  confirmed samples anywhere in the merge (a legacy/pre-backfill profile
  with a raw average_embedding but no sample rows) does the old
  weighted-average blend of the raw blobs run as a fallback, so no legacy
  data is silently dropped.
- All snippets are transferred to the target speaker.
- Usage statistics are combined.
- Source speakers are deleted.
"""

from datetime import datetime

import numpy as np
from src.database import db
from src.models import Speaker, SpeakerSnippet, SpeakerEmbeddingSample
from src.services.speaker_embedding_matcher import (
    serialize_embedding,
    deserialize_embedding,
    calculate_confidence
)
from src.services.speaker_embedding_samples import rebuild_speaker_profile


def merge_speakers(target_id, source_ids, user_id):
    """
    Merge multiple speaker profiles into one target speaker.

    All embeddings, snippets, and usage data from source speakers are
    combined into the target speaker. Source speakers are then deleted.

    Args:
        target_id: ID of the speaker to keep (receives all merged data)
        source_ids: List of speaker IDs to merge into target
        user_id: ID of the user (for security check)

    Returns:
        Speaker: The updated target speaker

    Raises:
        ValueError: If speakers don't exist or don't belong to user
    """
    # Validate target speaker
    target = Speaker.query.filter_by(id=target_id, user_id=user_id).first()
    if not target:
        raise ValueError(f"Target speaker {target_id} not found or doesn't belong to user")

    # Validate source speakers
    sources = Speaker.query.filter(
        Speaker.id.in_(source_ids),
        Speaker.user_id == user_id
    ).all()

    if len(sources) == 0:
        raise ValueError("No valid source speakers found")

    if len(sources) != len(source_ids):
        raise ValueError("Some source speakers don't exist or don't belong to user")

    # Can't merge a speaker with itself
    if target_id in source_ids:
        raise ValueError("Cannot merge a speaker with itself")

    # Re-parent every individual voice sample from the sources onto the
    # target BEFORE anything else touches embeddings or deletes a source --
    # this must run first so no sample is ever cascade-deleted (see
    # _transfer_embedding_samples' own docstring for why) and so the
    # confirmed-sample rebuild right after sees the fully merged set.
    _transfer_embedding_samples(target, sources)

    # Prefer a clean, deterministic rebuild from confirmed samples (now
    # including every confirmed sample re-parented from the sources above).
    # Only fall back to the legacy weighted-average blend of raw
    # average_embedding blobs when NEITHER the target nor any source has a
    # single confirmed sample anywhere -- a legacy/pre-backfill edge case
    # where average_embedding carries data no sample row represents yet.
    rebuilt_from_samples = rebuild_speaker_profile(target, commit=False)
    if not rebuilt_from_samples:
        _combine_embeddings(target, sources)

    # Transfer snippets
    for source in sources:
        SpeakerSnippet.query.filter_by(speaker_id=source.id).update(
            {'speaker_id': target_id}
        )

    # Combine usage statistics
    for source in sources:
        target.use_count += source.use_count

        # Update last_used to most recent
        if source.last_used and (not target.last_used or source.last_used > target.last_used):
            target.last_used = source.last_used

        # Combine embedding histories
        if source.embeddings_history:
            target_history = target.embeddings_history or []
            source_history = source.embeddings_history or []
            combined_history = target_history + source_history

            # Sort by timestamp (most recent last) and keep last 10
            try:
                combined_history.sort(key=lambda x: x.get('timestamp', ''))
                target.embeddings_history = combined_history[-10:]
            except:
                # If sorting fails, just concatenate and truncate
                target.embeddings_history = (target_history + source_history)[-10:]

    # rebuild_speaker_profile() already set confidence_score correctly (from
    # confirmed samples) when it ran. Only recompute via the legacy
    # history-based calculation when the legacy blend path was used instead.
    if not rebuilt_from_samples:
        target.confidence_score = calculate_confidence(target)

    # Delete source speakers. Safe: _transfer_embedding_samples() already
    # flushed every sample re-parent/status change to the DB above, so each
    # source has zero remaining speaker_embedding_sample rows by the time
    # this cascade is evaluated.
    for source in sources:
        db.session.delete(source)

    # Commit all changes
    db.session.commit()

    return target


def _transfer_embedding_samples(target, sources):
    """
    Re-parent every SpeakerEmbeddingSample row from *sources* onto *target*.

    Confirmed and rejected samples can never collide (the partial unique
    index only covers review_status='pending'), so they're re-parented in a
    single bulk UPDATE -- the same proven pattern already used below for
    SpeakerSnippet, which sidesteps SQLAlchemy's delete-orphan cascade
    entirely by writing the FK directly rather than mutating loaded ORM
    objects' relationship collections.

    Pending samples are handled one at a time: a source and the target (or
    two sources in the same merge) can each hold an independent pending
    guess for the exact same (recording_id, source_speaker_label), which
    would violate the partial unique index once both land on target.id.
    Resolved deterministically -- the newer sample (by created_at, then id
    as a tiebreak) stays 'pending'; the other is marked 'rejected' (
    superseded by the merge, not silently deleted, so the audit trail
    survives) before being re-parented too. Session autoflush means each
    iteration's query sees every earlier iteration's re-parent/rejection in
    this same call, so this also resolves chains across more than two
    colliding samples (e.g. two sources both duplicating the target).

    Finally flushes explicitly so every change above is written to the DB
    before the caller deletes any source speaker -- otherwise SQLAlchemy's
    delete-orphan cascade (declared on Speaker.embedding_samples) could
    still find a sample via a stale in-memory collection and delete it
    despite the FK having "already" been reassigned in Python.
    """
    source_ids = [s.id for s in sources]

    db.session.query(SpeakerEmbeddingSample).filter(
        SpeakerEmbeddingSample.speaker_id.in_(source_ids),
        SpeakerEmbeddingSample.review_status != 'pending',
    ).update({SpeakerEmbeddingSample.speaker_id: target.id}, synchronize_session=False)

    pending_from_sources = SpeakerEmbeddingSample.query.filter(
        SpeakerEmbeddingSample.speaker_id.in_(source_ids),
        SpeakerEmbeddingSample.review_status == 'pending',
    ).all()

    for sample in pending_from_sources:
        existing = SpeakerEmbeddingSample.query.filter_by(
            speaker_id=target.id,
            recording_id=sample.recording_id,
            source_speaker_label=sample.source_speaker_label,
            review_status='pending',
        ).first()

        if existing is not None:
            incoming_key = (sample.created_at, sample.id)
            existing_key = (existing.created_at, existing.id)
            loser = existing if incoming_key > existing_key else sample
            loser.review_status = 'rejected'
            loser.reviewed_at = datetime.utcnow()

        sample.speaker_id = target.id

    db.session.flush()


def _combine_embeddings(target, sources):
    """
    Combine embeddings from multiple speakers using weighted average.

    Weight is based on embedding_count (more samples = more weight).

    Args:
        target: Target Speaker instance
        sources: List of source Speaker instances
    """
    all_embeddings = []
    all_counts = []

    # Add target's embedding if it exists
    if target.average_embedding:
        all_embeddings.append(deserialize_embedding(target.average_embedding))
        all_counts.append(target.embedding_count or 1)

    # Add all source embeddings
    for source in sources:
        if source.average_embedding:
            all_embeddings.append(deserialize_embedding(source.average_embedding))
            all_counts.append(source.embedding_count or 1)

    if not all_embeddings:
        # No embeddings to combine
        return

    # Calculate weighted average
    total_count = sum(all_counts)
    weights = [c / total_count for c in all_counts]

    combined_emb = np.average(all_embeddings, axis=0, weights=weights)

    # Update target
    target.average_embedding = serialize_embedding(combined_emb)
    target.embedding_count = total_count


def preview_merge(target_id, source_ids, user_id):
    """
    Preview what a merge would look like without executing it.

    Args:
        target_id: ID of the target speaker
        source_ids: List of source speaker IDs
        user_id: ID of the user

    Returns:
        dict: Preview of the merge results
              {
                  'target_name': '...',
                  'source_names': [...],
                  'combined_use_count': 123,
                  'combined_embedding_count': 45,
                  'total_snippets': 67
              }
    """
    # Validate speakers
    target = Speaker.query.filter_by(id=target_id, user_id=user_id).first()
    if not target:
        raise ValueError("Target speaker not found")

    sources = Speaker.query.filter(
        Speaker.id.in_(source_ids),
        Speaker.user_id == user_id
    ).all()

    if len(sources) == 0:
        raise ValueError("No valid source speakers found")

    # Calculate combined statistics
    combined_use_count = target.use_count
    combined_embedding_count = target.embedding_count or 0
    total_snippets = SpeakerSnippet.query.filter_by(speaker_id=target_id).count()

    source_names = []
    for source in sources:
        combined_use_count += source.use_count
        combined_embedding_count += (source.embedding_count or 0)
        total_snippets += SpeakerSnippet.query.filter_by(speaker_id=source.id).count()
        source_names.append(source.name)

    return {
        'target_name': target.name,
        'source_names': source_names,
        'combined_use_count': combined_use_count,
        'combined_embedding_count': combined_embedding_count,
        'total_snippets': total_snippets,
        'has_embeddings': target.average_embedding is not None or any(s.average_embedding for s in sources)
    }


def can_merge_speakers(speaker_ids, user_id):
    """
    Check if speakers can be merged (all belong to same user, no duplicates).

    Args:
        speaker_ids: List of speaker IDs
        user_id: ID of the user

    Returns:
        tuple: (bool, str) - (can_merge, error_message)
    """
    if len(speaker_ids) < 2:
        return False, "Need at least 2 speakers to merge"

    if len(speaker_ids) != len(set(speaker_ids)):
        return False, "Duplicate speaker IDs provided"

    speakers = Speaker.query.filter(
        Speaker.id.in_(speaker_ids),
        Speaker.user_id == user_id
    ).all()

    if len(speakers) != len(speaker_ids):
        return False, "Some speakers don't exist or don't belong to user"

    return True, ""
