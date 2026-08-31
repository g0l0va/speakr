"""
SpeakerEmbeddingSample database model.

Speaker.average_embedding is a single blended vector -- once a bad or
ambiguous sample has been folded into it, there is no way to take it back
out. This table keeps every contributing voice embedding as its own row, so
a profile can be rebuilt from exactly the samples a human has confirmed,
and a bad sample can be rejected or deleted without disturbing the rest.

review_status is the mechanism that stops the auto-labelling feedback loop:
an automatic voice match creates a 'pending' sample here and does NOT touch
Speaker.average_embedding. Only a 'confirmed' sample (created directly by a
manual name assignment, or promoted from 'pending' via explicit review)
counts toward matching or a profile rebuild.

source/review_status/embedding_dim are also enforced with CHECK constraints
at the DB layer (not just src/services/speaker_embedding_samples.py's
Python-side validation), and a partial unique index forbids more than one
'pending' sample per (speaker, recording, label) so re-processing the same
recording can't pile up duplicate review-queue entries.
"""

from datetime import datetime
from src.database import db


# Where a sample came from.
#   manual    - a user explicitly assigned this name to this speaker label
#   automatic - voice-match auto-labelling proposed this identity (unreviewed)
#   import    - backfilled once from a pre-existing Speaker.average_embedding
SAMPLE_SOURCES = ('manual', 'automatic', 'import')

# Review lifecycle. Only 'confirmed' samples feed matching/rebuilds.
SAMPLE_REVIEW_STATUSES = ('confirmed', 'pending', 'rejected')


class SpeakerEmbeddingSample(db.Model):
    """One voice embedding sample contributing to a Speaker's voice profile."""

    __tablename__ = 'speaker_embedding_sample'

    id = db.Column(db.Integer, primary_key=True)
    speaker_id = db.Column(db.Integer, db.ForeignKey('speaker.id', ondelete='CASCADE'), nullable=False)

    # Denormalized alongside speaker_id: every query and ownership check this
    # feature does filters on user_id directly, so a join bug elsewhere can
    # never leak another user's sample into a match or a review list.
    user_id = db.Column(db.Integer, db.ForeignKey('user.id', ondelete='CASCADE'), nullable=False)

    # Nullable: an imported (backfilled) sample has no single originating
    # recording, and a recording can later be deleted without the sample
    # itself becoming meaningless.
    recording_id = db.Column(db.Integer, db.ForeignKey('recording.id', ondelete='SET NULL'), nullable=True)

    # The generic diarization label (e.g. "SPEAKER_00") this sample was
    # extracted under, before it was attributed to a named Speaker.
    source_speaker_label = db.Column(db.String(50), nullable=True)

    # Binary numpy float32 array -- same on-disk format as
    # Speaker.average_embedding (see speaker_embedding_matcher.serialize_embedding).
    embedding = db.Column(db.LargeBinary, nullable=False)
    embedding_dim = db.Column(db.Integer, nullable=False, default=256)

    source = db.Column(db.String(20), nullable=False, default='manual')
    review_status = db.Column(db.String(20), nullable=False, default='pending')

    # Cosine similarity (0-1) to the speaker's best-known profile at capture
    # time. Descriptive metadata only, never re-derived after the fact.
    similarity_to_profile = db.Column(db.Float, nullable=True)

    # Open-ended slot for future audio-quality signals (duration, SNR, clip
    # count, ...) so a future filter has somewhere to read from. Nothing
    # populates this yet -- no such metric is measured anywhere in the
    # pipeline today, and this increment does not invent one.
    quality_metadata = db.Column(db.JSON, nullable=True)

    # Which embedding/diarization model produced this sample, when known.
    model_name = db.Column(db.String(200), nullable=True)
    model_version = db.Column(db.String(50), nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    reviewed_at = db.Column(db.DateTime, nullable=True)

    __table_args__ = (
        db.CheckConstraint(
            "source IN ('manual','automatic','import')",
            name='ck_speaker_embedding_sample_source',
        ),
        db.CheckConstraint(
            "review_status IN ('confirmed','pending','rejected')",
            name='ck_speaker_embedding_sample_review_status',
        ),
        # The whole pipeline (serialize_embedding/deserialize_embedding,
        # calculate_similarity, np.stack in rebuild_speaker_profile) assumes
        # a fixed 256-dim float32 vector. Nothing upstream checks this, so
        # without a DB-level guard a malformed embedding_dim would silently
        # corrupt a rebuild (np.stack requires uniform shape) rather than
        # fail loudly at the point the bad row was written.
        db.CheckConstraint(
            "embedding_dim = 256",
            name='ck_speaker_embedding_sample_dim',
        ),
        db.Index('idx_speaker_embedding_sample_speaker_review', 'speaker_id', 'review_status'),
        db.Index('idx_speaker_embedding_sample_user', 'user_id'),
        db.Index('idx_speaker_embedding_sample_recording', 'recording_id'),
        # At most one PENDING sample per (speaker, recording, label): a
        # DB-level backstop against re-processing the same recording
        # creating duplicate review-queue entries for the same automatic
        # guess. Confirmed/rejected rows are exempt (a speaker can have many
        # confirmed samples from the same recording/label over time, and a
        # rejected-then-recreated pending sample is a legitimate new guess).
        db.Index(
            'uq_speaker_embedding_sample_pending_dedup',
            'speaker_id', 'recording_id', 'source_speaker_label',
            unique=True,
            postgresql_where=db.text("review_status = 'pending'"),
            sqlite_where=db.text("review_status = 'pending'"),
        ),
    )

    speaker = db.relationship(
        'Speaker',
        backref=db.backref('embedding_samples', lazy=True, cascade='all, delete-orphan')
    )
    recording = db.relationship('Recording')

    def to_dict(self):
        """Convert to dict representation. Never includes the raw embedding bytes."""
        return {
            'id': self.id,
            'speaker_id': self.speaker_id,
            'recording_id': self.recording_id,
            # Same convention as SpeakerSnippet.to_dict(): a bare numeric
            # recording_id isn't useful in a review UI, and the recording
            # may since have been deleted (ondelete='SET NULL' above).
            'recording_title': self.recording.title if self.recording else None,
            'source_speaker_label': self.source_speaker_label,
            'source': self.source,
            'review_status': self.review_status,
            'similarity_to_profile': self.similarity_to_profile,
            'quality_metadata': self.quality_metadata,
            'model_name': self.model_name,
            'model_version': self.model_version,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'reviewed_at': self.reviewed_at.isoformat() if self.reviewed_at else None,
        }

    def __repr__(self):
        return (
            f"SpeakerEmbeddingSample(speaker_id={self.speaker_id}, "
            f"source='{self.source}', review_status='{self.review_status}')"
        )
