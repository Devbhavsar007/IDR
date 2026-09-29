"""
IDR Cloud Database Models (SQLAlchemy).

Complies with DPDP Act 2023:
  - Strictly stores anonymized session IDs (no PII, no IMEI, no phone numbers).
  - Explicit consent audit trail with cryptographic purpose verification.
"""

from __future__ import annotations

import time
from sqlalchemy import Boolean, Column, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class DpdpConsentAudit(Base):
    """Immutable audit trail for user consent records under DPDP Act 2023 §6."""

    __tablename__ = "dpdp_consent_audits"

    id = Column(Integer, primary_key=True, autoincrement=True)
    device_pseudonym = Column(String(64), index=True, nullable=False)
    consent_version = Column(Integer, nullable=False)
    purpose_navigation = Column(Boolean, default=True)
    purpose_logging = Column(Boolean, default=False)
    purpose_cloud_upload = Column(Boolean, default=False)
    granted_at_utc = Column(Integer, nullable=False)
    withdrawn_at_utc = Column(Integer, nullable=True)
    status = Column(String(16), default="ACTIVE")  # ACTIVE or WITHDRAWN


class DriveSession(Base):
    """Aggregated anonymized driving telemetry for fleet analytics and model validation."""

    __tablename__ = "drive_sessions"

    session_id = Column(String(64), primary_key=True)
    vehicle_type = Column(String(32), default="car")
    duration_sec = Column(Float, nullable=False)
    distance_km = Column(Float, nullable=False)
    avg_speed_kmh = Column(Float, default=0.0)
    outage_count = Column(Integer, default=0)
    max_outage_sec = Column(Float, default=0.0)
    pct_dead_reckoning = Column(Float, default=0.0)
    uploaded_at_utc = Column(Integer, default=lambda: int(time.time()))


class ModelRelease(Base):
    """Registry of cryptographically signed ONNX motion models."""

    __tablename__ = "model_releases"

    version = Column(String(32), primary_key=True)
    filename = Column(String(128), nullable=False)
    sha256 = Column(String(64), nullable=False)
    signature = Column(String(128), nullable=False)
    download_url = Column(String(256), nullable=False)
    is_active = Column(Boolean, default=True)
    released_at_utc = Column(Integer, default=lambda: int(time.time()))


# Database session factory
DATABASE_URL = "sqlite:///idr_cloud.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
