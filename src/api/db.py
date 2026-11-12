"""SQLAlchemy models: reports, clusters (with review state), audit log."""
import os
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, LargeBinary, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from src.common import DATA_DIR


def make_engine(url=None):
    url = url or os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR / 'hazard.db'}")
    kw = {"connect_args": {"check_same_thread": False}} if url.startswith("sqlite") else {}
    return create_engine(url, **kw)


def make_session_factory(engine):
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class ReportRow(Base):
    __tablename__ = "reports"
    report_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    text: Mapped[str] = mapped_column(Text)
    language: Mapped[str | None] = mapped_column(String(8), nullable=True)
    latitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    longitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    location_text: Mapped[str | None] = mapped_column(String(120), nullable=True)
    timestamp: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    source: Mapped[str] = mapped_column(String(10), default="app")
    media_flag: Mapped[bool] = mapped_column(Boolean, default=False)
    received_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    in_triage: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    triage_reasons: Mapped[str] = mapped_column(Text, default="[]")
    pred_type: Mapped[str] = mapped_column(String(30), default="not_hazard")
    p_hazard: Mapped[float] = mapped_column(Float, default=0.0)
    relevant: Mapped[bool] = mapped_column(Boolean, default=False)
    severity_pred: Mapped[str] = mapped_column(String(10), default="none")
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    model_used: Mapped[str] = mapped_column(String(30), default="")
    type_probs: Mapped[str] = mapped_column(Text, default="{}")
    flags: Mapped[str] = mapped_column(Text, default="[]")
    contradictions: Mapped[str] = mapped_column(Text, default="[]")
    emb: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    cluster_id: Mapped[str | None] = mapped_column(String(20), nullable=True, index=True)
    true_incident_id: Mapped[str | None] = mapped_column(String(20), nullable=True)  # only set for seeded demo data


class ClusterRow(Base):
    __tablename__ = "clusters"
    cluster_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    payload: Mapped[str] = mapped_column(Text)
    review_status: Mapped[str] = mapped_column(String(20), default="Unverified", index=True)
    priority: Mapped[str] = mapped_column(String(10), default="Low")
    dominant_type: Mapped[str] = mapped_column(String(30), default="")
    first_report_time: Mapped[datetime] = mapped_column(DateTime)
    last_report_time: Mapped[datetime] = mapped_column(DateTime)
    n_unique: Mapped[int] = mapped_column(Integer, default=1)
    superseded: Mapped[bool] = mapped_column(Boolean, default=False)
    changed_since_review: Mapped[bool] = mapped_column(Boolean, default=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class AuditRow(Base):
    __tablename__ = "audit_log"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    cluster_id: Mapped[str] = mapped_column(String(20), index=True)
    old_status: Mapped[str] = mapped_column(String(20))
    new_status: Mapped[str] = mapped_column(String(20))
    reviewer: Mapped[str] = mapped_column(String(80))
    notes: Mapped[str] = mapped_column(Text, default="")
    at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
