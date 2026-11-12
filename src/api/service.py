"""Business logic: ingest -> classify/check -> triage or cluster -> priority -> human review."""
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import select

from src.api.db import AuditRow, ClusterRow, ReportRow
from src.classification.fallback import KeywordClassifier
from src.common import INDIA_BBOX, REPORTS_DIR
from src.data.clean_data import clean_text
from src.pipeline import cluster_df, enrich, load_classifier
from src.priority.agent import ORDER, assess
from src.semantic.embeddings import get_embedder

STATES = ["Unverified", "Under Review", "Confirmed", "Rejected", "Escalated", "Needs More Info"]
TRANSITIONS = {
    "Unverified": ["Under Review"],
    "Under Review": ["Confirmed", "Rejected", "Escalated", "Needs More Info"],
    "Needs More Info": ["Under Review", "Rejected"],
    "Escalated": ["Confirmed", "Rejected", "Under Review"],
    "Confirmed": ["Under Review"],
    "Rejected": ["Under Review"],
}
J = lambda o: json.dumps(o, default=lambda x: x.item() if hasattr(x, "item") else str(x))


class NotFound(KeyError):
    pass


class InvalidTransition(ValueError):
    pass


class HazardService:
    def __init__(self, session_factory, classifier=None):
        self.Session = session_factory
        self.clf = classifier or load_classifier()
        self.embedder = get_embedder()
        self.dim = int(len(self.embedder.encode(["probe"])[0]))
        self.clock = os.getenv("HAZARD_CLOCK", "data")  # "data": newest stored report = "now" (works for seeded history); "wall": real time

    # ---------------- ingestion ----------------
    def _norm(self, r):
        text = clean_text(r.get("text"))
        if not text:
            raise ValueError("text is empty after cleaning")
        la0, la1, lo0, lo1 = INDIA_BBOX
        lat, lon = r.get("latitude"), r.get("longitude")
        if lat is None or lon is None or not (la0 <= lat <= la1 and lo0 <= lon <= lo1):
            lat = lon = np.nan
        ts = r.get("timestamp")
        if ts is not None and getattr(ts, "tzinfo", None):
            ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
        lang = r.get("language") or ("hi" if any("\u0900" <= c <= "\u097f" for c in text) else "en")
        return dict(report_id=r.get("report_id") or "API-" + uuid.uuid4().hex[:8].upper(), text=text, language=lang, latitude=lat,
                    longitude=lon, location_text=clean_text(r.get("location_text")), timestamp=pd.Timestamp(ts) if ts is not None else pd.NaT,
                    source=r.get("source") or "app", media_flag=int(bool(r.get("media_flag"))), true_incident_id=r.get("true_incident_id"))

    def _enrich(self, df):
        warnings = []
        try:
            e = enrich(df, self.clf, self.embedder)
        except Exception as ex:  # embedder or model hard failure -> keyword rules + neutral embedding
            warnings.append(f"model failure ({type(ex).__name__}); keyword fallback used, embeddings neutral")
            class _Neutral:
                def encode(_, texts):
                    return np.ones((len(texts), self.dim), dtype=np.float32) / np.sqrt(self.dim)
            e = enrich(df, KeywordClassifier(), _Neutral())
        if (e.model_used == "fallback_keyword").any():
            warnings.append("classifier fallback (keyword rules) used; confidence is capped")
        return e, sorted(set(warnings))

    def ingest(self, records, recluster=True):
        df = pd.DataFrame([self._norm(r) for r in records])
        e, warnings = self._enrich(df)
        rows = []
        for r in e.to_dict("records"):
            reasons = [f for f in r["flags"] if f in ("missing_location", "missing_time")]
            rows.append(ReportRow(
                report_id=r["report_id"], text=r["text"], language=r["language"],
                latitude=None if pd.isna(r["latitude"]) else float(r["latitude"]), longitude=None if pd.isna(r["longitude"]) else float(r["longitude"]),
                location_text=r["location_text"], timestamp=None if pd.isna(r["timestamp"]) else r["timestamp"].to_pydatetime(),
                source=r["source"], media_flag=bool(r["media_flag"]), in_triage=bool(reasons), triage_reasons=J(reasons), pred_type=r["pred_type"],
                p_hazard=float(r["p_hazard"]), relevant=bool(r["relevant"]), severity_pred=r["severity_pred"], confidence=float(r["confidence"]),
                model_used=r["model_used"], type_probs=J(r["type_probs"]), flags=J(r["flags"]), contradictions=J(r["contradictions"]),
                emb=np.asarray(r["emb"], dtype=np.float32).tobytes(),
                true_incident_id=r.get("true_incident_id") if isinstance(r.get("true_incident_id"), str) else None))
        with self.Session() as s:
            s.add_all(rows)
            s.commit()
        if recluster:
            self.recluster()
        with self.Session() as s:
            return [self.report_out(s.get(ReportRow, x.report_id)) for x in rows], warnings

    # ---------------- clustering (+ stable ids + review state preserved) ----------------
    def _frame(self, rows):
        return pd.DataFrame([dict(report_id=r.report_id, text=r.text, latitude=r.latitude, longitude=r.longitude, timestamp=pd.Timestamp(r.timestamp),
                                  source=r.source, media_flag=int(r.media_flag), pred_type=r.pred_type, confidence=r.confidence,
                                  severity_pred=r.severity_pred, flags=json.loads(r.flags), contradictions=json.loads(r.contradictions),
                                  type_probs=json.loads(r.type_probs), emb=np.frombuffer(r.emb, dtype=np.float32)) for r in rows])

    def _now(self, s):
        if self.clock == "wall":
            return datetime.utcnow()
        latest = max((r.timestamp for r in s.scalars(select(ReportRow)) if r.timestamp), default=None)
        return latest or datetime.utcnow()

    def recluster(self):
        with self.Session() as s:
            rows = list(s.scalars(select(ReportRow).where(ReportRow.in_triage.is_(False), ReportRow.relevant.is_(True))))
            for r in s.scalars(select(ReportRow)):
                r.cluster_id = None
            clusters = []
            if rows:
                _, clusters, _ = cluster_df(self._frame(rows))
                now = self._now(s)
                for c in clusters:
                    c.update(assess(c, now))
            old = {c.cluster_id: c for c in s.scalars(select(ClusterRow).where(ClusterRow.superseded.is_(False)))}
            old_members = {k: set(json.loads(v.payload)["report_ids"]) for k, v in old.items()}
            cand = sorted(((len(set(c["report_ids"]) & m) / len(set(c["report_ids"]) | m), i, k)
                           for i, c in enumerate(clusters) for k, m in old_members.items() if set(c["report_ids"]) & m), reverse=True)
            assign, used = {}, set()
            for jac, i, k in cand:
                if jac >= 0.4 and i not in assign and k not in used:
                    assign[i] = k
                    used.add(k)
            nums = [int(x.cluster_id[1:]) for x in s.scalars(select(ClusterRow)) if x.cluster_id[1:].isdigit()]
            nxt = max(nums, default=0) + 1
            for i, c in enumerate(clusters):
                cid = assign.get(i)
                if cid is None:
                    cid, nxt = f"C{nxt:04d}", nxt + 1
                c["cluster_id"] = cid
                row = old.get(cid)
                first, last = datetime.fromisoformat(c["first_report_time"]), datetime.fromisoformat(c["last_report_time"])
                if row is None:
                    row = ClusterRow(cluster_id=cid, review_status="Unverified")
                    s.add(row)
                elif row.review_status != "Unverified" and set(c["report_ids"]) != old_members[cid]:
                    row.changed_since_review = True  # evidence changed after a human looked at it
                row.payload, row.priority, row.dominant_type = J(c), c["priority"], c["dominant_type"]
                row.first_report_time, row.last_report_time, row.n_unique = first, last, c["n_unique_reports"]
                row.updated_at = datetime.utcnow()
            members = {rid: c["cluster_id"] for c in clusters for rid in c["report_ids"]}
            for r in s.scalars(select(ReportRow)):
                r.cluster_id = members.get(r.report_id)
            for k, row in old.items():
                if k not in {c["cluster_id"] for c in clusters}:
                    if row.review_status == "Unverified":
                        s.delete(row)
                    else:
                        row.superseded = True  # keep human decisions + audit trail even if evidence regrouped
            s.commit()
        return len(clusters)

    # ---------------- serialisation ----------------
    def report_out(self, r):
        return dict(report_id=r.report_id, text=r.text, language=r.language, latitude=r.latitude, longitude=r.longitude, location_text=r.location_text,
                    timestamp=r.timestamp.isoformat() if r.timestamp else None, source=r.source, media_flag=r.media_flag,
                    queue="triage" if r.in_triage else ("clustered" if r.relevant else "not_relevant"), triage_reasons=json.loads(r.triage_reasons),
                    predicted_type=r.pred_type, p_hazard=round(r.p_hazard, 4), severity=r.severity_pred, confidence=round(r.confidence, 4),
                    model_used=r.model_used, flags=json.loads(r.flags), contradictions=json.loads(r.contradictions), cluster_id=r.cluster_id)

    def cluster_out(self, row, s=None, detail=False):
        d = json.loads(row.payload)
        d.update(review_status=row.review_status, superseded=row.superseded, changed_since_review=row.changed_since_review,
                 allowed_transitions=TRANSITIONS[row.review_status],
                 status_note="REPORTED by the public - not confirmed" if row.review_status != "Confirmed" else "Confirmed by a human reviewer")
        if detail:
            members = list(s.scalars(select(ReportRow).where(ReportRow.report_id.in_(d["report_ids"]))))
            d["members"] = sorted((self.report_out(m) for m in members), key=lambda x: x["timestamp"] or "")
            d["audit_log"] = [dict(old_status=a.old_status, new_status=a.new_status, reviewer=a.reviewer, notes=a.notes, at=a.at.isoformat())
                              for a in s.scalars(select(AuditRow).where(AuditRow.cluster_id == row.cluster_id).order_by(AuditRow.id))]
        return d

    # ---------------- queries ----------------
    def list_clusters(self, hazard_type=None, priority=None, status=None, since=None, until=None, min_unique=1, include_superseded=False):
        csv = lambda v: set(v.split(",")) if v else None
        types, pris, sts = csv(hazard_type), csv(priority), csv(status)
        with self.Session() as s:
            out = []
            for r in s.scalars(select(ClusterRow)):
                if (r.superseded and not include_superseded) or r.n_unique < min_unique:
                    continue
                if (types and r.dominant_type not in types) or (pris and r.priority not in pris) or (sts and r.review_status not in sts):
                    continue
                if (since and r.last_report_time < since) or (until and r.first_report_time > until):
                    continue
                out.append(r)
            out.sort(key=lambda r: (-ORDER.index(r.priority), -r.last_report_time.timestamp()))
            return [self.cluster_out(r, s) for r in out]

    def get_cluster(self, cid):
        with self.Session() as s:
            row = s.get(ClusterRow, cid)
            if row is None:
                raise NotFound(cid)
            return self.cluster_out(row, s, detail=True)

    def list_reports(self, triage=None, cluster_id=None, limit=100, offset=0):
        with self.Session() as s:
            q = select(ReportRow).order_by(ReportRow.received_at.desc(), ReportRow.report_id)
            if triage is not None:
                q = q.where(ReportRow.in_triage.is_(triage))
            if cluster_id:
                q = q.where(ReportRow.cluster_id == cluster_id)
            return [self.report_out(r) for r in s.scalars(q.limit(limit).offset(offset))]

    def review(self, cid, new_state, reviewer, notes=""):
        with self.Session() as s:
            row = s.get(ClusterRow, cid)
            if row is None:
                raise NotFound(cid)
            if new_state not in TRANSITIONS[row.review_status]:
                raise InvalidTransition(f"{row.review_status} -> {new_state} not allowed; allowed: {TRANSITIONS[row.review_status]}")
            s.add(AuditRow(cluster_id=cid, old_status=row.review_status, new_status=new_state, reviewer=reviewer, notes=notes))
            row.review_status, row.changed_since_review, row.updated_at = new_state, False, datetime.utcnow()
            s.commit()
        return self.get_cluster(cid)

    def live_stats(self):
        with self.Session() as s:
            reps = list(s.scalars(select(ReportRow)))
            cl = [c for c in s.scalars(select(ClusterRow)) if not c.superseded]
            by = lambda xs: {k: sum(1 for x in xs if x == k) for k in sorted(set(xs))}
            return dict(n_reports=len(reps), n_triage=sum(r.in_triage for r in reps), n_not_relevant=sum((not r.relevant) and not r.in_triage for r in reps),
                        n_clusters=len(cl), n_multi_report_clusters=sum(c.n_unique >= 2 for c in cl), by_status=by([c.review_status for c in cl]),
                        by_priority=by([c.priority for c in cl]), classifier=getattr(self.clf, "model_used", getattr(self.clf, "name", "?")),
                        embedder=self.embedder.name)

    def offline_metrics(self):
        p = Path(REPORTS_DIR) / "metrics.json"
        return json.loads(p.read_text()) if p.exists() else None
