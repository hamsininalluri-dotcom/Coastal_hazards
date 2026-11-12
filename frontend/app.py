"""Response map + human review dashboard.  Run: streamlit run frontend/app.py  (API_URL env, default http://localhost:8000)"""
import math
import os
from datetime import datetime, time, timedelta

import folium
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium

API = os.getenv("API_URL", "http://localhost:8000")
HAZARDS = ["tsunami", "storm_surge", "flooding", "high_waves", "coastal_erosion", "oil_spill", "cyclone_damage"]
STATES = ["Unverified", "Under Review", "Confirmed", "Rejected", "Escalated", "Needs More Info"]
PCOL = {"Critical": "#b71c1c", "High": "#ef6c00", "Medium": "#f9a825", "Low": "#2e7d32"}

st.set_page_config(layout="wide", page_title="Coastal Hazard Response Map", page_icon="🌊")


def call(method, path, **kw):
    try:
        r = requests.request(method, f"{API}{path}", timeout=30, **kw)
        if r.status_code >= 400:
            st.error(f"{r.status_code}: {r.json().get('detail', r.text)}")
            return None
        return r.json()
    except Exception as e:  # API down: degrade gracefully instead of crashing the page
        st.error(f"Cannot reach API at {API} ({type(e).__name__}). Start it with `uvicorn src.api.main:app`.")
        return None


st.title("🌊 Coastal Hazard Response Map")
st.warning("**Everything on this map is REPORTED by the public, not confirmed.** A cluster only becomes *Confirmed* after a human reviewer "
           "confirms it. Priority means *urgency if the reports are true*, not how likely they are to be true. Dashed outline = unverified.")

tab_map, tab_triage, tab_submit, tab_metrics = st.tabs(["Map & review", "Manual triage queue", "Submit test report", "Metrics"])

with st.sidebar:
    st.header("Filters")
    types = st.multiselect("Hazard type", HAZARDS)
    pris = st.multiselect("Priority", list(PCOL))
    stat = st.multiselect("Review status", STATES)
    min_unique = st.slider("Min. distinct reports per cluster", 1, 10, 2, help="1 shows isolated single reports, which are always 'unconfirmed'.")
    allc = call("GET", "/clusters", params={"min_unique": 1}) or []
    if allc:
        lo = min(datetime.fromisoformat(c["first_report_time"]) for c in allc).date()
        hi = max(datetime.fromisoformat(c["last_report_time"]) for c in allc).date()
        rng = st.date_input("Time range", (lo, hi), min_value=lo, max_value=hi)
    else:
        rng = None
    if st.button("Refresh"):
        st.rerun()

params = {"min_unique": min_unique}
if types:
    params["hazard_type"] = ",".join(types)
if pris:
    params["priority"] = ",".join(pris)
if stat:
    params["status"] = ",".join(stat)
if rng and len(rng) == 2:
    params["since"] = datetime.combine(rng[0], time.min).isoformat()
    params["until"] = datetime.combine(rng[1], time.max).isoformat()
clusters = call("GET", "/clusters", params=params) or []

with tab_map:
    left, right = st.columns([3, 2])
    with left:
        st.caption(f"{len(clusters)} incident cluster(s) shown. Marker size = number of reports, colour = priority.")
        if clusters:
            m = folium.Map(location=[sum(c["centroid_lat"] for c in clusters) / len(clusters), sum(c["centroid_lon"] for c in clusters) / len(clusters)],
                           zoom_start=5, tiles="OpenStreetMap")
            for c in clusters:
                done = c["review_status"] == "Confirmed"
                folium.CircleMarker(
                    [c["centroid_lat"], c["centroid_lon"]], radius=min(40, 6 + 4 * math.sqrt(c["n_reports"])), color="#000" if done else PCOL[c["priority"]],
                    weight=4 if done else 2, dash_array=None if done else "5", fill=True, fill_color=PCOL[c["priority"]], fill_opacity=0.55,
                    tooltip=f"{c['cluster_id']} | {c['dominant_type']} | {c['n_reports']} reports ({c['n_unique_reports']} distinct) | {c['priority']} | {c['review_status']}",
                ).add_to(m)
            out = st_folium(m, height=560, use_container_width=True, returned_objects=["last_object_clicked_tooltip"])
            clicked = (out or {}).get("last_object_clicked_tooltip")
            if clicked and clicked != st.session_state.get("last_click"):
                st.session_state["last_click"] = clicked
                st.session_state["sel"] = clicked.split(" | ")[0].strip()
        else:
            st.info("No clusters match the current filters.")
    with right:
        ids = [c["cluster_id"] for c in clusters]
        if ids:
            if st.session_state.get("sel") not in ids:
                st.session_state["sel"] = ids[0]
            sel = st.selectbox("Selected cluster (click a marker or choose)", ids, key="sel")
            d = call("GET", f"/clusters/{sel}")
        else:
            d = None
        if d:
            st.subheader(f"{d['cluster_id']} · {d['dominant_type'].replace('_', ' ')}")
            st.markdown(f"**Status:** `{d['review_status']}`  ·  **Priority:** `{d['priority']}`  ·  *{d['status_note']}*")
            if d.get("changed_since_review"):
                st.warning("New evidence joined this cluster after the last review - please re-check.")
            a, b, c3 = st.columns(3)
            a.metric("Reports", d["n_reports"])
            b.metric("Distinct", d["n_unique_reports"], help="after collapsing re-posts / duplicates")
            c3.metric("Duplicates", d["duplicate_count"])
            st.progress(min(1.0, d["confidence"]), text=f"Confidence {d['confidence']:.2f} - {d['confidence_label']}")
            with st.expander("Why this confidence?", expanded=True):
                st.write("\n".join(f"- {r}" for r in d["confidence_reasons"]))
            if d["contradictions"]:
                st.error("Contradictions found:\n" + "\n".join(f"- {x}" for x in d["contradictions"]))
            with st.expander("Why this priority?"):
                st.write(d["explanation"])
                st.dataframe(pd.DataFrame(d["factors"]), hide_index=True, use_container_width=True)
            st.caption(f"First report {d['first_report_time']} · last {d['last_report_time']} · spread {d['spread_km']:.1f} km · "
                       f"media items {d['media_count']} · sources {d['sources']}")
            with st.form("review", clear_on_submit=True):
                st.markdown("**Human review**")
                who = st.text_input("Reviewer name")
                new = st.selectbox("Move to", d["allowed_transitions"])
                notes = st.text_area("Notes (evidence, official bulletin, calls made...)")
                if st.form_submit_button("Submit decision"):
                    if not who.strip():
                        st.error("Reviewer name is required for the audit log.")
                    elif call("POST", f"/clusters/{sel}/review", json=dict(state=new, reviewer=who.strip(), notes=notes)):
                        st.rerun()
    if d:
        st.subheader("Supporting evidence")
        mem = pd.DataFrame(d["members"])[["report_id", "timestamp", "source", "media_flag", "predicted_type", "confidence", "text", "flags"]]
        st.dataframe(mem, hide_index=True, use_container_width=True)
        st.subheader("Audit log")
        st.dataframe(pd.DataFrame(d["audit_log"]) if d["audit_log"] else pd.DataFrame(columns=["at", "reviewer", "old_status", "new_status", "notes"]),
                     hide_index=True, use_container_width=True)

with tab_triage:
    st.markdown("Reports with **missing location or time** are never clustered automatically. A human can complete them or discard them.")
    tri = call("GET", "/reports", params={"triage": True, "limit": 500}) or []
    st.metric("Waiting for manual triage", len(tri))
    if tri:
        st.dataframe(pd.DataFrame(tri)[["report_id", "text", "source", "triage_reasons", "predicted_type", "confidence", "timestamp"]], hide_index=True, use_container_width=True)

with tab_submit:
    with st.form("submit"):
        txt = st.text_area("Report text", "Sea water entering streets at Puri, roads submerged")
        c1, c2, c3 = st.columns(3)
        lat = c1.text_input("Latitude (blank = missing)", "19.81")
        lon = c2.text_input("Longitude (blank = missing)", "85.83")
        src = c3.selectbox("Source", ["app", "sms", "social"])
        when = st.text_input("Timestamp ISO (blank = now; type 'none' for missing)", "")
        if st.form_submit_button("Send"):
            body = dict(text=txt, source=src)
            try:
                body["latitude"], body["longitude"] = (float(lat), float(lon)) if lat and lon else (None, None)
            except ValueError:
                st.error("Latitude/longitude must be numbers")
                st.stop()
            if when.strip().lower() != "none":
                body["timestamp"] = when.strip() or datetime.utcnow().isoformat()
            r = call("POST", "/reports", json=body)
            if r:
                st.success(r["queue_note"])
                st.json(r)

with tab_metrics:
    mt = call("GET", "/metrics")
    if mt:
        st.subheader("Live system")
        st.json(mt["live"])
        off = mt.get("offline_evaluation")
        if off:
            st.subheader("Offline evaluation (test split) - selected vs baseline")
            cls = off["classification"]
            st.dataframe(pd.DataFrame({k: {m: v[m] for m in ["accuracy", "macro_precision", "macro_recall", "macro_f1", "relevance_f1", "severity_macro_f1", "ECE"]}
                                       for k, v in cls.items()}).T, use_container_width=True)
            rows = [{"scenario": s, "method": m, **{k: v[k] for k in ["ARI", "NMI", "purity", "n_true_incidents", "n_pred_clusters"]}}
                    for s, d in off["clustering"].items() for m, v in d.items()]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            st.caption("Synthetic data: absolute numbers are optimistic. See reports/evaluation_report.md for failure analysis.")
        else:
            st.info("Run `make evaluate` to produce reports/metrics.json.")
