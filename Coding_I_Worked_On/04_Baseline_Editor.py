# 04_Baseline_Editor.py — FR-05/06/07: embedded Dash baseline block + polynomial order editor.

from __future__ import annotations

import os
from typing import Any, Mapping
from urllib.parse import urlencode

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from dash_apps.baseline_editor import start_server
from utils import session_store, sidebar, styles
from utils.baseline_editor_bridge import (
    initialize_bridge,
    read_bridge,
    spectrum_fingerprint,
)
from utils.baseline_editor_logic import THIS_STAGE_KEY, stage_payload

CURRENT_STAGE = 4
PAGE_STAGE_KEY = THIS_STAGE_KEY
PAGE_TITLE = "Baseline Block Editor"
PAGE_SUBTITLE = "FR-05 / 06 / 07 — Dash-embedded interactive editor"
IFRAME_HEIGHT = 736

# Page 4 may only run after the three preceding pipeline stages are complete.
UPSTREAM_STAGE_KEYS = (
    "01_source_browser",
    "file_loader",
    "03_bandpass_correction",
)
_REQUIRED_ARRAY_KEYS = ("Vlsr", "LCP", "RCP")


def _upstream_complete() -> bool:
    """Return True only when Pages 1–3 are all recorded as complete."""
    completed = st.session_state["completed_stages"]
    return all(stage_key in completed for stage_key in UPSTREAM_STAGE_KEYS)


def _completed_upstream_count() -> int:
    """Count only stages strictly before Page 4 for the shared progress bar."""
    completed = st.session_state["completed_stages"]
    return sum(stage_key in completed for stage_key in UPSTREAM_STAGE_KEYS)


def _normalise_handoff_spectrum(spectrum: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Validate the Page 3 in-memory handoff before mounting Dash."""
    arrays: dict[str, np.ndarray] = {}
    for key in _REQUIRED_ARRAY_KEYS:
        if key not in spectrum:
            raise ValueError(f"Page 3 handoff is missing required field: {key}.")
        try:
            array = np.asarray(spectrum[key], dtype=float).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Page 3 handoff field {key} is not a numeric array.") from exc
        if array.size == 0:
            raise ValueError(
                "Page 3 handed Page 4 an empty corrected spectrum. "
                "Return to Bandpass Correction and resolve the zero-channel output first."
            )
        arrays[key] = array.copy()

    lengths = {key: arrays[key].size for key in _REQUIRED_ARRAY_KEYS}
    if len(set(lengths.values())) != 1:
        raise ValueError(
            "Page 3 handoff arrays have inconsistent lengths: "
            + ", ".join(f"{key}={size}" for key, size in lengths.items())
        )
    return arrays


def _clear_page4_completion() -> None:
    """Remove stale Page 4 completion/output without altering upstream stages."""
    st.session_state["completed_stages"].discard(PAGE_STAGE_KEY)
    st.session_state["pipeline_data"].pop(PAGE_STAGE_KEY, None)
    st.session_state.pop("baseline_corrected_spectrum", None)


def _sync_dash_state_into_streamlit() -> None:
    """Pull the current Dash bridge state into Streamlit session/audit state.

    A stale bridge is ignored when its spectrum fingerprint does not match
    the current Page 3 handoff. This prevents an old baseline fit from being
    re-marked complete after the user changes the bandpass result.
    """
    if not _upstream_complete():
        _clear_page4_completion()
        return

    current_spectrum = st.session_state.get("bandpass_corrected_spectrum")
    if not current_spectrum:
        _clear_page4_completion()
        return

    try:
        current_fingerprint = spectrum_fingerprint(current_spectrum)
    except (KeyError, TypeError, ValueError):
        _clear_page4_completion()
        return

    session_id = st.session_state["session_id"]
    bridge = read_bridge(session_id)
    if not bridge:
        return

    if bridge.get("spectrum_fingerprint") != current_fingerprint:
        _clear_page4_completion()
        return

    payload = stage_payload(
        bridge.get("blocks", []),
        bridge.get("selection_mode", "Automatic"),
        bridge.get("selected_orders"),
        bridge.get("fit_result"),
    )
    st.session_state["pipeline_data"][PAGE_STAGE_KEY] = payload

    flattened = bridge.get("flattened_spectrum")
    if payload["complete"] and flattened:
        st.session_state["baseline_corrected_spectrum"] = flattened
        session_store.mark_stage_complete(PAGE_STAGE_KEY)
    else:
        st.session_state.pop("baseline_corrected_spectrum", None)
        st.session_state["completed_stages"].discard(PAGE_STAGE_KEY)

    seen_event_ids = set(
        st.session_state.setdefault("baseline_editor_seen_event_ids", [])
    )
    for event in bridge.get("events", []):
        event_id = event.get("id")
        if event_id and event_id not in seen_event_ids:
            session_store.log_event(
                event.get("message", "Baseline editor updated")
            )
            seen_event_ids.add(event_id)
    st.session_state["baseline_editor_seen_event_ids"] = sorted(seen_event_ids)


def _dash_url(port: int, session_id: str, revision: str) -> str:
    """Return the Dash iframe URL with a revision that forces fresh hydration.

    The revision changes when the Page 3 spectrum or theme changes. Without
    it, Streamlit can reuse the same iframe URL and the Dash app can keep an
    old/empty hydrated state even though the bridge JSON has been refreshed.
    """
    base = os.environ.get("DASH_BRIDGE_URL", f"http://localhost:{port}").rstrip("/")
    query = urlencode({"session_id": session_id, "rev": revision})
    return f"{base}/?{query}"


def _render_upstream_error(missing_stages: list[str]) -> None:
    """Show a normal Streamlit error rather than mounting an empty editor."""
    missing_text = ", ".join(missing_stages)
    st.error(
        "Baseline editing requires Pages 1–3 to be completed first. "
        f"Missing stage keys: {missing_text}."
    )
    if st.button("Return to Bandpass Correction", key="return_to_bandpass"):
        st.switch_page("pages/03_Bandpass_Correction.py")


def main() -> None:
    """Render Streamlit Page 4 and mount the Dash baseline editor iframe."""
    st.set_page_config(
        page_title="HARTSPEC — Baseline Block Editor",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    session_store.init_session_state()
    _sync_dash_state_into_streamlit()
    sidebar.render_sidebar(current_stage=CURRENT_STAGE)
    styles.inject_base_css()

    # Page 4 itself remains in-progress in its own header. Once FR-05/06/07
    # is complete, Page 5 will show 4/8 complete.
    styles.render_page_header(
        title=PAGE_TITLE,
        subtitle=PAGE_SUBTITLE,
        current_stage=CURRENT_STAGE,
        completed_stages=_completed_upstream_count(),
    )

    completed = st.session_state["completed_stages"]
    missing_stages = [
        stage_key
        for stage_key in UPSTREAM_STAGE_KEYS
        if stage_key not in completed
    ]
    if missing_stages:
        _clear_page4_completion()
        _render_upstream_error(missing_stages)
        st.stop()

    raw_handoff = st.session_state.get("bandpass_corrected_spectrum")
    if raw_handoff is None:
        _clear_page4_completion()
        st.error(
            "The Page 3 corrected spectrum is not present in this Streamlit session. "
            "Return to Bandpass Correction and let FR-04 run again."
        )
        if st.button("Return to Bandpass Correction", key="restore_bandpass_handoff"):
            st.switch_page("pages/03_Bandpass_Correction.py")
        st.stop()

    try:
        corrected = _normalise_handoff_spectrum(raw_handoff)
    except ValueError as exc:
        _clear_page4_completion()
        st.error(str(exc))
        if st.button("Return to Bandpass Correction", key="repair_bandpass_handoff"):
            st.switch_page("pages/03_Bandpass_Correction.py")
        st.stop()

    bandpass_state = st.session_state["pipeline_data"].get(
        "03_bandpass_correction",
        {},
    )
    bandpass_mode = bandpass_state.get("mode")
    if bandpass_mode not in ("Frequency Switch", "Position Switch"):
        _clear_page4_completion()
        st.error(
            "The Page 3 completion record has no valid bandpass mode. "
            "Return to Bandpass Correction and run FR-04 again."
        )
        st.stop()

    bridge = initialize_bridge(
        session_id=st.session_state["session_id"],
        spectrum=corrected,
        theme=styles.get_theme_name(),
        bandpass_mode=bandpass_mode,
    )

    # start_server() is already a process-level singleton guarded by a lock.
    # Calling it directly avoids Streamlit cache lifecycle warnings while
    # still ensuring only one Dash server is started per process.
    port = start_server()

    # Changing the revision changes the iframe src, forcing Dash's hydrate
    # callback to re-read the bridge instead of leaving an old blank graph.
    revision = (
        f"{bridge['spectrum_fingerprint'][:12]}-"
        f"{styles.get_theme_name()}"
    )
    components.iframe(
        _dash_url(port, st.session_state["session_id"], revision),
        height=IFRAME_HEIGHT,
        scrolling=False,
    )


if __name__ == "__main__":
    main()
