# baseline_editor.py — FR-05/06/07 Dash sub-app embedded by Streamlit Page 4.

from __future__ import annotations

import os
import threading
from typing import Any, Mapping
from urllib.parse import parse_qs

import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dash_table, dcc, html, no_update

from utils import styles
from utils.baseline_editor_bridge import append_event, read_bridge, update_bridge
from utils.baseline_editor_logic import (
    POLYNOMIAL_ORDERS,
    POLARIZATIONS,
    auto_select_orders,
    comparison_panels,
    normalize_blocks,
    rows_equal,
    selected_data_to_block,
    stage_payload,
)
from utils.baseline_polynomial_adapter import (
    CLCBaselineIntegrationUnavailable,
    fit_polynomial_comparison,
    flatten_spectrum_for_orders,
)

_SERVER_LOCK = threading.Lock()
_SERVER_THREAD: threading.Thread | None = None
_SERVER_PORT: int | None = None


def _theme_tokens(theme_name: str) -> dict[str, str]:
    """Return a valid shared theme token set for the iframe."""
    return styles.THEMES["dark" if theme_name == "dark" else "light"]


def _root_style(theme_name: str) -> dict[str, str]:
    """Expose shared Streamlit theme tokens to Dash CSS custom properties."""
    theme = _theme_tokens(theme_name)
    error_fill, error_text = styles.STATUS_BADGES["dark" if theme_name == "dark" else "light"]["error"]
    return {
        "--hs-bg-primary": theme["bg-primary"],
        "--hs-bg-panel": theme["bg-panel"],
        "--hs-bg-panel-alt": theme["bg-panel-alt"],
        "--hs-border-default": theme["border-default"],
        "--hs-border-subtle": theme["border-subtle"],
        "--hs-text-body": theme["text-body"],
        "--hs-text-muted": theme["text-muted"],
        "--hs-text-faint": theme["text-faint"],
        "--hs-accent": theme["accent"],
        "--hs-error-fill": error_fill,
        "--hs-error-text": error_text,
    }


def _parse_session_id(search: str | None) -> str | None:
    """Extract the session id from the iframe query string."""
    if not search:
        return None
    values = parse_qs(search.lstrip("?"))
    session_ids = values.get("session_id") or []
    return session_ids[0] if session_ids else None


def _empty_figure(theme_name: str, height: int = 78) -> go.Figure:
    """Return a blank, theme-correct fit panel without inventing placeholder science."""
    theme = _theme_tokens(theme_name)
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor=theme["bg-panel"],
        plot_bgcolor=theme["bg-panel"],
        height=height,
        margin=dict(l=28, r=8, t=18, b=18),
        showlegend=False,
        font=dict(size=9, color=theme["text-muted"]),
        xaxis=dict(
            showgrid=False, showticklabels=False, zeroline=False, showline=True,
            linecolor=theme["border-subtle"],
        ),
        yaxis=dict(
            showgrid=False, showticklabels=False, zeroline=False, showline=True,
            linecolor=theme["border-subtle"],
        ),
    )
    return fig


def _spectrum_figure(state: Mapping[str, Any], rows: list[dict[str, Any]], theme_name: str) -> go.Figure:
    """Build the FR-05 selectable corrected-spectrum canvas with committed block overlays."""
    theme = _theme_tokens(theme_name)
    spectrum = state.get("spectrum") or {}
    vlsr = spectrum.get("Vlsr") or []
    lcp = spectrum.get("LCP") or []
    rcp = spectrum.get("RCP") or []

    fig = go.Figure()
    if vlsr and lcp:
        fig.add_scatter(
            x=vlsr,
            y=lcp,
            mode="lines",
            name="LCP",
            line=dict(color=styles.LCP_COLOR, width=1.6),
            hovertemplate="Vlsr=%{x:.2f} km/s<br>LCP=%{y:.3f}<extra></extra>",
        )
    if vlsr and rcp:
        fig.add_scatter(
            x=vlsr,
            y=rcp,
            mode="lines",
            name="RCP",
            line=dict(color=styles.RCP_COLOR, width=1.6),
            hovertemplate="Vlsr=%{x:.2f} km/s<br>RCP=%{y:.3f}<extra></extra>",
        )

    normalized, _errors = normalize_blocks(rows)
    for row in normalized:
        fig.add_shape(
            type="rect",
            x0=row["start"],
            x1=row["end"],
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            fillcolor=theme["accent"],
            opacity=0.20,
            line=dict(width=0),
            layer="below",
        )

    fig.update_layout(
        paper_bgcolor=theme["bg-panel"],
        plot_bgcolor=theme["bg-panel"],
        height=270,
        margin=dict(l=46, r=10, t=4, b=34),
        dragmode="select",
        selectdirection="h",
        showlegend=False,
        font=dict(family="Inter, -apple-system, 'Segoe UI', sans-serif", size=10, color=theme["text-body"]),
        xaxis=dict(
            title="Vlsr (km/s)",
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor=theme["border-subtle"],
            color=theme["text-muted"],
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor=theme["border-subtle"],
            color=theme["text-muted"],
        ),
        # Plotly 5.20.0 only supports ``line`` and ``mode`` on layout.newselection.
        # Committed selections are already rendered as accent-tinted shapes above,
        # so the live drag selection only needs an accent outline here.
        newselection=dict(line=dict(color=theme["accent"], width=1)),
    )
    return fig


def _fit_panel_figure(
    panel_identity,
    fit_result: Mapping[str, Any] | None,
    theme_name: str,
    selected_orders: Mapping[str, Any] | None,
) -> go.Figure:
    """Render one FR-06 fit/chi-squared panel from the CLC adapter output."""
    theme = _theme_tokens(theme_name)
    panel_data = None
    for candidate in (fit_result or {}).get("panels", []):
        same_order = int(candidate.get("order", -1)) == panel_identity.order
        same_polarization = candidate.get("polarization") == panel_identity.polarization
        if same_order and same_polarization:
            panel_data = candidate
            break

    if not panel_data:
        fig = _empty_figure(theme_name)
        fig.update_layout(
            title=dict(text=panel_identity.title, x=0.03, y=0.98, font=dict(size=11, color=theme["text-body"]))
        )
        return fig

    fig = go.Figure()
    fig.add_scatter(
        x=panel_data.get("x", []),
        y=panel_data.get("observed", []),
        mode="lines",
        line=dict(
            color=styles.LCP_COLOR if panel_identity.polarization == "LCP" else styles.RCP_COLOR,
            width=1.1,
        ),
        hoverinfo="skip",
    )
    fig.add_scatter(
        x=panel_data.get("x", []),
        y=panel_data.get("fit", []),
        mode="lines",
        line=dict(
            color=styles.FIT_LINE_COLOR["dark" if theme_name == "dark" else "light"],
            width=1.1,
            dash="dash",
        ),
        hoverinfo="skip",
    )
    selected = dict(selected_orders or {})
    selected_marker = "  ✓" if selected.get(panel_identity.polarization) == panel_identity.order else ""
    chi_squared = panel_data.get("chi_squared")
    if isinstance(chi_squared, (int, float)):
        title = f"{panel_identity.title}{selected_marker} · χ² {chi_squared:.3g}"
    else:
        title = f"{panel_identity.title}{selected_marker}"

    fig.update_layout(
        title=dict(text=title, x=0.03, y=0.98, font=dict(size=11, color=theme["text-body"])),
        paper_bgcolor=theme["bg-panel"],
        plot_bgcolor=theme["bg-panel"],
        height=78,
        margin=dict(l=28, r=8, t=20, b=18),
        showlegend=False,
        font=dict(size=8, color=theme["text-muted"]),
        xaxis=dict(
            showgrid=False, showticklabels=False, zeroline=False, showline=True,
            linecolor=theme["border-subtle"],
        ),
        yaxis=dict(
            showgrid=False, showticklabels=False, zeroline=False, showline=True,
            linecolor=theme["border-subtle"],
        ),
    )
    return fig


def _table() -> dash_table.DataTable:
    """Create the editable FR-05 block table with inline × delete cells."""
    return dash_table.DataTable(
        id="block-table",
        columns=[
            {"name": "Block", "id": "block", "editable": False, "type": "numeric"},
            {"name": "Start (km/s)", "id": "start", "editable": True, "type": "numeric"},
            {"name": "End (km/s)", "id": "end", "editable": True, "type": "numeric"},
            {"name": "", "id": "delete", "editable": False},
        ],
        data=[],
        editable=True,
        cell_selectable=True,
        page_action="none",
        fixed_rows={"headers": True},
        style_table={"height": "88px", "overflowY": "auto", "overflowX": "hidden"},
        style_header={
            "height": "32px",
            "minHeight": "32px",
            "maxHeight": "32px",
            "backgroundColor": styles.HARTRAO_DARK_BLUE,
            "color": "#F1F5F9",
            "fontSize": "10px",
            "fontWeight": 600,
            "border": "0",
            "paddingLeft": "10px",
            "textAlign": "left",
        },
        style_cell={
            "height": "28px",
            "minHeight": "28px",
            "maxHeight": "28px",
            "fontSize": "10.5px",
            "fontFamily": "Inter, -apple-system, 'Segoe UI', sans-serif",
            "backgroundColor": "var(--hs-bg-panel)",
            "color": "var(--hs-text-body)",
            "border": "0",
            "borderBottom": "1px solid var(--hs-border-subtle)",
            "paddingLeft": "10px",
            "textAlign": "left",
        },
        style_cell_conditional=[
            {"if": {"column_id": "block"}, "width": "14%"},
            {"if": {"column_id": "start"}, "width": "30%"},
            {"if": {"column_id": "end"}, "width": "30%"},
            {"if": {"column_id": "delete"}, "width": "26%", "textAlign": "center", "cursor": "pointer"},
        ],
        style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "var(--hs-bg-panel-alt)"}],
    )


def _order_dropdown(component_id: str) -> dcc.Dropdown:
    """Return one compact order-3/4/5 dropdown for manual FR-07 selection."""
    return dcc.Dropdown(
        id=component_id,
        options=[{"label": str(order), "value": order} for order in POLYNOMIAL_ORDERS],
        value=3,
        clearable=False,
        searchable=False,
        disabled=True,
        style={"width": "72px", "fontSize": "11px"},
    )


def create_app() -> Dash:
    """Create the single Dash application used for all Streamlit sessions."""
    app = Dash(__name__, suppress_callback_exceptions=True, title="HARTSPEC Baseline Editor")

    app.layout = html.Div(
        id="root",
        className="hs-dash-root",
        children=[
            dcc.Location(id="url", refresh=False),
            dcc.Store(id="session-store"),
            dcc.Store(id="fit-result-store"),
            dcc.Store(id="selected-orders-store"),
            dcc.Store(id="persist-version"),
            html.Div(
                className="hs-dash-panel hs-canvas-panel",
                children=[
                    html.Div(
                        "Interactive canvas — click-drag to select a velocity range (0.1 km/s precision)",
                        className="hs-panel-title",
                    ),
                    dcc.Graph(
                        id="spectrum-graph",
                        figure=_empty_figure("light", height=270),
                        config={"displayModeBar": False, "scrollZoom": True},
                        style={"height": "270px"},
                    ),
                ],
            ),
            html.Div(className="hs-dash-panel hs-block-panel", children=[_table()]),
            html.Div("Six-panel polynomial order comparison", className="hs-grid-label"),
            html.Div(
                className="hs-comparison-grid",
                children=[
                    html.Div(
                        dcc.Graph(
                            id=f"fit-panel-{index}",
                            figure=_empty_figure("light"),
                            config={"displayModeBar": False},
                            style={"height": "98px"},
                        ),
                        className="hs-fit-cell",
                    )
                    for index in range(6)
                ],
            ),
            html.Div(
                className="hs-controls-row",
                children=[
                    html.Span("Order selection", className="hs-controls-label"),
                    dcc.RadioItems(
                        id="selection-mode",
                        options=["Automatic", "Manual"],
                        value="Automatic",
                        inline=True,
                        className="hs-radio",
                    ),
                    html.Span("LCP", className="hs-order-label"),
                    _order_dropdown("manual-order-lcp"),
                    html.Span("RCP", className="hs-order-label"),
                    _order_dropdown("manual-order-rcp"),
                    html.Button("Fit Polynomial", id="fit-polynomial", className="hs-fit-button", n_clicks=0),
                    html.Span(id="selection-feedback", className="hs-selection-feedback"),
                    html.Span(id="validation-message", className="hs-validation"),
                ],
            ),
        ],
    )

    @app.callback(
        Output("root", "style"),
        Output("session-store", "data"),
        Output("block-table", "data"),
        Output("selection-mode", "value"),
        Output("manual-order-lcp", "value"),
        Output("manual-order-rcp", "value"),
        Output("fit-result-store", "data"),
        Input("url", "search"),
    )
    def hydrate(search: str | None):
        session_id = _parse_session_id(search)
        state = read_bridge(session_id) if session_id else None
        state = state or {
            "theme": "light",
            "blocks": [],
            "selection_mode": "Automatic",
            "manual_orders": {"LCP": 3, "RCP": 3},
        }
        manual_orders = state.get("manual_orders") or {"LCP": 3, "RCP": 3}
        return (
            _root_style(state.get("theme", "light")),
            state,
            state.get("blocks", []),
            state.get("selection_mode", "Automatic"),
            manual_orders.get("LCP", 3),
            manual_orders.get("RCP", 3),
            state.get("fit_result"),
        )

    @app.callback(
        Output("block-table", "data", allow_duplicate=True),
        Input("spectrum-graph", "selectedData"),
        Input("block-table", "active_cell"),
        State("block-table", "data"),
        prevent_initial_call=True,
    )
    def add_or_delete_block(selected_data, active_cell, rows):
        rows = list(rows or [])
        trigger = ctx.triggered_id
        if trigger == "spectrum-graph":
            block = selected_data_to_block(selected_data, len(rows) + 1)
            if block is not None:
                rows.append(block.as_row())
            return rows

        if trigger == "block-table" and active_cell and active_cell.get("column_id") == "delete":
            row_index = active_cell.get("row")
            if isinstance(row_index, int) and 0 <= row_index < len(rows):
                rows.pop(row_index)
                normalized, _ = normalize_blocks(rows)
                return normalized
        return no_update

    @app.callback(
        Output("fit-result-store", "data", allow_duplicate=True),
        Input("block-table", "data"),
        State("fit-result-store", "data"),
        prevent_initial_call=True,
    )
    def invalidate_fit_when_blocks_change(_rows, fit_result):
        """A fit belongs to one exact block set; edits/deletes require a fresh fit."""
        return None if fit_result else no_update

    @app.callback(
        Output("fit-result-store", "data", allow_duplicate=True),
        Input("fit-polynomial", "n_clicks"),
        State("session-store", "data"),
        State("block-table", "data"),
        prevent_initial_call=True,
    )
    def run_fit(n_clicks, state, rows):
        if not n_clicks or not state:
            return no_update
        normalized, errors = normalize_blocks(rows)
        if errors or not normalized:
            return no_update
        blocks = [{"start": row["start"], "end": row["end"]} for row in normalized]
        try:
            result = fit_polynomial_comparison(state.get("spectrum") or {}, blocks)
        except CLCBaselineIntegrationUnavailable as exc:
            session_id = state.get("session_id")
            if session_id:
                def _mutate(bridge_state):
                    bridge_state["integration_error"] = str(exc)
                    bridge_state["fit_result"] = None
                    bridge_state["selected_orders"] = None
                    bridge_state["flattened_spectrum"] = None
                    bridge_state["complete"] = False

                update_bridge(session_id, _mutate)
            return None

        session_id = state.get("session_id")
        if session_id:
            def _clear_error(bridge_state):
                bridge_state["integration_error"] = None

            update_bridge(session_id, _clear_error)
        return result

    @app.callback(
        Output("spectrum-graph", "figure"),
        Output("fit-panel-0", "figure"),
        Output("fit-panel-1", "figure"),
        Output("fit-panel-2", "figure"),
        Output("fit-panel-3", "figure"),
        Output("fit-panel-4", "figure"),
        Output("fit-panel-5", "figure"),
        Output("manual-order-lcp", "disabled"),
        Output("manual-order-rcp", "disabled"),
        Output("selection-feedback", "children"),
        Output("validation-message", "children"),
        Output("validation-message", "className"),
        Output("selected-orders-store", "data"),
        Input("block-table", "data"),
        Input("selection-mode", "value"),
        Input("manual-order-lcp", "value"),
        Input("manual-order-rcp", "value"),
        Input("fit-result-store", "data"),
        State("session-store", "data"),
    )
    def render_editor(rows, selection_mode, manual_order_lcp, manual_order_rcp, fit_result, state):
        state = state or {"theme": "light"}
        theme_name = state.get("theme", "light")
        normalized, errors = normalize_blocks(rows)

        if selection_mode == "Automatic":
            selected_orders = auto_select_orders(fit_result)
        else:
            selected_orders = {"LCP": manual_order_lcp, "RCP": manual_order_rcp}
            if any(selected_orders.get(pol) not in POLYNOMIAL_ORDERS for pol in POLARIZATIONS):
                selected_orders = None

        spectrum_fig = _spectrum_figure(state, rows or [], theme_name)
        fit_figures = [
            _fit_panel_figure(panel, fit_result, theme_name, selected_orders)
            for panel in comparison_panels()
        ]

        integration_error = None
        session_id = state.get("session_id")
        bridge_state = read_bridge(session_id) if session_id else None
        if bridge_state:
            integration_error = bridge_state.get("integration_error")

        if errors:
            validation_text = errors[0] if len(errors) == 1 else f"{len(errors)} invalid blocks"
            validation_class = "hs-validation hs-validation-error"
        elif not normalized:
            validation_text = "Drag on the spectrum to add a block"
            validation_class = "hs-validation"
        elif integration_error and not fit_result:
            validation_text = integration_error
            validation_class = "hs-validation hs-validation-error"
        elif not fit_result:
            validation_text = f"{len(normalized)} block(s) valid · fit required"
            validation_class = "hs-validation"
        else:
            validation_text = f"{len(normalized)} block(s) valid"
            validation_class = "hs-validation"

        feedback = ""
        if selected_orders and fit_result:
            feedback = f"Selected LCP {selected_orders['LCP']} · RCP {selected_orders['RCP']}"

        manual_disabled = selection_mode != "Manual"
        return (
            spectrum_fig,
            *fit_figures,
            manual_disabled,
            manual_disabled,
            feedback,
            validation_text,
            validation_class,
            selected_orders,
        )

    @app.callback(
        Output("persist-version", "data"),
        Input("block-table", "data"),
        Input("selection-mode", "value"),
        Input("manual-order-lcp", "value"),
        Input("manual-order-rcp", "value"),
        Input("fit-result-store", "data"),
        Input("selected-orders-store", "data"),
        State("session-store", "data"),
        prevent_initial_call=True,
    )
    def persist(rows, selection_mode, manual_order_lcp, manual_order_rcp, fit_result, selected_orders, state):
        if not state or not state.get("session_id"):
            return no_update

        session_id = state["session_id"]
        normalized, errors = normalize_blocks(rows)
        payload = stage_payload(rows, selection_mode, selected_orders, fit_result)
        flattened_spectrum = None
        flatten_error = None

        if payload["complete"]:
            try:
                flattened_spectrum = flatten_spectrum_for_orders(
                    state.get("spectrum") or {}, fit_result or {}, selected_orders or {}
                )
            except CLCBaselineIntegrationUnavailable as exc:
                flatten_error = str(exc)
                payload["complete"] = False

        manual_orders = {"LCP": manual_order_lcp, "RCP": manual_order_rcp}

        def _mutate(bridge_state):
            previous_blocks = bridge_state.get("blocks", [])
            previous_orders = bridge_state.get("selected_orders")
            previous_mode = bridge_state.get("selection_mode")
            bridge_state["blocks"] = normalized if not errors else list(rows or [])
            bridge_state["selection_mode"] = selection_mode
            bridge_state["manual_orders"] = manual_orders
            bridge_state["selected_orders"] = selected_orders
            bridge_state["fit_result"] = fit_result
            bridge_state["flattened_spectrum"] = flattened_spectrum
            bridge_state["complete"] = payload["complete"]
            bridge_state["integration_error"] = flatten_error

            if not rows_equal(previous_blocks, bridge_state["blocks"]):
                append_event(bridge_state, "Baseline blocks updated")
            if selected_orders != previous_orders and selected_orders:
                append_event(
                    bridge_state,
                    f"Polynomial orders set — LCP {selected_orders.get('LCP')}, RCP {selected_orders.get('RCP')}",
                )
            if selection_mode != previous_mode:
                append_event(bridge_state, f"Polynomial selection mode set to {selection_mode}")

        written = update_bridge(session_id, _mutate)
        return written.get("updated_at")

    return app


_APP = create_app()


def start_server(port: int | None = None) -> int:
    """Start the singleton Dash server in a daemon thread and return its port."""
    global _SERVER_PORT, _SERVER_THREAD
    requested_port = int(port or os.environ.get("DASH_BRIDGE_PORT", "8050"))
    with _SERVER_LOCK:
        if _SERVER_THREAD and _SERVER_THREAD.is_alive():
            if _SERVER_PORT != requested_port:
                raise RuntimeError(
                    f"Baseline Dash server already running on port {_SERVER_PORT}, not {requested_port}."
                )
            return requested_port

        def _run() -> None:
            _APP.run(host="0.0.0.0", port=requested_port, debug=False, use_reloader=False)

        _SERVER_PORT = requested_port
        _SERVER_THREAD = threading.Thread(target=_run, name="hartspec-baseline-dash", daemon=True)
        _SERVER_THREAD.start()
        return requested_port
