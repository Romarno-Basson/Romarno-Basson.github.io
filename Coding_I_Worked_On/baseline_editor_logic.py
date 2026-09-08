# baseline_editor_logic.py — FR-05/06/07: pure/testable state logic for the Dash baseline editor.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

THIS_STAGE_KEY = "04_baseline_editor"
POLYNOMIAL_ORDERS = (3, 4, 5)
POLARIZATIONS = ("LCP", "RCP")
VELOCITY_PRECISION_DECIMALS = 1  # 0.1 km/s, per Page 4 build spec.


@dataclass(frozen=True)
class BaselineBlock:
    """One selected velocity interval used by the baseline editor."""

    block: int
    start: float
    end: float

    def as_row(self) -> dict[str, Any]:
        """Return the row shape consumed by Dash DataTable."""
        return {"block": self.block, "start": self.start, "end": self.end, "delete": "×"}


@dataclass(frozen=True)
class ComparisonPanel:
    """Semantic identity of one FR-06 comparison cell."""

    order: int
    polarization: str

    @property
    def key(self) -> str:
        """Stable key used in JSON fit-result payloads."""
        return f"order_{self.order}_{self.polarization.lower()}"

    @property
    def title(self) -> str:
        """Human-facing title for the six-panel comparison grid."""
        return f"Order {self.order} · {self.polarization}"


def comparison_panels() -> tuple[ComparisonPanel, ...]:
    """Return polynomial orders 3–5 for LCP then RCP in a 3×2 grid."""
    return tuple(
        ComparisonPanel(order=order, polarization=polarization)
        for polarization in POLARIZATIONS
        for order in POLYNOMIAL_ORDERS
    )


def round_velocity(value: float) -> float:
    """Round a velocity to the Page 4 drag-selection precision (0.1 km/s)."""
    return round(float(value), VELOCITY_PRECISION_DECIMALS)


def selected_data_to_block(selected_data: Mapping[str, Any] | None, block_number: int) -> BaselineBlock | None:
    """Translate Plotly ``selectedData`` into a normalized baseline block."""
    if not selected_data:
        return None
    try:
        raw_start, raw_end = selected_data["range"]["x"]
        start = round_velocity(min(float(raw_start), float(raw_end)))
        end = round_velocity(max(float(raw_start), float(raw_end)))
    except (KeyError, TypeError, ValueError):
        return None
    if start >= end:
        return None
    return BaselineBlock(block=block_number, start=start, end=end)


def normalize_blocks(rows: Sequence[Mapping[str, Any]] | None) -> tuple[list[dict[str, Any]], list[str]]:
    """Normalize/validate editable block-table rows without deleting invalid edits."""
    if not rows:
        return [], []

    normalized: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, row in enumerate(rows, start=1):
        try:
            start = round_velocity(float(row.get("start")))
            end = round_velocity(float(row.get("end")))
        except (TypeError, ValueError):
            errors.append(f"Block {index}: start and end must be numeric.")
            continue
        if start >= end:
            errors.append(f"Block {index}: start must be less than end.")
            continue
        normalized.append(BaselineBlock(index, start, end).as_row())
    return normalized, errors


def chi_squared_by_polarization(fit_result: Mapping[str, Any] | None) -> dict[str, dict[int, float]]:
    """Collect per-order FR-06 chi-squared scores separately for LCP/RCP."""
    scores: dict[str, dict[int, float]] = {polarization: {} for polarization in POLARIZATIONS}
    for panel in (fit_result or {}).get("panels") or []:
        try:
            order = int(panel["order"])
            polarization = str(panel["polarization"])
            chi_squared = float(panel["chi_squared"])
        except (KeyError, TypeError, ValueError):
            continue
        if polarization in POLARIZATIONS and order in POLYNOMIAL_ORDERS:
            scores[polarization][order] = chi_squared
    return scores


def auto_select_orders(fit_result: Mapping[str, Any] | None) -> dict[str, int] | None:
    """Select the CLC minimum-absolute chi-squared order for LCP/RCP."""
    if not fit_result:
        return None

    supplied = fit_result.get("auto_orders") or {}
    if all(supplied.get(pol) in POLYNOMIAL_ORDERS for pol in POLARIZATIONS):
        return {pol: int(supplied[pol]) for pol in POLARIZATIONS}

    scores = chi_squared_by_polarization(fit_result)
    if any(not scores[polarization] for polarization in POLARIZATIONS):
        return None
    return {
        polarization: min(
            scores[polarization], key=lambda order: abs(scores[polarization][order])
        )
        for polarization in POLARIZATIONS
    }


def auto_select_order(fit_result: Mapping[str, Any] | None) -> int | None:
    """Backward-compatible common-order helper used by older tests/callers.

    Returns a value only when automatic LCP and RCP selections agree. New Page
    4 code uses ``auto_select_orders`` because the supplied Blocks.py chooses
    LCP and RCP orders independently.
    """
    selected = auto_select_orders(fit_result)
    if not selected:
        return None
    return selected["LCP"] if selected["LCP"] == selected["RCP"] else None


def stage_payload(
    rows: Sequence[Mapping[str, Any]] | None,
    selection_mode: str,
    selected_orders: Mapping[str, Any] | None,
    fit_result: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build the JSON-serializable Page 4 payload stored in session state."""
    normalized, errors = normalize_blocks(rows)
    fit_provider = (fit_result or {}).get("provider")
    orders = dict(selected_orders or {})
    valid_orders = all(orders.get(pol) in POLYNOMIAL_ORDERS for pol in POLARIZATIONS)
    complete = bool(normalized) and not errors and valid_orders and fit_provider == "clc"
    return {
        "blocks": [
            {"block": row["block"], "start": row["start"], "end": row["end"]}
            for row in normalized
        ],
        "selection_mode": selection_mode,
        "selected_orders": {pol: orders.get(pol) for pol in POLARIZATIONS},
        "chi_squared": chi_squared_by_polarization(fit_result),
        "fit_provider": fit_provider,
        "complete": complete,
        "validation_errors": errors,
    }


def rows_equal(left: Iterable[Mapping[str, Any]], right: Iterable[Mapping[str, Any]]) -> bool:
    """Compare block rows after normalizing away display-only delete cells."""
    def _compact(rows: Iterable[Mapping[str, Any]]) -> list[tuple[Any, Any, Any]]:
        return [(row.get("block"), row.get("start"), row.get("end")) for row in rows]

    return _compact(left) == _compact(right)
