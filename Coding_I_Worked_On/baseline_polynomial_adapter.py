# baseline_polynomial_adapter.py — FR-06/07 GUI-safe adapter for CLC Blocks.py / Flat_base.py / Option.py.

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

POLYNOMIAL_ORDERS = (3, 4, 5)
POLARIZATIONS = ("LCP", "RCP")


class CLCBaselineIntegrationUnavailable(RuntimeError):
    """Raised when Page 4 cannot reproduce the supplied CLC baseline-fit contract safely."""


def _as_finite_1d(values: Sequence[float], name: str) -> np.ndarray:
    """Return one finite 1-D float array or raise a user-facing integration error."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise CLCBaselineIntegrationUnavailable(f"{name} must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(array)):
        raise CLCBaselineIntegrationUnavailable(f"{name} contains NaN or infinite values.")
    return array


def _validated_spectrum(spectrum: Mapping[str, Sequence[float]]) -> dict[str, np.ndarray]:
    """Validate the exact Vlsr/LCP/RCP contract used by Blocks.py and Flat_base.py."""
    missing = [key for key in ("Vlsr", "LCP", "RCP") if key not in spectrum]
    if missing:
        raise CLCBaselineIntegrationUnavailable(
            "Baseline fitting requires the CLC spectrum keys: " + ", ".join(missing)
        )

    arrays = {key: _as_finite_1d(spectrum[key], key) for key in ("Vlsr", "LCP", "RCP")}
    size = arrays["Vlsr"].size
    if arrays["LCP"].size != size or arrays["RCP"].size != size:
        raise CLCBaselineIntegrationUnavailable("Vlsr, LCP and RCP must contain the same number of channels.")
    return arrays


def _validated_blocks(blocks: Sequence[Mapping[str, float]]) -> list[tuple[float, float]]:
    """Validate block limits without changing their table order.

    Blocks.py uses the first block's lower bound and the last block's upper bound
    as its overall baseline envelope, so the GUI adapter deliberately preserves
    row order rather than silently sorting scientific inputs.
    """
    if not blocks:
        raise CLCBaselineIntegrationUnavailable("Choose at least one valid baseline block before fitting.")

    validated: list[tuple[float, float]] = []
    for index, block in enumerate(blocks, start=1):
        try:
            start = float(block["start"])
            end = float(block["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CLCBaselineIntegrationUnavailable(f"Block {index} must contain numeric start/end limits.") from exc
        if not np.isfinite(start) or not np.isfinite(end):
            raise CLCBaselineIntegrationUnavailable(f"Block {index} contains a non-finite velocity limit.")
        if start >= end:
            raise CLCBaselineIntegrationUnavailable(f"Block {index}: start must be less than end.")
        validated.append((start, end))
    return validated


def build_clc_masks(
    vlsr: Sequence[float],
    blocks: Sequence[Mapping[str, float]],
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Build the two inclusion masks exactly as the supplied CLC modules do.

    In Blocks.py the arrays are initialized to ones, changed to zero inside
    user-selected baseline blocks, then passed directly as NumPy masked-array
    masks. Therefore ``0`` means *included in the polynomial fit* and ``1``
    means excluded. Both polarizations initially use the same block mask.

    Returns
    -------
    mask, mask2, vll, vrl
        The LCP/RCP masks plus the first block's start and last block's end,
        matching Blocks.py / Flat_base.py naming and semantics.
    """
    velocity = _as_finite_1d(vlsr, "Vlsr")
    validated = _validated_blocks(blocks)

    mask = np.ones(velocity.size, dtype=int)
    for start, end in validated:
        inside = np.logical_and(velocity >= start, velocity <= end)
        mask[np.where(inside)] = 0

    # Blocks.py creates mask and mask2 independently but with identical ranges
    # before later sigma-clipping modifies them. FR-08 owns that later step.
    mask2 = mask.copy()
    vll = validated[0][0]
    vrl = validated[-1][1]
    return mask, mask2, vll, vrl


def _clc_chi_squared_scores(
    fits_lcp: Mapping[int, np.ndarray],
    fits_rcp: Mapping[int, np.ndarray],
    vlsr: np.ndarray,
    lcp: np.ndarray,
    rcp: np.ndarray,
    vll: float,
    vrl: float,
    mask: np.ndarray,
) -> tuple[dict[int, float], dict[int, float], dict[str, int]]:
    """Reproduce ``Stats_manager.chi_S`` and expose its six hidden scores.

    The original CLC helper accumulates ``((observed - fit) ** 2) / fit`` on
    channels where ``mask == 0`` between the outer ``vll``/``vrl`` limits,
    then chooses the polynomial order with the smallest *absolute* accumulated
    score independently for LCP and RCP. ``chi_S`` only returns the two chosen
    orders, so Page 4 performs the same loop here to also provide the six FR-06
    scores required beside the fit overlays.

    This is deliberately a literal GUI-safe extraction of the supplied CLC
    algorithm rather than a statistically reinterpreted/reduced chi-squared.
    """
    vl: int | None = None
    vr: int | None = None
    mask3 = np.ones(len(vlsr))
    mask3[np.where(np.logical_and(vlsr >= vll, vlsr <= vrl))] = 0

    # Same boundary search as Stats_manager.chi_S. The GUI validates the
    # resulting indices explicitly so an out-of-range block becomes a useful
    # error instead of the CLC's UnboundLocalError.
    for index in range(0, len(mask) - 1, 1):
        if vlsr[index] <= vll and vlsr[index + 1] >= vll:
            vl = index
        if vlsr[index] <= vrl and vlsr[index + 1] >= vrl and vrl < vlsr[len(mask3) - 1]:
            vr = index
        if vlsr[len(mask) - 1] < vrl:
            vr = len(mask3) - 2

    if vl is None or vr is None:
        raise CLCBaselineIntegrationUnavailable(
            "Baseline limits must fall within the ascending Vlsr grid for Stats_manager.chi_S."
        )

    scores_lcp = {order: 0.0 for order in POLYNOMIAL_ORDERS}
    scores_rcp = {order: 0.0 for order in POLYNOMIAL_ORDERS}

    with np.errstate(divide="ignore", invalid="ignore"):
        for index in range(vl, vr, 1):
            if mask[index] == 0:
                for order in POLYNOMIAL_ORDERS:
                    lcp_fit = fits_lcp[order][index]
                    rcp_fit = fits_rcp[order][index]
                    scores_lcp[order] += float(((lcp[index] - lcp_fit) ** 2) / lcp_fit)
                    scores_rcp[order] += float(((rcp[index] - rcp_fit) ** 2) / rcp_fit)

    auto_orders = {
        "LCP": min(POLYNOMIAL_ORDERS, key=lambda order: abs(scores_lcp[order])),
        "RCP": min(POLYNOMIAL_ORDERS, key=lambda order: abs(scores_rcp[order])),
    }
    return scores_lcp, scores_rcp, auto_orders


def fit_polynomial_comparison(
    spectrum: Mapping[str, Sequence[float]],
    blocks: Sequence[Mapping[str, float]],
) -> dict[str, Any]:
    """Generate the CLC-aligned six-panel polynomial comparison for Page 4.

    This is a GUI-safe extraction of the exact non-interactive numerical core
    visible in the supplied ``Blocks.py`` and ``Flat_base.py``:

    * mask = 1 everywhere, 0 inside each selected velocity block;
    * ``np.ma.array(..., mask=mask)`` for Vlsr and each polarization;
    * ``np.ma.polyfit`` at orders 3, 4 and 5;
    * ``np.poly1d(coefficients)(Vlsr)`` to evaluate each baseline fit.

    Calling ``Blocks.Blocks`` directly is intentionally avoided because that
    function still performs terminal ``input()`` prompts, Matplotlib window/
    PDF side effects, and baseline-file writes. ``Option.Option`` likewise
    still prompts on stdin. The Dash controls replace those interaction paths
    while preserving the CLC polynomial mathematics unchanged.
    """
    arrays = _validated_spectrum(spectrum)
    validated_blocks = _validated_blocks(blocks)
    block_rows = [{"start": start, "end": end} for start, end in validated_blocks]
    mask, mask2, vll, vrl = build_clc_masks(arrays["Vlsr"], block_rows)

    included_lcp = mask == 0
    included_rcp = mask2 == 0
    minimum_points = max(POLYNOMIAL_ORDERS) + 1
    if int(np.count_nonzero(included_lcp)) < minimum_points:
        raise CLCBaselineIntegrationUnavailable(
            f"The selected baseline blocks contain fewer than {minimum_points} channels; "
            "order-5 fitting is not defined."
        )

    panels: list[dict[str, Any]] = []
    fitted_by_polarization: dict[str, dict[int, np.ndarray]] = {
        polarization: {} for polarization in POLARIZATIONS
    }
    coefficients_by_polarization: dict[str, dict[int, np.ndarray]] = {
        polarization: {} for polarization in POLARIZATIONS
    }

    for polarization, current_mask in (("LCP", mask), ("RCP", mask2)):
        observed = arrays[polarization]
        velocity_masked = np.ma.array(arrays["Vlsr"], mask=current_mask)
        amplitude_masked = np.ma.array(observed, mask=current_mask)

        for order in POLYNOMIAL_ORDERS:
            try:
                coefficients = np.ma.polyfit(velocity_masked, amplitude_masked, order)
                polynomial = np.poly1d(coefficients)
                fitted = np.asarray(polynomial(arrays["Vlsr"]), dtype=float)
            except (TypeError, ValueError, np.linalg.LinAlgError) as exc:
                raise CLCBaselineIntegrationUnavailable(
                    f"CLC order-{order} {polarization} polynomial fit failed: {exc}"
                ) from exc

            if not np.all(np.isfinite(fitted)):
                raise CLCBaselineIntegrationUnavailable(
                    f"CLC order-{order} {polarization} polynomial produced non-finite values."
                )

            fitted_by_polarization[polarization][order] = fitted
            coefficients_by_polarization[polarization][order] = np.asarray(coefficients, dtype=float)

    scores_lcp, scores_rcp, auto_orders = _clc_chi_squared_scores(
        fitted_by_polarization["LCP"],
        fitted_by_polarization["RCP"],
        arrays["Vlsr"],
        arrays["LCP"],
        arrays["RCP"],
        vll,
        vrl,
        mask,
    )
    scores = {"LCP": scores_lcp, "RCP": scores_rcp}

    for polarization in POLARIZATIONS:
        observed = arrays[polarization]
        for order in POLYNOMIAL_ORDERS:
            panels.append(
                {
                    "order": order,
                    "polarization": polarization,
                    "x": arrays["Vlsr"].tolist(),
                    "observed": observed.tolist(),
                    "fit": fitted_by_polarization[polarization][order].tolist(),
                    "coefficients": [
                        float(value) for value in coefficients_by_polarization[polarization][order]
                    ],
                    "chi_squared": float(scores[polarization][order]),
                }
            )

    return {
        "provider": "clc",
        "source_modules": ["Blocks.py", "Flat_base.py", "Option.py", "Stats_manager.py"],
        "algorithm": (
            "CLC np.ma.polyfit orders 3-5 + Stats_manager.chi_S order selection "
            "on selected baseline blocks"
        ),
        "vll": float(vll),
        "vrl": float(vrl),
        "baseline_channel_count": int(np.count_nonzero(included_lcp)),
        "auto_orders": auto_orders,
        "panels": panels,
    }


def flatten_spectrum_for_orders(
    spectrum: Mapping[str, Sequence[float]],
    fit_result: Mapping[str, Any],
    selected_orders: Mapping[str, int],
) -> dict[str, list[float]]:
    """Subtract the selected polynomial baselines using Flat_base.py semantics.

    ``Flat_base.flatten_baseline`` ultimately subtracts the polynomial selected
    for LCP and RCP from each full amplitude array. Page 4 already has all six
    fits, so manual re-selection can re-render instantly without rerunning
    ``np.ma.polyfit`` or invoking stdin-driven CLC functions.
    """
    arrays = _validated_spectrum(spectrum)
    panels = fit_result.get("panels") or []
    flattened: dict[str, list[float]] = {"Vlsr": arrays["Vlsr"].tolist()}

    for polarization in POLARIZATIONS:
        try:
            order = int(selected_orders[polarization])
        except (KeyError, TypeError, ValueError) as exc:
            raise CLCBaselineIntegrationUnavailable(
                f"A selected polynomial order is required for {polarization}."
            ) from exc
        if order not in POLYNOMIAL_ORDERS:
            raise CLCBaselineIntegrationUnavailable(
                f"{polarization} order must be one of {POLYNOMIAL_ORDERS}, got {order}."
            )

        panel = next(
            (
                candidate
                for candidate in panels
                if int(candidate.get("order", -1)) == order
                and candidate.get("polarization") == polarization
            ),
            None,
        )
        if panel is None:
            raise CLCBaselineIntegrationUnavailable(
                f"No order-{order} {polarization} fit is available. Click Fit Polynomial again."
            )

        fitted = _as_finite_1d(panel.get("fit") or [], f"{polarization} fitted baseline")
        if fitted.size != arrays[polarization].size:
            raise CLCBaselineIntegrationUnavailable(
                f"{polarization} fitted baseline length does not match the spectrum."
            )
        flattened[polarization] = (arrays[polarization] - fitted).tolist()

    return flattened
