"""
vpin_core.py
============
Pure computation core for the Volume-Synchronized Probability of Informed
Trading (VPIN) framework of Easley, O'Hara & Lopez de Prado (2012).

This module is deliberately UI-agnostic so it can be unit-tested, imported
into research notebooks, or driven by any front-end (the Streamlit app in
``app.py`` is one such front-end).

Pipeline
--------
raw ticks -> clean_tick_data -> compute_adv -> create_volume_buckets
          -> compute_bvc -> calculate_vpin -> plot_toxicity
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

__all__ = [
    "VPINConfig",
    "VPINResult",
    "clean_tick_data",
    "compute_adv",
    "create_volume_buckets",
    "compute_bvc",
    "calculate_vpin",
    "derive_threshold",
    "run_vpin_pipeline",
    "plot_toxicity",
    "generate_synthetic_ticks",
]

# --------------------------------------------------------------------------- #
# Module constants
# --------------------------------------------------------------------------- #

_SQRT2: float = math.sqrt(2.0)
#: Relative floating-point tolerance used when detecting bucket boundaries.
_FP_EPS: float = 1e-9

_TOXIC_FILL = "rgba(214, 39, 40, 0.14)"
_PRICE_COLOR = "#1f77b4"
_VPIN_COLOR = "#d62728"


# --------------------------------------------------------------------------- #
# Configuration / result containers
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class VPINConfig:
    """Immutable configuration for a single VPIN run.

    Attributes
    ----------
    bucket_count:
        ``N`` in ``V_bucket = ADV / N``. Default 50 (one "volume day" of 50
        buckets), matching the original paper.
    window:
        ``V_window`` -- number of volume buckets in the rolling VPIN window.
    sigma_mode:
        ``"global"`` (single full-sample sigma) or ``"rolling"`` (trailing
        lookback std of bucket price changes).
    sigma_lookback:
        Lookback in buckets when ``sigma_mode == "rolling"``.
    bucket_size:
        Optional manual override of ``V_bucket``. When ``None`` the ADV / N
        rule is used.
    drop_partial_last:
        Drop the trailing, incompletely filled bucket so every bucket has an
        identical volume. Recommended (keeps the VPIN denominator exact).
    threshold_mode:
        ``"fixed"`` or ``"quantile"``.
    threshold_fixed:
        Threshold used when ``threshold_mode == "fixed"``.
    threshold_quantile:
        Quantile used when ``threshold_mode == "quantile"``.
    """

    bucket_count: int = 50
    window: int = 50
    sigma_mode: str = "global"
    sigma_lookback: int = 50
    bucket_size: Optional[float] = None
    drop_partial_last: bool = True
    threshold_mode: str = "fixed"
    threshold_fixed: float = 0.85
    threshold_quantile: float = 0.95

    def validate(self) -> None:
        if self.bucket_count < 1:
            raise ValueError("bucket_count (N) must be >= 1.")
        if self.window < 1:
            raise ValueError("window (V_window) must be >= 1.")
        if self.sigma_mode not in {"global", "rolling"}:
            raise ValueError("sigma_mode must be 'global' or 'rolling'.")
        if self.sigma_mode == "rolling" and self.sigma_lookback < 2:
            raise ValueError("sigma_lookback must be >= 2 for a rolling std.")
        if self.bucket_size is not None and self.bucket_size <= 0:
            raise ValueError("bucket_size override must be strictly positive.")
        if self.threshold_mode not in {"fixed", "quantile"}:
            raise ValueError("threshold_mode must be 'fixed' or 'quantile'.")


@dataclass
class VPINResult:
    """Container for every artefact produced by :func:`run_vpin_pipeline`."""

    ticks: pd.DataFrame
    frame: pd.DataFrame
    adv: float
    bucket_size: float
    threshold: float
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def vpin(self) -> pd.Series:
        return self.frame["vpin"]


# --------------------------------------------------------------------------- #
# 1. Data ingestion & cleaning
# --------------------------------------------------------------------------- #


def _coerce_timestamps(series: pd.Series) -> pd.Series:
    """Coerce a timestamp column to ``datetime64[ns]``.

    Handles ISO strings, ``datetime`` objects and raw numeric epochs. For
    numeric input the unit (s / ms / us / ns) is inferred from the median
    magnitude, which is robust for both seconds-since-epoch feeds and
    nanosecond exchange timestamps.
    """
    if pd.api.types.is_numeric_dtype(series):
        arr = series.to_numpy(dtype="float64", na_value=np.nan)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return pd.to_datetime(series, errors="coerce")
        magnitude = float(np.median(np.abs(finite)))
        if magnitude >= 1e17:
            unit = "ns"
        elif magnitude >= 1e14:
            unit = "us"
        elif magnitude >= 1e11:
            unit = "ms"
        else:
            unit = "s"
        return pd.to_datetime(arr, unit=unit, errors="coerce")
    return pd.to_datetime(series, errors="coerce")


def clean_tick_data(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    price_col: str = "price",
    volume_col: str = "volume",
) -> pd.DataFrame:
    """Normalise a raw tick feed into a canonical ``[timestamp, price, volume]`` frame.

    The cleaning layer performs, in order:

    1. Column projection + renaming to the canonical schema.
    2. Timestamp coercion (string / epoch / datetime).
    3. Numeric coercion of price and volume.
    4. Removal of NaN rows, non-positive prices and non-positive volumes
       (zero-volume heartbeats, cancelled prints, bad exchange ticks).
    5. Stable chronological sort (preserves the original ordering of
       equal timestamps, i.e. sub-millisecond sequencing).
    6. De-duplication of identical timestamps by aggregating volume and
       computing a volume-weighted average price (VWAP) for the group.

    Parameters
    ----------
    df:
        Raw tick DataFrame.
    timestamp_col, price_col, volume_col:
        Source column names.

    Returns
    -------
    pandas.DataFrame
        Columns ``timestamp`` (datetime64), ``price`` (float64),
        ``volume`` (float64), reset to a clean RangeIndex.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["timestamp", "price", "volume"])

    if len({timestamp_col, price_col, volume_col}) != 3:
        raise ValueError("timestamp_col, price_col and volume_col must be distinct.")

    missing = [c for c in (timestamp_col, price_col, volume_col) if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required column(s): {missing}. Available: {list(df.columns)}")

    out = df.loc[:, [timestamp_col, price_col, volume_col]].copy()
    out.columns = ["timestamp", "price", "volume"]

    out["timestamp"] = _coerce_timestamps(out["timestamp"])
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce")

    out = out.dropna(subset=["timestamp", "price", "volume"])
    out = out[(out["price"] > 0.0) & (out["volume"] > 0.0)]

    if out.empty:
        return out.reset_index(drop=True)

    # ``kind="stable"`` keeps intra-timestamp arrival order intact.
    out = out.sort_values("timestamp", kind="stable")

    if bool(out["timestamp"].duplicated().any()):
        ts = out["timestamp"].to_numpy()
        vol = out["volume"].to_numpy(dtype=np.float64)
        notional = out["price"].to_numpy(dtype=np.float64) * vol
        grouped = (
            pd.DataFrame({"timestamp": ts, "volume": vol, "notional": notional})
            .groupby("timestamp", as_index=False, sort=True)
            .sum()
        )
        grouped["price"] = grouped["notional"] / grouped["volume"]
        out = grouped.loc[:, ["timestamp", "price", "volume"]]

    return out.reset_index(drop=True)


def compute_adv(ticks: pd.DataFrame, freq: str = "1D") -> float:
    """Average daily volume over the (already cleaned) tick frame.

    Days with zero traded volume (weekends, holidays) are excluded so they do
    not dilute the estimate.
    """
    if ticks.empty:
        return 0.0
    daily = ticks.set_index("timestamp")["volume"].resample(freq).sum()
    daily = daily[daily > 0.0]
    if daily.empty:
        return float(ticks["volume"].sum())
    return float(daily.mean())


# --------------------------------------------------------------------------- #
# 2. Volume-time bucketization (exact print splitting)
# --------------------------------------------------------------------------- #


def create_volume_buckets(
    ticks: pd.DataFrame,
    bucket_size: float,
    drop_partial_last: bool = True,
) -> pd.DataFrame:
    """Build volume buckets, splitting prints that straddle bucket boundaries.

    Algorithm
    ---------
    Let ``C_i`` be the cumulative traded volume *after* trade ``i`` and
    ``C_{i-1}`` the cumulative volume *before* it. Bucket ``k`` occupies the
    cumulative-volume interval ``[k*B, (k+1)*B)``. Trade ``i`` therefore
    touches buckets ``floor(C_{i-1}/B)`` through ``floor((C_i - eps)/B)``.

    Every touched (trade, bucket) pair is materialised with ``np.repeat`` and
    the exact overlap

        ``alloc = min(C_i, (k+1)*B) - max(C_{i-1}, k*B)``

    is accumulated with ``np.add.reduceat``. Because trades are chronologically
    sorted the bucket index array is monotonically non-decreasing, which makes
    the reduction O(n) instead of requiring a hash-based groupby.

    Parameters
    ----------
    ticks:
        Cleaned tick frame (see :func:`clean_tick_data`).
    bucket_size:
        Target volume per bucket, ``V_bucket``.
    drop_partial_last:
        Discard the final bucket if it is not completely filled. Since every
        non-terminal bucket is filled to exactly ``V_bucket`` by construction,
        this guarantees a constant VPIN denominator.

    Returns
    -------
    pandas.DataFrame
        One row per bucket with columns ``bucket_id``, ``timestamp`` (close
        time = timestamp of the last print allocated to the bucket),
        ``price`` (bucket close price) and ``volume``.
    """
    if bucket_size <= 0.0 or not np.isfinite(bucket_size):
        raise ValueError("bucket_size must be a finite, strictly positive number.")
    if ticks.empty:
        return pd.DataFrame(columns=["bucket_id", "timestamp", "price", "volume"])

    price = ticks["price"].to_numpy(dtype=np.float64)
    volume = ticks["volume"].to_numpy(dtype=np.float64)
    ts = ticks["timestamp"].to_numpy()

    cum = np.cumsum(volume)
    cum_prev = cum - volume

    eps = bucket_size * _FP_EPS

    # ``+eps`` / ``-eps`` make boundary-exact prints deterministic despite
    # floating point representation error in the running cumulative sum.
    first_bucket = np.floor((cum_prev + eps) / bucket_size).astype(np.int64)
    last_bucket = np.floor((cum - eps) / bucket_size).astype(np.int64)
    np.maximum(last_bucket, first_bucket, out=last_bucket)

    counts = (last_bucket - first_bucket + 1).astype(np.int64)
    total_segments = int(counts.sum())
    if total_segments == 0:
        return pd.DataFrame(columns=["bucket_id", "timestamp", "price", "volume"])

    # Expand each trade into one entry per bucket it touches.
    trade_idx = np.repeat(np.arange(ticks.shape[0], dtype=np.int64), counts)
    group_offsets = np.repeat(np.cumsum(counts) - counts, counts)
    seg_offset = np.arange(total_segments, dtype=np.int64) - group_offsets
    bucket_idx = np.repeat(first_bucket, counts) + seg_offset

    # Exact volume allocated to each (trade, bucket) segment.
    lower = np.maximum(cum_prev[trade_idx], bucket_idx.astype(np.float64) * bucket_size)
    upper = np.minimum(cum[trade_idx], (bucket_idx + 1).astype(np.float64) * bucket_size)
    seg_volume = np.maximum(upper - lower, 0.0)

    # Reduceat over monotonically increasing group starts -> O(n).
    starts = np.concatenate(([0], np.flatnonzero(np.diff(bucket_idx)) + 1))
    ends = np.concatenate((starts[1:] - 1, [total_segments - 1]))

    bucket_volume = np.add.reduceat(seg_volume, starts)
    close_trade = trade_idx[ends]

    buckets = pd.DataFrame(
        {
            "bucket_id": bucket_idx[starts],
            "timestamp": ts[close_trade],
            "price": price[close_trade],
            "volume": bucket_volume,
        }
    )

    if drop_partial_last and len(buckets) > 1:
        tolerance = bucket_size * 1e-6
        if buckets["volume"].iloc[-1] < bucket_size - tolerance:
            buckets = buckets.iloc[:-1]

    return buckets.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 3. Bulk Volume Classification
# --------------------------------------------------------------------------- #


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF, vectorised, NaN/inf-safe (returns 0.5 there)."""
    x = np.asarray(x, dtype=np.float64)
    out = np.full(x.shape, 0.5, dtype=np.float64)
    mask = np.isfinite(x)
    if mask.any():
        out[mask] = [0.5 * (1.0 + math.erf(v / _SQRT2)) for v in x[mask]]
    return out


def compute_bvc(
    buckets: pd.DataFrame,
    sigma_mode: str = "global",
    sigma_lookback: int = 50,
) -> pd.DataFrame:
    """Bulk Volume Classification over volume buckets.

    ``V_B = V * CDF(dP / sigma_dP)`` and ``V_S = V - V_B``.

    The first bucket has no predecessor, so its price change is defined as
    exactly zero, yielding a neutral 50/50 split. If ``sigma_dP`` is zero or
    undefined (a perfectly flat price path) the split also defaults to 50/50
    rather than raising a division-by-zero.

    Parameters
    ----------
    buckets:
        Output of :func:`create_volume_buckets`.
    sigma_mode:
        ``"global"`` or ``"rolling"``.
    sigma_lookback:
        Trailing window length for ``"rolling"``.

    Returns
    -------
    pandas.DataFrame
        Copy of ``buckets`` enriched with ``delta_p``, ``sigma_delta_p``,
        ``buy_prob``, ``buy_volume``, ``sell_volume`` and ``imbalance``.
    """
    if buckets.empty:
        return buckets.assign(
            delta_p=pd.Series(dtype=float),
            sigma_delta_p=pd.Series(dtype=float),
            buy_prob=pd.Series(dtype=float),
            buy_volume=pd.Series(dtype=float),
            sell_volume=pd.Series(dtype=float),
            imbalance=pd.Series(dtype=float),
        )

    out = buckets.copy()
    delta_p = out["price"].diff()
    if len(delta_p) > 0:
        delta_p.iloc[0] = 0.0

    if sigma_mode == "global":
        values = delta_p.to_numpy(dtype=np.float64)
        sigma_value = float(np.nanstd(values, ddof=1)) if values.size > 1 else 0.0
        sigma = pd.Series(sigma_value, index=delta_p.index, dtype=np.float64)
    elif sigma_mode == "rolling":
        sigma = delta_p.rolling(window=int(sigma_lookback), min_periods=2).std()
    else:  # pragma: no cover - guarded by VPINConfig.validate()
        raise ValueError("sigma_mode must be 'global' or 'rolling'.")

    # Zero / undefined volatility -> neutral classification (handled by _norm_cdf).
    sigma = sigma.replace(0.0, np.nan)

    z = delta_p / sigma
    z = z.replace([np.inf, -np.inf], np.nan)

    buy_prob = pd.Series(_norm_cdf(z.to_numpy(dtype=np.float64)), index=out.index).clip(0.0, 1.0)

    out["delta_p"] = delta_p
    out["sigma_delta_p"] = sigma
    out["buy_prob"] = buy_prob
    out["buy_volume"] = out["volume"] * buy_prob
    out["sell_volume"] = out["volume"] - out["buy_volume"]
    out["imbalance"] = (out["buy_volume"] - out["sell_volume"]).abs()
    return out


# --------------------------------------------------------------------------- #
# 4. Rolling VPIN
# --------------------------------------------------------------------------- #


def calculate_vpin(bvc: pd.DataFrame, bucket_size: float, window: int = 50) -> pd.DataFrame:
    """Rolling VPIN.

    ``VPIN_t = sum_{i=t-w+1..t} |V_B,i - V_S,i| / (V_window * V_bucket)``

    The output is clipped to ``[0, 1]``. Since ``|V_B - V_S| <= V_bucket`` by
    construction the bound is mathematically guaranteed; the clip is a purely
    defensive guard against floating-point drift.
    """
    if window < 1:
        raise ValueError("window must be >= 1.")
    if bucket_size <= 0.0:
        raise ValueError("bucket_size must be strictly positive.")

    out = bvc.copy()
    rolling_imbalance = out["imbalance"].rolling(window=window, min_periods=window).sum()
    denominator = float(window) * float(bucket_size)
    out["vpin"] = (rolling_imbalance / denominator).clip(0.0, 1.0)
    return out


def derive_threshold(
    vpin: pd.Series,
    mode: str = "fixed",
    fixed: float = 0.85,
    quantile: float = 0.95,
) -> float:
    """Resolve the toxicity threshold from a fixed level or a rolling quantile."""
    if mode == "fixed":
        return float(np.clip(fixed, 0.0, 1.0))
    clean = vpin.dropna()
    if clean.empty:
        return float(np.clip(fixed, 0.0, 1.0))
    return float(np.clip(clean.quantile(quantile), 0.0, 1.0))


# --------------------------------------------------------------------------- #
# 5. End-to-end pipeline
# --------------------------------------------------------------------------- #


def run_vpin_pipeline(
    raw: pd.DataFrame,
    timestamp_col: str = "timestamp",
    price_col: str = "price",
    volume_col: str = "volume",
    config: Optional[VPINConfig] = None,
) -> VPINResult:
    """Execute the full VPIN pipeline and return every intermediate artefact."""
    config = config or VPINConfig()
    config.validate()

    ticks = clean_tick_data(raw, timestamp_col, price_col, volume_col)
    if ticks.empty:
        raise ValueError("No valid ticks remain after the cleaning stage.")

    adv = compute_adv(ticks)

    if config.bucket_size is not None:
        bucket_size = float(config.bucket_size)
        bucket_source = "manual override"
    else:
        bucket_size = adv / float(config.bucket_count)
        bucket_source = f"ADV / N (N={config.bucket_count})"

    if not np.isfinite(bucket_size) or bucket_size <= 0.0:
        raise ValueError(
            "Resolved bucket size is non-positive. Increase the data span, "
            "lower N, or provide a manual bucket size."
        )

    buckets = create_volume_buckets(
        ticks, bucket_size=bucket_size, drop_partial_last=config.drop_partial_last
    )
    if buckets.empty:
        raise ValueError(
            "Not enough traded volume to fill a single complete volume bucket. "
            "Lower N or supply a manual bucket size."
        )

    bvc = compute_bvc(buckets, sigma_mode=config.sigma_mode, sigma_lookback=config.sigma_lookback)
    frame = calculate_vpin(bvc, bucket_size=bucket_size, window=config.window)

    threshold = derive_threshold(
        frame["vpin"],
        mode=config.threshold_mode,
        fixed=config.threshold_fixed,
        quantile=config.threshold_quantile,
    )

    valid_vpin = frame["vpin"].dropna()
    toxic_fraction = (
        float((valid_vpin >= threshold).mean()) if not valid_vpin.empty else 0.0
    )

    diagnostics: Dict[str, Any] = {
        "raw_ticks": int(len(raw)),
        "clean_ticks": int(len(ticks)),
        "removed_ticks": int(len(raw) - len(ticks)),
        "buckets": int(len(frame)),
        "adv": float(adv),
        "bucket_size": float(bucket_size),
        "bucket_source": bucket_source,
        "threshold": float(threshold),
        "vpin_points": int(valid_vpin.size),
        "vpin_mean": float(valid_vpin.mean()) if not valid_vpin.empty else float("nan"),
        "vpin_max": float(valid_vpin.max()) if not valid_vpin.empty else float("nan"),
        "vpin_last": float(valid_vpin.iloc[-1]) if not valid_vpin.empty else float("nan"),
        "toxic_fraction": toxic_fraction,
        "span_start": ticks["timestamp"].iloc[0],
        "span_end": ticks["timestamp"].iloc[-1],
    }

    return VPINResult(
        ticks=ticks,
        frame=frame,
        adv=float(adv),
        bucket_size=float(bucket_size),
        threshold=float(threshold),
        diagnostics=diagnostics,
    )


# --------------------------------------------------------------------------- #
# 6. Visualisation
# --------------------------------------------------------------------------- #


def _true_runs(mask: np.ndarray, limit: int = 300) -> List[Tuple[int, int]]:
    """Return ``(start, end)`` index pairs for each contiguous run of True."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([idx[0]], idx[breaks + 1]))
    ends = np.concatenate((idx[breaks], [idx[-1]]))
    return list(zip(starts.tolist(), ends.tolist()))[:limit]


def plot_toxicity(
    result: VPINResult,
    threshold: Optional[float] = None,
    title: str = "VPIN — Order-Flow Toxicity",
) -> go.Figure:
    """Two synchronized stacked subplots: price-with-toxic-regimes and rolling VPIN."""
    frame = result.frame
    if frame.empty:
        raise ValueError("Nothing to plot: the bucket frame is empty.")

    thr = float(result.threshold if threshold is None else threshold)
    x = frame["timestamp"]
    price = frame["price"]
    vpin = frame["vpin"]

    mask = (vpin.notna() & (vpin >= thr)).to_numpy()

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.62, 0.38],
        vertical_spacing=0.06,
        subplot_titles=("Price · toxicity regimes shaded", "Rolling VPIN"),
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=price,
            name="Bucket close price",
            mode="lines",
            line=dict(color=_PRICE_COLOR, width=1.2),
            hovertemplate="%{x|%Y-%m-%d %H:%M:%S}<br>Price %{y:.4f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    for start, end in _true_runs(mask):
        fig.add_vrect(
            x0=x.iloc[start],
            x1=x.iloc[end],
            fillcolor=_TOXIC_FILL,
            line_width=0,
            layer="below",
            row=1,
            col=1,
        )
        fig.add_vrect(
            x0=x.iloc[start],
            x1=x.iloc[end],
            fillcolor=_TOXIC_FILL,
            line_width=0,
            layer="below",
            row=2,
            col=1,
        )

    if mask.any():
        fig.add_trace(
            go.Scatter(
                x=x[mask],
                y=price[mask],
                name=f"Toxic bucket (VPIN ≥ {thr:.3f})",
                mode="markers",
                marker=dict(color=_VPIN_COLOR, size=4, symbol="circle", opacity=0.75),
                hovertemplate="%{x|%Y-%m-%d %H:%M:%S}<br>Price %{y:.4f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=vpin,
            name="VPIN",
            mode="lines",
            line=dict(color=_VPIN_COLOR, width=1.4),
            hovertemplate="%{x|%Y-%m-%d %H:%M:%S}<br>VPIN %{y:.4f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.add_hline(
        y=thr,
        line=dict(color="crimson", dash="dash", width=1.1),
        annotation_text=f"Toxicity threshold = {thr:.3f}",
        annotation_position="top left",
        row=2,
        col=1,
    )

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="VPIN", range=[0.0, 1.0], row=2, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=1)

    fig.update_layout(
        title=dict(text=title, x=0.01, xanchor="left"),
        height=760,
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        margin=dict(l=60, r=30, t=90, b=50),
    )
    return fig


# --------------------------------------------------------------------------- #
# 7. Synthetic data generator (runs out-of-the-box)
# --------------------------------------------------------------------------- #


def generate_synthetic_ticks(
    n_days: int = 4,
    trades_per_day: int = 50_000,
    seed: int = 7,
    start: str = "2024-03-04 09:30:00",
    start_price: float = 100.0,
    session_hours: float = 6.5,
    inject_dirt: bool = True,
) -> pd.DataFrame:
    """Generate a realistic multi-day synthetic tick feed.

    The generator deliberately injects *informed-trading bursts*: windows in
    which volatility, volume and directional drift all rise simultaneously.
    A correct VPIN implementation should light up during those windows.

    When ``inject_dirt`` is True the feed is also polluted with the artefacts
    of a real exchange feed -- duplicate timestamps, zero-volume heartbeats,
    negative prices and NaNs -- and the rows are shuffled so that the cleaning
    layer is genuinely exercised.
    """
    rng = np.random.default_rng(seed)
    session_seconds = session_hours * 3600.0
    mean_gap = session_seconds / max(int(trades_per_day), 1)

    frames: List[pd.DataFrame] = []
    price_level = float(start_price)
    session_start = pd.Timestamp(start)

    for day in range(int(n_days)):
        day_open = session_start + pd.Timedelta(days=day)

        gaps = rng.exponential(scale=mean_gap, size=int(trades_per_day))
        timestamps = day_open + pd.to_timedelta(np.cumsum(gaps), unit="s")

        # Latent toxicity intensity: two informed bursts per session.
        regime = np.zeros(int(trades_per_day), dtype=np.float64)
        direction = np.zeros(int(trades_per_day), dtype=np.float64)
        for centre_frac, width_frac, amplitude, sign in (
            (0.24, 0.030, 1.00, 1.0),
            (0.58, 0.045, 1.35, -1.0),
            (0.86, 0.020, 0.80, 1.0),
        ):
            centre = int(centre_frac * trades_per_day)
            half = max(int(width_frac * trades_per_day), 1)
            lo, hi = max(centre - half, 0), min(centre + half, int(trades_per_day))
            if hi <= lo:
                continue
            x = np.linspace(-1.0, 1.0, hi - lo)
            regime[lo:hi] += amplitude * np.exp(-4.0 * x * x)
            direction[lo:hi] = sign

        sigma = 0.00035 * (1.0 + 0.8 * regime)
        shocks = rng.normal(0.0, 1.0, size=int(trades_per_day)) * sigma * price_level
        drift = direction * regime * 0.0015
        price_level = max(price_level + float(np.sum(shocks) + np.sum(drift)), 1.0)

        # Rebuild the path deterministically from the realised end level.
        raw_path = np.cumsum(shocks + drift)
        raw_path = raw_path - raw_path[-1] + price_level
        prices = np.maximum(raw_path + (price_level - raw_path[-1]), 1.0)
        prices = np.maximum(prices + (price_level - prices[-1]), 1.0)

        base_volume = rng.lognormal(mean=4.2, sigma=0.9, size=int(trades_per_day))
        volumes = np.maximum(np.round(base_volume * (1.0 + 1.8 * regime)), 1.0)

        frames.append(
            pd.DataFrame(
                {"timestamp": timestamps, "price": prices, "volume": volumes}
            )
        )

    ticks = pd.concat(frames, ignore_index=True)

    if inject_dirt and len(ticks) > 100:
        n = len(ticks)

        duplicates = ticks.iloc[rng.choice(n, size=max(int(0.010 * n), 1), replace=False)].copy()

        zero_vol = ticks.iloc[rng.choice(n, size=max(int(0.004 * n), 1), replace=False)].copy()
        zero_vol["volume"] = 0.0

        negative = ticks.iloc[rng.choice(n, size=max(int(0.002 * n), 1), replace=False)].copy()
        negative["price"] = -negative["price"].abs()

        nan_rows = ticks.iloc[rng.choice(n, size=max(int(0.002 * n), 1), replace=False)].copy()
        nan_rows["price"] = np.nan

        ticks = pd.concat([ticks, duplicates, zero_vol, negative, nan_rows], ignore_index=True)

    return ticks.sample(frac=1.0, random_state=seed).reset_index(drop=True)
