"""
======
Streamlit front-end for the VPIN (Volume-Synchronized Probability of
Informed Trading) pipeline defined in ``vpin_core.py``.

Run locally:
    streamlit run app.py

Deploy (Streamlit Community Cloud):
    Main file path -> app.py
"""

from __future__ import annotations

import io
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from vpin_core import (
    VPINConfig,
    VPINResult,
    generate_synthetic_ticks,
    plot_toxicity,
    run_vpin_pipeline,
)

# --------------------------------------------------------------------------- #
# Page configuration
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="VPIN · Order-Flow Toxicity",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------- #
# Cached helpers
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def _cached_pipeline(
    raw: pd.DataFrame,
    ts_col: str,
    price_col: str,
    volume_col: str,
    config: VPINConfig,
) -> VPINResult:
    """Cache the whole pipeline so sidebar tweaks are instant."""
    return run_vpin_pipeline(
        raw,
        timestamp_col=ts_col,
        price_col=price_col,
        volume_col=volume_col,
        config=config,
    )


@st.cache_data(show_spinner=False)
def _cached_synthetic(
    n_days: int,
    trades_per_day: int,
    seed: int,
    inject_dirt: bool,
) -> pd.DataFrame:
    """Cache synthetic data generation."""
    return generate_synthetic_ticks(
        n_days=n_days,
        trades_per_day=trades_per_day,
        seed=seed,
        inject_dirt=inject_dirt,
    )


def _read_uploaded(uploaded) -> pd.DataFrame:
    """Best-effort reader for CSV / TSV / Parquet / Excel uploads."""
    name = uploaded.name.lower()
    data = uploaded.getvalue()

    if name.endswith(".parquet"):
        return pd.read_parquet(io.BytesIO(data))
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(data))

    # CSV / TSV / TXT: sniff the delimiter.
    for sep in (",", ";", "\t", "|"):
        try:
            df = pd.read_csv(io.BytesIO(data), sep=sep)
            if df.shape[1] > 1:
                return df
        except Exception:
            continue
    return pd.read_csv(io.BytesIO(data))


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("📊 VPIN · Volume-Synchronized Probability of Informed Trading")
st.caption(
    "Rolling order-flow toxicity from high-frequency trade prints — "
    "reproducing Easley, O'Hara & López de Prado (2012)."
)

with st.expander("ℹ️ How this app works", expanded=False):
    st.markdown(
        """
        1. **Clean** the raw tick feed — drop zero/negative volumes, bad prices,
           NaNs, and aggregate duplicate timestamps by VWAP + summed volume.
        2. **Bucket** trades into equal-volume buckets of size
           `V_bucket = ADV / N`. Prints that straddle a boundary are split
           exactly, not rounded.
        3. **Classify** each bucket's volume via Bulk Volume Classification:
           `V_B = V · Φ(ΔP / σ_ΔP)`, `V_S = V − V_B`.
        4. **Roll** the absolute imbalance over `V_window` buckets and
           normalise to `[0, 1]` to obtain VPIN.
        5. **Flag** buckets whose VPIN crosses a fixed (default 0.85) or
           quantile-based toxicity threshold.
        """
    )


# --------------------------------------------------------------------------- #
# Sidebar — data source
# --------------------------------------------------------------------------- #
st.sidebar.title("⚙️ Configuration")
st.sidebar.caption("VPIN pipeline · Easley, O'Hara & López de Prado (2012)")

st.sidebar.subheader("1 · Data source")
source = st.sidebar.radio(
    "Choose a feed",
    ("Synthetic demo", "Upload file"),
    index=0,
    label_visibility="collapsed",
)

raw_df: Optional[pd.DataFrame] = None
ts_col, price_col, volume_col = "timestamp", "price", "volume"

if source == "Synthetic demo":
    with st.sidebar:
        n_days = st.slider("Days of synthetic data", 1, 10, 3, 1)
        trades_per_day = st.slider("Trades per day", 5_000, 200_000, 25_000, 5_000)
        seed = st.number_input("Random seed", 0, 10_000, 7, 1)
        inject_dirt = st.checkbox(
            "Inject dirty ticks (duplicates, NaNs, zero-vol)",
            value=True,
        )
    raw_df = _cached_synthetic(
        int(n_days), int(trades_per_day), int(seed), bool(inject_dirt)
    )
else:
    uploaded = st.sidebar.file_uploader(
        "Upload tick file (CSV, TSV, Parquet, Excel)",
        type=["csv", "tsv", "txt", "parquet", "xlsx", "xls"],
    )
    if uploaded is None:
        st.info("⬅️ Upload a file or switch to the **Synthetic demo** to begin.")
        st.stop()

    try:
        raw_df = _read_uploaded(uploaded)
    except Exception as exc:
        st.error(f"Could not read the uploaded file: {exc}")
        st.stop()

    cols = list(raw_df.columns)
    ts_col = st.sidebar.selectbox("Timestamp column", cols, index=0)
    price_col = st.sidebar.selectbox(
        "Price column", cols, index=min(1, len(cols) - 1)
    )
    volume_col = st.sidebar.selectbox(
        "Volume column", cols, index=min(2, len(cols) - 1)
    )


# --------------------------------------------------------------------------- #
# Sidebar — VPIN configuration
# --------------------------------------------------------------------------- #
st.sidebar.subheader("2 · Volume bucketing")
bucket_count = st.sidebar.number_input(
    "N (buckets per ADV day)", min_value=1, max_value=1000, value=50, step=1
)
window = st.sidebar.number_input(
    "V_window (rolling VPIN buckets)", min_value=1, max_value=1000, value=50, step=1
)

use_manual_bucket = st.sidebar.checkbox("Override bucket size manually", value=False)
manual_bucket_size: Optional[float] = None
if use_manual_bucket:
    manual_bucket_size = float(
        st.sidebar.number_input(
            "Manual V_bucket",
            min_value=0.0001,
            value=10_000.0,
            step=100.0,
            format="%.4f",
        )
    )

st.sidebar.subheader("3 · Bulk Volume Classification")
sigma_mode = st.sidebar.selectbox("σ(ΔP) mode", ("global", "rolling"), index=0)
sigma_lookback = st.sidebar.number_input(
    "σ lookback (rolling only)", min_value=2, max_value=1000, value=50, step=1
)
drop_partial_last = st.sidebar.checkbox("Drop partial trailing bucket", value=True)

st.sidebar.subheader("4 · Toxicity threshold")
threshold_mode = st.sidebar.selectbox(
    "Threshold mode", ("fixed", "quantile"), index=0
)
if threshold_mode == "fixed":
    threshold_fixed = st.sidebar.slider("Fixed threshold", 0.0, 1.0, 0.85, 0.01)
    threshold_quantile = 0.95
else:
    threshold_quantile = st.sidebar.slider("Quantile", 0.50, 0.999, 0.95, 0.005)
    threshold_fixed = 0.85


# --------------------------------------------------------------------------- #
# Run pipeline
# --------------------------------------------------------------------------- #
config = VPINConfig(
    bucket_count=int(bucket_count),
    window=int(window),
    sigma_mode=str(sigma_mode),
    sigma_lookback=int(sigma_lookback),
    bucket_size=manual_bucket_size,
    drop_partial_last=bool(drop_partial_last),
    threshold_mode=str(threshold_mode),
    threshold_fixed=float(threshold_fixed),
    threshold_quantile=float(threshold_quantile),
)

if raw_df is None or raw_df.empty:
    st.warning("No data available. Upload a file or use the synthetic demo.")
    st.stop()

try:
    with st.spinner("Running VPIN pipeline…"):
        result = _cached_pipeline(raw_df, ts_col, price_col, volume_col, config)
except ValueError as exc:
    st.error(f"**Pipeline error:** {exc}")
    with st.expander("Preview the raw input"):
        st.dataframe(raw_df.head(50), use_container_width=True)
    st.stop()
except Exception as exc:  # pragma: no cover - defensive
    st.exception(exc)
    st.stop()


# --------------------------------------------------------------------------- #
# Metric strip
# --------------------------------------------------------------------------- #
d = result.diagnostics
removed = int(d["removed_ticks"])
delta_txt = f"-{removed:,} dirty rows" if removed else "0 removed"

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Clean ticks", f"{d['clean_ticks']:,}", delta_txt)
c2.metric("Volume buckets", f"{d['buckets']:,}")
c3.metric("ADV", f"{d['adv']:,.0f}")
c4.metric("V_bucket", f"{d['bucket_size']:,.2f}", d["bucket_source"])
c5.metric("Threshold", f"{d['threshold']:.3f}", threshold_mode)

v1, v2, v3, v4 = st.columns(4)
v1.metric(
    "VPIN mean",
    f"{d['vpin_mean']:.4f}" if np.isfinite(d["vpin_mean"]) else "—",
)
v2.metric(
    "VPIN max",
    f"{d['vpin_max']:.4f}" if np.isfinite(d["vpin_max"]) else "—",
)
v3.metric(
    "VPIN last",
    f"{d['vpin_last']:.4f}" if np.isfinite(d["vpin_last"]) else "—",
)
v4.metric("Toxic buckets", f"{d['toxic_fraction'] * 100:.2f}%")


# --------------------------------------------------------------------------- #
# Chart
# --------------------------------------------------------------------------- #
if result.frame["vpin"].notna().sum() == 0:
    st.warning(
        f"No VPIN values could be computed — you have {len(result.frame)} "
        f"bucket(s) but V_window = {config.window}. Increase the data span, "
        f"lower V_window, or lower N."
    )

fig = plot_toxicity(result, title="VPIN — Order-Flow Toxicity")
st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


# --------------------------------------------------------------------------- #
# Data tabs
# --------------------------------------------------------------------------- #
tab1, tab2, tab3 = st.tabs(["📦 Bucket frame", "🧹 Cleaned ticks", "🔍 Diagnostics"])

with tab1:
    show_cols = [
        "bucket_id",
        "timestamp",
        "price",
        "volume",
        "delta_p",
        "sigma_delta_p",
        "buy_prob",
        "buy_volume",
        "sell_volume",
        "imbalance",
        "vpin",
    ]
    show_cols = [c for c in show_cols if c in result.frame.columns]
    st.dataframe(result.frame[show_cols], use_container_width=True, height=420)

    csv_bytes = result.frame.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download bucket frame (CSV)",
        data=csv_bytes,
        file_name="vpin_buckets.csv",
        mime="text/csv",
    )

with tab2:
    st.dataframe(result.ticks.head(2000), use_container_width=True, height=420)
    st.caption(f"Showing first 2,000 of {len(result.ticks):,} cleaned ticks.")

with tab3:
    diag_df = pd.DataFrame(
        [{"metric": k, "value": str(v)} for k, v in d.items()]
    )
    st.dataframe(diag_df, use_container_width=True, hide_index=True)
