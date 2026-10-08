# VPIN · Order-Flow Toxicity Monitor

A Streamlit application for calculating the **Volume-Synchronized Probability of Informed Trading (VPIN)**, based on the original framework by Easley, O'Hara, and López de Prado (2012).

This tool processes high-frequency trade prints (tick data) to compute rolling order-flow toxicity, helping quants and market makers identify periods where liquidity providers are at high risk of being adversely selected by informed traders.

Application - https://satwik-vpin.streamlit.app/

## Features

- **Robust Data Ingestion:** Handles dirty exchange feeds by removing zero/negative volumes, NaNs, and bad prices.
- **Duplicate Timestamp Handling:** Aggregates duplicate timestamps by calculating the Volume-Weighted Average Price (VWAP) and summing the volume.
- **Exact Volume Bucketization:** Large trades that cross bucket boundaries are mathematically split. The portion filling the current bucket is allocated to it, and the remainder rolls over to the next bucket. No rough rounding.
- **Dynamic Bulk Volume Classification (BVC):** Supports both a global historical standard deviation of price changes, or a rolling lookback.
- **Interactive Visualization:** A dual-pane Plotly chart showing:
  - **Top:** Price action with red shading/markers indicating toxic regimes (VPIN > threshold).
  - **Bottom:** The rolling VPIN line with a dashed horizontal threshold line.
- **Downloadable Results:** Export the full bucket frame (including buy/sell volumes and VPIN) as a CSV.
- **Synthetic Data Generator:** Includes a built-in data generator with injected "dirty" ticks and informed-trading bursts, so you can test the app immediately without your own data.

## Mathematical Framework

The pipeline strictly adheres to the formulas defined in the original academic paper:

1. **Volume Bucket Size (V_bucket):**
<img width="172" height="68" alt="image" src="https://github.com/user-attachments/assets/2c365366-4249-4735-aae7-1ef0f9694cd6" />

   Where ADV is the Average Daily Volume and N is the number of buckets per day (default 50).

2. **Bulk Volume Classification (BVC):**
   Using the price change across a bucket (Delta P) and the standard deviation of those changes (sigma_Delta_P):
  <img width="268" height="124" alt="image" src="https://github.com/user-attachments/assets/7d7d55b6-efc1-4fff-ae74-be25becfb588" />

   Where CDF is the standard normal cumulative distribution function.

3. **Rolling VPIN:**
   Over a rolling window of V_window buckets (default 50):
<img width="367" height="70" alt="image" src="https://github.com/user-attachments/assets/d4fb1523-d127-4357-abd2-5683afbea4c5" />

   The output is strictly bounded between 0.0 and 1.0.

## Usage Guide

### 1. Data Source
You can run the pipeline using either the built-in synthetic generator or your own data.

- **Synthetic Demo:** Use this to explore the app immediately without any external files. You can adjust the number of days, trades per day, and toggle "Inject dirty ticks" to test the cleaning layer. The generator deliberately creates informed-trading bursts, so you will see VPIN spike during these periods.
- **Upload File:** Upload your own tick data (CSV, TSV, Parquet, or Excel). After uploading, use the sidebar dropdowns to map your specific `Timestamp`, `Price`, and `Volume` columns to the pipeline's required schema.

<img width="1440" height="832" alt="image" src="https://github.com/user-attachments/assets/49a98c6e-0f77-4c96-a66b-e1754b01a4c6" />

### 2. Sidebar Configuration
The sidebar allows you to fine-tune the VPIN calculation parameters:

- **N (Buckets per ADV day):** Default is 50. This determines the volume bucket size (`ADV / N`). Lower values create larger buckets (fewer, broader VPIN points). Higher values create smaller buckets (more granular VPIN).
- **V_window:** Default is 50. This is the rolling window size for VPIN. It must be less than or equal to the total number of buckets generated.
- **σ(ΔP) Mode:** Choose `global` for a single full-sample standard deviation of bucket price changes, or `rolling` for a trailing lookback of your chosen length.
- **Toxicity Threshold:** Choose `fixed` (e.g., 0.85) or `quantile` (e.g., 95th percentile). Periods where VPIN crosses this line are highlighted in red on the price chart.

<img width="205" height="805" alt="image" src="https://github.com/user-attachments/assets/7b6b2a7d-9ae8-43e4-af92-f3e1728aec3b" />

### 3. Interpreting the Charts
The dashboard features two synchronized, interactive Plotly charts:

- **Top Plot (Price & Toxicity):** Displays the cleaned asset price. Red shaded bands indicate periods where the rolling VPIN exceeded the toxicity threshold. Red dots mark the exact buckets where the threshold was crossed.
- **Bottom Plot (Rolling VPIN):** Displays the rolling VPIN line. The dashed horizontal line represents the historical high-toxicity threshold.

<img width="1145" height="644" alt="image" src="https://github.com/user-attachments/assets/74ca2f37-4dbc-46d1-96ab-15c052beddf7" />

<img width="1186" height="451" alt="image" src="https://github.com/user-attachments/assets/3bc5b449-0231-4dc6-a885-e5c4c1bf23f5" />

<img width="1179" height="415" alt="image" src="https://github.com/user-attachments/assets/38e5c4f6-bc89-418a-95da-3e59008aeec2" />

<img width="936" height="383" alt="image" src="https://github.com/user-attachments/assets/b14aac32-6e75-4295-b4c4-163c9735714d" />





## References
```
Easley, D., López de Prado, M., & O'Hara, M. (2012). Flow Toxicity and Liquidity in a High-Frequency World. The Review of Financial Studies, 25(5), 1457–1493.

Easley, D., López de Prado, M., & O'Hara, M. (2011). The Volume Clock: Insights into the High-Frequency Paradigm. The Journal of Portfolio Management, 39(1), 19-29.
