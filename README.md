# VPIN · Order-Flow Toxicity Monitor

A Streamlit app that computes the **Volume-Synchronized
Probability of Informed Trading (VPIN)** from high-frequency trade prints,
following Easley, O'Hara & López de Prado (2012).

## Features
- Robust tick cleaning (duplicates, NaNs, zero/negative volumes)
- Exact volume-bucket splitting (no rounding)
- Bulk Volume Classification with global or rolling σ(ΔP)
- Rolling, normalised VPIN bounded to [0, 1]
- Interactive Plotly chart with toxicity shading + markers
- CSV download of the full bucket frame

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Reference
Easley, D., López de Prado, M., & O'Hara, M. (2012).
Flow Toxicity and Liquidity in a High-Frequency World.
Review of Financial Studies, 25(5), 1457–1493.
