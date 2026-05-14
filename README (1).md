# GNN Tangency Portfolio — Time-Varying Correlation Graph Extension

> Replication and extension of:
> Liu, B., Li, H., \& Kang, L. (2026). **Tangency portfolios using graph neural networks.** Neural Networks, 193, 108043. (https://www.sciencedirect.com/science/article/pii/S0893608025009232?via%3Dihub)

## Overview
This project extends Tangency Portfolios Using Graph Neural Networks by improving how relationships between stocks are modeled in portfolio construction. The original study replaces traditional mean–variance optimisation with a Graph Neural Network (GNN), where stocks are represented as nodes and connected through a fixed industry-based graph, allowing the model to learn portfolio weights that maximise the Sharpe ratio without explicitly estimating covariance matrices.

In this extension, the key modification is that the adjacency matrix is no longer static. Instead, it is rebuilt at each rolling window using correlations between stock returns, creating a time-varying, data-driven graph. This makes the approach fully reproducible using publicly available price data and allows the model to adapt to changing market conditions.

All model components from the original paper are preserved unchanged:

* GNN embedding module (Equation 4)
* Mean and precision matrix fitting layer (Equations 5–6)
* Long-short portfolio weight prediction (Equations 3, 7)
* Combined loss function: `exp(-SR) + α₁·modu + α₂·ranking` (Equation 16)

**University of Johannesburg | PORA9X1 -- Portfolio Optimisation and Risk Allocation**

---

## What This Project Does

This repository replicates the GNN-based tangency portfolio framework of Liu et al. (2026)
and extends it by replacing the original paper's static industry-chain adjacency matrix
with a **time-varying Pearson correlation graph** that is rebuilt at every 40-day rolling
window. The extension requires only publicly available OHLCV data via Yahoo Finance.

**Key results on the CSI-300 proxy dataset (test period Jul 2020 - Dec 2023, ~900 days):**

| Model | Sharpe (ann) | Return (ann) | Volatility |
|-------|-------------|-------------|-----------|
| GNN-GCN | 0.781 | 24.7% | 31.6% |
| GNN-SAGE | **0.877** | 21.9% | 25.0% |
| GNN-GAT | 0.706 | 22.2% | 31.5% |
| Equal-Weight | 0.264 | 4.4% | 16.7% |
| Momentum | -0.398 | -10.5% | 26.4% |
| Min-Variance | 0.080 | 1.3% | 15.8% |
| OLS-Factor | -0.154 | -3.8% | 24.7% |

All three GNN models outperform baselines by a factor of 2.7x-3.3x on the Sharpe ratio.

---

## Repository Structure

```
gnn_tangency/
├── GNN_Tangency_Portfolio_v2.ipynb   # Main Jupyter notebook (24 code cells)
├── gnn_tangency_colab.py             # Same code as a single Colab script
├── paper_diagrams_v3.py             # Publication-quality figure generator
├── requirements.txt                  # Package dependencies
└── README.md                         # This file
```

---

## Quick Start (Google Colab)

**Step 1:** Open Colab and set the runtime to GPU

```
Runtime > Change runtime type > Hardware accelerator: T4 GPU
```

**Step 2:** Install packages (first cell of the notebook)

```python
!pip install torch-geometric yfinance scipy openpyxl matplotlib -q
```

Restart the runtime after installation (`Runtime > Restart runtime`).

**Step 3:** Run cells top to bottom

The notebook is split into 24 code cells (Cells 1-24). Run them sequentially.
The full pipeline (Cell 24) trains all three GNN models, runs baselines, produces
all evaluation metrics, and saves all figures.

**Step 4 (optional):** Generate paper-style figures

Paste `paper_diagrams_v3.py` into a new cell after running Cell 24. All 8 diagrams
will display inline one at a time.

---

## Notebook Cell Map

| Cell | Contents |
|------|----------|
| 1 | Install packages |
| 2 | Library imports |
| 3 | Configuration (CFG dictionary) |
| 4 | Stock universe -- 94 CSI-300 proxy tickers |
| 5 | Feature engineering -- 7 Option-2 factors |
| 6 | Sliding window dataset |
| 7 | Time-varying Pearson correlation graph builder |
| 8 | MST graph builder (alternative) |
| 9 | GNN embedding module (GCN / GAT / SAGE) |
| 10 | Mean and precision fitting layer |
| 11 | Portfolio weight prediction |
| 12 | Loss functions (Sharpe + modularity + ranking) |
| 13 | VaR, CVaR, and Kupiec backtest |
| 14 | Data download with cubic spline interpolation |
| 15 | Inference helpers and basic metric computation |
| 16 | 6-panel diagnostic plot |
| 17 | Winsorise and full metric computation |
| 18 | Baseline models (4 strategies) |
| 19 | Single model training loop |
| 20 | Full model comparison (GCN + SAGE + GAT + baselines) |
| 21 | Comparison plots |
| 22 | Threshold sensitivity analysis |
| 23 | Publication-quality figures (paper-style) |
| 24 | Execute full pipeline |

---

## Configuration

All hyperparameters are in the `CFG` dictionary at the top of Cell 3:

```python
CFG = {
    "gnn_type":          "GAT",    # 'GCN', 'GAT', or 'SAGE'
    "graph_method":      "pearson",# 'pearson' or 'glasso'
    "threshold":         0.60,     # edge threshold -- see note below
    "hidden_dim":        64,
    "embed_dim":         64,
    "num_layers":        2,
    "dropout":           0.1,
    "heads":             4,        # GAT attention heads
    "lr":                1e-3,     # Adam learning rate
    "alpha1":            0.1,      # modularity loss weight
    "alpha2":            0.1,      # ranking loss weight
    "epochs":            200,
    "batch_size":        16,
    "val_len":           20,
    "train_len":         40,
    "prec_rank":         64,       # precision matrix rank cap
    "lr_warmup_epochs":  10,
}
```

**Why threshold = 0.60 (not the paper's 0.35):**
With N=94 Chinese stocks, threshold=0.35 produces 32.5% graph density (1,422 edges).
A dense graph causes the GNN to average all stocks together, producing identical
embeddings and NaN loss. Threshold=0.60 gives 6.8% density (297 edges).

---

## Data

Data is downloaded automatically from Yahoo Finance on first run and cached to
`/content/prices.csv`. Delete the cache to force a fresh download.

**Chinese A-share tickers:** 94 stocks with `.SS` (Shanghai) or `.SZ` (Shenzhen) suffix.

**Late-listing stocks:** Stocks listed after the dataset start date are handled by
cubic spline interpolation. Short gaps (<=10 days) after IPO are filled with a
smooth spline; longer gaps are forward-filled. No price data is fabricated before
a stock's IPO date.

---

## Outputs

After running the full pipeline, the following files are saved to `/content/`:

| File | Contents |
|------|----------|
| `all_model_results.json` | Full metrics for all 7 models |
| `all_model_metrics.csv` | Same data as CSV |
| `var_analysis.csv` | VaR, CVaR, and Kupiec test results |
| `threshold_sensitivity.csv` | Graph density at thresholds 0.20, 0.35, 0.50 |
| `fig3_replication.png` | Cumulative returns + drawdown (paper Fig. 3 style) |
| `fig4_adjacency_heatmap.png` | Correlation / Pearson / MST heatmaps (paper Fig. 4) |
| `metrics_dashboard.png` | 8-metric bar chart, all models |
| `var_dashboard.png` | VaR/CVaR visualisation |
| `rolling_sharpe.png` | 63-day rolling Sharpe ratio |
| `diagnostics.png` | 6-panel training diagnostic figure |

---

## Stability Fixes (addresses NaN loss and SAGE kurtosis issues)

The following engineering decisions prevent the numerical instability problems that
arise when applying the original paper's settings to a 94-stock Chinese A-share dataset:

1. **Threshold raised 0.35 -> 0.60**: prevents 51% graph density and NaN loss
2. **prec_rank = 64**: caps precision matrix rank; prevents ill-conditioning with N=94
3. **Input clipping to [-3, 3]**: prevents extreme features entering SAGE aggregation
4. **SAGE layer clamp [-5, 5]**: prevents embedding explosion across 2 message-passing layers
5. **w_raw z-scored before softmax**: removes SAGE scale difference; prevents exp() overflow
6. **Max weight = 1/N**: prevents >90% concentration in a single stock (kurtosis fix)
7. **Winsorised skewness/kurtosis at 5-std**: reports realistic distributional shape metrics
8. **NaN guards in all loss functions**: prevents training crash from early degenerate states
9. **LR warmup over 10 epochs**: prevents large gradient updates from random prec_hat at init

---

## Requirements

See `requirements.txt`. Key dependencies:

- Python >= 3.10
- PyTorch >= 2.2.1 (with CUDA 11.8 or 12.1)
- torch-geometric >= 2.5.0
- yfinance >= 0.2.50 (note: `show_errors` parameter removed in newer versions)
- scipy >= 1.11
- matplotlib >= 3.8

Install order matters -- see `requirements.txt` for the correct sequence.

---

## Known Issues

**yfinance MultiIndex columns:** yfinance >= 0.2.50 returns a MultiIndex even for
single-ticker downloads. The `load_or_download()` function handles this with
`raw.columns = raw.columns.droplevel(1)` followed by explicit `pd.Series` construction
with `.flatten()` to guarantee 1-D float arrays before concatenation.

**`show_errors` deprecation:** The `show_errors` parameter was removed from
`yf.download()` in newer versions. It has been removed from this code.

---

## Citation

```bibtex
@article{liu2026tangency,
  title   = {Tangency portfolios using graph neural networks},
  author  = {Liu, Bin and Li, Haolong and Kang, Linshuang},
  journal = {Neural Networks},
  volume  = {193},
  pages   = {108043},
  year    = {2026},
  doi     = {10.1016/j.neunet.2025.108043}
}
```

---

## References

- Liu, B., Li, H., & Kang, L. (2026). Tangency portfolios using graph neural networks. *Neural Networks*, 193, 108043.
- Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with GCN. *ICLR*.
- Hamilton, W., Ying, Z., & Leskovec, J. (2017). Inductive representation learning. *NeurIPS*.
- Velickovic, P., et al. (2018). Graph attention networks. *ICLR*.
- Kupiec, P. H. (1995). Techniques for verifying risk measurement models. *Journal of Derivatives*.
- Mantegna, R. N. (1999). Hierarchical structure in financial markets. *European Physical Journal B*.
