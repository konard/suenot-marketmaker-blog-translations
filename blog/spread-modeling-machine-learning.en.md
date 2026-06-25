---
title: "Bid-Ask Spread Modeling and Prediction with Machine Learning"
description: "Decomposing and predicting bid-ask spreads with ML — from Roll's implicit estimator to gradient boosting and neural networks — with the units, leakage, and benchmarking pitfalls that bite in production."
tags: ["microstructure", "spread", "market-making", "prediction", "machine-learning", "gradient-boosting", "deep-learning"]
date: "2026-06-03"
lang: "en"
image: "/images/blog/spread-modeling-machine-learning.webp"
slug: "spread-modeling-machine-learning"
authors: ["suenot"]
---

# Bid-Ask Spread Modeling and Prediction with Machine Learning

The bid-ask spread is the single most important variable a market maker controls. Set it too wide and you lose flow to competitors. Set it too narrow and adverse selection eats your inventory alive. Traditional microstructure theory gives us elegant decompositions of the spread into its economic components. Machine learning gives us the tools to predict how those components shift in real time. This post bridges both worlds: we start with the classical theory, build up to Roll's implicit spread estimator, then move into gradient boosting and deep learning models that predict spreads from order book features. Along the way we flag the units, leakage, and benchmarking traps that quietly invalidate spread models in practice.

## Why Spreads Matter for Market Makers

A market maker continuously quotes a bid price $P_b$ and an ask price $P_a$. The quoted spread is:

$$S = P_a - P_b$$

Every round-trip (buy at the maker's bid, sell at the maker's ask, both filled by takers) transfers up to $S$ from the takers to the maker — in theory. In practice, the maker earns less than $S$ because of adverse selection: some takers are informed and trade right before the price moves against the maker. The realized profit per round-trip is the **realized spread**, which equals the effective spread minus the price impact:

$$S_{\text{realized}} = S_{\text{effective}} - \text{PriceImpact}$$

We measure all three quantities on the same **full-spread** basis (not half), so the identity is dimensionally consistent. The effective spread for a single trade is:

$$S_{\text{effective}} = 2 \cdot d_t \cdot (P_t - M_t)$$

Here $d_t \in \{+1, -1\}$ is the trade direction (taker buy or sell), $P_t$ is the transaction price, and $M_t$ is the midpoint of the best bid and ask at the time of the trade. The price-impact term is defined symmetrically over a post-trade horizon $\tau$:

$$\text{PriceImpact} = 2 \cdot d_t \cdot (M_{t+\tau} - M_t)$$

where $M_{t+\tau}$ is the mid a horizon $\tau$ after the trade. The horizon must be stated explicitly — common choices are 5 minutes in equities and 30 seconds in crypto, where prices reprice faster. Subtracting the impact from the effective spread leaves the realized spread: what the maker keeps after the market has moved.

A market maker who can **predict** the spread — and its components — in the next 1, 5, or 60 seconds can dynamically adjust quotes to maximize realized spread while maintaining fill rates.

## The Three Components of the Spread

![The bid-ask spread decomposed into order-processing, inventory, and adverse-selection cost layers](/images/blog/spread-modeling-machine-learning-components.webp)

The market microstructure literature (Stoll 1978, Glosten and Milgrom 1985, Huang and Stoll 1997) decomposes the bid-ask spread into three economic components.

### 1. Order Processing Cost ($\alpha$)

This is the cost of providing the market-making service per filled side: the fee the maker actually pays plus technology infrastructure, regulatory compliance, and the opportunity cost of capital deployed. Demsetz (1968) and Tinic (1972) were the first to formalize this component.

The key distinction is *who pays which fee*. A maker quoting passively pays the **maker fee** $f_m$ on its own fills — and on many venues $f_m$ is a **rebate**, i.e. negative. It does **not** pay the taker fee $f_t$ on those passive fills; the counterparty that crosses the spread pays $f_t$. So the maker's per-side order-processing cost is:

$$\alpha \approx f_m + c_{\text{infra}}$$

where $f_m$ is signed (a rebate lowers $\alpha$ and can make it negative) and $c_{\text{infra}}$ covers connectivity, colocation, and compute. In modern electronic markets this component has shrunk dramatically — sub-penny in equities, a few basis points or a net rebate in crypto.

The taker fee $f_t$ matters for a different reason: it sets a floor on how tight the *full* spread can profitably be, because a taker who crosses pays $f_t$ on top of the spread. If you want a spread floor that keeps your quotes economically attractive relative to that taker cost, motivate it separately rather than folding $f_t$ into the maker's own cost. Conflating the two double-counts a round-trip fee inside a single half-spread.

### 2. Inventory Holding Cost ($\beta$)

When a market maker accumulates a directional position (long or short), they bear price risk. The inventory component compensates for this risk. Stoll (1978) and Amihud and Mendelson (1980) modeled this as a function of volatility and the maker's current inventory:

$$\beta \propto \sigma \cdot |Q|$$

where $\sigma$ is the asset's volatility and $Q$ is the maker's current inventory. As inventory grows, the maker widens the spread on the side where they are exposed and narrows it on the other, a technique called **inventory skewing**.

### 3. Adverse Selection Cost ($\gamma$)

This is the most dangerous component. Informed traders — those with superior information about imminent price moves — systematically pick off stale quotes. The adverse selection cost equals the expected loss per trade to informed counterparties. Copeland and Galai (1983) modeled this as the value of a free option the maker gives to informed traders. Glosten and Milgrom (1985) formalized it as the Bayesian revision in the maker's beliefs after observing a trade:

$$\gamma = E[V | \text{trade}] - M_t$$

where $V$ is the true fundamental value. In liquid markets, adverse selection can account for 30-60% of the total spread.

### The Full Decomposition

The quoted half-spread can be written as:

$$\frac{S}{2} = \alpha + \beta + \gamma$$

with $\alpha$, $\beta$, and $\gamma$ all expressed as per-side (half-spread) costs — that is what keeps the accounting consistent. Huang and Stoll (1997) proposed an econometric method to estimate these components from trade and quote data. The key insight: order processing costs create a fixed spread floor, inventory costs create a spread that varies with position and volatility, and adverse selection costs create a spread that varies with information asymmetry.

## Roll's Implicit Spread Model

![Roll's implicit spread model: transaction prices bouncing between bid and ask, leaving a negative serial-covariance signature](/images/blog/spread-modeling-machine-learning-roll.webp)

Before high-frequency data was widely available, Richard Roll (1984) proposed an elegant method to estimate the effective spread using only transaction prices. His insight: in an efficient market, the bid-ask bounce induces negative serial covariance in price changes, even when there is no new information.

### The Model

Assume the fundamental value $V_t$ follows a random walk:

$$V_t = V_{t-1} + u_t, \quad u_t \sim \text{i.i.d.}(0, \sigma_u^2)$$

The observed transaction price bounces between bid and ask:

$$P_t = V_t + \frac{S}{2} \cdot d_t$$

where $d_t \in \{-1, +1\}$ with equal probability (i.e., buys and sells are equally likely). The price change is:

$$\Delta P_t = u_t + \frac{S}{2}(d_t - d_{t-1})$$

Computing the first-order autocovariance:

$$\text{Cov}(\Delta P_t, \Delta P_{t-1}) = -\frac{S^2}{4}$$

The model is derived in **price units**: $S$ falls out of the autocovariance of price *changes*, not returns. This distinction is the single most common implementation error, and we keep the code faithful to it below.

### The Roll Estimator

Solving for $S$:

$$\hat{S}_{\text{Roll}} = 2\sqrt{-\text{Cov}(\Delta P_t, \Delta P_{t-1})}$$

When the sample autocovariance is positive (which happens frequently in practice due to noise or momentum), the estimator is undefined. A common fix is to set the estimate to zero or use the signed root:

$$\hat{S}_{\text{Roll}}^{*} = 2 \cdot \text{sign}(-\hat{\gamma}_1) \cdot \sqrt{|\hat{\gamma}_1|}$$

where $\hat{\gamma}_1$ is the sample first-order autocovariance.

### Implementation in Python

The estimator returns a spread in **price units**. To express it in basis points we divide by the midprice once — because, unlike a return-space estimator, it has not already been divided by price:

```python
import numpy as np
import pandas as pd

def roll_spread(prices: pd.Series, window: int = 200) -> pd.Series:
    """
    Rolling Roll (1984) spread estimator, in PRICE units.

    The model is P_t = V_t + (S/2) d_t with Cov(ΔP_t, ΔP_{t-1}) = -S^2/4,
    so S is recovered from the autocovariance of price CHANGES (diff),
    not returns (pct_change). Using returns rescales the estimate by the
    price level and is wrong by roughly that factor.

    Parameters
    ----------
    prices : pd.Series
        Transaction prices.
    window : int
        Rolling window size (number of price changes).

    Returns
    -------
    pd.Series
        Estimated spread per window, in price units.
    """
    dprice = prices.diff().dropna()
    autocov = dprice.rolling(window).apply(
        lambda x: np.cov(x[:-1], x[1:])[0, 1], raw=True
    )
    # Roll estimator: S = 2 * sqrt(-Cov) when Cov < 0, else 0
    return 2.0 * np.sqrt(np.maximum(-autocov, 0.0))


# Example usage with trade data
trades = pd.read_parquet("trades.parquet")
trades["roll_spread"] = roll_spread(trades["price"], window=200)

# Compare with quoted spread (both in price units, then in bps)
trades["quoted_spread"] = trades["ask"] - trades["bid"]
trades["midprice"] = 0.5 * (trades["ask"] + trades["bid"])
trades["quoted_spread_bps"] = trades["quoted_spread"] / trades["midprice"] * 1e4
trades["roll_spread_bps"] = trades["roll_spread"] / trades["midprice"] * 1e4
```

A quick sanity check on a simulated series — a fundamental random walk near a price of 100 with a true spread of $S = 0.10$ — recovers $\approx 0.0999$ from price changes. The returns-based variant would return $\approx 0.001$, off by the price level, and then dividing *that* by the midprice again to get bps compounds the error. If you prefer a return-space estimator, derive the model in log-price space and drop the second division by the midprice; pick one convention and make the code match the math.

### Limitations of Roll's Model

Roll's model assumes: (1) market efficiency, (2) no information asymmetry, (3) i.i.d. trade direction, and (4) constant spread. All of these are violated in practice. Harris (1990) showed that the estimator is severely biased due to Jensen's inequality when applied to noisy data. Despite these limitations, the Roll estimator remains useful as a quick baseline and is widely used in empirical finance research.

## ML Features for Spread Prediction

![Engineered microstructure features — volatility, order-flow imbalance, depth, trade intensity — feeding a spread predictor](/images/blog/spread-modeling-machine-learning-features.webp)

To move beyond static models, we need features that capture the dynamic drivers of spread variation. Here is a taxonomy of features organized by the spread component they proxy for.

### Order Book Features (Inventory & Adverse Selection)

| Feature | Formula | Proxies For |
|---------|---------|-------------|
| Book imbalance | $\text{BI} = \frac{V_b^1 - V_a^1}{V_b^1 + V_a^1}$ | Directional pressure |
| Weighted mid-price | $P_w = P_a \cdot \frac{V_b}{V_b + V_a} + P_b \cdot \frac{V_a}{V_b + V_a}$ | Short-term fair value |
| Depth ratio (levels 1-5) | $\text{DR}_5 = \frac{\sum_{i=1}^{5} V_b^i}{\sum_{i=1}^{5} V_a^i}$ | Multi-level supply/demand |
| Book pressure | $\text{BP} = \sum_{i=1}^{5} V_b^i\, w_i^b - \sum_{i=1}^{5} V_a^i\, w_i^a$ | Distance-weighted pressure |
| Spread / tick ratio | $S / \text{tick}$ | Tightness relative to minimum |

Book pressure here uses an **absolute distance-to-mid decay**, $w_i = e^{-\lambda |P_i - M|}$, so volume sitting near the touch counts more than deep volume, and both sides are weighted by a positive, decreasing function. This avoids the structural sign bias of dividing volume by the *signed* distance $P_i - M$ (which is negative on the bid side, positive on the ask side, and blows up as a level approaches the mid). Pick $\lambda$ from the typical book depth, or replace the exponential with any positive weight $w(|P_i - M|)$ that decreases in distance.

### Trade Flow Features (Adverse Selection)

| Feature | Formula | Proxies For |
|---------|---------|-------------|
| Trade imbalance | $\text{TI}_{n} = \frac{\sum_{i=1}^{n} d_i \cdot v_i}{\sum_{i=1}^{n} v_i}$ | Net informed flow |
| VPIN | Volume-synchronized probability of informed trading | Toxicity |
| Kyle's lambda | Regression of $\Delta M$ on signed volume | Price impact per unit |
| Large-trade frequency | Count of trades $> k \cdot \text{median}$ in window | Institutional activity |

### Volatility Features (Inventory Cost)

| Feature | Formula | Proxies For |
|---------|---------|-------------|
| Realized volatility | $\sigma_{\text{rv}} = \sqrt{\sum (\Delta \log M)^2}$ | Short-term risk |
| Garman-Klass vol | $\frac{1}{2}(\log H/L)^2 - (2\ln 2 - 1)(\log C/O)^2$ | Range-based vol |
| Vol-of-vol | Rolling std of $\sigma_{\text{rv}}$ | Regime uncertainty |
| Return autocorrelation | $\rho_1(\Delta M)$ | Momentum / mean-reversion |

### Market Regime Features

| Feature | Description | Proxies For |
|---------|-------------|-------------|
| Time-of-day encoding | $\sin(2\pi t / T), \cos(2\pi t / T)$ | Intraday seasonality |
| Seconds since last trade | Time gap | Activity level |
| Cross-asset correlation | Rolling corr with index/BTC | Systematic risk |
| Funding rate (crypto) | Perp funding rate | Leveraged positioning |

## Gradient Boosting for Spread Prediction

Gradient boosted trees (XGBoost, LightGBM, CatBoost) are the workhorse of tabular prediction in quantitative finance. They handle mixed feature types, capture nonlinear interactions, require minimal preprocessing, and train fast on millions of rows — provided the feature build itself is vectorized (see the autocorrelation note below).

### Problem Formulation

We frame spread prediction as a regression task. The target is the time-weighted average quoted spread over the next $\tau$ seconds:

$$y_t = \frac{1}{\tau} \int_{t}^{t+\tau} S(u) \, du$$

In practice, we approximate this with the volume-weighted average spread over the next $N$ snapshots:

$$y_t = \frac{\sum_{i=1}^{N} S_{t+i} \cdot V_{t+i}}{\sum_{i=1}^{N} V_{t+i}}$$

This target is a **forward window** of length $N$ (or `horizon`), which means adjacent rows share overlapping future windows. That overlap leaks information across a naive train/validation split — we handle it explicitly in the training code below.

### Full Pipeline

```python
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

# ── 1. Feature Engineering ───────────────────────────────────
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build spread-prediction features from L2 order book snapshots."""
    f = pd.DataFrame(index=df.index)

    # Book imbalance (level 1)
    f["imb1"] = (df["bid_vol_1"] - df["ask_vol_1"]) / (
        df["bid_vol_1"] + df["ask_vol_1"] + 1e-9
    )

    # Depth imbalance (levels 1-5)
    bid_depth = df[[f"bid_vol_{i}" for i in range(1, 6)]].sum(axis=1)
    ask_depth = df[[f"ask_vol_{i}" for i in range(1, 6)]].sum(axis=1)
    f["depth_imb5"] = (bid_depth - ask_depth) / (bid_depth + ask_depth + 1e-9)

    # Midprice and spread
    mid = 0.5 * (df["ask_1"] + df["bid_1"])
    f["spread_bps"] = (df["ask_1"] - df["bid_1"]) / mid * 1e4
    f["log_spread"] = np.log1p(df["ask_1"] - df["bid_1"])

    # Realized volatility: rolling std of log-returns. Trees are scale-
    # invariant, so we do NOT multiply by sqrt(window) — that only rescales
    # each column by a constant and would put rvol_50 and rvol_200 on
    # different, non-comparable horizons. These are window stds, matching
    # the sum-of-squared-returns definition in the feature table.
    log_ret = np.log(mid / mid.shift(1))
    f["rvol_50"] = log_ret.rolling(50).std()
    f["rvol_200"] = log_ret.rolling(200).std()

    # Trade flow imbalance
    if "trade_sign" in df.columns and "trade_vol" in df.columns:
        signed_vol = df["trade_sign"] * df["trade_vol"]
        total_vol = df["trade_vol"].rolling(50).sum()
        f["tfi_50"] = signed_vol.rolling(50).sum() / (total_vol + 1e-9)

    # Return autocorrelation (lag 1), VECTORIZED.
    # A rolling .apply(x.autocorr) runs a Python callback per 100-length
    # window across every row — it dominates feature-build time on millions
    # of rows. Instead compute the rolling correlation of log_ret with its
    # own lag-1 shift, which pandas evaluates in C.
    lag1 = log_ret.shift(1)
    f["ret_autocorr"] = log_ret.rolling(100).corr(lag1)

    # Time-of-day encoding
    if isinstance(df.index, pd.DatetimeIndex):
        seconds = df.index.hour * 3600 + df.index.minute * 60 + df.index.second
        f["tod_sin"] = np.sin(2 * np.pi * seconds / 86400)
        f["tod_cos"] = np.cos(2 * np.pi * seconds / 86400)

    # Lagged spreads (autoregressive component)
    for lag in [1, 5, 10, 50]:
        f[f"spread_lag_{lag}"] = f["spread_bps"].shift(lag)

    return f.dropna()


# ── 2. Target Construction ───────────────────────────────────
def build_target(df: pd.DataFrame, horizon: int = 10) -> pd.Series:
    """Forward mean spread over the next `horizon` snapshots (in bps).

    target[t] = mean(spread_bps[t+1 .. t+horizon]). Note that consecutive
    targets share an overlapping forward window of length `horizon`, which
    is why the CV below purges a gap of `horizon` rows around each fold.
    """
    mid = 0.5 * (df["ask_1"] + df["bid_1"])
    spread_bps = (df["ask_1"] - df["bid_1"]) / mid * 1e4
    # Forward rolling mean of the NEXT horizon snapshots.
    fwd = spread_bps.shift(-1).rolling(horizon).mean().shift(-(horizon - 1))
    return fwd


# ── 3. Purged, embargoed walk-forward CV ─────────────────────
def purged_walk_forward(n: int, n_splits: int, horizon: int):
    """Expanding-window splits with a purge/embargo gap of `horizon` rows.

    Because each target spans `horizon` future snapshots, rows straddling a
    train/val boundary share overlapping target windows. Dropping a gap of
    `horizon` rows between train and validation removes that leakage
    (Lopez de Prado-style purging). Without it, validation R²/MAE are
    optimistically biased by the target overlap.
    """
    fold_size = n // (n_splits + 1)
    for k in range(1, n_splits + 1):
        train_end = fold_size * k
        val_start = train_end + horizon       # embargo gap
        val_end = val_start + fold_size
        if val_end > n:
            break
        train_idx = np.arange(0, train_end - horizon)   # purge gap
        val_idx = np.arange(val_start, val_end)
        yield train_idx, val_idx


# ── 4. Model Training ────────────────────────────────────────
def train_spread_model(features: pd.DataFrame, target: pd.Series, horizon: int = 10):
    """Train LightGBM with purged, embargoed walk-forward validation."""
    # Align and drop NaN
    common = features.index.intersection(target.dropna().index)
    X = features.loc[common].reset_index(drop=True)
    y = target.loc[common].reset_index(drop=True)

    models, scores = [], []
    params = {
        "objective": "mae",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "verbose": -1,
    }

    for fold, (train_idx, val_idx) in enumerate(
        purged_walk_forward(len(X), n_splits=5, horizon=horizon)
    ):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        ds_tr = lgb.Dataset(X_tr, y_tr)
        ds_val = lgb.Dataset(X_val, y_val, reference=ds_tr)

        model = lgb.train(
            params,
            ds_tr,
            num_boost_round=2000,
            valid_sets=[ds_val],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)],
        )
        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        r2 = r2_score(y_val, preds)
        print(f"Fold {fold}: MAE={mae:.4f} bps, R²={r2:.4f}")
        models.append(model)
        scores.append({"mae": mae, "r2": r2})

    # Return the model from the last fold (most recent data)
    return models[-1], scores
```

The crucial detail is the **purge/embargo gap**. The forward-mean target means consecutive rows overlap by up to `horizon` snapshots, so a plain `TimeSeriesSplit` lets validation rows share future windows with training rows — leaking the answer and inflating validation R². Dropping a gap of at least `horizon` rows on both sides of each fold boundary (Lopez de Prado-style purged k-fold) removes that bias. This applies just as much to the gradient-boosting pipeline as to deep learning, even though the leakage is more commonly discussed for sequence models.

### Feature Importance Analysis

One of the key advantages of tree-based models is interpretability. After training, inspect SHAP values to understand which features drive spread predictions:

```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_val)
shap.summary_plot(shap_values, X_val, max_display=15)
```

Typical findings across asset classes:

1. **Lagged spread** ($\text{spread\_lag\_1}$) is almost always the most important feature — spreads are highly autocorrelated. This is also why headline R² looks high: much of the score is just persistence, so always benchmark against an AR/EWMA baseline (more on this below).
2. **Realized volatility** is the second most important — intraday volatility and spreads are strongly positively correlated, both contemporaneously and dynamically.
3. **Book imbalance** matters most during volatile periods — it signals imminent directional moves.
4. **Trade flow imbalance** captures short-term adverse selection — a burst of one-sided flow predicts spread widening.
5. **Time-of-day** captures the U-shaped intraday pattern (wider at open/close, tighter midday).

### Hyperparameter Considerations

For spread prediction specifically:

- Use **MAE or Huber loss** rather than MSE. Spread distributions are right-skewed with occasional extreme outliers (during news events). MAE is more robust.
- Set `min_child_samples` high (100+) to prevent the model from fitting to microstructure noise in individual snapshots.
- Use `subsample < 1.0` to decorrelate trees and improve generalization across different volatility regimes.

## Deep Learning Approaches

While gradient boosting excels on tabular features, deep learning can learn representations directly from raw order book data. Two architectures have proven effective for spread-related prediction tasks.

### Architecture 1: CNN-LSTM for Order Book Snapshots

The DeepLOB architecture (Zhang et al. 2019) uses stacked small-kernel convolutions — and an Inception module — to extract spatial patterns across order book levels while **preserving** that spatial structure, followed by LSTM layers to model temporal dependencies. The important design choice is *not* to global-pool the level axis away before the recurrent layer: doing so collapses exactly the cross-level structure the convolutions are meant to capture.

For spread prediction, the input is a tensor of shape $(T, L, F)$:
- $T$ = number of time steps (e.g., 100 snapshots)
- $L$ = number of price levels (e.g., 10 bid + 10 ask = 20)
- $F$ = features per level (price, volume, order count)

The model below keeps the convolutional feature map over levels and **flattens** it into the LSTM input (`input_size = 16 * L`), rather than averaging the level dimension into 16 channel means:

```python
import torch
import torch.nn as nn


class SpreadPredictor(nn.Module):
    """
    CNN-LSTM model for bid-ask spread prediction from L2 order book.

    Input: (batch, seq_len, n_levels, n_features)
    Output: (batch, 1) — predicted spread in bps
    """

    def __init__(
        self,
        n_levels: int = 20,
        n_features: int = 3,
        seq_len: int = 100,
        hidden_dim: int = 64,
        n_lstm_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.n_levels = n_levels

        # Convolutional block: extract cross-level patterns per timestep.
        # Padding keeps the level axis length L, so the (16, L) feature map
        # is preserved — no global pooling that would discard cross-level
        # structure before the LSTM ever sees it.
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1),
        )
        conv_out_dim = 16 * n_levels  # flattened (channels × levels)

        # LSTM: model temporal dynamics over the flattened conv features
        self.lstm = nn.LSTM(
            input_size=conv_out_dim,
            hidden_size=hidden_dim,
            num_layers=n_lstm_layers,
            batch_first=True,
            dropout=dropout,
        )

        # Regression head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (batch, seq_len, n_levels, n_features)

        Returns
        -------
        Tensor of shape (batch, 1) — predicted spread
        """
        batch, T, L, F = x.shape

        # Process each timestep through the CNN
        # Reshape: (batch * T, F, L) for Conv1d
        x = x.reshape(batch * T, L, F).permute(0, 2, 1)
        x = self.conv(x)                  # (batch * T, 16, L)
        x = x.reshape(batch, T, 16 * L)   # (batch, T, 16 * L) — keep levels

        # LSTM over timesteps
        lstm_out, _ = self.lstm(x)        # (batch, T, hidden_dim)
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_dim)

        return self.head(last_hidden)     # (batch, 1)
```

If you do want pooling on the level axis to control parameter count, use a strided or learned pooling that retains more than a single position — not `AdaptiveAvgPool1d(1)`, which averages every level into one number and throws away the spatial signal.

### Architecture 2: Transformer Encoder

Transformers can capture long-range dependencies in order book sequences without the sequential bottleneck of LSTMs. For spread prediction, a lightweight transformer encoder works well:

```python
class TransformerSpreadPredictor(nn.Module):
    """Transformer encoder for spread prediction from order book sequences."""

    def __init__(
        self,
        input_dim: int = 40,   # 20 levels * 2 features (price_offset, volume)
        d_model: int = 64,
        nhead: int = 4,
        n_layers: int = 3,
        seq_len: int = 100,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, input_dim) — flattened order book snapshots
        """
        x = self.input_proj(x) + self.pos_encoding[:, : x.size(1), :]
        x = self.encoder(x)
        # Use last token's representation
        return self.head(x[:, -1, :])
```

### Training Considerations

1. **Normalization**: Normalize prices as offsets from the midprice (in ticks or bps). Normalize volumes by their rolling mean. Raw prices and volumes cause training instability.

2. **Loss function**: Use Huber loss ($\delta = 1.0$) to handle spread spikes:

$$L_\delta(y, \hat{y}) = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \le \delta \\ \delta |y - \hat{y}| - \frac{1}{2}\delta^2 & \text{otherwise} \end{cases}$$

3. **Window sampling and leakage**: Use non-overlapping windows for training, and — exactly as in the gradient-boosting pipeline — purge/embargo a gap of at least `horizon` snapshots between train and validation. Both the forward-mean target and overlapping input windows leak future information across split boundaries and inflate apparent performance.

4. **Online adaptation**: In production, periodically fine-tune the model on recent data (last 1-2 hours) with a small learning rate. Market microstructure changes intraday, and a model trained on morning data may underperform in the afternoon.

### When to Use Deep Learning vs. Gradient Boosting

| Criterion | Gradient Boosting | Deep Learning |
|-----------|------------------|---------------|
| Input type | Tabular features | Raw order book sequences |
| Training data size | Works with 100K+ rows | Needs 1M+ rows |
| Feature engineering | Manual (high effort, high control) | Learned (lower effort, less interpretable) |
| Inference latency | Single-digit µs with a compiled predictor; tens of µs from Python | Hundreds of µs on GPU |
| Interpretability | High (SHAP) | Low (attention maps) |
| Regime adaptation | Retrain / online update | Fine-tune on recent data |
| Short-horizon spread skill | Broadly comparable to DL | Edge grows on longer horizons / larger data |

We deliberately avoid quoting specific R² figures: spread forecast accuracy depends heavily on the horizon, the asset, and how much of the score is simply spread autocorrelation. A model can post an impressive raw R² while adding almost nothing over a one-line EWMA. Report **skill above an AR/EWMA baseline** on the same data, with the horizon stated, rather than a headline R². Likewise, treat latency numbers as implementation-dependent: a 2000-round, 63-leaf LightGBM model predicts a single row in tens of microseconds from Python and only reaches a few microseconds with a compiled/C++ predictor.

In practice, many production systems use a **two-stage approach**: a fast gradient boosting model for real-time quoting (latency-critical), and a deep learning model running asynchronously to adjust the boosting model's parameters or provide a secondary signal.

## From Prediction to Quoting

![Turning a predicted spread into live bid/ask quotes, skewed by inventory around a fair mid-price](/images/blog/spread-modeling-machine-learning-quoting.webp)

A spread prediction is only valuable if it translates into better quotes. Here is a simplified quoting rule that uses the predicted spread:

```python
def compute_quotes(
    mid: float,
    predicted_spread_bps: float,
    inventory: float,
    max_inventory: float,
    skew_factor: float = 0.5,
    min_spread_bps: float = 1.0,
) -> tuple[float, float]:
    """
    Compute bid/ask quotes from predicted spread and inventory.

    Parameters
    ----------
    mid : float
        Current midprice.
    predicted_spread_bps : float
        Model-predicted spread in basis points.
    inventory : float
        Current inventory (positive = long).
    max_inventory : float
        Maximum allowed inventory.
    skew_factor : float
        How aggressively to skew quotes toward inventory neutrality.
    min_spread_bps : float
        Minimum spread floor (covers order processing costs).

    Returns
    -------
    (bid, ask) : tuple[float, float]
    """
    spread_bps = max(predicted_spread_bps, min_spread_bps)
    half_spread = mid * spread_bps / 2e4

    # Inventory skew: shift quotes to discourage further accumulation
    inv_ratio = inventory / max_inventory  # in [-1, 1]
    skew = skew_factor * inv_ratio * half_spread

    bid = mid - half_spread - skew
    ask = mid + half_spread - skew

    return bid, ask
```

When inventory is long ($Q > 0$), the skew lowers both the bid and the ask. From a single, consistent perspective: a lower ask makes it cheaper for takers to buy from us, which offloads our long inventory; a lower bid makes us less likely to be hit by sellers, slowing further accumulation. The predicted spread controls the overall width — widening when the model expects volatility or adverse selection, narrowing when conditions are calm.

## Evaluation and Backtesting

![Evaluating a spread model: predicted-vs-realized spread calibration alongside a fill-rate and PnL gauge](/images/blog/spread-modeling-machine-learning-evaluation.webp)

### Spread Prediction Metrics

Beyond standard regression metrics (MAE, $R^2$), evaluate spread predictions with metrics that matter for market making:

- **Skill over a baseline**: Always report MAE/R² *relative to* an AR(1) or EWMA forecast of recent spreads. Because spreads are strongly persistent, the absolute score is dominated by autocorrelation; only the improvement over a trivial baseline reflects real predictive content.
- **Directional accuracy**: Does the model correctly predict whether the spread will widen or narrow? A model with mediocre MAE but high directional accuracy can still be profitable.
- **Tail coverage**: Does the model predict spread spikes? Compute MAE separately for the top 5% of spread values — this is where adverse selection losses concentrate.
- **Calibration**: Plot predicted vs. realized spread quantiles. A well-calibrated model's 90th percentile prediction should match the 90th percentile of realized spreads.

### PnL-Based Evaluation

Ultimately, the only metric that matters is realized PnL. Backtest the full loop:

1. At each timestamp, predict the spread
2. Compute quotes using the predicted spread + inventory skew
3. Simulate fills against historical trades
4. Track inventory, realized PnL, and Sharpe ratio

Compare against baselines: (a) constant spread (the time-series median), (b) EWMA of recent spreads, and (c) the Roll estimator.

## Conclusion

Spread modeling sits at the intersection of financial theory and applied ML. The classical decomposition into order processing, inventory, and adverse selection costs provides the economic intuition for **why** spreads vary. Roll's model gives an elegant baseline estimator from minimal data — as long as you compute it in price units. Gradient boosting models turn microstructure features into accurate short-horizon spread forecasts with low-latency inference. Deep learning architectures learn directly from raw order book data, capturing patterns that handcrafted features may miss — provided the architecture preserves cross-level structure rather than pooling it away.

For a production market-making system, the practical recommendation is layered:

1. Use the **Huang-Stoll decomposition** offline to understand your spread components and calibrate risk limits
2. Use **Roll's estimator** as a sanity check and for instruments where you lack order book data
3. Deploy a **LightGBM model** for real-time spread prediction — it is fast, interpretable, and robust — with purged walk-forward validation and an AR/EWMA benchmark
4. Run a **CNN-LSTM or Transformer model** in a secondary loop to detect regime changes and adjust the primary model

The spread is not a number — it is a signal. The better you model it (and the more honestly you measure that model), the more precisely you can price liquidity provision.

---

*This post is part of the [marketmaker.cc](https://marketmaker.cc) series on algorithmic market making and microstructure.*
