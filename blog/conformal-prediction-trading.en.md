---
title: "Conformal Prediction for Risk-Aware Position Sizing"
description: "Distribution-free prediction intervals with guaranteed coverage. We use split conformal, jackknife+, and adaptive conformal inference to calibrate trading risk and size positions without parametric assumptions."
tags: ["uncertainty", "conformal-prediction", "risk", "position-sizing", "statistics", "algorithmic-trading"]
date: "2026-06-12"
lang: "en"
image: "/images/blog/conformal-prediction-trading.webp"
slug: "conformal-prediction-trading"
authors: ["suenot"]
---

# Conformal Prediction for Risk-Aware Position Sizing

Every position sizing formula needs an estimate of uncertainty. The Kelly criterion needs a win probability and a payoff ratio (see [The Kelly criterion for strategies](/en/blog/post/kelly-criterion-strategy-sizing)). Mean-variance optimization needs a covariance matrix. VaR needs a return distribution. All of these require assumptions about the data-generating process — assumptions that financial markets routinely violate.

Conformal prediction offers something different: **prediction intervals with finite-sample coverage guarantees, without any parametric distributional assumptions**. If you ask for 90% coverage, you get at least 90% coverage — regardless of whether returns are Gaussian, fat-tailed, skewed, or heteroskedastic. The only requirement is exchangeability (or weaker conditions, as we will see).

This post covers the theory, the key variants, and a practical implementation for position sizing in Python.

## The Core Idea: Nonconformity Scores

![Distribution of nonconformity (residual) scores with a quantile threshold marked — the core of conformal calibration](/images/blog/conformal-prediction-trading-nonconformity.webp)

Conformal prediction works by measuring how "strange" a new observation is relative to past data. The strangeness is quantified by a **nonconformity score** — any function that measures how poorly a data point conforms to the pattern seen in the rest of the data.

For regression (predicting a continuous value like returns), the simplest nonconformity score is the absolute residual:

$$R_i = |Y_i - \hat{\mu}(X_i)|$$

where $\hat{\mu}$ is any point predictor (linear regression, random forest, neural network — it does not matter) and $(X_i, Y_i)$ is a data point.

The key insight: if the data points $(X_1, Y_1), \ldots, (X_n, Y_n), (X_{n+1}, Y_{n+1})$ are exchangeable, then the rank of $R_{n+1}$ among $R_1, \ldots, R_n, R_{n+1}$ is uniformly distributed over $\{1, \ldots, n+1\}$. This is a purely combinatorial fact — it requires no assumptions about the distribution of $X$ or $Y$.

From this rank uniformity, we can construct prediction intervals with finite-sample coverage.

## Split Conformal Prediction

![Split conformal prediction: data split into training and calibration folds, the calibration residuals producing prediction intervals](/images/blog/conformal-prediction-trading-split.webp)

Split conformal prediction (Papadopoulos et al., 2002; Lei et al., 2018) is the most practical variant. The algorithm is simple:

**Step 1.** Split the data into a training set $\mathcal{D}_{\text{train}}$ and a calibration set $\mathcal{D}_{\text{cal}} = \{(X_1, Y_1), \ldots, (X_n, Y_n)\}$.

**Step 2.** Fit any model $\hat{\mu}$ on $\mathcal{D}_{\text{train}}$.

**Step 3.** Compute nonconformity scores on the calibration set:

$$R_i = |Y_i - \hat{\mu}(X_i)|, \quad i = 1, \ldots, n$$

**Step 4.** For a desired miscoverage level $\alpha \in (0, 1)$, take $\hat{q}$ as the $\frac{\lceil (1 - \alpha)(n + 1) \rceil}{n}$ empirical quantile of $R_1, \ldots, R_n$. Concretely, this is the $\lceil (1 - \alpha)(n + 1) \rceil$-th smallest residual (and $\hat{q} = +\infty$ whenever $\lceil (1 - \alpha)(n + 1) \rceil > n$, i.e. for very small $n$).

**Step 5.** The prediction interval for a new point $X_{n+1}$ is:

$$C(X_{n+1}) = \left[\hat{\mu}(X_{n+1}) - \hat{q}, \; \hat{\mu}(X_{n+1}) + \hat{q}\right]$$

### The Coverage Guarantee

Under exchangeability of the calibration data and the new test point:

$$\mathbb{P}\left(Y_{n+1} \in C(X_{n+1})\right) \geq 1 - \alpha$$

This is a finite-sample guarantee — not an asymptotic approximation. It holds for any model $\hat{\mu}$, any distribution of the data, and any sample size $n$. If $\hat{\mu}$ is a terrible predictor, the intervals will simply be wider. The coverage guarantee still holds.

There is also an upper bound when the scores have no ties: $\mathbb{P}(Y_{n+1} \in C(X_{n+1})) \leq 1 - \alpha + \frac{1}{n+1}$, so the coverage is not wastefully conservative.

### Why This Matters for Trading

Traditional prediction intervals from, say, a linear regression assume Gaussian errors. A Gaussian interval calibrated on the bulk of the data can badly misjudge the tails when the true residuals are heavy-tailed (e.g. Student-$t$ with a few degrees of freedom): the central mass is thinner than Gaussian, so a variance-matched Gaussian interval over-covers near the center yet under-covers in the tails, and a tail-fit one does the opposite. The point is not a single magic number — it is that the realized coverage of a parametric interval depends on a distributional assumption you have not verified.

Conformal prediction intervals sidestep this. They automatically widen when the model is uncertain, and they maintain marginal coverage regardless of the true error distribution. For a trader, this means:

- If you size positions inversely proportional to the interval width, you **automatically reduce exposure** when the model is uncertain.
- The coverage guarantee means your risk estimates are **honest** — if you say "90% of realized returns will fall within this interval," that statement is statistically valid (marginally, under exchangeability).

## Full Conformal and Jackknife+

Split conformal is simple but wastes data: the calibration set cannot be used for training. Two alternatives address this.

### Full Conformal Prediction

Full conformal prediction (Vovk et al., 2005) uses all the data for both training and calibration. For each candidate value $y$ of $Y_{n+1}$:

1. Augment the dataset with $(X_{n+1}, y)$.
2. Refit the model on the augmented dataset.
3. Compute all nonconformity scores.
4. Include $y$ in the prediction set if the score for $(X_{n+1}, y)$ is not too extreme.

The prediction set is:

$$C(X_{n+1}) = \left\{y : \frac{|\{i : R_i^y \geq R_{n+1}^y\}|}{n+1} > \alpha \right\}$$

where $R_i^y$ are the nonconformity scores computed with the augmented dataset.

Full conformal provides the tightest intervals but is computationally prohibitive for most models — you must refit the model for every candidate $y$ on a grid. For a return prediction, this could mean thousands of refits per prediction.

### Jackknife+ (Barber et al., 2021)

The jackknife+ strikes a balance. It uses leave-one-out (LOO) residuals but accounts for the variability in the fitted model across LOO folds.

Let $\hat{\mu}_{-i}$ denote the model trained on all data except point $i$. Define the LOO nonconformity score with the single absolute residual:

$$R_i = |Y_i - \hat{\mu}_{-i}(X_i)|$$

The jackknife+ prediction interval is then built from the LOO predictions at the test point, widened by these residuals:

$$C(X_{n+1}) = \left[\, q_{\alpha}^{-}\!\left\{\hat{\mu}_{-i}(X_{n+1}) - R_i\right\}, \;\; q_{1-\alpha}^{+}\!\left\{\hat{\mu}_{-i}(X_{n+1}) + R_i\right\} \right]$$

Here $q_{1-\alpha}^{+}\{v_i\}$ denotes the $\lceil (1-\alpha)(n+1)\rceil$-th smallest value of the set $\{v_i\}_{i=1}^n$, and $q_{\alpha}^{-}\{v_i\}$ the $\lfloor \alpha(n+1)\rfloor$-th smallest value. The lower bound **subtracts** the residual from each LOO prediction; the upper bound **adds** it. That asymmetry is the whole point — collapsing both bounds to $\hat{\mu}_{-i} + R_i$ would put the lower bound above the prediction, which is wrong.

The jackknife+ provides a coverage guarantee of at least $1 - 2\alpha$ (slightly weaker than split conformal's $1-\alpha$), but it uses all the data for both training and calibration. In practice the coverage is typically close to $1-\alpha$.

For trading models trained on limited data (e.g., regime-specific models with only a few hundred observations), the jackknife+ is often the best choice — it does not sacrifice scarce data for calibration. The cost is $n$ model refits.

## The Problem With Financial Time Series: Non-Exchangeability

![Non-exchangeability in financial time series: a non-stationary series with a regime shift, coverage cracking as the distribution drifts](/images/blog/conformal-prediction-trading-non-exchangeability.webp)

The standard conformal guarantee requires exchangeability: the joint distribution of $(Z_1, \ldots, Z_{n+1})$ is invariant under permutations. For i.i.d. data, this holds trivially.

Financial time series are not exchangeable. Returns exhibit:

- **Volatility clustering:** High-volatility periods follow high-volatility periods (GARCH effects).
- **Momentum and mean reversion:** Autocorrelation in returns or squared returns.
- **Regime changes:** The distribution shifts over time (bull vs. bear markets).

If you naively apply split conformal to a time series — using a random calibration split — you violate the temporal structure. The calibration scores from a calm 2017 will not reflect the uncertainty of a volatile 2020. Your coverage guarantee breaks.

## Adaptive Conformal Inference (ACI)

![Adaptive conformal inference: a prediction interval that widens and narrows via a feedback loop tracking realized coverage toward target](/images/blog/conformal-prediction-trading-adaptive.webp)

Gibbs and Candes (2021, NeurIPS) introduced **Adaptive Conformal Inference (ACI)** to handle distribution shift and non-exchangeable data. The idea is elegant: instead of using a fixed coverage level, adapt the *target miscoverage level* online based on whether recent intervals covered the true outcome, and re-derive the quantile from the score distribution at each step.

### The ACI Algorithm

ACI does **not** nudge the interval width directly. It maintains an adaptive miscoverage parameter $\alpha_t$ and recomputes the conformal quantile from it. At each time step $t$:

1. Compute the conformal threshold as the empirical $(1 - \alpha_t)$-quantile of the current residual set (the calibration scores, plus any realized scores so far): $\hat{q}_t = \widehat{\text{Quantile}}_{1-\alpha_t}(\{R_j\})$.
2. Observe features $X_t$, produce the interval $C_t(X_t) = [\hat{\mu}(X_t) - \hat{q}_t, \; \hat{\mu}(X_t) + \hat{q}_t]$.
3. Observe the true value $Y_t$ and compute the error indicator $\text{err}_t = \mathbf{1}\{Y_t \notin C_t(X_t)\}$.
4. Update the **level** (not the width):

$$\alpha_{t+1} = \text{clip}\!\left(\alpha_t + \gamma\,(\alpha - \text{err}_t),\; 0,\; 1\right)$$

where $\gamma > 0$ is a step size and $\alpha$ is the target miscoverage. If an interval missed ($\text{err}_t = 1$), $\alpha_t$ shrinks, which pushes the next quantile higher and widens the interval; if it covered, $\alpha_t$ grows and intervals tighten. Crucially, $\gamma$ here is in **probability units** — it nudges a level in $[0,1]$, not the threshold in raw return units — so the same $\gamma$ behaves sensibly whether residuals are on the order of $10^{-3}$ or not.

### Coverage Guarantee for ACI

ACI provides a long-run coverage guarantee that does not depend on a distributional model:

$$\left|\frac{1}{T}\sum_{t=1}^{T} \text{err}_t - \alpha\right| \leq \frac{|\alpha_{T+1} - \alpha_1|}{\gamma T}$$

Because $\alpha_t$ is clipped to $[0,1]$, the numerator is bounded by $1/\gamma$ times a constant, so the right-hand side is $O(1/T)$ and the empirical miscoverage frequency converges to $\alpha$. The precise statement: ACI guarantees the long-run empirical miscoverage frequency converges to $\alpha$ for arbitrary (including adversarial) sequences, provided the adapted levels stay bounded — which the clip enforces. It is a guarantee on coverage *frequency*, not on interval *informativeness*: under a truly adversarial sequence the intervals can grow uninformatively wide while still hitting the coverage target.

### Dynamically-Tuned ACI (DtACI)

Gibbs and Candes (2024, JMLR) refined ACI with **dynamic tuning** of the step size $\gamma$. Instead of fixing $\gamma$, they maintain a candidate set $\Gamma = \{\gamma_1, \ldots, \gamma_K\}$ and combine them via an expert-aggregation rule, favoring the $\gamma$ whose recent coverage is closest to the target.

This addresses a practical problem: a large $\gamma$ adapts quickly to regime changes but produces volatile interval widths; a small $\gamma$ is stable but slow to adapt. DtACI trades these off automatically.

### Why This Matters for Trading

Consider a market-making strategy that uses a return forecasting model. During calm markets, the conformal intervals are tight — the model is confident, and you can take larger positions. When volatility spikes (earnings season, FOMC announcements, geopolitical shocks), the ACI level adapts and the intervals widen within a few time steps. Your position sizing shrinks in response, **without** any explicit volatility model or regime-detection logic.

This is uncertainty quantification as a first-class signal, not an afterthought.

## Position Sizing With Conformal Intervals

![Mapping a calibrated uncertainty interval to position size: a tight interval drives a large position, a wide one a small position](/images/blog/conformal-prediction-trading-position-sizing.webp)

Now let us connect conformal prediction to concrete position sizing. The key variable is the **prediction interval half-width** relative to the symmetric absolute-residual case. With the symmetric interval $[\hat{\mu}(X_t) - \hat{q}_t, \; \hat{\mu}(X_t) + \hat{q}_t]$, the full width is $w_t = 2\hat{q}_t$. To keep formulas and code consistent, we measure everything against the **full width** $w_t$ throughout.

### Inverse-Width Sizing

The simplest approach: size inversely proportional to interval width.

$$\text{position\_size}_t = \frac{k}{w_t}$$

where $k$ is a scaling constant calibrated to your risk budget. When the model is confident (narrow interval), you take a larger position. When uncertain (wide interval), you take a smaller one.

This is analogous to volatility targeting ($\text{size} \propto 1/\sigma$), but with a crucial difference: the conformal interval width is a distribution-free uncertainty measure, not a parametric volatility estimate. It captures predictive uncertainty under the coverage guarantee, not just return variance.

### Edge-Ratio Sizing and the No-Trade Filter

Pure inverse-width sizing ignores the strength of the signal itself. A natural refinement scales by the **edge ratio** — the point prediction relative to the interval width:

$$e_t = \frac{|\hat{\mu}(X_t)|}{w_t}$$

This is a conformal analog of a signal-to-noise ratio: expected return divided by a distribution-free uncertainty measure. We use it both for sizing and for a no-trade filter.

The filter is principled. If the interval straddles zero,

$$\text{lower}_t < 0 < \text{upper}_t,$$

then the $(1-\alpha)$ prediction interval includes both positive and negative returns — the realized return can plausibly have the opposite sign of your prediction. Define a **minimum edge threshold** $\theta$ and trade only when $e_t > \theta$. Note that the geometric content of "$e_t$ large enough that the interval no longer straddles zero" is exactly $e_t > 1/2$ (since the interval clears zero when $|\hat{\mu}| > \hat{q}_t = w_t/2$). Choose $\theta$ on the actual scale of $e_t = |\hat{\mu}|/w_t$ via backtesting; for daily-return residuals $e_t$ is usually well below $1/2$, so a tiny $\theta$ may admit nearly all trades and a large one may admit none. Calibrate it to your data.

### On "Conformal Kelly"

It is tempting to bolt conformal intervals onto the Kelly fraction $f^* = \frac{pb - (1-p)}{b}$. But $f^*$ is already a complete, bounded fraction derived from a win probability $p$ and a payoff ratio $b$; multiplying it by an unbounded ratio like $\hat{\mu}/\hat{q}$ has no decision-theoretic justification — it can exceed 1 or flip sign independently of $f^*$, and it double-counts the edge that $f^*$ already encodes. So we do **not** present a "conformal Kelly" multiplier.

If you want to drive Kelly from the interval, you must actually derive $p$ and $b$ from it, which requires an explicit assumption about the distribution *within* the interval (conformal intervals deliberately say nothing about that — see Limitations). For example, under an assumed within-interval shape you can approximate $p \approx \mathbb{P}(\text{return} > 0)$ and a payoff ratio from the interval geometry — but flag that assumption loudly, because it reintroduces exactly the parametric commitment conformal prediction was meant to avoid.

The honest, assumption-light alternative is to use the edge ratio $e_t = |\hat{\mu}|/w_t$ as a **fractional-Kelly shrinkage**: size up when the expected return is large relative to the interval, size down when it is small, and apply this on top of a hard cap — explicitly as a heuristic, not as "the Kelly fraction."

## Python Implementation

Here is a practical implementation. We show both the split/prefit path and the temporal (EnbPI) path, since the whole point of this post is that financial data is non-exchangeable.

### Setup and Data Preparation

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold

# MAPIE 0.8.x API. For MAPIE >= 1.0 (2025) the classes are
# SplitConformalRegressor / CrossConformalRegressor with a
# .conformalize() / .predict_interval() API -- pin the version.
from mapie.regression import MapieRegressor, MapieTimeSeriesRegressor
from mapie.subsample import BlockBootstrap


def prepare_features(prices: pd.Series, lookback: int = 20) -> pd.DataFrame:
    """Create features from a price series."""
    df = pd.DataFrame()
    returns = prices.pct_change()

    # Lagged returns
    for lag in range(1, lookback + 1):
        df[f"ret_lag_{lag}"] = returns.shift(lag)

    # Rolling statistics (shifted by 1 to avoid look-ahead)
    for window in [5, 10, 20]:
        df[f"ret_mean_{window}"] = returns.rolling(window).mean().shift(1)
        df[f"ret_std_{window}"] = returns.rolling(window).std().shift(1)
        df[f"ret_skew_{window}"] = returns.rolling(window).skew().shift(1)

    # Target: next-period return
    df["target"] = returns.shift(-1)

    return df.dropna()
```

### Split Conformal With MAPIE (Prefit)

For split/prefit conformal, `cv="prefit"` requires `method="base"` (the naive split-conformal estimator). The `method="plus"` option is the CV+/jackknife+ estimator and is incompatible with `cv="prefit"` — it needs a cross-validation object instead. We use the correct combination here, and standardize sizing on the full width.

```python
def split_conformal_sizing(
    prices: pd.Series,
    alpha: float = 0.1,
    k: float = 1e-3,          # scaling constant, in width units
    max_position: float = 1.0,
    min_edge: float = 0.05,   # threshold on |pred| / width
) -> pd.DataFrame:
    """
    Position sizing using split (prefit) conformal prediction intervals.

    Sizing rule (consistent with the prose):
        edge_t = |pred_t| / width_t
        size_t = clip(k / width_t, 0, max_position)   # inverse-width
        size_t = 0 if edge_t < min_edge               # no-trade filter
    """
    df = prepare_features(prices)
    X = df.drop(columns=["target"])
    y = df["target"]

    # Time-respecting split: train on earlier data, calibrate on later.
    n_train = int(len(X) * 0.6)
    n_cal = int(len(X) * 0.2)

    X_train, y_train = X.iloc[:n_train], y.iloc[:n_train]
    X_cal, y_cal = X.iloc[n_train:n_train + n_cal], y.iloc[n_train:n_train + n_cal]
    X_test, y_test = X.iloc[n_train + n_cal:], y.iloc[n_train + n_cal:]

    base_model = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42,
    )
    base_model.fit(X_train, y_train)

    # Prefit split conformal: cv="prefit" REQUIRES method="base".
    mapie = MapieRegressor(estimator=base_model, cv="prefit", method="base")
    mapie.fit(X_cal, y_cal)

    y_pred, y_intervals = mapie.predict(X_test, alpha=alpha)
    lower = y_intervals[:, 0, 0]
    upper = y_intervals[:, 1, 0]
    width = upper - lower

    # Inverse-width sizing scaled by a FIXED k, then clipped.
    # (We do NOT normalize by the mean before clipping -- that would
    #  flatten every above-average point to max_position and destroy
    #  the gradation we are trying to express.)
    raw_size = k / np.where(width > 0, width, np.inf)
    position_size = np.clip(raw_size, 0.0, max_position)

    # No-trade filter on the SAME quantity used elsewhere: |pred| / width.
    edge_ratio = np.abs(y_pred) / np.where(width > 0, width, np.inf)
    position_size = np.where(edge_ratio < min_edge, 0.0, position_size)

    # Direction from the sign of the point prediction.
    position_size = position_size * np.sign(y_pred)

    return pd.DataFrame({
        "prediction": y_pred,
        "lower": lower,
        "upper": upper,
        "width": width,
        "edge_ratio": edge_ratio,
        "position_size": position_size,
        "actual": y_test.values,
    }, index=X_test.index)
```

### Time-Series Conformal With EnbPI

Because returns are non-exchangeable, the random/prefit split above is only a baseline. MAPIE's `MapieTimeSeriesRegressor` with `method="enbpi"` (Xu & Xie, 2021) uses block bootstrap and residual updating designed for temporal dependence. This is the tool that matches the post's own argument.

```python
def enbpi_sizing(
    prices: pd.Series,
    alpha: float = 0.1,
    k: float = 1e-3,
    max_position: float = 1.0,
    min_edge: float = 0.05,
) -> pd.DataFrame:
    """Position sizing with EnbPI (block-bootstrap, time-series conformal)."""
    df = prepare_features(prices)
    X = df.drop(columns=["target"])
    y = df["target"]

    n_train = int(len(X) * 0.7)
    X_train, y_train = X.iloc[:n_train], y.iloc[:n_train]
    X_test, y_test = X.iloc[n_train:], y.iloc[n_train:]

    base_model = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42,
    )

    cv = BlockBootstrap(n_resamplings=30, length=20, overlapping=False, random_state=42)
    mapie_ts = MapieTimeSeriesRegressor(base_model, method="enbpi", cv=cv, agg_function="mean")
    mapie_ts.fit(X_train, y_train)

    # Predict; partial_fit lets the residuals update online as truth arrives.
    y_pred, y_intervals = mapie_ts.predict(X_test, alpha=alpha, ensemble=True)
    lower = y_intervals[:, 0, 0]
    upper = y_intervals[:, 1, 0]
    width = upper - lower

    raw_size = k / np.where(width > 0, width, np.inf)
    position_size = np.clip(raw_size, 0.0, max_position)
    edge_ratio = np.abs(y_pred) / np.where(width > 0, width, np.inf)
    position_size = np.where(edge_ratio < min_edge, 0.0, position_size)
    position_size = position_size * np.sign(y_pred)

    return pd.DataFrame({
        "prediction": y_pred, "lower": lower, "upper": upper,
        "width": width, "edge_ratio": edge_ratio,
        "position_size": position_size, "actual": y_test.values,
    }, index=X_test.index)
```

### Adaptive Conformal Inference (Online)

For live trading we implement true ACI from scratch: maintain the miscoverage **level** $\alpha_t$, update it additively, and re-derive the quantile from the residual set each step. Two finite-sample details matter:

- Use the **order statistic**, not an interpolated quantile. `np.quantile` interpolates by default, which can fall just below the required rank and undercover; pass `method="higher"` (equivalently `"inverted_cdf"`).
- When the required rank exceeds $n$ (small $n$, high target coverage), the correct threshold is $+\infty$ (interval = whole line), not a clamp to the largest residual. Clamping silently breaks the $\ge 1-\alpha_t$ guarantee.

```python
class AdaptiveConformalSizer:
    """
    Online position sizing with Adaptive Conformal Inference (Gibbs & Candes,
    2021). Updates the miscoverage LEVEL alpha_t and re-derives the quantile
    from the residual set each step -- gamma is in probability units.
    """

    def __init__(self, base_model, alpha=0.1, gamma=0.02,
                 max_position=1.0, min_edge=0.05, k=1e-3):
        self.base_model = base_model
        self.alpha_target = alpha     # target miscoverage
        self.alpha_t = alpha          # adaptive miscoverage level
        self.gamma = gamma            # step size, in [0, 1] units
        self.max_position = max_position
        self.min_edge = min_edge
        self.k = k
        self.residuals = []
        self.q_hat = np.inf
        self.coverage_history = []

    @staticmethod
    def _conformal_quantile(residuals, alpha_t):
        """(1 - alpha_t) conformal quantile via the order statistic."""
        n = len(residuals)
        if n == 0:
            return np.inf
        rank = int(np.ceil((1.0 - alpha_t) * (n + 1)))
        if rank > n:                  # required order statistic does not exist
            return np.inf             # -> interval is the whole line
        # method="higher" returns the exact order statistic (no interpolation).
        level = rank / n
        return float(np.quantile(residuals, min(level, 1.0), method="higher"))

    def calibrate(self, X_cal, y_cal):
        preds = self.base_model.predict(X_cal)
        self.residuals = list(np.abs(np.asarray(y_cal) - preds))
        self.q_hat = self._conformal_quantile(self.residuals, self.alpha_t)

    def predict_and_size(self, X_t) -> dict:
        mu_hat = self.base_model.predict(np.asarray(X_t).reshape(1, -1))[0]
        lower, upper = mu_hat - self.q_hat, mu_hat + self.q_hat
        width = upper - lower                      # = 2 * q_hat

        edge_ratio = abs(mu_hat) / width if np.isfinite(width) and width > 0 else 0.0

        if edge_ratio < self.min_edge:
            size = 0.0
        else:
            size = min(self.k / width, self.max_position) if width > 0 else 0.0

        return {
            "prediction": mu_hat, "lower": lower, "upper": upper,
            "width": width, "edge_ratio": edge_ratio,
            "position_size": size * np.sign(mu_hat),
            "alpha_t": self.alpha_t, "q_hat": self.q_hat,
        }

    def update(self, X_t, y_t: float):
        """ACI update: adapt the LEVEL, then re-derive the quantile."""
        mu_hat = self.base_model.predict(np.asarray(X_t).reshape(1, -1))[0]
        residual = abs(y_t - mu_hat)

        covered = int(residual <= self.q_hat)
        err_t = 1 - covered

        # Update the miscoverage level (NOT the threshold directly), clipped.
        self.alpha_t = float(np.clip(
            self.alpha_t + self.gamma * (self.alpha_target - err_t), 0.0, 1.0
        ))

        # Grow the residual set and re-derive the quantile from alpha_t.
        self.residuals.append(residual)
        self.q_hat = self._conformal_quantile(self.residuals, self.alpha_t)

        self.coverage_history.append(covered)

    @property
    def running_coverage(self) -> float:
        if not self.coverage_history:
            return float("nan")
        return float(np.mean(self.coverage_history))
```

### Putting It Together: Backtest Loop

```python
def backtest_aci_sizing(prices: pd.Series, alpha=0.1, gamma=0.02) -> pd.DataFrame:
    """Backtest position sizing with Adaptive Conformal Inference."""
    df = prepare_features(prices)
    X = df.drop(columns=["target"]).values
    y = df["target"].values
    index = df.index

    n_train = int(len(X) * 0.5)
    n_cal = int(len(X) * 0.2)

    X_train, y_train = X[:n_train], y[:n_train]
    X_cal, y_cal = X[n_train:n_train + n_cal], y[n_train:n_train + n_cal]
    X_test, y_test = X[n_train + n_cal:], y[n_train + n_cal:]
    test_index = index[n_train + n_cal:]

    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42,
    )
    model.fit(X_train, y_train)

    sizer = AdaptiveConformalSizer(base_model=model, alpha=alpha, gamma=gamma)
    sizer.calibrate(X_cal, y_cal)

    records = []
    for i in range(len(X_test)):
        result = sizer.predict_and_size(X_test[i])
        result["actual"] = y_test[i]
        result["pnl"] = result["position_size"] * y_test[i]
        records.append(result)
        sizer.update(X_test[i], y_test[i])   # online residual + level update

    results = pd.DataFrame(records, index=test_index)
    results["cumulative_pnl"] = results["pnl"].cumsum()
    results["running_coverage"] = (
        ((results["lower"] <= results["actual"]) &
         (results["actual"] <= results["upper"])).expanding().mean()
    )
    return results
```

### Evaluating the Results

```python
def evaluate(results: pd.DataFrame, alpha: float = 0.1):
    """Print evaluation metrics for conformal position sizing."""
    covered = ((results["actual"] >= results["lower"]) &
               (results["actual"] <= results["upper"]))

    print(f"Target coverage:      {1 - alpha:.1%}")
    print(f"Empirical coverage:   {covered.mean():.1%}")
    print(f"Mean interval width:  {results['width'].mean():.6f}")
    print(f"Median position size: {results['position_size'].abs().median():.4f}")
    print(f"Fraction no-trade:    {(results['position_size'] == 0).mean():.1%}")
    print(f"Total PnL (bps):      {results['pnl'].sum() * 10000:.1f}")
    sd = results['pnl'].std()
    sharpe = results['pnl'].mean() / sd * np.sqrt(252) if sd > 0 else float('nan')
    print(f"Sharpe (annualized):  {sharpe:.2f}")
```

## Practical Considerations

### Choosing the Nonconformity Score

The absolute residual $|Y - \hat{\mu}(X)|$ is the default, but it assumes the prediction interval should be symmetric around the point prediction. For financial returns, asymmetric intervals often make more sense:

- **Conformalized Quantile Regression (CQR):** Fit quantile regressors at levels $\alpha/2$ and $1 - \alpha/2$, then conformalize (Romano et al., 2019). The intervals adapt their shape to the local distribution — wider on the downside during drawdowns, wider on the upside during rallies. (With CQR the interval is no longer symmetric, so $w_t$ is the genuine upper-minus-lower width — keep using $w_t$ as the denominator everywhere.)
- **Normalized scores:** $R_i = |Y_i - \hat{\mu}(X_i)| / \hat{\sigma}(X_i)$, where $\hat{\sigma}$ is a local volatility estimate. This produces intervals that are conditionally adaptive — tighter during low-volatility regimes, wider during high-volatility ones — while maintaining marginal coverage.

### Calibration Set Size

The coverage guarantee of split conformal holds for any calibration set size $n$, but the interval width decreases as $n$ increases. For very small $n$ the required order statistic may not exist, in which case the honest threshold is $+\infty$ (an uninformative but valid interval) — clamping to the largest residual quietly voids the guarantee. For practical trading:

- $n \geq 100$ calibration points gives reasonably tight intervals.
- $n \geq 500$ is preferred for stable quantile estimation.
- With ACI, the calibration set is only for initialization; the online level updates handle the rest.

### Retraining Frequency

The base model $\hat{\mu}$ can go stale. Two approaches:

1. **Retrain periodically** (e.g., monthly) and re-calibrate the conformal quantile.
2. **Use ACI** and let the adaptive level compensate for model staleness — the intervals widen automatically as the model's residuals grow.

Option 2 is simpler and surprisingly effective. The conformal layer acts as a safety net: even if the model degrades, the long-run ACI coverage frequency still converges to the target.

### Transaction Costs

Conformal intervals interact with transaction costs in a useful way. When intervals are wide (high uncertainty), positions are small, so turnover is low. When intervals narrow (the model is confident), positions grow — but the model is more likely to be right, so the turnover is worth paying for.

You can also incorporate transaction costs directly into the no-trade filter:

$$\text{trade only if } |\hat{\mu}(X_t)| - \text{cost} > \theta \cdot w_t$$

This ensures you only trade when the expected net edge exceeds a threshold scaled by the conformal width — using the same $w_t$ denominator as everywhere else.

## Comparison With Traditional Methods

| Property | Gaussian CI | Bootstrap CI | Conformal PI |
|----------|-------------|--------------|--------------|
| Distribution assumption | Normal errors | i.i.d. + asymptotic | None (exchangeability) |
| Finite-sample guarantee | No | No | Yes |
| Works with any model | No | Yes | Yes |
| Adapts to heteroskedasticity | No | Partially | With CQR / normalized scores |
| Handles distribution shift | No | No | ACI / EnbPI variant |
| Computational cost | Low | Medium | Split: low; jackknife+: $O(n)$ refits; full: prohibitive |

The bootstrap is "asymptotic" only in its *guarantee*; it still **assumes** i.i.d./exchangeable data and smoothness, so its distribution-assumption cell is not "assumption-free." And the single "conformal" column hides very different costs across variants, which the cost row now spells out.

## Limitations

Conformal prediction is not magic. Key limitations for trading:

1. **Marginal, not conditional coverage.** The guarantee is that $\mathbb{P}(Y_{n+1} \in C(X_{n+1})) \geq 1 - \alpha$ marginally — averaged over the randomness in both $X$ and $Y$. For a specific conditioning value $X = x$, the local coverage may be above or below $1 - \alpha$. Conformalized quantile regression partially addresses this.

2. **Exchangeability is a real requirement.** For split conformal, the calibration data and test point must be exchangeable. Financial data is not. ACI and EnbPI relax this to a long-run guarantee, but short-run coverage can deviate.

3. **Interval width is not a probability density.** A conformal interval tells you where $Y$ will fall with probability $1-\alpha$, but it says nothing about the distribution *within* the interval. It is not a substitute for a full predictive distribution — which is exactly why deriving a Kelly $p$ and $b$ from it requires an extra, explicit assumption.

4. **Garbage in, wider out.** A bad base model produces wide intervals. Conformal prediction guarantees coverage but not usefulness. If your model has no predictive power, the intervals will be so wide that the position sizer never trades.

## Summary

Conformal prediction provides a rigorous, distribution-free framework for uncertainty quantification that slots naturally into position sizing:

- **Split conformal** for static, offline calibration with finite-sample coverage.
- **Jackknife+** when calibration data is scarce and you want to use all observations (at the cost of $n$ refits).
- **Adaptive conformal inference / EnbPI** for online trading with non-stationary markets.
- **Position sizing** via inverse interval width and an edge-ratio no-trade filter — and, if you go to Kelly, only after deriving $p$ and $b$ honestly, not via an unjustified multiplier.

The key advantage over parametric alternatives: you never need to specify or validate a parametric distributional assumption. The intervals are honest by construction (marginally, under exchangeability). For a systematic trader, this means one fewer source of model risk — and in a business where model risk is existential, that matters.

---

**References:**

- Papadopoulos, H., Proedrou, K., Vovk, V., Gammerman, A. (2002). Inductive confidence machines for regression. *ECML*.
- Vovk, V., Gammerman, A., Shafer, G. (2005). *Algorithmic Learning in a Random World.* Springer.
- Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R.J., Wasserman, L. (2018). Distribution-free predictive inference for regression. *JASA*.
- Xu, C., Xie, Y. (2021). Conformal prediction interval for dynamic time-series (EnbPI). *ICML*.
- Barber, R.F., Candes, E.J., Ramdas, A., Tibshirani, R.J. (2021). Predictive inference with the jackknife+. *Annals of Statistics*.
- Gibbs, I., Candes, E.J. (2021). Adaptive conformal inference under distribution shift. *NeurIPS*.
- Gibbs, I., Candes, E.J. (2024). Conformal inference for online prediction with arbitrary distribution shifts. *JMLR*.
- Romano, Y., Patterson, E., Candes, E.J. (2019). Conformalized quantile regression. *NeurIPS*.
- Cordier, T. et al. (2022). MAPIE: an open-source library for distribution-free uncertainty quantification. *arXiv:2207.12274*.
