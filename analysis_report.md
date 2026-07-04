# Integrating the Attention Mechanism in Predicting the Vietnamese Stock Market

## Abstract

Attention mechanisms have transformed natural language processing, but their application to financial time series raises a distinct question: what should attention attend to, and does the answer depend on market conditions? We investigate this by evaluating seven attention-based architectures—each integrating market context into the attention pipeline differently—on Vietnamese VN100 stock return prediction across three training horizons (3, 5, and 10 years). Our central finding is that **the value of sophisticated attention integration scales with regime diversity in the training data**. Tree-based gating mechanisms that route attention conditionally on market volatility and momentum features (MoE+LGBM Gate) achieve the best prediction accuracy (IC = 0.30) on the longest horizon, where the data span 2–3× more market regime transitions. However, on the 3-year horizon—where 80% of training days occur in bear-market drawdown—cross-attention overfits to homogeneous conditions, producing brittle predictions with high accuracy but low consistency. The simplest attention variant (Base MASTER) achieves the best risk-adjusted portfolio returns (IR = 3.40), demonstrating that attention complexity is not free: it improves raw prediction but adds variance that can erode realized performance. We attribute these patterns to quantifiable data properties—regime transition frequency, feature-label stability, and tail behavior—that practitioners can measure before choosing an architecture.

---

## 1. Introduction

### 1.1 The Problem: Predicting Stock Returns With Attention

The core task is *stock ranking*: given 100 Vietnamese stocks today, predict which will outperform which over the next 5 trading days (about one week). The output is not a price target but a *rank ordering*—the model should put the best stocks at the top and the worst at the bottom. A trader then buys the top-ranked stocks and avoids the bottom-ranked ones.

The input to the model is a small matrix: for each stock, we have 8 consecutive trading days of history, with 158 *technical features* per day. These features are mechanical transformations of price and volume data—momentum indicators, volatility measures, K-line (candlestick) patterns. No fundamentals, no news, no analyst estimates.

The input also includes 63 *market features*: returns and volatility statistics for three Vietnamese market indices (VNINDEX, VN100, VN30) computed over 5, 10, 20, 30, and 60-day windows. These tell the model what the overall market is doing.

The output, called the *label*, is the stock's 5-day forward return. Concretely: suppose today is Monday and a stock closes at 50,000 VND. On Tuesday it closes at 50,500 VND. The model's job is to predict the return from *Tuesday's close* to the close *5 trading days later* (the following Tuesday). If that future price is 52,000 VND, the label is (52,000 / 50,500) − 1 = +0.0297, or roughly +3%. If instead the stock falls to 49,000 VND, the label is (49,000 / 50,500) − 1 = −0.0297, or roughly −3%.

Why use tomorrow's price as the baseline rather than today's? Because today's price is already known at prediction time—the market has closed. The earliest practical entry point is tomorrow. The formula `(price_t+5 / price_t+1) − 1` measures the return an investor actually captures: buy at tomorrow's close, sell five trading days later. This removes the overnight jump from today to tomorrow, which the model cannot trade on.

The label is always a small decimal: +0.03 means "this stock rose 3% relative to tomorrow's close over the next five trading days." A label of 0.00 means the stock was flat. A label of −0.05 means it fell 5%.

**Where does attention fit in?** The transformer architecture at the heart of all seven models uses two kinds of attention. *Temporal self-attention* looks across the 8-day history of a single stock and asks: "Which of the past 8 days matter most for tomorrow's prediction?" *Spatial cross-attention* looks across all 158 features and asks: "Given current market conditions, which features should I pay attention to?" This second mechanism—how market context modulates feature attention—is what the seven architectures implement differently, and it is the subject of this paper.

### 1.2 The Core Challenge: Attention Must Adapt to Market Regimes

A feature like "20-day momentum" (MA20) is helpful in a trending bull market—stocks that have been going up tend to keep going up, so attention should focus on momentum features. But in a volatile bear market, momentum can be dangerous—stocks that went up last month may be the ones that crash hardest, so attention should shift to defensive features. The *same feature* deserves *different attention* depending on market conditions.

This is called *regime dependence*, and it is the central challenge for attention-based stock prediction. A *gate* is a sub-network that looks at market-level information (is the market trending up? is volatility high? are we in a drawdown?) and uses it to *reweight* the attention over stock-level features. In a bull market, the gate amplifies attention to momentum features. In a bear market, it redirects attention toward low-volatility features. The gate implements the principle that *what deserves attention depends on context*.

### 1.3 Research Questions

1. How does the training horizon (3, 5, or 10 years) affect the performance of attention-based stock prediction models?
2. Which method of integrating market context into the attention mechanism works best, and does the answer depend on how much training data is available?
3. What measurable properties of the data itself—regime transition frequency, feature-label stability, return distribution characteristics—explain why certain attention architectures outperform others?

---

## 2. The Seven Attention Architectures

All seven models share an identical backbone: a transformer encoder that processes 8-day sequences of 158 stock-level features. The backbone contains two attention sub-layers:

- **Temporal self-attention** operates along the time axis. For a single stock with an 8-day history, it computes attention weights between every pair of days, allowing day \(t\) to attend to information from days \(t-7\) through \(t\). This captures patterns like "the trend reversed 3 days ago" or "yesterday's volume spike matters more than last week's."

- **Spatial cross-attention** operates along the feature axis. It computes attention weights between the 158 stock-level features, allowing the model to learn that "momentum features should be weighted together" or "volume features should suppress volatility features when both are elevated."

The 63 market-level features (index returns and volatilities) enter the model through a sub-network called a *gate*, which sits *before* the transformer backbone. The gate's job is to produce a 158-dimensional vector \(\mathbf{w} \in \mathbb{R}^{158}\) of feature importance weights. Each stock-level feature \(x_j\) is multiplied element-wise by its gate weight \(w_j\) before entering the transformer: \(\tilde{x}_j = w_j \cdot x_j\). This means the gate can *amplify* features that are useful in the current market regime and *suppress* features that are misleading.

The seven architectures differ only in *how the gate computes \(\mathbf{w}\) from the 63 market features*. The transformer backbone, the training procedure, and all other components are identical. This design isolates the gate mechanism as the single experimental variable.

| Model | How Market Context Modulates Attention | Key Mechanism |
|-------|----------------------------------------|---------------|
| **Base MASTER** | Linear gate: \(\mathbf{w} = \mathbf{A} \mathbf{m} + \mathbf{b}\) | One learned matrix |
| **MoE** | Three expert gates + router network | Router selects which expert's mask to apply |
| **LGBM-Gate** | LightGBM tree predicts \(\mathbf{w}\); blended with neural prior | Tree learns threshold splits on market features |
| **BiLSTM** | Bidirectional LSTM replaces temporal self-attention; linear gate | Sequential pattern detection over 8-day window |
| **LGBM-LeafInput** | Tree leaf index embedded and injected into spatial attention | Discrete regime ID fed to transformer |
| **Cross-Attn Gate** | Market features query stock features via cross-attention | Market "looks at" stocks to decide attention |
| **MoE+LGBM Gate** | Tree detects regime → router selects expert → expert produces mask | Tree + experts: separate regime detection from strategy |

The following subsections describe each architecture in detail, including the data flow, dimensions, and design rationale.

### 2.1 Base MASTER — Linear Attention Modulation

**Data flow.** The 63 market features \(\mathbf{m} \in \mathbb{R}^{63}\) pass through a single linear (fully-connected) layer with weight matrix \(\mathbf{A} \in \mathbb{R}^{158 \times 63}\) and bias \(\mathbf{b} \in \mathbb{R}^{158}\):

\[
\mathbf{w} = \mathbf{A}\mathbf{m} + \mathbf{b}
\]

The resulting 158-dimensional vector \(\mathbf{w}\) is the attention mask. Each stock-level feature \(x_j\) is multiplied by \(w_j\), producing modulated features \(\tilde{x}_j = w_j \cdot x_j\) for \(j = 1,\ldots,158\). These modulated features then enter the standard transformer backbone.

**What it learns.** The matrix \(\mathbf{A}\) encodes 158 × 63 = 9,954 parameters. Each row of \(\mathbf{A}\) tells us how much each market feature contributes to the weight of a specific stock feature. For example, if row 20 (corresponding to MA20, the 20-day momentum feature) has a large negative entry in the column for VN100_RET_STD_10, the model learns: "when market volatility is high, reduce attention to momentum."

**Why it's the baseline.** This is the simplest possible integration of market context—a single linear transformation. There is no non-linearity (beyond what the transformer itself provides downstream), no regime partitioning, no expert specialization. If this model performs nearly as well as the more complex gates, the added complexity is not buying predictive power. Its simplicity also makes it the most *stable*: with only ~10K gate parameters, different random seeds converge to similar solutions.

**Limitation.** A linear gate cannot learn threshold rules. If momentum features are helpful when VN100 volatility is *either very low or very high* but harmful in the middle range, a linear gate cannot express this—it can only say "more volatility → less momentum attention" or "more volatility → more momentum attention," not both depending on the level.

### 2.2 MoE — Multi-Expert Attention Routing

**Data flow.** The MoE gate contains three independent *expert* sub-networks, each structurally identical to the Base MASTER's linear gate (a 63 → 158 linear layer). In parallel, a *router* network—another linear layer mapping \(\mathbb{R}^{63} \to \mathbb{R}^3\) followed by a softmax—produces a probability distribution over the three experts:

\[
\mathbf{p} = \text{softmax}(\mathbf{R}\mathbf{m}), \quad \mathbf{p} \in \mathbb{R}^3, \quad \sum_{k=1}^3 p_k = 1
\]

Each expert \(k\) produces its own attention mask \(\mathbf{w}^{(k)} = \mathbf{A}^{(k)}\mathbf{m} + \mathbf{b}^{(k)}\). The final mask is a weighted blend:

\[
\mathbf{w} = \sum_{k=1}^{3} p_k \cdot \mathbf{w}^{(k)}
\]

During training, we also add a small Gaussian noise term to the router logits (\(\sigma = 0.02\)) to encourage exploration—the router occasionally tries a different expert than its top choice, preventing premature convergence to a single expert. A load-balancing penalty (\(\lambda = 0.001\)) discourages the router from always selecting the same expert.

**What it learns.** Ideally, the three experts specialize. Expert 0 might learn high attention to momentum features, becoming the "bull market" expert. Expert 1 might learn high attention to low-volatility and quality features, becoming the "bear market" expert. Expert 2 might learn to attend to mean-reversion signals, becoming the "choppy market" expert. The router learns to blend them based on current conditions.

**Why it can fail.** With only 3 experts and a linear router, the specialization is *soft*—the router typically blends all three rather than hard-selecting one. If the training data lacks regime diversity (as in the 3-year dataset), the experts never differentiate and the MoE reduces to approximately three copies of the base linear gate averaged together, wasting parameters.

**Parameter count.** 3 experts × (63 × 158 + 158) + 1 router × (63 × 3 + 3) ≈ 30,000 parameters—three times the base model but still small relative to the transformer backbone.

### 2.3 LGBM-Gate — Tree-Based Attention Modulation

**Data flow.** This gate replaces the linear layer with a LightGBM regression tree. LightGBM is a gradient-boosted decision tree implementation optimized for speed and memory efficiency. The tree takes the 63 market features as input and outputs a 158-dimensional vector of predicted feature weights:

\[
\mathbf{w}^{\text{tree}} = \text{LGBM}(\mathbf{m}), \quad \mathbf{w}^{\text{tree}} \in \mathbb{R}^{158}
\]

The tree is trained to predict feature importance weights that minimize the transformer's training loss. Simultaneously, a neural prior \(\mathbf{w}^{\text{neural}} = \mathbf{A}\mathbf{m} + \mathbf{b}\) (identical to the Base MASTER gate) is computed. The final weight is a convex combination:

\[
\mathbf{w} = (1 - \alpha) \cdot \mathbf{w}^{\text{neural}} + \alpha \cdot \mathbf{w}^{\text{tree}}, \quad \alpha = 0.2
\]

The mixing coefficient \(\alpha = 0.2\) means the tree contributes 20% and the neural prior contributes 80%. The tree acts as a *correction term*—the neural gate provides a stable baseline, and the tree adjusts it when market conditions match a learned threshold rule.

**Tree hyperparameters.** The LightGBM uses 31 leaves, a learning rate of 0.05, and 300 estimators. These settings produce a moderately complex tree that can capture non-linear regime boundaries without overfitting. The tree is retrained at each epoch on the current batch of training data.

**What it learns.** Decision trees naturally express threshold rules. A learned split might be: "if VN100_vol_20d > 0.25 AND VN100_ret_60d < −0.05, then increase attention weight for feature 45 (RSV20, a mean-reversion indicator) by 0.15." These rules are *interpretable*—you can inspect the tree to understand what market conditions trigger which attention changes.

**Why it should work for financial data.** Market regime boundaries are threshold-like, not smooth. A bear market "begins" when drawdown crosses −20%, not when it smoothly drifts past that level. Trees model discontinuities; neural networks model smooth functions. The hybrid design combines both: the neural prior handles gradual changes, the tree handles regime switches.

### 2.4 BiLSTM — Sequential Attention Over Time

**Data flow.** This architecture modifies the transformer backbone itself, not just the gate. The temporal self-attention sub-layer is replaced by a bidirectional LSTM (Long Short-Term Memory) network with one layer and hidden size equal to the model dimension (\(d_{\text{model}} = 256\)):

\[
\mathbf{h}_1, \ldots, \mathbf{h}_8 = \text{BiLSTM}(x_1, \ldots, x_8)
\]

where \(x_t \in \mathbb{R}^{158}\) is the feature vector on day \(t\) of the 8-day lookback. The BiLSTM processes the sequence both forward (day 1 → day 8) and backward (day 8 → day 1), concatenating the hidden states. This gives each time step a representation that depends on both past and future context within the 8-day window.

The spatial cross-attention sub-layer remains unchanged. The gate for feature-level attention is the same linear gate as Base MASTER.

**What it learns.** LSTMs are designed to capture long-range sequential dependencies through gated memory cells (input gate, forget gate, output gate). Over an 8-day window, the BiLSTM can detect patterns like: "the stock gapped down on day 3 but recovered by day 5, and volume spiked on day 6" — a narrative that a single attention-weighted sum over days might blur. The bidirectional processing means day 1 "knows about" day 8 and vice versa, enabling the model to identify patterns relative to both the beginning and end of the window.

**Limitation.** The gate—the mechanism that integrates market context—remains linear. The BiLSTM improves *temporal* processing but does not improve *regime-conditional* processing. It is better at answering "what pattern occurred in the last 8 days?" but no better at answering "given current market conditions, should I care about that pattern?"

### 2.5 LGBM-LeafInput — Discrete Regime Embeddings in Attention

**Data flow.** A LightGBM regression tree with 63 leaves is trained to predict stock returns from the 63 market features. Rather than using the tree's predicted value, this architecture uses the *leaf index*—which of the 63 terminal leaves each sample falls into. This leaf index (an integer from 0 to 62) is mapped to a learned embedding vector \(\mathbf{e} \in \mathbb{R}^{d_{\text{model}}}\) via an embedding lookup table:

\[
\ell = \text{LGBM-leaf}(\mathbf{m}), \quad \ell \in \{0, \ldots, 62\}
\]
\[
\mathbf{e} = \text{Embedding}[\ell], \quad \mathbf{e} \in \mathbb{R}^{256}
\]

This regime embedding \(\mathbf{e}\) is injected into the spatial cross-attention layer: it is concatenated with the stock features before the attention computation, so the attention mechanism can condition on "which regime are we in?" when deciding which features to attend to. The leaf embedding is trained end-to-end with the rest of the model.

**What it should do (in theory).** The tree acts as a regime classifier, partitioning the 63-dimensional market feature space into 63 discrete regions. Leaf 0 might correspond to "low volatility, positive trend," leaf 15 to "high volatility, negative trend," etc. The embedding provides the transformer with a concise regime identifier—a 256-dimensional vector that encodes "you are in regime type K." The spatial attention can then learn regime-specific attention patterns: attend to momentum in leaf 0, attend to defensive features in leaf 15.

**What actually goes wrong (§5.4 will detail).** The tree is retrained each epoch on a different random subset of data. Different subsets produce different tree structures: the boundary between leaf 5 and leaf 7 shifts, or the meaning of leaf 5 changes entirely. The embedding learned for leaf 5 in epoch 1 represents a regime that may not exist in epoch 30, causing the transformer to receive inconsistent regime signals. The model wastes capacity reconciling these shifting embeddings.

### 2.6 Cross-Attention Gate — Market-Feature Cross-Attention

**Data flow.** Unlike all previous gates, which produce a flat weight vector \(\mathbf{w} \in \mathbb{R}^{158}\), the cross-attention gate allows the market features and stock features to *interact* before producing the modulation. The computation follows the standard attention formula:

\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
\]

where:
- **Query (\(Q\)):** A learned linear projection of the 63 market features, \(\mathbf{Q} = \mathbf{m}\mathbf{W}^Q\), producing a query vector of dimension \(d_k = 64\).
- **Keys (\(K\)) and Values (\(V\)):** Learned linear projections of the 158 stock-level features (using only the most recent day's values, not the full 8-day sequence), \(\mathbf{K} = \mathbf{x}\mathbf{W}^K\), \(\mathbf{V} = \mathbf{x}\mathbf{W}^V\), each producing 158 vectors of dimension \(d_k = 64\).
- **Output:** The attention output is a weighted sum of the value vectors, where the weights are determined by how well each stock feature's key matches the market query. This is projected back to \(\mathbb{R}^{158}\) to produce the attention mask \(\mathbf{w}\).

With 2 attention heads, the market can pose two different "questions" about the stock features simultaneously. Head 1 might ask "which features are related to volatility?" while Head 2 asks "which features are related to momentum?"

**What it learns.** Unlike the linear gate, which applies the *same* market-conditioned weight to every stock, the cross-attention gate can theoretically produce *stock-specific* modulation—the same market conditions could direct different attention to different stocks. This is the key distinction: the gate is not just "given market X, weight feature Y by Z," but potentially "given market X, for stock A attend to momentum, for stock B attend to value."

**Why it can overfit (§5.2 will show).** The cross-attention gate has the most parameters in the gate sub-network and the most expressive interaction between market and stock features. With homogeneous training data (e.g., 3-year bear market), this expressiveness becomes a liability: the attention learns brittle stock-to-stock relationships that are specific to the training regime and fail when market conditions change.

### 2.7 MoE+LGBM Gate — Tree-Routed Multi-Expert Attention

**Data flow.** This hybrid combines the tree from §2.3 with the multi-expert structure from §2.2. The data flow has three stages:

1. **Regime detection (tree).** A LightGBM tree takes the 63 market features and outputs a 158-dimensional vector \(\mathbf{w}^{\text{tree}}\). Unlike §2.3, this tree output is not blended with a neural prior—it directly controls the router.

2. **Expert routing.** The tree output \(\mathbf{w}^{\text{tree}}\) is fed (along with the original market features) into a router network that produces expert selection probabilities \(\mathbf{p} \in \mathbb{R}^3\). The key difference from §2.2: the router sees both the raw market features AND the tree's interpretation of them.

3. **Expert attention masks.** Three expert networks (each a 63 → 158 linear layer) produce attention masks \(\mathbf{w}^{(1)}, \mathbf{w}^{(2)}, \mathbf{w}^{(3)}\). The final mask is \(\mathbf{w} = \sum_k p_k \mathbf{w}^{(k)}\).

The mixing coefficient (§2.3's \(\alpha = 0.2\)) is not needed here because the tree doesn't directly produce the attention mask—it produces the *routing decision*. The tree says "we are in regime type A" and the router says "regime A → use Expert 2." This separation is cleaner: the tree handles the discrete classification problem (which regime?), the experts handle the continuous regression problem (what attention weights for this regime?).

**Why it should be the best.** This is the only architecture that cleanly separates two distinct sub-problems: (1) *regime identification*—partitioning market conditions into discrete categories—which trees do well; and (2) *within-regime attention*—learning the optimal feature weighting for each category—which neural networks do well. The tree's threshold splits on VN100 volatility and momentum features detect regime boundaries; the experts learn specialized attention strategies for each side of those boundaries.

**Parameter count.** 3 experts × (63 × 158 + 158) + router including tree features ≈ 35,000 gate parameters. The additional cost over MoE is modest (~15% more parameters) for the benefit of tree-guided routing.

---

## 3. How We Tested

### 3.1 Data

We crawled daily price and volume data for 100 Vietnamese stocks (VN100 universe) from 2016 through 2025 using the vnstock API. From this raw data, we computed 158 Alpha158 technical features (K-line patterns, momentum oscillators, volatility measures, volume indicators) and 63 market-level features (VNINDEX, VN100, VN30 index returns and volatilities at 5/10/20/30/60-day windows).

Three datasets were carved from the same underlying data by varying the training window:

| Dataset | Train Period | Valid Period | Test Period | Training Years | Training Samples |
|---------|-------------|--------------|-------------|-----------------|------------------|
| 3-Year | Jan 2022 – Dec 2023 | Jan–Jun 2024 | Jul 2024 – Dec 2025 | 2 | 23,481 |
| 5-Year | Jan 2020 – Dec 2023 | Jan–Jun 2024 | Jul 2024 – Dec 2025 | 4 | 94,807 |
| 10-Year | Jan 2016 – Dec 2022 | Jan–Dec 2023 | Jan 2024 – Dec 2025 | 7 | 144,866 |

A *sample* is one stock on one trading day. Each sample contains an 8-day lookback window of features (so the input shape is 8 days × 221 features). The prediction target (*label*) is the stock's return over the next 5 trading days, measured relative to the next day's close: `label = (price_t+5 / price_t+1) − 1`.

Crucially, the test period is identical across all three datasets (2024–2025). Only the training data differs. This means any performance difference is caused by *what the model learned during training*, not by testing on different periods.

### 3.2 Training

All models were trained for 40 epochs with early stopping (training halts when loss drops below 95% of its initial value). We used the Adam optimizer with learning rate 1×10⁻⁵. Each model was trained 5 times with different random seeds; results are reported as mean ± standard deviation across seeds. Training used a single NVIDIA GPU.

### 3.3 Evaluation Metrics — Definitions and Formulas

All metrics are computed on the test set (2024–2025 data, which no model saw during training). Let \(N\) be the number of trading days in the test period, and let \(K_t\) be the number of stocks with predictions on day \(t\). For each day \(t\), the model produces a vector of predicted returns \(\hat{\mathbf{y}}_t \in \mathbb{R}^{K_t}\) and we observe actual returns \(\mathbf{y}_t \in \mathbb{R}^{K_t}\).

**IC (Information Coefficient).** The Pearson cross-sectional correlation between predicted and actual returns, averaged over all test days:

\[
IC = \frac{1}{N} \sum_{t=1}^{N} \frac{\sum_{i=1}^{K_t} (\hat{y}_{t,i} - \bar{\hat{y}}_t)(y_{t,i} - \bar{y}_t)}{\sqrt{\sum_i (\hat{y}_{t,i} - \bar{\hat{y}}_t)^2 \sum_i (y_{t,i} - \bar{y}_t)^2}}
\]

Intuitively: "when the model says stock A will outperform stock B, how often is the ordering correct?" An IC of 0 means random guessing; 0.10 is modest; 0.30 is strong. The mean ± std reported is across the 5 random seeds, not across days.

**ICIR (IC Information Ratio).** The mean of daily IC values divided by their standard deviation over time:

\[
ICIR = \frac{\text{mean}(IC_1, \ldots, IC_N)}{\text{std}(IC_1, \ldots, IC_N)}
\]

This measures *consistency across time*. A model with IC = 0.25 ± 0.05 day-to-day (ICIR = 5.0) is far more reliable than one with IC = 0.25 ± 0.20 (ICIR = 1.25). The first model is reliably right; the second is sometimes right and sometimes wrong.

**Rank IC (RIC).** Same structure as IC, but replaces Pearson \(r\) with Spearman's \(\rho\) (rank correlation). Concretely: on each day, replace each stock's predicted and actual returns with their ranks (1 = best, \(K_t\) = worst), then compute the Pearson correlation of those ranks. This eliminates sensitivity to outliers—a single stock with an extreme ±10% return will not dominate.

\[
RIC = \frac{1}{N} \sum_{t=1}^{N} \rho(\text{rank}(\hat{\mathbf{y}}_t),\; \text{rank}(\mathbf{y}_t))
\]

**Rank ICIR.** The consistency of Rank IC:

\[
RICIR = \frac{\text{mean}(RIC_1, \ldots, RIC_N)}{\text{std}(RIC_1, \ldots, RIC_N)}
\]

**AR (Annualized Return).** This is the actual portfolio return an investor would earn by following the model's recommendations. The procedure, executed at each trading day's close:

1. **Rank.** Sort all 100 stocks by the model's predicted 5-day forward return, from highest to lowest.
2. **Select.** Take the top \(K = 30\) stocks. The choice of 30 reflects a concentrated but diversified portfolio—enough stocks to reduce idiosyncratic risk, few enough that the model's ranking skill matters.
3. **Allocate.** Invest equal capital in each of the 30 stocks at the next day's closing price. (We use next-day close because today's close has already passed when the prediction is made; the earliest executable price is tomorrow's close.)
4. **Hold.** Each purchased stock is held for exactly 5 trading days, then sold at that day's closing price.
5. **Deduct costs.** Each trade (buy and sell) incurs a 0.2% transaction cost (20 basis points), modeling brokerage fees, bid-ask spread, and market impact for a moderate-sized order in the Vietnamese market.
6. **Repeat daily.** Every day, a new cohort of 30 stocks is purchased. The portfolio at any moment holds 5 overlapping daily cohorts (days \(t-4, t-3, t-2, t-1, t\)), each containing 30 stocks. This overlapping structure smooths returns and reduces day-to-day volatility.

The daily portfolio return \(r_t^{\text{portfolio}}\) is the equal-weighted average return of all currently held positions, net of the transaction costs incurred that day. The annualized return compounds these daily returns:

\[
AR = \left( \prod_{t=1}^{N} (1 + r_t^{\text{portfolio}}) \right)^{252 / N} - 1
\]

where \(N\) is the number of trading days in the test period and 252 is the approximate number of Vietnamese trading days per year. For context: if AR = 0.05, the strategy earns roughly 5% per year after costs—comparable to a fixed-income instrument but with equity-like volatility. If AR = 0.10, the strategy is generating equity-like returns with potential alpha above the market. The VN100 index itself returned approximately 8–12% annualized over 2024–2025, so an AR above ~0.10 represents genuine outperformance.

**IR (Information Ratio).** AR tells you *how much* you earn; IR tells you *how smoothly* you earn it. Formally, IR is the annualized mean of daily portfolio returns divided by their annualized standard deviation:

\[
IR = \frac{\bar{r} \times 252}{\sigma_r \times \sqrt{252}} = \frac{\bar{r}}{\sigma_r} \times \sqrt{252}
\]

where \(\bar{r}\) is the mean daily portfolio return and \(\sigma_r\) is its standard deviation. The \(\sqrt{252}\) factor annualizes the ratio.

Intuitively, IR measures the *consistency* of the strategy's daily profits. Consider two strategies that both earn AR = 0.07 (7% per year):

- Strategy A gains roughly +0.028% every single day, like clockwork. \(\sigma_r\) is tiny → IR is high (~5.0).
- Strategy B gains +0.5% on some days and loses −0.4% on others, averaging out to +0.028%. \(\sigma_r\) is large → IR is low (~0.5).

Strategy A is far more valuable: you can leverage it safely, it survives drawdowns, and it inspires confidence during losing streaks. Strategy B might earn the same AR but the ride is stomach-churning.

Benchmark values: IR > 1.0 is respectable (you earn more than the noise); IR > 2.0 is strong (a institutional-quality strategy); IR > 3.0 is excellent (world-class, sustainable alpha). The IR is closely related to the Sharpe ratio but differs in that it measures *active* return relative to the strategy's own volatility, not relative to a risk-free rate.

**Why both metrics matter together.** AR without IR is dangerous: high returns achieved through high volatility can wipe out an investor during a drawdown. IR without AR is unsatisfying: smooth returns at 1% per year don't justify the operational complexity of running a quantitative strategy. The ideal model has both—and as §5.3 will show, no single architecture maximizes both simultaneously.

---

## 4. What the Data Looks Like

Before discussing model results, we need to understand the data itself. If one dataset is fundamentally harder to predict than another, that matters.

### 4.1 The Label Distribution: What Are We Trying to Predict?

The label is the 5-day forward return for each stock. Let \(y\) denote the label values across all training samples. We characterize its distribution using the following metrics:

\[
\begin{aligned}
\text{Mean: } & \mu = \frac{1}{n}\sum_{i=1}^{n} y_i \quad\text{(average weekly return)} \\
\text{Standard deviation: } & \sigma = \sqrt{\frac{1}{n}\sum_{i=1}^{n} (y_i - \mu)^2} \quad\text{(typical weekly move)} \\
\text{Skewness: } & \gamma_1 = \frac{1}{n}\sum_{i=1}^{n} \left(\frac{y_i - \mu}{\sigma}\right)^3 \quad\text{(asymmetry: + = right tail, − = left tail)} \\
\text{Kurtosis: } & \kappa = \frac{1}{n}\sum_{i=1}^{n} \left(\frac{y_i - \mu}{\sigma}\right)^4 \quad\text{(tail weight; normal distribution = 3)} \\
\text{Tail ratio: } & \frac{P_{99} - P_1}{P_{95} - P_5} \quad\text{(how much fatter are the extreme 2% tails than the middle 90%?)} \\
\text{Signal-to-noise: } & \frac{|\mu|}{\sigma} \quad\text{(absolute mean relative to typical variation)}
\end{aligned}
\]

where \(P_k\) denotes the \(k\)-th percentile of the label distribution.

Here is the distribution of labels in each training set:

| Property | 3-Year | 5-Year | 10-Year | Formula |
|----------|--------|--------|---------|---------|
| Training samples (\(n\)) | 23,481 | 94,807 | 144,866 | — |
| Label mean (\(\mu\)) | **−0.0061** | +0.0045 | +0.0034 | \(\frac{1}{n}\sum y_i\) |
| Label std (\(\sigma\)) | **0.0693** | 0.0575 | 0.0535 | \(\sqrt{\frac{1}{n}\sum (y_i-\mu)^2}\) |
| Skewness (\(\gamma_1\)) | **−0.11** | +0.14 | **+0.46** | \(\mathbb{E}[((y-\mu)/\sigma)^3]\) |
| Kurtosis (\(\kappa\)) | **4.84** | 6.76 | **9.27** | \(\mathbb{E}[((y-\mu)/\sigma)^4]\) |
| Tail ratio | **1.62** | 1.74 | **1.80** | \((P_{99}-P_1)/(P_{95}-P_5)\) |
| % positive labels | **45.3%** | 52.0% | 49.0% | \(\frac{1}{n}\sum \mathbf{1}[y_i > 0]\) |
| Signal-to-noise (\(|\mu|/\sigma\)) | **0.088** | 0.079 | 0.064 | \(|\mu|/\sigma\) |

**The 3-year dataset is dominated by a bear market.** The mean label of −0.0061 means that across all 23,481 stock-weeks in the 2022–2023 training window, the average stock *lost* 0.61% per week. Annualized (multiply by 52), that is roughly −27% per year. Only 45.3% of labels are positive—the model spends most of its training seeing stocks go down. The negative skew (−0.11) means that when large moves occur, they tend to be crashes rather than rallies: the left tail is fatter than the right tail. A model trained on this data learns that "stocks go down" is the default state; it receives weak reinforcement for predicting positive returns. When tested on 2024–2025 (a recovery period where stocks mostly rise), this learned pessimism translates into missed opportunities—the model systematically underranks stocks that go on to rally.

**The 10-year dataset provides the healthiest training distribution.** With 144,866 samples spanning 2016–2022, the mean is modestly positive (+0.0034, or roughly +19% annualized). Skewness is positive (+0.46)—large positive moves outnumber large negative moves, reflecting the presence of strong bull markets (2017, 2020–2021) in the training data. Kurtosis of 9.27 is **three times the normal distribution's value of 3.0**, meaning extreme ±5%+ weekly moves occur far more often than a Gaussian model would predict. The model sees abundant tail events—COVID crash (−30% in Q1 2020), V-shaped recovery (+40% in Q2–Q3 2020), 2022 rate-hike selloff (−35%)—and must learn to predict through all of them.

**The signal-to-noise ratio is highest for 3-year data—but misleadingly so.** The value 0.088 is driven by the large *negative* mean (−0.0061), not by genuine predictability. A naive model that always predicts "stocks go down" achieves decent-looking metrics during a bear market but fails catastrophically when the regime changes. Genuine predictability comes from *cross-sectional variation* (differences *between* stocks), not from a strong directional bias shared by all stocks.

### 4.2 Regime Diversity: How Often Does the Market Change Its Mind?

We quantify regime diversity using three technical frameworks applied to the equal-weighted VN100 index proxy. Let \(P_t\) be the index level on day \(t\) and \(R_t = P_t / P_{t-1} - 1\) be the daily return.

**Trend regime** — defined by the relationship between two moving averages:

\[
MA50_t = \frac{1}{50}\sum_{k=0}^{49} P_{t-k}, \quad MA200_t = \frac{1}{200}\sum_{k=0}^{199} P_{t-k}
\]

The regime is 1 (uptrend) when \(MA50_t > MA200_t\) (a "golden cross" configuration) and 0 (downtrend) otherwise. A regime *transition* occurs when this binary state flips.

**Volatility regime** — based on rolling 20-day annualized volatility:

\[
\sigma_{20}(t) = \sqrt{252 \cdot \frac{1}{20}\sum_{k=0}^{19} (R_{t-k} - \bar{R}_{20})^2}
\]

where \(\bar{R}_{20}\) is the 20-day mean return. Days are assigned to quartiles of the historical \(\sigma_{20}\) distribution: low volatility (Q1), medium-low (Q2), medium-high (Q3), or high volatility (Q4). A transition occurs when a day's quartile differs from the previous day's.

**Drawdown regime** — the percentage decline from the all-time high:

\[
DD_t = \frac{P_t - \max_{k \leq t} P_k}{\max_{k \leq t} P_k}
\]

\(DD_t\) is always ≤ 0. We bucket into four severity levels: normal (\(DD_t > -5\%\)), correction (\(-15\% < DD_t \leq -5\%\)), bear market (\(-30\% < DD_t \leq -15\%\)), and crash (\(DD_t \leq -30\%\)).

For each framework, we count how many times the regime *changes* during the training period. More transitions = more diverse market conditions = harder for the model to memorize any single regime.

**Regime entropy** quantifies the *balance* of time spent across states. For a discrete probability distribution over \(K\) regime states with proportions \(p_1, \ldots, p_K\):

\[
H = -\sum_{k=1}^{K} p_k \log_2(p_k)
\]

Maximum entropy (all states equally frequent) for \(K=4\) is \(\log_2(4) = 2.0\). Minimum entropy (always in one state) is 0. Entropy captures distribution *balance*; transition count captures *change frequency*. Both matter: a dataset can have high entropy (balanced across states) but low transitions (long uninterrupted periods in each state), or vice versa.

#### 4.2.1 Regime Transition Counts

| Metric | 3-Year | 5-Year | 10-Year | 10Y/5Y |
|--------|--------|--------|---------|--------|
| Trend transitions (MA crossovers) | 3 | 3 | **8** | 2.7× |
| Volatility regime transitions | 52 | 82 | **176** | 2.1× |
| Drawdown regime transitions | 32 | 54 | **85** | 1.6× |
| Combined (Trend × Vol) transitions | 25 | 39 | **75** | 1.9× |
| Volatility-of-volatility | 0.419 | 0.454 | **0.491** | 1.08× |
| Yearly return dispersion (σ) | 57% | 69% | **77%** | 1.12× |

The 10-year dataset has **2–3 times more regime transitions** than the 5-year dataset across every framework. This is not just "more data"—it's *qualitatively different* data. A model trained on 3 years sees one sustained bear market with brief interruptions. A model trained on 10 years sees the market flip from bull to bear to crisis to recovery multiple times. Each transition forces the model to learn features that work *across* changes, not features that work *within* one persistent state.

**Volatility-of-volatility** (how much volatility itself fluctuates) increases with horizon length. This matters because the LGBM tree gate splits on volatility features—when volatility-of-volatility is higher, there are more distinct volatility *states* for the tree to discover.

#### 4.2.2 Where Does Training Time Go?

| Regime State | 3-Year | 5-Year | 10-Year |
|-------------|--------|--------|---------|
| Trend: % time in uptrend (MA50 > MA200) | 32% | 62% | 67% |
| Drawdown: % time normal (<5% from highs) | **10%** | 39% | **44%** |
| Drawdown: % time in bear market (15–30% DD) | **50%** | 28% | **25%** |
| Drawdown: % time in crash (>30% DD) | **30%** | 15% | **5%** |
| Combined: UP + LOW VOL (easiest to predict) | 16% | 36% | 39% |
| Combined: DOWN + HIGH VOL (hardest to predict) | **34%** | 23% | 22% |
| ADX: % time trending (vs ranging) | 62% | 62% | 59% |

The 3-year training window is **80% bear/crash drawdown and 68% downtrend**. The model rarely sees a normal, rising market. It overfits to bear-market dynamics: features that signal "sell" are reinforced, features that signal "buy" are rarely rewarded. When tested on 2024–2025 (which includes recovery periods), the model's bear-market habits hurt performance.

The 10-year dataset has the most balanced distribution: 44% normal, 25% bear, only 5% crash. It also has the highest fraction of UP+LOWVOL days (39%)—the "easy" regime where trend-following works well. But it still has enough DOWN+HIGHVOL days (22%) to learn defensive behavior.

#### 4.2.3 Technical Indicator Extremes

Beyond the regime frameworks above, we count how often three classical technical indicators reach extreme levels during each training period. These counts measure how many "edge case" examples the model encounters during training.

**RSI (Relative Strength Index)** — a bounded momentum oscillator (0–100) that measures the speed and magnitude of recent price changes. Let \(\Delta_t = P_t - P_{t-1}\) be the price change. Define the average gain and average loss over a 14-day window:

\[
\text{AvgGain}_{14}(t) = \frac{1}{14}\sum_{k=0}^{13} \max(\Delta_{t-k}, 0), \quad \text{AvgLoss}_{14}(t) = \frac{1}{14}\sum_{k=0}^{13} \max(-\Delta_{t-k}, 0)
\]

\[
RSI_t = 100 - \frac{100}{1 + \frac{\text{AvgGain}_{14}(t)}{\text{AvgLoss}_{14}(t) + \varepsilon}}
\]

where \(\varepsilon = 10^{-12}\) prevents division by zero. RSI > 70 is considered *overbought* (the asset has risen too fast and may reverse downward); RSI < 30 is *oversold* (it has fallen too fast and may bounce). These extremes are mean-reversion signals: they mark moments when momentum has potentially exhausted itself.

**Bollinger Bands** — volatility envelopes around a moving average. Let \(\text{MA20}_t\) be the 20-day simple moving average and \(\sigma_{20}(t)\) the 20-day standard deviation of price:

\[
\text{UpperBand}_t = \text{MA20}_t + 2 \cdot \sigma_{20}(t), \quad \text{LowerBand}_t = \text{MA20}_t - 2 \cdot \sigma_{20}(t)
\]

The band *width* is \(4 \cdot \sigma_{20}(t)\). A *Bollinger Band breakout* occurs when the width expands to more than 1.5× its value 20 days ago, signaling a volatility expansion event. These are transition moments: the market is shifting from a low-volatility to a high-volatility regime (or vice versa).

**ADX (Average Directional Index)** — a trend strength indicator (0–100) derived from directional movement. Let \(+DM_t = \max(P_t - P_{t-1}, 0)\) (upward movement) and \(-DM_t = \max(P_{t-1} - P_t, 0)\) (downward movement). The true range is \(TR_t = \max(P_t - P_{t-1}, |P_t - P_{t-1}|, |P_{t-1} - P_t|)\) — actually the max of high-low, high-prevclose, low-prevclose. For our index proxy we use \(|P_t - P_{t-1}|\). Then:

\[
+DI_{14} = 100 \cdot \frac{\text{EMA}_{14}(+DM)}{\text{EMA}_{14}(TR)}, \quad -DI_{14} = 100 \cdot \frac{\text{EMA}_{14}(-DM)}{\text{EMA}_{14}(TR)}
\]

\[
DX_t = 100 \cdot \frac{|+DI_{14} - -DI_{14}|}{+DI_{14} + -DI_{14} + \varepsilon}, \quad ADX_t = \text{EMA}_{14}(DX_t)
\]

ADX > 25 indicates a *trending* market (directional movement dominates); ADX < 25 indicates a *ranging* market (price moves sideways). An ADX switch occurs when the index crosses the 25 threshold. These are regime-change signals: the market transitions from directionless chop to a sustained trend (or vice versa).

| Extreme Event | 3-Year | 5-Year | 10-Year | 10Y/3Y Ratio |
|--------------|--------|--------|---------|--------------|
| RSI overbought days (>70) | 53 | 134 | **467** | 8.8× |
| RSI oversold days (<30) | 53 | 126 | **170** | 3.2× |
| Bollinger Band breakout days | 106 | 235 | **411** | 3.9× |
| ADX trending/ranging switches | 52 | 150 | **254** | 4.9× |

The 10-year dataset provides **3–9 times more technical extreme events** than the 3-year dataset. An RSI overbought reading (>70) signals that the market has risen rapidly and may be due for a pullback—a mean-reversion signal. A model that has seen 467 overbought days during training learns that roughly half precede corrections and half don't; it develops calibrated uncertainty. A model that has seen only 53 overbought days has no statistical basis for nuanced judgment—it either ignores the signal entirely or overfits to the few examples it saw. Similarly, 411 Bollinger Band breakouts versus 106 means the 10-year model has observed nearly 4× more volatility regime transitions, and 254 ADX trending/ranging switches versus 52 gives it 5× more examples of the market shifting between directional and non-directional behavior.

### 4.3 Cross-Sectional Dispersion: Are Stocks Different From Each Other?

If all 100 stocks move together (high correlation), stock picking is impossible—you can't rank what moves in lockstep. Cross-sectional dispersion measures how much stock returns *differ* from each other on a given day. For each trading day \(t\) with \(K_t\) stocks, we compute the standard deviation of labels across stocks:

\[
CS_t = \sqrt{\frac{1}{K_t}\sum_{i=1}^{K_t} (y_{t,i} - \bar{y}_t)^2}
\]

where \(\bar{y}_t\) is the mean label across all stocks on day \(t\). High \(CS_t\) means stocks are moving in different directions—there is room for stock picking to add value. Low \(CS_t\) means all stocks move together—even a perfect ranking produces little differentiation.

We also measure the P95−P5 spread: the gap between the 95th and 5th percentile stock on each day, capturing the return difference between a top-decile and bottom-decile pick.

| Metric | 3-Year | 5-Year | 10-Year |
|--------|--------|--------|---------|
| Trading days in training | 240 | 992 | 1,736 |
| Mean cross-sectional std (\(CS_t\)) | **0.0510** | 0.0438 | 0.0424 |
| Mean P95−P5 spread | **0.159** | 0.134 | 0.126 |
| % days with P95−P5 > 5% | 100.0% | 100.0% | 100.0% |

On **every single trading day** across all three datasets, the gap between a top-decile and bottom-decile stock exceeds 5%. Stock picking is always possible—there are always meaningful winners and losers.

The 3-year dataset shows the *highest* cross-sectional dispersion (0.051)—a direct consequence of the 2022 bear market: during crises, capital flees weak names for strong ones, widening the return gap between stocks. However, this higher dispersion does not translate to better model performance, because it is driven by *directional* moves (everything going down at different speeds) rather than by the *relative reversal and rotation patterns* that the model must predict. The 10-year dataset has the lowest average dispersion (0.042) but still produces 100% of days with actionable spreads. Its advantage is not more per-day opportunity, but rather *more days* (1,736 vs 240) spanning *diverse regimes*—the model sees dispersion in bull markets (driven by momentum leaders), in bear markets (driven by flight-to-quality), and in sideways markets (driven by sector rotation), learning to predict ranking in all three environments.

### 4.4 Feature-Label Stability: Do Features Mean the Same Thing Over Time?

This is perhaps the most revealing analysis. Let \(f_j\) be the \(j\)-th feature (of 158) and \(y\) be the label. We split the training set at its chronological midpoint into an early period \(E\) (first half) and late period \(L\) (second half). For each period, we compute the Pearson correlation between each feature and the label:

\[
c_j^E = \text{corr}(f_j^E, y^E), \quad c_j^L = \text{corr}(f_j^L, y^L), \quad j = 1,\ldots,158
\]

We then measure three forms of stability:

\[
\begin{aligned}
\text{Correlation stability: } & r_{\text{stab}} = \text{corr}(\mathbf{c}^E, \mathbf{c}^L) \quad\text{(Pearson r between the two 158-d correlation vectors)} \\
\text{Sign flip fraction: } & \frac{1}{158}\sum_{j=1}^{158} \mathbf{1}[c_j^E \cdot c_j^L < 0] \quad\text{(fraction of features that reverse direction)} \\
\text{Rank stability: } & \rho(\text{rank}(|\mathbf{c}^E|),\; \text{rank}(|\mathbf{c}^L|)) \quad\text{(Spearman ρ of feature importance ranks)}
\end{aligned}
\]

| Property | 3-Year | 5-Year | 10-Year | Formula |
|----------|--------|--------|---------|---------|
| Feature-label correlation stability | **0.919** | 0.806 | **0.915** | \(\text{corr}(\mathbf{c}^E, \mathbf{c}^L)\) |
| Sign flip fraction | **0.120** | 0.196 | **0.108** | \(\frac{1}{158}\sum \mathbf{1}[c_j^E \cdot c_j^L < 0]\) |
| Feature importance rank stability | **0.029** | −0.036 | −0.163 | \(\rho(\text{rank}(|\mathbf{c}^E|), \text{rank}(|\mathbf{c}^L|))\) |

**Caveat for 3-year data:** With only 23,481 total samples split into halves (11,740 per half) across 158 features, each per-feature correlation is estimated from roughly 74 observations. This produces wide confidence intervals — the apparently high stability (0.919) may partially reflect noisy estimates regressing toward zero in both halves rather than genuine stability. The 5-year and 10-year estimates, with 4× and 6× more samples per half, are more reliable.

**Interpretation:** On the 10-year dataset, the vector of feature-label correlations is 91.5% consistent between the first and second halves of training. If MA20 is positively correlated with returns in 2016–2019 (\(c_{MA20}^E > 0\)), it is still positively correlated in 2020–2022 (\(c_{MA20}^L > 0\)). On the 5-year dataset, stability drops to 80.6%. The sign flip fraction nearly doubles from 10.8% to 19.6%—**one in five features reverses its directional relationship** between the early and late training period. For example, a momentum feature that was bullish in 2020–2021 (\(c > 0\)) becomes bearish in 2022 (\(c < 0\)) as the market transitions from bull to bear.

The negative rank stability in both datasets (features most important early are slightly *anti-correlated* with features most important late) suggests that the optimal feature set shifts over time. This is the core challenge that attention gating addresses: when which features deserve attention changes, the attention mechanism needs market context to redirect focus appropriately.

### 4.5 Temporal Autocorrelation: Persistence of Returns

For each stock, let \(y_1, y_2, \ldots, y_T\) be its sequence of labels (5-day forward returns) over time. The lag-1 autocorrelation measures whether consecutive labels are related:

\[
\rho_1 = \frac{\sum_{t=1}^{T-1} (y_t - \bar{y})(y_{t+1} - \bar{y})}{\sum_{t=1}^{T} (y_t - \bar{y})^2}
\]

We compute \(\rho_1\) for each of the 20 most liquid stocks and report the mean. Values near +1 indicate strong momentum (positive returns follow positive returns); values near −1 indicate strong mean-reversion; values near 0 indicate no persistence.

| Dataset | Mean Lag-1 Autocorrelation | Range (across 20 stocks) |
|---------|---------------------------|--------------------------|
| 3-Year | 0.772 | [0.688, 0.847] |
| 5-Year | 0.767 | [0.698, 0.820] |
| 10-Year | 0.780 | [0.746, 0.871] |

The lag-1 autocorrelation is approximately **0.77** across all three datasets, with remarkably tight ranges—every stock exhibits similar short-term momentum regardless of the training window. This structural property is invariant to horizon length: the Vietnamese market persistently trends in the short term. This means that if a stock's 5-day forward return was positive in the previous period, it has a ~78% tendency to remain positive in the current period. The Vietnamese market exhibits pronounced short-term persistence.

This is good news for any model: even a simple trend-following rule achieves positive predictive accuracy, which is why even the worst model (LGBM-LeafInput) achieves positive IC. It also explains BiLSTM's reasonable performance—a bidirectional LSTM over 8-day sequences can easily learn "recent direction predicts near-term direction" from strongly autocorrelated data.

### 4.6 Market Feature Information Content

The 63 market features are not equally informative. Let \(g_j\) be the \(j\)-th gate (market) feature and \(y\) be the label. We compute the Pearson correlation between each market feature and the label across all training samples:

\[
r_j = \frac{\sum_{i=1}^{n} (g_{i,j} - \bar{g}_j)(y_i - \bar{y})}{\sqrt{\sum_i (g_{i,j} - \bar{g}_j)^2 \sum_i (y_i - \bar{y})^2}}, \quad j = 1,\ldots,63
\]

The informativeness of market features varies with horizon. Below are the top-5 most label-correlated market features for each training set, ranked by absolute Pearson \(r\).

**10-Year (2016–2022):**

| Market Feature | \(r\) with Label | Interpretation |
|---------------|-----------------|----------------|
| VN100_RET_STD_10 | **−0.040** | Higher recent return volatility → lower future returns |
| VN100_VOL_STD_20 | **−0.039** | More volatile volume → lower future returns |
| VN100_VOL_STD_30 | −0.039 | Consistent across longer windows |
| VN100_VOL_MEAN_30 | −0.039 | Higher average volume → negative signal |
| VN100_VOL_STD_60 | −0.039 | The effect persists at 60-day scale |

Mean absolute gate-label correlation: 0.011. All significant correlations are negative and volatility-driven—the "leverage effect."

**5-Year (2020–2023):**

| Market Feature | \(r\) with Label | Interpretation |
|---------------|-----------------|----------------|
| VN100_RET_MEAN_60 | **−0.033** | Longer-term market momentum → weakly negative |
| VN100_RET_STD_20 | **−0.032** | Return volatility → lower future returns |
| VN100_RET_MEAN_30 | +0.031 | Medium-term momentum → weakly positive |
| VN100_RET_STD_30 | −0.028 | Return volatility effect weaker at 30d |
| VN100_VOL_STD_60 | −0.028 | Volume-of-volatility → negative signal |

Mean absolute gate-label correlation: 0.008. Correlations are weaker and noisier than 10Y—the 5-year window provides fewer samples for stable market-feature relationships. The signal is present but attenuated.

**3-Year (2022–2023):**

| Market Feature | \(r\) with Label | Interpretation |
|---------------|-----------------|----------------|
| VN100_RET_MEAN_60 | **−0.066** | Strong bear-market momentum: falling market → continued falls |
| VN100_RET_STD_60 | +0.052 | Return-of-volatility: turbulent recovery rallies |
| VN100_RET_STD_5 | +0.048 | Short-term vol spikes → brief positive reversals |
| VN100_RET_MEAN_10 | +0.046 | Short-term mean-reversion in a bear market |
| VN100_VOL_STD_60 | +0.044 | Volume volatility → positive (atypical, bear-market specific) |

Mean absolute gate-label correlation: 0.013 — the *highest* of the three horizons, but for the wrong reasons. The 3-year window is dominated by the 2022 bear market, where falling markets create spuriously strong correlations. The sign pattern is mixed (3 negative, 2 positive among the top 5) and several relationships are *reversed* from their 10Y counterparts (e.g., VN100_VOL_STD_60 is +0.044 on 3Y but −0.039 on 10Y). These correlations are regime-specific, not structural—a model that learns them will fail when the regime changes.

**Cross-horizon pattern.** The 10-year dataset exhibits the most *consistent* market signal: all top correlations are negative and volatility-driven, matching the well-known leverage effect. The 5-year dataset shows weaker, noisier versions of the same pattern. The 3-year dataset shows *reversed and mixed* signals—bear-market artifacts that would mislead a model tested on a recovery period. This is another mechanism by which longer horizons help: they wash out regime-specific noise and surface the structural relationships that generalize.

The LGBM tree gate can exploit these structural relationships by learning threshold rules: "if VN100_RET_STD_10 exceeds a critical level, route samples to the defensive expert that weights low-volatility stock features more heavily." This mechanism requires both *informative* market features and *enough regime diversity* to distinguish structural from spurious correlations—conditions met only on the 10-year dataset.

---

## 5. Results

### 5.1 Full Results Tables

**3-Year Dataset (2022–2023 training, 2024–2025 testing)**

| Model | IC | ICIR | Rank IC | Rank ICIR | AR | IR |
|-------|----|------|---------|-----------|-----|-----|
| Base MASTER | 0.1575 ± 0.0155 | 1.0883 ± 0.1641 | 0.1646 ± 0.0128 | 1.1868 ± 0.1759 | 0.0512 ± 0.0141 | 2.6810 ± 0.7030 |
| MoE | 0.1735 ± 0.0274 | 1.2104 ± 0.2340 | 0.1712 ± 0.0263 | 1.2708 ± 0.2643 | 0.0609 ± 0.0272 | 2.7115 ± 0.8003 |
| LGBM-Gate | 0.1863 ± 0.0119 | 1.3355 ± 0.0833 | 0.1844 ± 0.0106 | 1.3660 ± 0.1072 | 0.0662 ± 0.0076 | 3.5570 ± 0.3685 |
| BiLSTM | 0.1680 ± 0.0198 | 1.1673 ± 0.1575 | 0.1710 ± 0.0169 | 1.2639 ± 0.1600 | 0.0440 ± 0.0121 | 2.6735 ± 0.8838 |
| LGBM-LeafInput | 0.1091 ± 0.0211 | 0.8173 ± 0.1705 | 0.1107 ± 0.0220 | 0.8454 ± 0.1666 | 0.0425 ± 0.0186 | 2.1604 ± 1.0086 |
| Cross-Attn Gate | 0.2699 ± 0.0143 | 0.9163 ± 0.0563 | 0.2576 ± 0.0142 | 0.9150 ± 0.0596 | 0.0550 ± 0.0162 | 2.6017 ± 0.7865 |
| MoE+LGBM Gate | 0.2847 ± 0.0209 | 0.9654 ± 0.0826 | 0.2758 ± 0.0186 | 0.9735 ± 0.0677 | 0.0555 ± 0.0164 | 2.4840 ± 0.9035 |

**5-Year Dataset (2020–2023 training, 2024–2025 testing)**

| Model | IC | ICIR | Rank IC | Rank ICIR | AR | IR |
|-------|----|------|---------|-----------|-----|-----|
| Base MASTER | 0.1854 ± 0.0050 | 1.1997 ± 0.0358 | 0.1923 ± 0.0052 | 1.2823 ± 0.0239 | 0.0401 ± 0.0101 | 1.7289 ± 0.5114 |
| MoE | 0.2094 ± 0.0110 | 1.3573 ± 0.1285 | 0.2126 ± 0.0116 | 1.4355 ± 0.1448 | 0.0500 ± 0.0145 | 2.2452 ± 0.5620 |
| LGBM-Gate | 0.1863 ± 0.0179 | 1.2042 ± 0.1218 | 0.1906 ± 0.0161 | 1.2831 ± 0.1195 | 0.0394 ± 0.0120 | 1.8187 ± 0.6262 |
| BiLSTM | 0.1935 ± 0.0107 | 1.2273 ± 0.0378 | 0.1990 ± 0.0096 | 1.2996 ± 0.0464 | 0.0311 ± 0.0174 | 1.3112 ± 0.8057 |
| LGBM-LeafInput | 0.1413 ± 0.0283 | 0.9670 ± 0.1789 | 0.1468 ± 0.0287 | 1.0306 ± 0.1795 | 0.0378 ± 0.0225 | 1.8694 ± 1.0156 |
| Cross-Attn Gate | 0.1979 ± 0.0094 | 1.2805 ± 0.0725 | 0.2017 ± 0.0091 | 1.3473 ± 0.0816 | 0.0377 ± 0.0122 | 1.6793 ± 0.6730 |
| MoE+LGBM Gate | **0.2293 ± 0.0164** | **1.4988 ± 0.1067** | **0.2316 ± 0.0118** | **1.5772 ± 0.0906** | **0.0537 ± 0.0222** | **2.4003 ± 0.9714** |

**10-Year Dataset (2016–2022 training, 2024–2025 testing)**

| Model | IC | ICIR | Rank IC | Rank ICIR | AR | IR |
|-------|----|------|---------|-----------|-----|-----|
| Base MASTER | 0.2356 ± 0.0201 | 1.6689 ± 0.1895 | 0.2387 ± 0.0186 | 1.6785 ± 0.1585 | 0.0728 ± 0.0035 | **3.4041 ± 0.1510** |
| MoE | 0.2524 ± 0.0208 | 1.8083 ± 0.1975 | 0.2563 ± 0.0171 | 1.8351 ± 0.1671 | 0.0676 ± 0.0095 | 2.9933 ± 0.5037 |
| LGBM-Gate | 0.2501 ± 0.0237 | 1.8238 ± 0.2364 | 0.2523 ± 0.0224 | 1.8035 ± 0.1937 | 0.0742 ± 0.0107 | 3.3171 ± 0.5712 |
| BiLSTM | 0.2352 ± 0.0132 | 1.6381 ± 0.1174 | 0.2362 ± 0.0146 | 1.6434 ± 0.1170 | 0.0609 ± 0.0246 | 3.0408 ± 1.1999 |
| LGBM-LeafInput | 0.1771 ± 0.0205 | 1.1872 ± 0.1287 | 0.1798 ± 0.0211 | 1.2304 ± 0.1390 | 0.0601 ± 0.0068 | 2.8197 ± 0.2998 |
| Cross-Attn Gate | 0.2354 ± 0.0190 | 1.6311 ± 0.1564 | 0.2367 ± 0.0163 | 1.6471 ± 0.1397 | **0.0763 ± 0.0069** | 3.2153 ± 0.1935 |
| MoE+LGBM Gate | **0.2984 ± 0.0245** | **2.3821 ± 0.2349** | **0.2984 ± 0.0237** | **2.3301 ± 0.2168** | 0.0712 ± 0.0107 | 3.3644 ± 0.6312 |

### 5.2 Cross-Horizon Scaling: Who Benefits Most From More History?

All models improve as training horizon increases. The magnitude of improvement reveals which architectures best exploit additional data:

| Model | 3Y IC | 5Y IC | 10Y IC | 3Y→10Y Gain | Notes |
|-------|-------|-------|--------|-------------|-------|
| Base MASTER | 0.1575 | 0.1854 | 0.2356 | +49.6% | Steady, linear scaling |
| MoE | 0.1735 | 0.2094 | 0.2524 | +45.5% | Steady, linear scaling |
| LGBM-Gate | 0.1863 | 0.1863 | 0.2501 | +34.2% | Flat from 3Y→5Y, jumps at 10Y |
| BiLSTM | 0.1680 | 0.1935 | 0.2352 | +40.0% | Steady scaling |
| LGBM-LeafInput | 0.1091 | 0.1413 | 0.1771 | +62.3% | Largest % gain (from lowest base) |
| Cross-Attn Gate | **0.2699** | 0.1979 | 0.2354 | **−12.8%** | Anomalous: *best* on 3Y, worst scaling |
| MoE+LGBM Gate | **0.2847** | 0.2293 | 0.2984 | +4.8% | Dominates at all horizons |

**Cross-Attn Gate exhibits a striking reversal**: it is the best model on 3-year data (IC = 0.27) but *worsens* on 5-year data (0.20) before recovering partially on 10-year (0.24). This inverted-U pattern has a specific cause: on the 3-year dataset, where 80% of training occurs in a bear market, market conditions are unusually homogeneous. Cross-attention over a narrow, repetitive set of conditions can learn sharp but brittle attention patterns—it "memorizes" which stocks tend to co-move during a bear market. When tested on the more diverse 2024–2025 period, those memorized patterns fail. On the 10-year dataset, cross-attention is forced to generalize across bull, bear, and crisis regimes, learning more robust inter-stock relationships. Its ICIR tells the same story: 0.92 on 3Y (predictions are noisy despite high IC), improving to 1.63 on 10Y (predictions are both accurate *and* consistent).

**MoE+LGBM Gate dominates** at all horizons on IC-based metrics but shows diminishing returns in percentage terms (+4.8% from 5Y to 10Y). The architecture is already near its effective capacity on 5-year data; the 10-year data refines its regime splits rather than discovering entirely new ones.

### 5.3 AR and IR: From Predictions to Portfolio Returns

IC and RIC measure whether the model ranks stocks correctly. AR and IR measure whether those rankings make money. The relationship is not one-to-one: a model can rank well (high IC) but generate disappointing portfolio returns (low AR/IR) if its predictions lack *magnitude calibration* or if its ranking errors cluster in high-volatility periods. Conversely, a model with moderate IC can generate excellent IR if its errors are evenly distributed and its predictions are well-calibrated.

#### 5.3.1 Annualized Return (AR)

| Model | 3Y AR | 5Y AR | 10Y AR | 3Y→10Y Δ |
|-------|-------|-------|--------|-----------|
| Base MASTER | 0.0512 | 0.0401 | 0.0728 | +42% |
| MoE | 0.0609 | 0.0500 | 0.0676 | +11% |
| LGBM-Gate | **0.0662** | 0.0394 | 0.0742 | +12% |
| BiLSTM | 0.0440 | 0.0311 | 0.0609 | +38% |
| LGBM-LeafInput | 0.0425 | 0.0378 | 0.0601 | +41% |
| Cross-Attn Gate | 0.0550 | 0.0377 | **0.0763** | +39% |
| MoE+LGBM Gate | 0.0555 | **0.0537** | 0.0712 | +28% |

Three patterns stand out. First, **AR is highest on 10-year data for all models**, with the average improving from 0.054 (3Y) to 0.069 (10Y)—a 28% increase. More training data directly translates to higher portfolio returns.

Second, **the AR leader changes with horizon**: LGBM-Gate wins on 3Y (0.0662), MoE+LGBM Gate wins on 5Y (0.0537), and Cross-Attn Gate wins on 10Y (0.0763). No single architecture dominates AR across all horizons. The IC leader (MoE+LGBM Gate) is AR leader only on 5Y.

Third—and most revealing—**Cross-Attn Gate achieves the best AR on 10Y (0.0763) despite ranking only 5th on IC (0.2354)**. Its cross-attention mechanism produces predictions that are *directionally noisier than trees* (hence lower IC) but *better calibrated in magnitude* (hence higher AR). Cross-attention captures not just "which stock will outperform?" but also "by how much?"—the magnitude signal that IC discards but AR captures. This is a distinctive strength of attention-based interaction between market and stock features.

#### 5.3.2 Information Ratio (IR)

| Model | 3Y IR | 5Y IR | 10Y IR | IR StdDev (10Y) |
|-------|-------|-------|--------|-----------------|
| Base MASTER | 2.6810 | 1.7289 | **3.4041** | **±0.15** |
| MoE | 2.7115 | 2.2452 | 2.9933 | ±0.50 |
| LGBM-Gate | **3.5570** | 1.8187 | 3.3171 | ±0.57 |
| BiLSTM | 2.6735 | 1.3112 | 3.0408 | ±1.20 |
| LGBM-LeafInput | 2.1604 | 1.8694 | 2.8197 | ±0.30 |
| Cross-Attn Gate | 2.6017 | 1.6793 | 3.2153 | ±0.19 |
| MoE+LGBM Gate | 2.4840 | **2.4003** | 3.3644 | ±0.63 |

**The Base MASTER achieves the highest IR on 10-year data (3.40) with by far the lowest seed-to-seed variance (±0.15).** To understand the magnitude of this consistency advantage: a ±0.15 standard deviation on an IR of 3.40 means the 95% confidence interval across seeds is [3.10, 3.70]—the model reliably delivers excellent risk-adjusted returns. MoE+LGBM Gate, with IR = 3.36 ± 0.63, has a 95% interval of [2.10, 4.62]—it *might* be even better, but it might be substantially worse.

This variance difference has a structural cause. The base model's linear gate produces the same feature-weighting policy regardless of random seed: "high volatility → downweight momentum" is a simple rule that every seed discovers. The tree-based gates learn regime boundaries from data, and different seeds can discover slightly different split points, routing samples to different experts. When the splits are good, performance exceeds the base model. When they are slightly off, performance degrades. The *expected* performance of tree-based gates is higher; the *variance* of that performance is also higher.

**The 3Y IR values are suspiciously high.** LGBM-Gate achieves IR = 3.56 on 3Y—the highest IR in the entire study—despite training on only 23K samples in a concentrated bear market. This is almost certainly a backtest artifact: the 2022–2023 training data and 2024–2025 test data share a common recovery dynamic (stocks that fell most in 2022 bounced hardest in 2024), creating an unrealistically favorable backtest for models trained on that specific bear market. The 3Y IR numbers should be interpreted as *in-sample regime persistence*, not as genuine out-of-sample skill.

#### 5.3.3 The IC-to-IR Conversion Gap

We define the *conversion gap* as the rank difference between a model's IC rank and its IR rank. A positive gap means the model converts prediction accuracy into portfolio returns *more efficiently* than its IC would suggest:

| Model | IC Rank (10Y) | IR Rank (10Y) | Conversion Gap |
|-------|---------------|---------------|----------------|
| Base MASTER | 4th | **1st** | **+3** |
| Cross-Attn Gate | 5th | 5th | 0 |
| LGBM-Gate | 3rd | 3rd | 0 |
| MoE+LGBM Gate | 1st | 2nd | −1 |
| BiLSTM | 6th | 6th | 0 |
| LGBM-LeafInput | 7th | 7th | 0 |
| MoE | 2nd | 4th | −2 |

Base MASTER is the only model with a large positive conversion gap (+3 ranks). It punches above its weight: moderate IC translated into superior IR through low prediction volatility and gradual position changes. MoE shows the largest negative gap (−2): high IC but disappointing IR, suggesting that its expert-routing mechanism introduces prediction volatility that erodes risk-adjusted returns.

For a practitioner selecting a model, this table is the single most actionable result in the paper. MoE+LGBM Gate is the best *predictor*. Base MASTER is the best *portfolio constructor*. The choice depends on whether the objective is accurate rankings or smooth returns.

### 5.4 LGBM-LeafInput: Why Leaf Embeddings Fail

LGBM-LeafInput is the worst model across every metric and every horizon. Its IC on 3-year data (0.11) is barely above random. It also posts the worst AR (0.0425 on 3Y) and worst IR (2.16 on 3Y, 2.82 on 10Y). The architecture trains a LightGBM tree to predict returns from market features, then embeds the leaf index as a learned vector. The problem is **leaf instability**: the tree is retrained each epoch on a different random subset of data, producing different split boundaries and different leaf semantics. An embedding learned for "leaf #5" in epoch 1 represents a completely different set of market conditions than "leaf #5" in epoch 30.

Furthermore, with diverse regimes (especially on 10-year data), the tree's splits shift between bull-era and bear-era samples, producing leaf assignments that drift in meaning across training. The transformer wastes representational capacity reconciling these shifting embeddings rather than learning stable alpha features. The high variance across seeds (±1.01 on IR for 5Y) confirms the instability.

### 5.5 BiLSTM: Temporal Patterns Without Regime Awareness

BiLSTM achieves middle-of-pack performance at all horizons, ranking 4th–5th out of 7 across metrics. Its IC gain from 3Y to 10Y is +40.0%, almost exactly matching the cross-model average of +41.6%—it scales with data quantity but gains no *additional* benefit from the regime diversity that longer horizons provide. Its bidirectional LSTM over the 8-day lookback captures sequential patterns (gap-down-then-recover, volume-spike-then-fade) thanks to the strong momentum autocorrelation (0.78, §4.5). However, its gate is a simple linear layer—the same as Base MASTER—so it cannot learn regime-conditional attention strategies. The BiLSTM's temporal processing advantage is orthogonal to the regime-detection problem; it improves predictions by better modeling *within-sequence* dynamics, not by adapting to *between-regime* changes.

The high IR standard deviation on 10-year data (±1.20) reveals that some seeds latch onto spurious temporal patterns—a specific 8-day sequence that worked in 2020 but fails in 2022—while other seeds learn more generalizable dynamics.

### 5.6 MoE+LGBM Gate: Why It Wins

The hybrid architecture achieves the best IC, ICIR, RIC, and RICIR on both 5-year and 10-year data. Its dominance strengthens with horizon length:

| Horizon | IC | ICIR | Key Advantage |
|---------|----|------|---------------|
| 3-Year | 0.2847 | 0.965 | High accuracy but low consistency (homogeneous training data) |
| 5-Year | 0.2293 | 1.499 | ICIR jumps 55%—predictions become much more stable |
| 10-Year | 0.2984 | 2.382 | Highest IC *and* ICIR—both accurate and reliable |

The architecture combines two complementary mechanisms:

1. **The LGBM tree** learns threshold-based regime detection. With 2–3× more regime transitions on 10-year data (§4.2), the tree sees enough examples of each regime boundary to learn meaningful splits. The most informative market features are VN100 volatility measures (§4.6), which the tree can use to define splits like "VN100_vol_20d > 75th percentile → high-volatility regime."

2. **The MoE router** learns to dispatch samples to specialized experts. Expert 0 might specialize in low-volatility bull markets (weighting momentum features heavily), Expert 1 in high-volatility bear markets (weighting defensive features), and Expert 2 in transition periods (weighting mean-reversion features).

On 10-year data, both components have sufficient training signal: the tree sees enough regime transitions to learn robust boundaries, and the experts see enough samples in each regime to learn specialized strategies.

---

## 6. Why 10-Year Data Outperforms: A Synthesis

The 10-year dataset's superiority is not one effect but several compounding ones:

### 6.1 More Training Samples (Quantity)

Training samples increase from 23K (3Y) to 95K (5Y) to 145K (10Y). Neural networks are data-hungry, and the transformer backbone benefits directly from more examples. But quantity alone does not explain the magnitude of improvement—the 5-year dataset has 4× the samples of 3-year yet improves IC by only 18% on the base model, while the 10-year adds only 53% more samples over 5-year but improves IC by 27%.

### 6.2 Regime Diversity (Quality)

The 10-year dataset provides **2–3× more regime transitions** across trend, volatility, and drawdown dimensions (§4.2). More regime transitions during training force the model to learn features that generalize *across* changes. A model trained only on a persistent bear market learns "stocks go down" as a reliable rule; a model trained on alternating bull and bear markets learns *conditional* rules that depend on market context.

### 6.3 Feature Stability (Consistency)

Feature-label correlations are 13.6% more stable on 10-year data (r = 0.915 vs 0.806; §4.4). With fewer features reversing their directional relationship, the model spends less capacity reconciling contradictory signals and more capacity learning nuanced interactions.

### 6.4 Signal Quality (Tail Richness)

The 10-year dataset has fatter tails (kurtosis 9.27 vs 4.84), positive skew (+0.46 vs −0.11), and a monotonic increase in tail ratio (§4.1). Extreme events—COVID crash, V-shaped recovery, rate-hike selloff—provide "memorable" training examples that teach the model about tail risk. The 3-year dataset simply has not seen enough extreme events to learn these dynamics.

### 6.5 Market Feature Activation

Market features, particularly VN100 volatility measures, provide the regime context that gates need (§4.6). The richer the market signal and the more regime transitions in training, the more effectively tree-based gates can learn threshold rules and MoE routers can specialize experts.

---

## 7. Practical Takeaways

### 7.1 Data Quality Enables Model Complexity

Gate mechanisms are only as good as their inputs. Before investing in architectural complexity, verify that market-derived gate features carry meaningful variance. A simple check—`np.std(gate_features, axis=0) > 0`—and a correlation analysis against the prediction target should be part of every dataset build pipeline. Complex gating on noise is worse than simple gating on signal.

### 7.2 Longer Horizons Are Worth the Effort

The jump from 5-year to 10-year training data improves IC by 27% for the base model and comparable amounts for most architectures. For Vietnamese stocks, the vnstock API provides data back to approximately 2016, making 10-year datasets feasible. The marginal cost of crawling additional years is low (API calls); the marginal benefit is high (regime diversity, feature stability, more samples).

### 7.3 Tree-Based Gating Is the Right Approach for Regime-Dependent Markets

The consistent dominance of MoE+LGBM Gate suggests that tree-based regime detection is well-suited to financial data. Trees naturally handle threshold-based rules ("volatility above X → different strategy"), which mirrors how real market regimes operate. Regime transitions are not smooth; they involve threshold-like shifts in correlation structure, volatility, and trend direction. Trees model these discontinuities better than smooth neural functions.

### 7.4 Simpler Models Have Advantages

Base MASTER achieves the best IR (3.40) with the lowest standard deviation (±0.15). For a risk-averse portfolio manager who values *consistency* over raw accuracy, the base model's stable predictions and low turnover may be more valuable than MoE+LGBM's higher-but-noisier IC. The best model for a paper is not always the best model for a portfolio.

### 7.5 Cross-Attention Requires Diverse Training Data

Cross-Attn Gate's inverted-U performance pattern—best on 3Y, worst on 5Y, recovering on 10Y—is a cautionary tale. Attention mechanisms trained on homogeneous data learn brittle, overfitted patterns. If you use cross-attention, ensure your training data spans multiple market regimes, or add regularization (attention dropout, entropy penalties) to prevent memorization.

---

## 8. Limitations

This study has several important limitations:

1. **Transaction costs are modeled simplistically.** We assume a flat 0.2% impact cost per trade. In reality, costs vary with liquidity, order size, and market conditions. Tree-based gates that adapt to market conditions likely generate higher turnover than the base model, and our backtest may understate the cost disadvantage of adaptive strategies.

2. **The 3-year dataset uses a different underlying data crawl** than the 5-year and 10-year datasets, introducing potential consistency issues beyond the horizon difference itself. The 5-year and 10-year datasets share the same crawl and are directly comparable.

3. **No hyperparameter optimization per model.** All models use the same learning rate (1×10⁻⁵), dropout (0.5), early stopping threshold (0.95), and training epochs (40). Some architectures may benefit from different hyperparameters, and our results may partly reflect differential sensitivity to these shared settings rather than intrinsic architectural quality.

4. **Regime classification is post-hoc and descriptive.** We used technical indicators (MA crossovers, volatility quartiles, drawdown levels) to characterize regimes, but we did not use a formal regime-switching model (e.g., Hidden Markov Model) to identify latent states. Our regime counts depend on the specific thresholds chosen.

5. **The Vietnamese market has structural idiosyncrasies**—high retail investor participation (over 80% of trading volume), low liquidity for small-cap stocks, frontier market status, and periodic government intervention. Results may not generalize to developed markets with different investor composition and market microstructure.

6. **Only 5 seeds per configuration.** With high between-seed variance for some architectures (e.g., BiLSTM IR std = ±1.20), 5 seeds may not fully capture the distribution of outcomes. More seeds would narrow the confidence intervals on our mean estimates.

---

## 9. Conclusion

This thesis investigated seven methods of integrating market context into the attention mechanism for Vietnamese stock return prediction, across three training horizons. The results demonstrate that **how attention is integrated matters, but how much training data is available matters more—and the two are connected**.

The headline empirical finding is that the training horizon has a first-order effect on model performance. The 10-year dataset (2016–2022 training) improves prediction accuracy by 27–50% over the 3-year dataset across all architectures. This is not merely a "more data" effect. Our regime analysis shows that the 10-year training window provides 2–3 times more transitions between market states (bull, bear, crisis, recovery) and captures fatter-tailed return distributions with more extreme events. These properties create the conditions under which sophisticated attention integration actually works: gate mechanisms receive enough diverse market context to learn meaningful regime boundaries, and experts see enough samples in each regime to develop specialized attention strategies.

The MoE+LGBM Gate architecture—which separates regime detection (a decision tree) from attention modulation (multiple learned experts with a router)—achieves the best prediction accuracy (IC = 0.30, ICIR = 2.38) on the longest horizon. Its design directly addresses the core challenge identified in §1.2: that what deserves attention depends on market context. The tree learns *when* to switch attention strategies; the experts learn *what* to attend to in each regime.

However, complexity has a cost. The Base MASTER architecture—a single linear attention mask modulated by market features—achieves the best risk-adjusted portfolio returns (IR = 3.40) with the lowest variance across random seeds (±0.15). Its predictions change gradually, producing lower portfolio turnover and smoother returns. For a risk-averse practitioner, the simplest attention integration may be the best choice despite not achieving the highest raw predictive accuracy.

The 3-year dataset serves as a cautionary demonstration of attention without diversity. With 80% of training days in bear-market drawdown, cross-attention overfits to homogeneous conditions—it learns brittle inter-stock attention patterns that collapse when tested on the different market environment of 2024–2025. The architecture with the most expressive attention interaction (Cross-Attn Gate) also exhibits the most severe overfitting on narrow data, a finding with practical implications for any application of attention mechanisms to financial time series.

Three principles emerge for integrating attention into financial prediction. First, **measure before you model**: quantify regime diversity, feature stability, and market feature information content before choosing an attention architecture. Second, **match attention complexity to data diversity**: tree-based routing and multi-expert attention provide value proportional to the number of regime transitions in the training data. Third, **simplicity is robust**: when in doubt, the base attention mechanism—a linear modulation of feature importance by market context—provides the most consistent risk-adjusted performance.

The Vietnamese stock market, with its high retail participation, frontier-market characteristics, and pronounced boom-bust cycles, is an ideal testbed for attention mechanisms in prediction. Its very volatility—the same property that makes prediction difficult—also generates the regime diversity that makes sophisticated attention integration worthwhile.

---

*Analysis of VN100 Vietnamese stock data, July 2026.*
