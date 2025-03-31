---
Category: ML Trading
Title: Adventures in ML trading - Part 2
Layout: post
Name: Adventures in ML trading - Part 2
slug: adventures-in-ml-trading-part-2
date: 2024-12-25
banner: "SimpleStrategy1-Banner.webp"
cover:
  image: "SimpleStrategy1-Banner.webp"
tags: [ML, Trading, AI, Beginner, AlgorthmicTrading, Basics]
keywords:
  [
    ML,
    machine-learning,
    AI,
    Transformers,
    Trading,
    AlgorithmicTrading,
    Beginner,
    Basics,
  ]
---

# Preface

In my previous [post](/2024/12/19/adventures-in-ml-trading-part-1/), I developed a simple mean-reversion strategy based on an oscillating signal calculated from a stock's distance to its 50-day simple moving average.

However, the results revealed a key shortcoming: the algorithm struggled to account for momentum, leading to poorly timed exits during parabolic moves—either too early or too late.

{{< glightbox href="/image_1734691320767_0.png" src="/image_1734691320767_0.png" alt="image.png" >}}

In this post, we’ll dive into momentum and conduct an analysis to validate our assumption. If we can confirm that incorporating momentum enhances the strategy, we’ll move forward with developing a more advanced approach to leverage it effectively.

**Note** : This article builds on the insights from the first [post](/2024/12/19/adventures-in-ml-trading-part-1/). If you haven’t read it yet, consider giving it a quick skim to catch up!

# Hypothesis

In the previous strategy, we analyzed large periods of stock market history (1–2 years) to build probability statistics around a stock's likelihood of mean-reversion. These probabilities informed our trading decisions. However, such broad timeframes encompass various market conditions—bull, bear, and flat phases—which the overall probability statistic fails to account for, instead averaging everything out.

Take TSLA as an example: it often experiences prolonged sideways consolidations that frequently precede exponential upward movements. By calculating probability statistics across many years, we overlook these distinct "states" of the stock, treating them as a single homogeneous dataset.

> **Key question:** What if we could classify a stock into specific momentum states—such as ultra-bear, bear, flat, bull, and ultra-bull—and refine the previous strategy to incorporate these states?

_Our hypothesis is that the data distribution differs significantly across these momentum states. Relying on aggregated statistics over a large period blends these distributions, leading to inaccurate conclusions and suboptimal trading decisions._

# Method

This leads us to a straightforward plan of action:

1. **Classify Momentum States:** Identify the stock's momentum states (e.g., ultra-bear, bear, flat, bull, ultra-bull) across defined window intervals.

2. **Analyze Probability Distributions:** Calculate probability statistics for each momentum state, treating them as distinct data distributions.

3. **Incorporate Momentum into Strategy:** When applying the strategy from the previous [post](/2024/12/19/adventures-in-ml-trading-part-1/), factor in the stock’s current momentum state and use its corresponding probability distribution to make trading decisions.

The goal is simple: validate our hypothesis that momentum states offer more precise insights, enhancing the strategy. And if the hypothesis doesn’t hold, we’ll still gain valuable insights.

Now, let’s crunch the numbers—grab your favourite drink and settle in!

# Step 1: Defining the Momentum States

To start, let’s give each momentum state a clear, intuitive meaning:

1. **ULTRA-BEAR**: Severe downward movements leading to significant losses. Avoid holding positions during this time.

2. **BEAR**: Noticeable downward momentum causing losses. Consider avoiding positions or possibly going short.

3. **FLAT**: Sideways movement with minimal gains or losses. This phase is uneventful—better to allocate resources elsewhere or exploit consistent patterns here.

4. **BULL**: Upward momentum with notable positive returns. A good time to hold a position in the stock.

5. **ULTRA-BULL**: Intense market greed leading to parabolic upward movements. Ideal for holding positions, and perhaps leveraging with call options.

> **The challenge:** We need mathematical definitions for these states. Let’s analyze TSLA to establish these definitions more broadly.

---

## Return Distributions Over 3-Day Intervals

Our approach is straightforward: calculate stock returns over 3-day intervals and use those returns to classify momentum states.  
While indicators like RSI or MACD could be used, actual returns better represent the “truth.”

IMO - Higher momentum should equate to higher profits, and vice versa.

> _I’m open to discussion on this point if you have alternative suggestions!_

---

### Absolute Distribution Over 15-Day Intervals

The chart below shows the dollar return for every point on the stock price, looking 3 days forward:
{{< glightbox href="/image_1735391127954_0.png" src="/image_1735391127954_0.png" alt="image.png" >}}

---

### Percentage-Gain Distribution Over 3-Day Intervals

As noted in the previous post, absolute values introduce bias toward recent dates with higher stock prices. Instead, we analyze _percentage gains_ for a more balanced view:
{{< glightbox href="/image_1735391146939_0.png" src="/image_1735391146939_0.png" alt="image.png" >}}

---

### Determining Threshold Values

To divide the returns into five momentum states (ULTRA-BEAR to ULTRA-BULL), we calculate threshold values that evenly segment the data into five groups of ~500 days each:

```python
thresholds = np.percentile(percentage_returns, np.linspace(0, 100, num_states + 1))
```

The resulting thresholds:

```
- [-21.29815971, -3.22985358, -0.61682303, 1.2045505, 3.61522128, 36.35119642]
```

### Visualizing the Momentum States

The price chart below illustrates these momentum states, based on our thresholds:
{{< glightbox href="/image_1735391229928_0.png" src="/image_1735391229928_0.png" alt="image.png" >}}

At this stage, we have a solid mathematical definition for each momentum state. The segmentation aligns well with intuitive patterns in TSLA's price action, and we can move forward

# Step 2: Probability Statistics for Each State

Now that we have identified 5 distinct momentum states:

> Let’s validate our assumption that these states represent separate data distributions by analyzing the `Log Distance to SMA 50`, as we did earlier.

{{< glightbox href="/image_1735391253380_0.png" src="/image_1735391253380_0.png" alt="image.png" >}}

---

## Visualizing Data Distributions

Instead of examining raw log prices, we visualize the data distributions as histograms for clarity:
{{< glightbox href="/image_1735391262210_0.png" src="/image_1735391262210_0.png" alt="image.png" >}}

---

### Key Observations

    1. **ULTRA-BEAR** Skewed to the left, indicating most values are negative and below the SMA50.

2. **BEAR**: Symmetrical but slightly shifted to the left, with a narrower spread compared to other states.

3. **FLAT**:Nearly symmetrical and centered around zero, reflecting minimal deviation from the SMA50.

4. **BULL**: Symmetrical but slightly shifted to the right, with a narrower spread.  
   5. **ULTRA-BULL**: Skewed to the right, indicating most values are positive and above the SMA50.

---

## Summary

The data confirms that all momentum states exhibit distinct probability distributions.  
**These differences validate the need to consider separate statistics for each state when analyzing or trading.**

# Step 3: Simulating/Backtesting Based on Momentum States

## Now that we have momentum-state definitions, let’s use this information to simulate trades.

## Challenge: Estimating the Current Momentum State

To estimate the current momentum state, I use a simple heuristic for now (more in next blog):

1. Look at the past 3 dates, `[t-6, t-5, t-4]`, and calculate the percentage returns and corresponding momentum states for those points.

2. Pick the majority (highest frequency) momentum state.

3. In case of a tie, randomly select one to avoid bias.  
   Once the estimated momentum state is determined, **I use only the probability statistics for that specific state.**
   This allows the algorithm to contextualize current price movements based on similar past price structures, improving decision-making.

---

## Simulation Using Momentum States

### Baseline: Simple BUY-and-HOLD Strategy

First, let’s consider a baseline where we buy and never sell:
{{< glightbox href="/image_1735391846341_0.png" src="/image_1735391846341_0.png" alt="image.png" >}}

---

### Our Momentum-Based Strategy

Next, we run the simulation using the momentum-state approach:
{{< glightbox href="/image_1735391911417_0.png" src="/image_1735391911417_0.png" alt="image.png" >}}

---

### Side-by-Side Comparison

Comparing the two strategies:
{{< glightbox href="/image_1735392004631_0.png" src="/image_1735392004631_0.png" alt="image.png" >}}

---

# Observations

While the momentum-based strategy didn’t outperform BUY-and-HOLD overall, it demonstrated interesting results:

1. During significant price surges, the a*lgorithm adapted and avoided blindly selling.*

2. It showed agility by correcting itself, _entering buy/sell positions after observing strong directional surges._

---

### Examples of Strategy Adjustments

**Changed to BUY after detecting a price surge:**  
{{< glightbox href="/image_1735392902602_0.png" src="/image_1735392902602_0.png" alt="image.png" >}}

**Avoided a big crash by adapting to the price structure:**  
{{< glightbox href="/image_1735392912063_0.png" src="/image_1735392912063_0.png" alt="image.png" >}}

Compare this with trades from the previous strategy (without momentum consideration):
{{< glightbox href="/image_1735393119125_0.png" src="/image_1735393119125_0.png" alt="image.png" >}}

The older strategy exited parabolic moves prematurely because the statistics didn’t align with the observed price structure.

## Conclusion

Although the momentum-based strategy didn’t beat the traditional BUY-and-HOLD, it showed clear improvements in agility and adaptability:

**Agility:** The algorithm adjusted to price surges and crashes dynamically.

**Improved Contextualization:** Considering momentum alongside oscillating signals enabled better alignment with price structures.

In the next post, I’ll explore building an ML model to _predict_ the current momentum state, aiming to further enhance the algorithm. Stay tuned!
