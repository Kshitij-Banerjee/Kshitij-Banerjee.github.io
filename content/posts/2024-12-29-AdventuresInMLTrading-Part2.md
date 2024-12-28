---
Category: ML Trading
Title: Adventures in ML trading - Part 2
Layout: post
Name: Adventures in ML trading - Part 2
date: 2024-12-25
banner: "Momentum_Strategy_2.png"
cover:
  image: "Momentum_Strategy_2.png"
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

In the last [post](2024/12/19/adventures-in-ml-trading-part-1/), I tried building a simple mean reversion strategy based on distance from SMA 50, and it was clear that without considering momentum - our algorithm was failing to recognise, and exited parabolic moves early/late.

{{< glightbox href="/image_1734691320767_0.png" src="/image_1734691320767_0.png" alt="image.png" >}}

So this time, we'll take a look into momentum and do some analysis to validate our assumption. If can successfully validate that adding momentum will be beneficial to the strategy - we can then continue building our advanced strategy to utilise it.

Note: This article builds up from learnings from the first [post](2024/12/19/adventures-in-ml-trading-part-1/), so do skim-read that if you haven't already

# Hypothesis

In the previous strategy - we would look at large periods of stock market history (1-2 years), and build probability statistics around the stock reverting, and we would use that information for taking trades.

However in such large periods of time, the stock is going through many bull, bear and flat markets - but the overall probability statistic is not considering those finer movements and averaging everything out.

Consider TSLA stock, it goes through many periods of large sideways consolidations, and they typically lead into large upside exponential movements. By calculating probability statistics across many years, we are not taking these "states" of the stock into consideration.

> **Key question:** What if we could classify the stock to be in certain momentum states (ultra-bear, bear, flat, bull, ultra-bull), and enhance the previous strategy using these internal states.

_Our assumption is that the data distribution is quite different in each of these momentum-states, and calculating statistics over a large period averages out the statistics from various distributions - leading to incorrect outcomes_

# Method

So the above leads to a simple line of attack..

1. Let's classify the stock to be in various momentum-states over certain window intervals.

2. We'll find probability statistics of the stock for each of these momentum-states, and consider them to be different data distributions.

3. When running the strategy from the previous [post](2024/12/19/adventures-in-ml-trading-part-1/), we'll consider the current momentum-state of the stock - and use that state specific probability distribution to take the trades.

Hopefully, our hypothesis turns out to be true, and if not - we'll learn something new.

Let's crunch the numbers, and I'd recommend getting your favourite drink here.

# Step 1 - Defining the Momentum-States

Let's give these states some english meaning.

1. **ULTRA-BEAR **: Extreme downward movements, causing considerable loss. We want to avoid holding a position in this time.

2. **BEAR**: Downward momentum causing loss. We also want to avoid holding a position, and potentially be short.

3. **FLAT**: We expect sideways movement of the stock, this will be boring. Better invest somewhere else, or utilise the consistent patterns here.

4. **BULL**: Considerable positive returns. We should be holding a position in the stock.

5. **ULTRA-BULL**: Extreme levels of greed in the market, the stock is parabolic. We should be in the stock and maybe even have some call options.

> **But we need to find the mathematical definition of these states**. Let's run some analysis on TSLA and define these more broadly

## Return distributions over 3 day intervals

This method is quite simple. We find the stock returns over a 3-day interval and use those returns as a classification of the momentum.

(We could use some 'indicator' like RSI / MACD, but IMO the actual return is a better representation of the "truth". Higher momentum, should give higher profits and vice versa - I welcome debate and discussion here)

### Absolute distribution over 15 day intervals

{{< glightbox href="/image_1735391127954_0.png" src="/image_1735391127954_0.png" alt="image.png" >}}

_This is the $ return, for every point of the stock, looking 3 days forward._

### Percentage-gain distribution over 3 day interval

As with our previous post, we shouldn't take absolute values, as that causes a skew in our data towards recent dates as they have higher stock prices.

So let's look at _percentage gain_ for each point instead.

{{< glightbox href="/image_1735391146939_0.png" src="/image_1735391146939_0.png" alt="image.png" >}}

Nice, Let's now find the threshold values that divide these return in 5 equal buckets from 'ULTRA_BEAR' -> 'ULTRA_BULL'

```
 thresholds = np.percentile(percentage_returns, np.linspace(0, 100, num_states + 1))

Equal Thresholds: [-21.29815971  -3.22985358  -0.61682303   1.2045505    3.61522128
  36.35119642]
```

This divides our 2516 days of data, into groups roughly 500 in size. Let's plot these states on the price chart and see if they're making sense.

{{< glightbox href="/image_1735391229928_0.png" src="/image_1735391229928_0.png" alt="image.png" >}}

At this point, I think we have a good definition of each of these momentum states.

# Step 2 - Probability Statistics for each state

Now that we have 5 distinct groups,

> Let's validate our assumption that these are indeed separate data distributions, by showing the different `Log Distance to SMA 50` as we did before

{{< glightbox href="/image_1735391253380_0.png" src="/image_1735391253380_0.png" alt="image.png" >}}

Instead of raw log prices, let's see the data distribution as histograms

{{< glightbox href="/image_1735391262210_0.png" src="/image_1735391262210_0.png" alt="image.png" >}}

##### Considerations

1. **ULTRA_BEAR**: Skewed to the left, indicating most values are negative and below the SMA50.

2. **BEAR**: Symmetrical but shifted slightly to the left, with a narrower spread compared to other states.

3. **FLAT**: Nearly symmetrical and centered around zero, indicating minimal deviation from the SMA50.

4. **BULL**: Symmetrical but shifted slightly to the right, with a narrower spread.

5. **ULTRA_BULL**: Skewed to the right, indicating most values are positive and above the SMA50.

### Conclusion

**Indeed all the momentum states do have different probability distributions, and hence different statistics to consider.**

# Step 3 - Simulating / Backtesting basis momentum-state

Now let's try to use this this new information to simulate the trades. But there's a problem :-

### How do we estimate the current momentum state ?

I've taken a simple strategy of taking an average of recently seen momentum-states

1. I look at 3 dates, `[ t-6, t-5, t-4 ]`, and then get the percentage returns and momentum states for those 3 points.

2. I then pick the majority (highest frequency) momentum state.

3. In case of a tie, I randomly pick one to have no bias.

Once I have an estimated momentum-state, **I consider only the probability statistics for that specific momentum state.**

This allows us to look at the current price movements, in context of similar price structures in the past, and make better decisions than looking at everything as one.

## Simulation with nearby momentum states

### Simulation results for a simple BUY-and-HOLD

Let's consider a baseline first - what if buy and never sell. Below is the backtrader plot of such a strategy.

{{< glightbox href="/image_1735391846341_0.png" src="/image_1735391846341_0.png" alt="image.png" >}}

### Simulation results for our strategy

This time,

{{< glightbox href="/image_1735391911417_0.png" src="/image_1735391911417_0.png" alt="image.png" >}}

### Side by Side comparison

{{< glightbox href="/image_1735392004631_0.png" src="/image_1735392004631_0.png" alt="image.png" >}}

## Conclusion

While this strategy did not beat the BUY-and-HOLD, it has some really interesting results.

Notice in the big price surges, when the price structure has changed, the new algorithm _IS_ adapting to the surge and not blindly saying SELL as we saw in the previous post.

Showing some instances of the strategy correcting itself and entering buy/sell positions after a strong surge in one direction

_Changed to a BUY, once price surge was seen_

{{< glightbox href="/image_1735392902602_0.png" src="/image_1735392902602_0.png" alt="image.png" >}}

_Changed to see, and avoided a big crash_

{{< glightbox href="/image_1735392912063_0.png" src="/image_1735392912063_0.png" alt="image.png" >}}

Compare this to trades from our previous post, that did not consider momentum - the strategy was exiting parabolic moves because the statistics didn't match the price structure being seen.

{{< glightbox href="/image_1735393119125_0.png" src="/image_1735393119125_0.png" alt="image.png" >}}

**So overall - this agility, and considering the price movements feels like this is an improvement to me, even though our strategy currently still fails the traditional BUY-and-HOLD**

It's also clear that using momentum in addition to oscillating signals - makes the algorithm more agile and adaptable.

In the next post, I'll build an ML model to _predict_ the current momentum-state, and see if that improves our algorithm further.
