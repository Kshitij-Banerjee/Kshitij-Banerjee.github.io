---
Category: ML Trading
Title: Adventures in ML trading - Part 1
Layout: post
Name: Adventures in ML trading - Part 1
date: 2024-12-19
banner: "SimpleStrategy1-Banner.webp"
cover:
  image: "SimpleStrategy1-Banner.webp"
tags: [ML, Trading, AI, Beginner, AlgorthmicTrading, Basics]
popular: True
Summary: Part 1/3 - Exporing the mathematical, statistical, and probabilistic nature of the market. Specifically, I attempt building a mean-reversion probability model, backtesting it against historical data, and understanding where/why it falls short. The results explain why simple statistical models fail to capture the complex beast that is the financial market. Nevertheless, this helps with foundational understanding and there is much to learn that I then iterate in the subsequent posts on the topic of ML based Trading.
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

Those who know me know, that I spend a lot of time reading financial statements and analysing technical signals. While this is probably the known path, and the perhaps the wiser way to make money in the market - I am particularly interested in the mathematical, statistical and probabilistic nature of the market.

Everyone who has tried their hand at trading, can atleast imagine how computers would be able to crunch through numbers better than humans. Hedge funds and algo trading firms have proliferated after [Jim Simmons](https://en.wikipedia.org/wiki/Jim_Simons) showed that computers can indeed make gigantic returns.

I believe that between the high-frequency trading firms, and the average Joe that watches CNBC and trades as a hobby - there is probably an unexplored middle. Hedge funds won't divulge their secrets, and the average Joe doesn't have the Math and Computer science skills to to understand how to use statistics and ML to make money.

So if like me, you're curious how can statistics make money - hopefully I scratch that mental itch for you in this post. I'll crunch the numbers, write the code, and show the insights - and I suggest you get your favourite drink at this point. Let's get started with a very simple strategy.

# Strategy 1 - Distance from moving average

While stock prices generally move up and to the right, they typically oscillate around a moving average.

The 50 day moving average, is perhaps the most ubiquitous signal that traders take note of when making investment decisions.

{{< glightbox href="/image_1734624144744_0.png" src="/image_1734624144744_0.png" alt="image.png" >}}

We have 3 numbers here of note.

1. The stock price

2. The 50 SMA price

3. The difference between 1) and 2)

_While the price and SMA are not oscillating number, their difference - the distance of the stocks price to its 50 day moving average is an oscillating number we can observe_

> **Key insight** What if we statistically identify when the stock is at the extremes - too far above the MA / too far below the MA from normal. Those could be good entry / exit points on the stock.

For the rest of the article, we will focus on number-3, the difference of price to its 50 day simple moving average (SMA)

# Oscillations from mean

{{< glightbox href="/image_1734624431836_0.png" src="/image_1734624431836_0.png" alt="image.png" >}}

Plotting the _Distance from SMA 50 for SPY_ gives us the above chart.

We've successfully found an oscillating pattern in the price structure.

Intuitively, its easy to visualise an average line and see the number oscillating above and below.

Let's get more formal , and define some simple statistics of this number , namely - its average, median, standard deviation, and some percentiles.

## Normal Distributions

{{< glightbox href="/image_1734624768328_0.png" src="/image_1734624768328_0.png" alt="image.png" >}}

When we plot the frequency distribution of this number as a histogram, its easy to see that it fits a normal distribution (bell curve)

Note that the distribution is skewed towards the positive ( bull markets in the last 5 years ) , but its easy to see the sigmas.

I'm interested in the 95/15th percentiles - namely the 95 and 15 percentiles, let's plot those too

{{< glightbox href="/image_1734624919409_0.png" src="/image_1734624919409_0.png" alt="image.png" >}}

#### Strategy insight

1. When the stock is more than $24.4 above its 50 SMA - There is a 95% chance it will go back to its median price soon.

2. Conversely, when the stock is more than $16.76 below its 50 SMA - There is a 85% chance that it will go back near its median price soon.

_Caveats:_

1. It's also possible that the stock price itself keeps going down or remains stable, and the Moving average comes closer to the price to reduce the gap

2. The price will eventually come closer to the SMA, but we don't know when. Stocks can remain irrational much longer than we can be solvent.

Nevertheless, the caveats may or may not happen - Let's find out what actually happens.

# Simulating the strategy

Instead of visualising this oscillating number, let's plot these extremes on the stock chart itself

{{< glightbox href="/image_1734625924321_0.png" src="/image_1734625924321_0.png" alt="image.png" >}}

The lower indicator, shows the percentiles and mean overlayed on the oscillating chart. While the chart above highlights these extremes on the price chart itself.

Looks pretty good mostly - reds are sells, greens are buys. Problem solved? - Not really.

Let's run the simulation, on what happens if we use this strategy. We buy on the green, if we have no stock - and sell on the reds if we have stock already with a simple simulation

```python
def simulate_trades(data, highlight_low_indices, highlight_indices):
    # Initialize variables
    trades = []
    shares_owned = 100
    total_profit = 0
    buy_price = data['Close'][0] # Assuming starting prices

    # Simulate trades over the period
    for i in data.index:
        if i in highlight_low_indices and shares_owned == 0:
            # Buy action
            buy_price = data['Close'][i]
            shares_owned = 100
            trades.append({'Date': i, 'Action': 'Buy', 'Price': buy_price, 'Shares': shares_owned, 'Profit': None})
            print(f"Bought 100 shares at: {buy_price:.2f} on {i}")

        elif i in highlight_indices and shares_owned > 0:
            # Sell action
            sell_price = data['Close'][i]
            profit = (sell_price - buy_price) * shares_owned
            total_profit += profit
            trades.append({'Date': i, 'Action': 'Sell', 'Price': sell_price, 'Shares': shares_owned, 'Profit': profit})
            print(f"Sold 100 shares at: {sell_price:.2f} on {i}")
            print(f"Profit/Loss for this trade: {profit:.2f}")
            shares_owned = 0

    # Sell remaining shares at the last available price
    if shares_owned > 0:
        sell_price = data['Close'].iloc[-1]
        profit = (sell_price - buy_price) * shares_owned
        total_profit += profit
        trades.append({'Date': data.index[-1], 'Action': 'Sell', 'Price': sell_price, 'Shares': shares_owned})
        print(f"Sold remaining shares at: {sell_price:.2f} on {data.index[-1]}")
        print(f"Profit for this trade: {profit:.2f}")

    return total_profit, trades
```

Which gives us :-

```txt
Bought 100 shares at: 302.4599914550781 on 2020-03-05 00:00:00
Sold 100 shares at: 296.92999267578125 on 2020-05-20 00:00:00
Profit for this trade: -552.9998779296875
Bought 100 shares at: 446.75 on 2022-01-20 00:00:00
Sold 100 shares at: 419.989990234375 on 2022-08-10 00:00:00
Profit for this trade: -2676.0009765625
Bought 100 shares at: 385.55999755859375 on 2022-09-16 00:00:00
Sold 100 shares at: 402.4200134277344 on 2022-11-23 00:00:00
Profit for this trade: 1686.0015869140625
Bought 100 shares at: 425.8800048828125 on 2023-09-26 00:00:00
Sold 100 shares at: 464.1000061035156 on 2023-12-12 00:00:00
Profit for this trade: 3822.0001220703125
Bought 100 shares at: 517.3800048828125 on 2024-08-05 00:00:00
Sold 100 shares at: 584.3200073242188 on 2024-10-14 00:00:00
Profit for this trade: 6694.000244140625

Total Profit over 5 years: 8973.001098632812

```

**Pretty good! We made $8,973!!**

Naah, you'd do better just to buy and hold the shares. **If you just held , your profit was - $27,014.99**

So in affect, this strategy is actually pretty bad: **-$18,041.99** bad. Woops.

# What went wrong ?

## Problem 1 - Momentum

{{< glightbox href="/image_1734626539846_0.png" src="/image_1734626539846_0.png" alt="image.png" >}}

If we plot the trades on the chart - the problem is clear. The red markers are failed trades. Notice that after the red markers - the stock continues to climb much farther and they typically are in bull runs.

At times, the strategy works - but other times, especially in periods of high momentum - low stock prices go lower, and high stock prices go even higher. These period of high momentum, make us exit the strategy too soon and we lose out on massive gains.

{{< glightbox href="/image_1734691320767_0.png" src="/image_1734691320767_0.png" alt="image.png" >}}

## Problem 2 - Bias

There is an obvious problem with our solution. We are calculating the statistics on absolute numbers with 10/10 hindsight bias.

Since we already know the prices historically, ofcourse the math will line up. But future price movements can be quite different from the past.

SPY is a very stable stock, and so this problem is not so obvious - let's plot the extremes on TSLA chart instead, to show how a much more volatile stock behaves

{{< glightbox href="/image_1734626952657_0.png" src="/image_1734626952657_0.png" alt="image.png" >}}

Notice what happens at the end, when TSLA stock recently went parabolic, every points on the chart are showing as SELL!

In absolute terms -

10% gain on a $500 stock = $50

10% gain on a $50 stock = $1

those numbers are a factor of 50 apart! for the same percentage gain.

Another way to see this bias is by plotting a longer timeframe on the oscillating distance chart

{{< glightbox href="/image_1734627692454_0.png" src="/image_1734627692454_0.png" alt="image.png" >}}

_When the stock price was in absolute terms low, the variation on the distance converges to 0 on the left._

# Let's fix the issues

## Fixing the bias

We could simply compare the %age movement, instead of the absolute values to fix this bias.

But, what if we can do a simple transformation that can be more broadly applicable to all the strategies in the future and remove this problem altogether?

Log is an simple solution here -

When we compare Log(price), the differences have the same relative magnitude and remove this problem.

To demonstrate:-

{{< glightbox href="/image_1734627403318_0.png" src="/image_1734627403318_0.png" alt="image.png" >}}

It's also a simple one line fix - we Log the close prices right after fetching them in the beginning. (but we'll keep separate columns to make things maintainable)

```python
data["Close"] = np.log(data["Close"])
```

Now the Long-term chart has more uniform oscillations

{{< glightbox href="/image_1734627832641_0.png" src="/image_1734627832641_0.png" alt="image.png" >}}

And our distributions are also less skewed

{{< glightbox href="/image_1734627869439_0.png" src="/image_1734627869439_0.png" alt="image.png" >}}

Plotting extremes on Log prices

{{< glightbox href="/image_1734628342474_0.png" src="/image_1734628342474_0.png" alt="image.png" >}}

Just for comparison, this was the before image (note the y scales are different, so its not an exact comparison)

{{< glightbox href="/image_1734626952657_0.png" src="/image_1734626952657_0.png" alt="image.png" >}}

Also - we should split the data in train, test and validation - because we don't want to have the hindsight bias, and this reflects actual conditions much bette.r

Let's run the statistics for SPY again, before we run the simulation again.

Statistics for Log Distances:

{{< glightbox href="/image_1734847258387_0.png" src="/image_1734847258387_0.png" alt="image.png" >}}

Highlighting extreme points on `test` price chart

{{< glightbox href="/image_1734847295284_0.png" src="/image_1734847295284_0.png" alt="image.png" >}}

### Simulating for SPY over 5 years again, this time with the bias fixes

We'll use the validation dataset now : Here's how the trades look for the validation dataset

{{< glightbox href="/image_1734847328466_0.png" src="/image_1734847328466_0.png" alt="image.png" >}}

```
Sold 100 shares at: 427.92 on 2023-06-02 00:00:00
Profit/Loss for this trade: 4169.00
Bought 100 shares at: 439.64 on 2023-08-16 00:00:00
Sold 100 shares at: 449.68 on 2023-11-15 00:00:00
Profit/Loss for this trade: 1004.00
Bought 100 shares at: 504.45 on 2024-04-15 00:00:00
Sold 100 shares at: 541.36 on 2024-06-12 00:00:00
Profit/Loss for this trade: 3691.00
Bought 100 shares at: 538.41 on 2024-07-25 00:00:00
Sold 100 shares at: 579.58 on 2024-10-11 00:00:00
Profit/Loss for this trade: 4117.00
Bought 100 shares at: 586.28 on 2024-12-18 00:00:00
Sold remaining shares at: 591.15 on 2024-12-20 00:00:00
Profit for this trade: 487.00

Total Profit over 728 days: 13468.00
```

Buy and hold total profit: $20,492.00

**Strategy Outcome: $-7,024.00**

Which is a marked improvement from our -$18041.99 strategy deficit earlier - but still negative.

**In summary:** This seems slightly better on the extremes - but momentum is still making us buy on continued downward pressure, and also making us sell sooner, making the overall strategy loss making.

## Fixing for momentum

A proper solution would be to factor the momentum into the equation when making our buy and sell decisions.

But that gets complex, by adding another variable to the mix - so i'll delve into it in the future.

For now, let's see if we can use options to fix the problem instead

> **Key logic:** We don't want to have the opportunity loss of missing out on a strong bull run. But in cases when this indicator is right - it should generate us some extra cash. So let's try to use a covered call to capitalise on the extremes. When the strategy is losing, we buy back our shares on the expiry date - and count the difference of prices as an opportunity loss.

### Simulating a 1% premiums Covered-call option strategy

```python
# Utility fn to find the next Friday
def get_next_friday(date):
    date = pd.to_datetime(date)
    # Get the weekday (Monday=0, ..., Sunday=6)
    weekday = date.weekday()
    # Calculate days to next Friday (weekday 4)
    days_to_friday = (4 - weekday + 7) % 7
    # If already Friday, move to the next Friday
    if days_to_friday == 0:
        days_to_friday = 7
    # Add the offset to the original date
    next_friday = date + pd.Timedelta(days=days_to_friday)
    return next_friday

def simulate_options(data, highlight_indices, highlight_low_indices):
  premium_percentage = 0.01  # 1% premium
  premiums_collected = 0  # Initialize total premiums collected
  opportunity_loss = 0  # Track opportunity loss when buying back shares
  strike_price_buffer = 0.01 # 1% above current price

  trade_log = []
  holding_option_until = pd.Timestamp.min  # Track until when we are holding an option

  for idx in highlight_indices:
      # Skip this index if we are already holding an option that hasn't expired
      if idx <= holding_option_until:
          continue

      # Current stock price at the sell signal
      stock_price = data.loc[idx, 'Close']
      # Premium collected from selling the option
      premium = stock_price * premium_percentage * 100
      # ATM strike price is the same as the current stock price
      strike_price = stock_price * (1 + strike_price_buffer)

      # Simulate option expiry
      holding_option_until = get_next_friday(idx)  # Get the price on the next Friday
      if holding_option_until not in data.index:
            continue
      next_friday_price = data['Close'][holding_option_until]

      if next_friday_price > strike_price:  # ITM
          # Calculate opportunity loss (difference between next Friday's price and strike price for 100 shares)
          # loss = premium * 1.5 # Bought back the option at 50% higher cost
          premiums_collected += premium
          loss = (next_friday_price - strike_price) * 100 # Got sold at strike price since
          opportunity_loss += loss
          # Log the transaction with the loss
          trade_log.append({
              'Index': idx,
              'Next Friday Date': holding_option_until,
              'Price': stock_price,
              'Strike Price': strike_price,
              'Next Friday Price': next_friday_price,
              'Outcome': 'ITM - Bought Back option',
              'Premium Collected': premium,
              'Opportunity Loss': loss
          })
      else:  # OTM
          # Add premium to total collected
          premiums_collected += premium
          # Log the transaction without loss
          trade_log.append({
              'Index': idx,
              'Next Friday Date': holding_option_until,
              'Price': stock_price,
              'Strike Price': strike_price,
              'Next Friday Price': next_friday_price,
              'Outcome': 'OTM - Premium Retained',
              'Premium Collected': premium,
              'Opportunity Loss': 0
          })

    # Convert log to DataFrame
  trade_log_df = pd.DataFrame(trade_log)

  return opportunity_loss, trade_log_df, premiums_collected
```

#### Results

```txt
Buy and Hold Strategy Base: 20492.00
-Opportunity Loss: 1312.18
Total profit excluding opp loss: 19179.82
+Total Premium: 17445.38
Strategy Delta: 16133.20
Wins: 28
Losses: 9
Win rate: 0.7567567567567568
```

**So we made an additional ~$16,133!**

This sounds great, but note that I've made a lot of assumptions here and this will likely fail when it goes to production against actual live data.

Here's the graphical representation of all the trades. We (almost) never sell the stock, and we make more profitable covered call premiums than opportunity losses.

{{< glightbox href="/image_1734855911680_0.png" src="/image_1734855911680_0.png" alt="image.png" >}}

# Conclusion

I think this simple strategy provides a lot of insights and learnings.

I made a ton of assumptions in the option simulations, that may not work with real datasets - but thats okay, the point is to learn for now and iterate.

Ignoring the momentum seems silly, let's try to incorporate it next.

_Better still, why add data one by one - let's use ML to make those non-linear relationships into probabilistic decisions._
