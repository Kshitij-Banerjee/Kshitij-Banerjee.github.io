---
Category: ML Trading
Title: Short post - Santa Rally Myths vs Facts
Layout: post
Name: Short Post - Santa Rally Myths vs Facts
date: 2025-01-01
banner: "SantaRally.png"
cover:
  image: "SantaRally.png"
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

## Facts and Myths of the Santa Claus Rally

The **Santa Claus Rally** is a well-known phenomenon in the stock market where investors often see positive returns during the final week of the year, from **December 25th to January 2nd**. But is it a real pattern or just a market myth?

In this post, we’ll analyze the facts and separate them from the myths.

### Data Analysis: 7-Day Returns (December 25th to January 2nd)

To understand whether the Santa Claus Rally holds any weight, let’s look at the data. I’ve plotted a histogram showing the **7-day returns** from December 25th to January 2nd over the last 35 years for SPY.

{{< glightbox href="/image_1735725319980_0.png" src="/image_1735725319980_0.png" alt="image.png" >}}

From the histogram, we can see that the returns during this period are quite variable. In fact, **only 48% of the time** have we seen positive returns over the last 7 days of the year.

## Normalised returns - considering duration

Now, 7 days and 365 days are quite different, so its not an apples to apples comparison.

Let's normalise the returns - i.e, divide the returns by the duration 7, or 365 to make it an apples to apples comparison

{{< glightbox href="/image_1735725522646_0.png" src="/image_1735725522646_0.png" alt="image.png" >}}

> Insight: So it's clear that the last 7 days have a huge positive/negative impact return and the market is more volatile.

But its probably unfair to say that its mostly always "Positive" as the conventional wisdom says - In fact, its equal times negative, as it is positive - with a large amplitude of returns.

If you only consider the last 10 years, its an exact 50% probability of positive vs negative last 7 day returns.

### How about the annual return ?

Thats more interesting - its a split of 70% positive, and 30% negative - which is fair and well established fact. Market goes up more than it goes down.

### Correlation with Next Year's Annual Returns

Next, let’s look at the **annual returns** of the following year again and examine if there’s any relationship between the 7-day returns at the end of the year and the performance for the next year.

{{< glightbox href="/image_1735725319980_0.png" src="/image_1735725319980_0.png" alt="image.png" >}}
**The correlation coefficient between the 7-day returns and the next year's annual returns is 0.37**. This suggests that, while there is a moderate positive relationship, it’s not a guarantee that the last 7 days’ performance will always predict next year’s overall performance.

In fact, if we only take the last 10 years into consideration the correlation is even lower - 0.34

{{< glightbox href="/image_1735725830219_0.png" src="/image_1735725830219_0.png" alt="image.png" >}}

### Conclusion: Myth or Fact?

While the Santa Claus Rally has become part of market lore, the data reveals that it’s not always true.

In fact, the 7-day return from December 25th to January 2nd is positive less than **half the time**.

Moreover, the correlation with the following year's annual returns is moderate (0.37), indicating that while there may be some relationship, it's not a strong or consistent predictor.

## _So, while the Santa Claus Rally is an intriguing concept, it’s important to remember that market performance can be unpredictable, and past patterns don’t always guarantee future results._

_Disclaimer: Past performance is not indicative of future results. Always conduct thorough research before making investment decisions._
