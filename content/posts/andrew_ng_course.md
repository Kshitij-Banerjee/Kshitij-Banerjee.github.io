---
category: Machine Learning
layout: post
name: Andrew Ng ML Overview Course Stanford
tags:
- ML
- AI
- Andrew
- Ng
- Stanford
date: 2023-01-24
title: Stanford Machine Learning Course from Andrew Ng Notes
fileName: andrew_ng_course
categories: Machine Learning
lastMod: 2023-01-24
---
## Introduction

  + Note: These are my rough notes, which are auto-synced from my private LogSeq, and is a WIP.

  + I'll update and make these more readable in the future.

#### Lecture 1 - Intro

  + {{< youtube jGwO_UgTS7I >}}

  + @00:40:31 Supervised learning - Has training examples, and learn model on that

  + @01:08:51 - Unsupervised- cocktail party problem, ICA Independent component analysis

  + @01:10:48 - Re-inforcement learning - Helicopter example

#### Lecture 2 - Linear Regression and Gradient Descent

  + {{< youtube 4b4MUYve_U8 >}}

  + @00:06:24 Linear Regression - H(x) = SUM( Theta[_j_] * X[_j_]) + Theta[ _0_ ]

  + @00:21:37 - Gradient descent visualisation

    + @00:17:13 Cost Function: J(ø) = Sum{ H(X) - Y )^2 }

    + @00:23:54 Optimisation: ø[_j_] = ø[_j_] - Learning_Rate * Partial_derivative( J(ø) )

    + @00:45:00 Stochastic Gradient Descent = Do for each examples

![image.png](/image_1674464592581_0.png)

#### Lecture 3 - Logistic Regression

  + {{< youtube het9HFqo1TQ >}}

  + @00:05:54 Locally Weighted Regression


![image.png](/image_1674465035497_0.png)

    + Keep the examples in memory, so takes more memory

    + @00:12:18 Weight based on nearness to prediction X

      + W[_i_] = Exp (  -(X[_i_] - X )^2 / 2 ) :- Is e^0=1 when close to the prediction, otherwise 0 if far

      + J(ø) = SUM( W[_i_] * (y[_i_] - ø * x[_i_])^2 )

  + @00:46:43 Logistic Regression

Sigmoid Function G(x) : ![image.png](/image_1674465233772_0.png)

    + {{< logseq/orgEXPORT >}} H(x) = G( ø^{T} * X ) 
{{< / logseq/orgEXPORT >}}

      + Where H(x) now gives the probability of the value being 1.

    + @00:54:20

![image.png](/image_1674532901491_0.png)

      + Taking Log of P(y|x;ø), we get @00:57:50

      + {{< logseq/orgEXPORT >}}Log Likelihood = Log(y) h(x) + Log(1-y) (1-h(x))
{{< / logseq/orgEXPORT >}}

      + @00:58:38 Choose ø that maximises the log likelyhood with Gr.Ascent

      + 

  + @01:07:39 Newton's method :

    + Get the tangent, and use tangent X boundary as next iteration X

  + {{video https://www.youtube.com/watch?v=het9HFqo1TQ&list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU&index=3}}

#### Lecture 4 - Generalized Linear Model

  + {{< youtube iZTeva0WSTQ >}}

  + Generalized Linear Models

    + @00:05:22

    + {{< logseq/orgEXPORT >}}ø[j] = ø[j] - Learning Rate * (y[i] - H_{ø}(x[i]))* x[j]
{{< / logseq/orgEXPORT >}}

![image.png](/image_1674465875349_0.png)
Adding Learning_Rate  * x[j] => Moves ø in the direction of x[j]

    + @00:43:14  Explaining the model

![image.png](/image_1674466215288_0.png)

    + ø - is the learnable parameter
ø^{T} X = H(x) is the prediction
Produces an exponential family, parameterized by b, a, and T

  + @01:14:09  Softmax

#### Lecture 5 - GDA & Naive Bayes

  + {{< youtube nt63k3bfXS0 >}}

  + @00:05:17  Generative Learning Algorithms, P(X|Y) instead of P(Y|X)

    + It also learns P(Y)- Called class prior, before you see anything, what is the chance

  + @00:10:09 GDA - Gaussian Discriminant Analysis

    + E[z] = Mean

    + Cov[z] = E[ (z-Mean)(z-Mean)^{T}]

    + @00:12:35  Visualization of mean and covariance

    + GDA Model: @00:19:33

![image.png](/image_1674466717189_0.png)

      + {{< logseq/orgEXPORT >}}P(Y) = ø^{y} (1 - ø^{1-y})
{{< / logseq/orgEXPORT >}}

@00:45:16  ![image.png](/image_1674466803649_0.png)

  + @00:25:52 Naive Bayes @01:08:08

  + GDA vs Logistic Regression @00:53:00

    + GDA assumes gaussian, logistic is more general, GDA will do better if assumption is true

#### Lecture 6 - Support Vector Machines

  + {{< youtube lDwow4aOrtg >}}

  + @00:47:59 - How SVN finds non-linear decision boundaries

  + {{< logseq/orgEXPORT >}} H(X) = G( W^{T}X + b )
{{< / logseq/orgEXPORT >}}

  + Functional Margin

    + {{< logseq/orgEXPORT >}}J[i] = y[i] * ( W^{T}X + b )
{{< / logseq/orgEXPORT >}}

  + Geometric Margin

    + {{< logseq/orgEXPORT >}}\frac{Functional Margin} {||W||} 
{{< / logseq/orgEXPORT >}}

![image.png](/image_1674467376775_0.png)

#### Lecture 7 - Kernels Trick

  + {{< youtube 8NYoQiRANpg >}}

  + @00:15:20  - ø is a linear combination of the examples

  + @00:29:21 kernel - trick

    + Write algo in terms of dot-product of Xi, Xj

    + Make the features higher dimentional using a kernal function

    + Solve and move back to original dimensions

    + @00:47:20  visualisation of higher dimension

  + Regularisation @01:05:00

    + @01:06:40  - Adding C to teh optimisation of W

#### Lecture 8 - Data Splits and Fitting

  + {{< youtube rjbkWSTjHzM >}}

  + @00:07:41  Fitting / underfit / overfit

  + Regularisation:  @00:14:30

    + Adding Lambda * ||ø||^{2}

    + Incentive term for algo to make ø parameters smaller

![image.png](/image_1674470357051_0.png)

  + @00:38:05 Model complexity tuning to get "just right"

![image.png](/image_1674470485278_0.png)

  + @01:05:20 - K-Fold CV

  + @01:19:40 Feature selection - adding features greedily

  + 

  + 

#### Lecture 9 - Estimation Errors & ERM

  + {{< youtube iVOxMcumR4A >}}

#### Lecture 10 - Decision Trees, Random Forests, Boosting

  + {{< youtube wr9gUr-eWdA >}}

  + @00:03:45 Decision tree loss function choice


    + We could find a non-linear boundary via higher dimension SVM's but decision tree would be better here

    + Greedy top-down recursive partitioning

    + @00:10:54

      + {{< logseq/orgEXPORT >}}Loss_{missclassification} = 1 - P^{∆}_{c} 
{{< / logseq/orgEXPORT >}}

      + Where P^{^}_{c} = most common class; and hence  1- becomes miss-classification

    + Cross entropy loss- @00:16:02, work much better

![image.png](/image_1674471183939_0.png)

  + @00:31:10  Regression trees - Real values instead of classification


    + @00:32:20 - On the leaf, produce the mean of the values in that leaf

![image.png](/image_1674471735082_0.png)

    + 

  + @00:35:30 Categorical Features can also be used easily


    + Each category makes 2^{z} categories, so need to have less of them

  + @00:38:31 Regularisation | stop at min leaf size, max depth, max nodes etc.


    + change of loss is not great, because some variables if dependent, then first question even if suboptimal is needed for next to be optimal

  + @01:00:27 Bagging - Bootstrapped Aggregation - Avg. of multiple samples


    + @01:04:50 Get different samples from training set and train on each and average for outputs

    + {{< logseq/orgEXPORT >}}Model = Sum( G_{m}(x) ) /div M
{{< / logseq/orgEXPORT >}}

  + @01:13:44 Random Forests - At each split, only consider some subset features

  + @01:15:39 Boosting - Increase weight of errored samples iteratively

![image.png](/image_1674474275037_0.png)

#### Lecture 11 - Neural Networks

  + {{< youtube MfIjxPh6Pys >}}

  + Nice recap of logistic regression: @00:03:25

  + @00:08:30 - Finding W(weights) and B(bias) by definition loss function


    + Loss Function: yLog(y^{y}) + (1-y)log(1-y^{^})

    + use gradient descent function

    + @00:12:12 - Neuron = Linear(Wx+b) + Activation(Sigmoid)

  + @00:17:08 - Multi classification by layers, where multiple can exist


![image.png](/image_1674475057522_0.png)

  + @00:25:36  - Softmax, Multi-class, only one exists, sum of output is 1 so a prob over all classes


    + @00:35:47  Cross entropy loss function for softmax

![image.png](/image_1674475728325_0.png)

  + @00:41:43  - Neural networks intro

  + @01:11:17 - Optimising function

![image.png](/image_1674533245743_0.png)

    + y^ is the final output frm NN

    + @01:13:00  backward propagation

  + 

#### Lecture 12 - Back propagation and NN Improvements

  + {{< youtube zUazLXZZA2U >}}

  + 
