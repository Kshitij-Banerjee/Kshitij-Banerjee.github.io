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
title: andrew_ng_course
fileName: andrew_ng_course
categories: Machine Learning
date: 2023-01-24
lastMod: 2023-01-24
---
Lecture 1 - Intro

  + {{< youtube jGwO_UgTS7I >}}

  + undefined Supervised learning - Has training examples, and learn model on that

  + undefined - Unsupervised- cocktail party problem, ICA Independent component analysis

  + undefined - Re-inforcement learning - Helicopter example

Lecture 2 - Linear Regression and Gradient Descent

  + {{< youtube 4b4MUYve_U8 >}}

  + undefined Linear Regression - H(x) = SUM( Theta[_j_] * X[_j_]) + Theta[ _0_ ]

  + undefined - Gradient descent visualisation

    + undefined Cost Function: J(ø) = Sum{ H(X) - Y )^2 }

    + undefined Optimisation: ø[_j_] = ø[_j_] - Learning_Rate * Partial_derivative( J(ø) )

    + undefined Stochastic Gradient Descent = Do for each examples

![image.png](/image_1674464592581_0.png)


Lecture 3

  + {{< youtube het9HFqo1TQ >}}

  + undefined Locally Weighted Regression

![image.png](/image_1674465035497_0.png)

    + Keep the examples in memory, so takes more memory

    + undefined Weight based on nearness to prediction X

      + W[_i_] = Exp (  -(X[_i_] - X )^2 / 2 ) :- Is e^0=1 when close to the prediction, otherwise 0 if far

      + J(ø) = SUM( W[_i_] * (y[_i_] - ø * x[_i_])^2 )

  + undefined Logistic Regression

Sigmoid Function G(x) : ![image.png](/image_1674465233772_0.png)

    + {{< logseq/orgEXPORT >}} H(x) = G( ø^{T} * X ) 
{{< / logseq/orgEXPORT >}}

      + Where H(x) now gives the probability of the value being 1.

    + undefined

![image.png](/image_1674532901491_0.png)

      + Taking Log of P(y|x;ø), we get undefined

      + {{< logseq/orgEXPORT >}}Log Likelihood = Log(y) h(x) + Log(1-y) (1-h(x))
{{< / logseq/orgEXPORT >}}

      + undefined Choose ø that maximises the log likelyhood with Gr.Ascent

      + 

  + undefined Newton's method :

    + Get teh tangent, and use tangent X boundary as next iteration X

  + {{video https://www.youtube.com/watch?v=het9HFqo1TQ&list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU&index=3}}

Lecture 4

  + {{< youtube iZTeva0WSTQ >}}

  + Generalized Linear Models

    + undefined

    + {{< logseq/orgEXPORT >}}ø[j] = ø[j] - Learning Rate * (y[i] - H_{ø}(x[i]))* x[j]
{{< / logseq/orgEXPORT >}}

![image.png](/image_1674465875349_0.png)
Adding Learning_Rate  * x[j] => Moves ø in the direction of x[j]

    + undefined  Explaining the model

![image.png](/image_1674466215288_0.png)

    + ø - is the learnable parameter
ø^{T} X = H(x) is the prediction
Produces an exponential family, parameterized by b, a, and T

  + undefined  Softmax

Lecture 5

  + {{< youtube nt63k3bfXS0 >}}

  + undefined  Generative Learning Algorithms, P(X|Y) instead of P(Y|X)

    + It also learns P(Y)- Called class prior, before you see anything, what is the chance

  + undefined GDA - Gaussian Discriminant Analysis

    + E[z] = Mean

    + Cov[z] = E[ (z-Mean)(z-Mean)^{T}]

    + undefined  Visualization of mean and covariance

    + GDA Model: undefined

![image.png](/image_1674466717189_0.png)

      + {{< logseq/orgEXPORT >}}P(Y) = ø^{y} (1 - ø^{1-y})
{{< / logseq/orgEXPORT >}}

undefined  ![image.png](/image_1674466803649_0.png)

      + 

  + undefined Naive Bayes undefined

  + GDA vs Logistic Regression undefined

    + GDA assumes gaussian, logistic is more general, GDA will do better if assumption is true

  + 

  + 

  + 

  + 

  + 

  + 

  + 

  + 

  + 

  + 

  + 

Lecture 6

  + {{< youtube lDwow4aOrtg >}}

  + undefined

  + {{< logseq/orgEXPORT >}} H(X) = G( W^{T}X + b )
{{< / logseq/orgEXPORT >}}

  + Functional Margin

    + {{< logseq/orgEXPORT >}}J[i] = y[i] * ( W^{T}X + b )
{{< / logseq/orgEXPORT >}}

  + Geometric Margin

    + {{< logseq/orgEXPORT >}}\frac{Functional Margin} {||W||} 
{{< / logseq/orgEXPORT >}}

![image.png](/image_1674467376775_0.png)

Lecture 7

  + {{< youtube 8NYoQiRANpg >}}

  + undefined  - ø is a linear combination of the examples

  + undefined kernel - trick

    + Write algo in terms of dot-product of Xi, Xj

    + Make the features higher dimentional using a kernal function

    + Solve and move back to original dimensions

    + undefined  visualisation of higher dimension

  + Regularisation undefined

    + undefined  - Adding C to teh optimisation of W

Lecture 8

  + {{< youtube rjbkWSTjHzM >}}

  + undefined  Fitting / underfit / overfit

  + Regularisation:  undefined

    + Adding Lambda * ||ø||^{2}

    + Incentive term for algo to make ø parameters smaller

![image.png](/image_1674470357051_0.png)

  + undefined Model complexity tuning to get "just right"

![image.png](/image_1674470485278_0.png)

  + undefined - K-Fold CV

  + undefined Feature selection - adding features greedily

  + 

  + 

Lecture 9

  + {{< youtube iVOxMcumR4A >}}

Lecture 10 - Decision Trees, Random Forests, Boosting

  + {{< youtube wr9gUr-eWdA >}}

  + undefined Decision tree loss function choice

    + We could find a non-linear boundary via higher dimension SVM's but decision tree would be better here

    + Greedy top-down recursive partitioning

    + undefined  Loss_{missclassification} = 1 - P^{^}_{c} Doesn't work welll

      + P^{^}_{c} = most common classification, so 1- is missclassification

    + Cross entropy loss- undefined

![image.png](/image_1674471494668_0.png)

      + 

    + 

    + 

    + 

![image.png](/image_1674471183939_0.png)

  + undefined  Regression trees - Real values instead of classification

    + undefined - On the leaf, produce the mean of the values in that leaf

![image.png](/image_1674471735082_0.png)

    + 

  + undefined Categorical Features can also be used easily

    + Each category makes 2^{z} categories, so need to have less of them

  + undefined Regularisation | stop at min leaf size, max depth, max nodes etc.

    + change of loss is not great, because some variables if dependent, then first question even if suboptimal is needed for next to be optimal

  + undefined Bagging - Bootstrapped Aggregation - Avg. of multiple samples

    + undefined Get different samples from training set and train on each and average for outputs

    + {{< logseq/orgEXPORT >}}Model = Sum( G_{m}(x) ) /div M
{{< / logseq/orgEXPORT >}}

  + undefined Random Forests - At each split, only consider some subset features

  + undefined Boosting - Increase weight of errored samples iteratively

![image.png](/image_1674474275037_0.png)

    + 

    + 

    + 

    + 

    + 

  + 

  + 

  + 

Lecture 11 - Neural Networks

  + {{< youtube MfIjxPh6Pys >}}

  + Nice recap of logistic regression: undefined

  + undefined - Finding W(weights) and B(bias) by definition loss function

    + Loss Function: yLog(y^{y}) + (1-y)log(1-y^{^})

    + use gradient descent function

    + undefined - Neuron = Linear(Wx+b) + Activation(Sigmoid)

  + undefined - Multi classification by layers, where multiple can exist

![image.png](/image_1674475057522_0.png)

  + undefined  - Softmax, Multi-class, only one exists, sum of output is 1 so a prob over all classes

    + undefined  Cross entropy loss function for softmax

![image.png](/image_1674475728325_0.png)

  + undefined  - Neural networks intro

  + undefined - Optimising function

![image.png](/image_1674533245743_0.png)

    + y^ is the final output frm NN

    + undefined  backward propagation

  + 

Lecture 12 - Back propagation and NN Improvements

  + {{< youtube zUazLXZZA2U >}}

  + 
