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
date: 2023-01-20
title: Intro to ML
fileName: andrew_ng_course
categories: Machine Learning
lastMod: 2023-02-05
---
## Introduction

  + Note: These are my rough notes, which are auto-synced from my private LogSeq, and is a WIP.

  + I'll update and make these more readable in the future (which possibly means never :D)

### Lecture Notes:

  + https://cs229.stanford.edu/notes2022fall/main_notes.pdf

![AndrewNgMLCourseNotes.pdf](/andrewngmlcoursenotes_1674652188746_0.pdf)

### Notations

  + A pair (x^{(i)}, y^{(i)}) is called a training example, the superscript “(i)” in the
notation is simply an index into the training set, and has nothing to do with
exponentiation.

  + The notation “a := b” to denote an assignment operation

  + _h_ is called a hypothesis

  + The notation “p(y(i)|x(i); θ)” indicates that this is the distribution of y(i)
given x(i) and parameterized by θ.

#### Lecture 1 - Intro

  + {{< youtube jGwO_UgTS7I >}}

  + @00:40:31 Supervised learning - Has training examples, and learn model on that

  + @01:08:51 - Unsupervised- cocktail party problem, ICA Independent component analysis

  + @01:10:48 - Re-inforcement learning - Helicopter example

#### Lecture 2 - Linear Regression and Gradient Descent

  + {{< youtube 4b4MUYve_U8 >}}

  + @00:06:24 Linear Regression

    + {{< logseq/orgEXPORT >}}h_{θ}(x) = θ_0 + θ_1x_1 + θ_2x_2
{{< / logseq/orgEXPORT >}}

    + {{< logseq/orgEXPORT >}}J(θ) = \sum_{n=1}^{n}
(h_θ( x^{(i)} ) − y^{(i)})^2
{{< / logseq/orgEXPORT >}}

  + @00:21:37 - Gradient descent visualisation

    + @00:17:13 Cost Function:

      + {{< logseq/orgEXPORT >}}J(θ) = \sum_{n=1}^{n}
(h_θ( x^{(i)} ) − y^{(i)})^2
{{< / logseq/orgEXPORT >}}

    + @00:23:54 Optimisation: ø[_j_] = ø[_j_] - Learning_Rate * Partial_derivative( J(ø) )

      + {{< logseq/orgEXPORT >}}θj := θj − α \dfrac{∂J(θ)} {∂θj}
{{< / logseq/orgEXPORT >}}

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

  + @00:05:17  Generative Learning Algorithms, try to find the P(X|Y) instead of P(Y|X)

    + It also learns P(Y)- Called class prior, before you see anything, what is the chance

    + Example:

      + If building classifier for elephants vs dogs. First, looking at elephants, we can build a
model of what elephants look like. Then, looking at dogs, we can build a
separate model of what dogs look like. Finally, to classify a new animal, we
can match the new animal against the elephant model, and match it against
the dog model, to see whether the new animal looks more like the elephants
or more like the dogs we had seen in the training set.

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

#### Lecture 12 - Back propagation and NN Optimisations

  + {{< youtube zUazLXZZA2U >}}

  + @00:06:08 - Loss function and partial derivates for back propagation

![image.png](/image_1674556024293_0.png)

    + @00:17:38 Finally, derivative of cost function comes out as

    + {{< logseq/orgEXPORT >}}\frac {\partial LossFn} {\partial W^{[3]}} = -\frac{1}{m} \sum_{n=1}^{m} (y^{i} - a^{[3]})(a^{[2]})^{T}
{{< / logseq/orgEXPORT >}}

Then we continue to previous layers ![image.png](/image_1674560524973_0.png)

  + @00:34:37  Improving the network

    + Sigmoid has problems because the derivate is very small with large X,

    + RelU, has better derivates and helps with the backpropagation

    + Normalising Inputs @00:48:20

      + if Xs are big, then Wx + B is big, So Z is big, so derivates on sigmoid are very saturated

      + Normalise (by reducing X by the means), makes the X smaller, but still variance could be high in Y @00:49:39

      + Dividing X by Sigma, it becomes more homogeneous around the axis

![image.png](/image_1674561947221_0.png)

    + @01:05:26 Optimisations - Mini batch Gr.Desc

    + @01:11:48 - Momentum Algorithm

      + Looks at the past updates, and considers those for future iterations

![image.png](/image_1674564963664_0.png)

    + 

#### Lecture 13 - Debugging ML models and Error Analysis

  + {{< youtube ORrStCArmP4 >}}

  + @00:09:56  Bias vs Variance Diagnostics

    + Understand how much of the problem comes from bias vs variance

    + High Variance model problems

![image.png](/image_1674565738786_0.png)

      + Training error, also increases as training examples increase, because the perfect fit becomes harder with more training data

      + If there is a huge gap between training and test error, then the model has a high variance

        + Because, the model is able to fit the training set well, i.e: overfitting (like a higher degree polynomial)

    + High Bias model problems

![image.png](/image_1674565874183_0.png)

      + Even on training set, the error is high - that means the model is not fitting the data well

  + @00:27:00 Optimisation Algorithm

    + If another algorithm is able to get better results, then possibly the current algorithm is not converging

      + Maybe a different cost function is needed

    + 

  + 

#### Lecture 14 - Expectation Maximization Algorithms

  + {{< youtube rVfZHWTwXSA >}}

  + @00:01:37  Unsupervised Learning

    + @00:02:33 K-means clustering

      + 1. Initialise cluster centroids randomly, usually take k random training sets for k clusters
2. Color the point, based on nearest centroid 
{{< logseq/orgEXPORT >}}C^{j} = arg min ||x^{i} - u_{j}||^{2}
{{< / logseq/orgEXPORT >}}

      + 3. Update centroid with new values

  + @00:17:26 Anomaly Detection

    + To detect anomaly, first find the Probability density function of the given data P(x)

    + Then, if P(x) on the new sample is close to 0, then the new sample has an anomaly

  + @00:24:49  Expectation maximisation Algorithm

    + Allows us to find a joint distribution that models the data without knowing anything about the distribution of the data.

    + {{< logseq/orgEXPORT >}}p(x(i), z(i)) = p(x(i)|z(i))p(z(i))
{{< / logseq/orgEXPORT >}}

    + 

    + @00:31:21 2 steps in EM algorithms

      + Guess the value, by setting W[j] via bayes rule

      + 

    + EM implements a softer way to assign to classes, and updates with probability instead of hard assigment like K-means did

  + @00:48:17 Jensen's Inequality

#### Lecture 15 - EM + Factor Analysis

  + {{< youtube tw6cmL5STuY >}}

  + Factory Analysis:

    + Definitions online:

      + @00:17:33  Factor analysis (FA)

        + allows us to simplify a set of complex variables or items using statistical procedures to explore the underlying dimensions that explain the relationships between the multiple variables/items.

      + Factor analysis is **one of the unsupervised machine learning algorithms which is used for dimensionality reduction**. This algorithm creates factors from the observed variables to represent the common variance i.e. variance due to correlation among the observed variables.

#### Lecture 16 - Independent Component Analysis & Re-enforcement Learning

  + {{< youtube YQA9lLdLig8 >}}

    + Why do we need non-gaussian distributions in ICA

      + @00:02:59 It's important that the training set doesn't follow a gaussian density for ICA to work

      + @00:20:00 - So, we take a sigmoid function as the choice of our cumulative density function, as its derivate is non-gaussian and modes normal speech beter - as it has fatter edges, i.e more outliers wrt to the mean.

    + @00:51:40 Re-inforcement Learning

      + @00:53:40 Reward Function = R(S)

      + @00:54:04 = Credit assign problem : how do we attribute previous actions to the result in reward functions?

    + @00:56:10 Markov Decision Process

      + @00:56:49 S - set of all states

      + @00:57:04 - A set of actions

      + @00:57:21 - P_{sa} - Taking action a on s, what is the probability of getting to next state

      + @00:58:22  - R = Reward Function

      + @00:59:10 - Example illustrating MDP with maze problem

      + @01:07:45 - Discount factor

      + @01:11:52 Policy / Controller

#### Lecture 17 - MDPs & Value/Policy Iteration

  + {{< youtube d5gaWTo6kDM >}}

    + @00:06:50 - Value function

      + V_{pi}s = E[R(s_0) + ...)]

      + @00:08:29 - What is the expected total payoff, if you start at S and execute _pi_

      + @00:09:20 - Explanation of Value function over the maze example

    + @00:11:20 - Bellman Equations

      + @00:12:07 Intuition behind it

        + Immediate Reward of R(s_{0}) + DiscountFactor * [R(s_{1}) + DF*...]

        + @00:19:30  DP formulation = V_{s0} = R(S_{0}) + DiscountFactor * V_{s1}

        + s' is used to denote the next step

        + @00:21:05  Illustration of V over the maze

        + @00:22:50 - Matrix definition of Value function calculation

      + @00:25:20 - Optimal Value Function V^{*}

        + [:span]
ls-type:: annotation
hl-page:: 180
hl-color:: yellow


      + @00:30:09 - PI^{*}(S) = Optimal Policy

        + [:span]
ls-type:: annotation
hl-page:: 180
hl-color:: red


    + @00:35:31 - Value Iteration

![image.png](/image_1675524729839_0.png)

      + @00:42:30 - Backup operator

        + takes a current estimate of
the value function, and maps it to a new estimate.

      + @00:43:30  - Converges to V*

      + @00:45:30 - Illustration of the iteration algorithms

    + @00:50:34 - Policy Iteration

![image.png](/image_1675524850899_0.png)

    + @00:55:10 - Value vs Policy Iteration

      + Small states - Policy iteration is fine, but large problems the Value iteration is faster

    + @00:58:40 - How to handle Unknown P_{sa}

      + Workflow is to find P from data,

      + Take a random policy and let the problem run, and see how much probability of moving to other directions when taken P for each decision as simple probability

    + @01:03:40  workflow

      + Take actions randomly to get P_{sa}

      + Solve Bellmans equations using value iteration to get V

      + Update Policy(S)  with argmax of V over all decisions

    + @01:08:20 Exploration vs Exploitation problem - Local optimas

      + @01:14:19

Lecture 18 - Continuous/Infinite State MDP & Model Simulation

  + {{< youtube QFu5nuc-S0s >}}

    + Value Function Approximation

      + Find a model / simulator that transitions s_t -> s_t+1

        + For example, one may choose to learn a linear model of the form
s_{t+1} = A s_{t} + B a_{t},

        + Typically a gaussian error is added to make it a stochastic model

    + @00:47:04 Fitted Value Iteration

      + The main idea of fitted value iteration is that we are going to approximately carry out value iteration step, over a finite sample of states s(1), . . . , s(n).

        + Specifically, we will use a supervised learning algorithm, to approximate the value function as a linear or non-linear function of the states:

      + @01:01:50 - illustration

      + @01:16:27  - Example of how using simulator and

    + 

Lecture 19 - Reward Model & Linear Dynamical System

  + {{< youtube 0rt2CsEQv6U >}}

    + @00:06:40 - State action Rewards, makes reward function depend on both state and actions

    + @00:11:42 Finite Horizon MDP, replaces Discount factor with Finite horizon time, and it assumes a finite number of steps taken

    + @00:16:40  - Non stationary transition probabilities

    + @00:23:30 Value iteration in non stationary problems

    + @00:30:33 Illustration of the algorithm

    + @00:32:30 - Linear Quadratic Regulation

    + 

Lecture 20 - RL Debugging

  + {{< youtube pLhPQynL0tY >}}

  + 
