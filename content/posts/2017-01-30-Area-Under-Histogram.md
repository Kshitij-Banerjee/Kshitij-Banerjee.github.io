---
layout: post
title:  "Largest Area Under a Histogram (and related concepts/problems)."
description: "A problem with a clever solution, with some insights to its construction"
name: "A problem with a clever solution, and the insights that I've gained from it"
author: Kshitij Banerjee
avatar: "img/authors/K_icon.png"
img: "histo_logo.jpg"
date: 2017-01-30
color: 2aa198
---

## Problem Statement:
- - - - - - - - -
_GfG quoted_: Find the largest rectangular area possible in a given histogram where the largest rectangle can be made of a number of contiguous bars. For simplicity, assume that all bars have same width and the width is 1 unit.
<img src="/images/histogram1.png" alt="example decode" />

## The Clever Solution
- - - - - - - - - - - - -

Sometimes, the nicest solutions come from clues we receive from the worst ones.

### What's the naive solution ?
* Iterate through all possible rectangles and calculate the area. How is the area bounded ?
* What is the extra information we need other than the "free" variables i,j ?
* We realize that min(i..j) is what constraints the area for every (i,j) pair.
<img src="/images/minArea.jpg"/>
* For every possible combination of left and right extremes. Find the maximum value of `(j-i+1)*min(i..j)`
* General way our brain thinks is :-
  * **Create every situation and try to find the value of the contraint that is needed to solve the problem.**
  * And we happily convert that to code as :-  find the value of contraint(min) for each situation(pair(i,j))

Or,

* `Max( (i,j) ->  (j-i+1)*min(i,i+1,i+2,...j) )`


### The clever solutions tries to flip the problem.
Hereon refered to as **inversion of constraint solution**

**For each `constraint/min` value of tha area, what is the best possible left and right extremes ?**

* So if we traverse over each possible `min` in the array. What are the left and right extremes for each value ?
  * Little thought says, the first left most value less than the `current min` and similarly the first rightmost value that is lesser than the current min.
  * Try some examples, if the above is difficult to validate.

<img src="/images/extreme_for_min.jpg"/>
  * This is still O(n^2)

* So now we need to see if we can find a clever way to find the first left and right values lesser than the current value.
* _To think_: If we have traversed the array partially say till min_i, how can the solution to min_i+1 be built?

take some time to think..
<br>

For each min_i.....

* We need the first value **less** than min_i to its left.
* **Inverting the statement** : we need to _ignore_ values to the left of it that are **greater** than min_i.
* _The troughs /\ in the curve hence become useless once we have crossed it_.
* **Example**: In histogram , (2 4 3) => if 3 is curr min_i being evaluated, the 4 before it being larger to it, is of no interest since we have crossed it in the previous calculations.
* _Corrollary_: Any area being considered on the right, with a min value larger than current j, will be binded at j.
* So in our processing, for each value being considered, we need the set of values before it that are less than it.
* The values of interest on the left form a monotonically increasing sequence with j being the largest value. (Values of interest here being possible values that may be of interest for the later array)
* This solves the left side. Lets concretize with an example
* If the array being evaluated is  : (1,6,2,56,_4_,23,7) [currently at 4]
  * To know the value just less than 4 the only interesting part we need to retain is (1, 2). i.e, 6 and 56 are useless for calculation of 4, and have been ignored

* Since, we are travelling from left to right, for each min value/ current value -  we do not know whether the right side of the array will have an element smaller than it.
  * So we have to keep it in memory until we get to know this value is useless.
* All this leads to a usage of our very own `stack` structure.
  * We keep on stack until we don't know its useless.
  * We remove from stack once we know the thing is crap.


- So for each min value to find its left smaller value, we do the following:-
  1. pop the elements larger to it (useless values)
  2. The first element smaller than the value is are leftmost extreme. The i to our min.

* We can do the same thing from the right side of the array and we will get j to each of our min.

* _Observation_ : If we observe the stack for each iteration i. What does it contain ?
  * A `monotonically increasing subsequence` with _i_ its max value. Notice how this becomes usefull later.

* Code examples are in plenty. But here is an implementation I did a while back
  * [https://github.com/Kshitij-Banerjee/CompetitiveCoding/blob/master/IB-LargestREctHistogram/IB-LargestREctHistogram/main.cpp](https://github.com/Kshitij-Banerjee/CompetitiveCoding/blob/master/IB-LargestREctHistogram/IB-LargestREctHistogram/main.cpp)

### What's the time complexity of one traversal ?
- - - - - - -

* While the constant pushing popping of the items seems cumbersome. Here is what helps me.
* The question to ask in general is: How many times is each item seen ?
  * Once when it is pushed.
  * Once when it is popped.
* Hence ->  O(n)


### Even better ?
* The above needs 2 traversals. One to get i for each min, and one to get j for each min.
* Can we do it in one traversal ?
* The trick is to infer the i and j values together from the stack.
* All min values fall between two smaller elements.
* Consider that we are traversing for each value j.
  * each value being popped is a potential min value with j -  the first value seen to its right that is smaller.
  * since at any point, the values in the stack are in monotonically increasing order. The i value to this min is the value just before it.

## More
- - - - - -
  1. [http://www.spoj.com/problems/C1TABOVI/](SPOJ problem CITABOVI)
  2. [http://www.geeksforgeeks.org/largest-rectangle-under-histogram/](Geeks for Geeks with code)

