---
Category: AI
Title: Understanding GPT - Transformers
Layout: post
Name: Understanding GPT - Transformers
date: 2023-07-07
banner: "Transformers_banner_1689490231707_0.png"
slug: understanding-gpt-transformers
popular: True
cover:
  image: "Transformers_banner_1689490231707_0.png"
tags: [ML, machine-learning, AI, Transformers]
keywords: [ML, machine-learning, AI, Transformers]
Summary: Part 2/3 - Understanding how modern LLMS work. From RNNs, to transformers, towards modern scaling laws.
---

# Introduction

The goal of this series of posts, is to form foundational knowledge that helps understanding modern state-of-the-art LLM models, and gain a comprehensive understanding of GPT. I prefer doing this via reading the seminal papers themselves, instead of online articles.

In my previous [post](/2023/06/18/understanding-gpt-rnn-attention), I covered some of the papers that formulated sequence based models from RNNs to the Attention mechanism in encoder-decoder architectures. 

This post will focus on the "Attention is all you need" paper that introduced the transformer architecture to the world, and has since had an exponential affect on the AI landscape.

# Papers to be covered in this series

1. **[THIS POST]** Transformers, following Vaswani et al. 2017 Google  {{< pdflink "https://arxiv.org/pdf/1706.03762.pdf" "Attention is all you need" >}}

2. GPT-1, following Radford et al. 2018 {{< pdflink "https://www.mikecaptain.com/resources/pdf/GPT-1.pdf" "Improving Language Understanding by Generative Pre-Training" >}}

3. GPT-2, following Radford et al. 2018 {{< pdflink "https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf" "Language Models are Unsupervised Multitask Learners" >}}

4. BERT, following Devlin et.al. 2019 Google {{< pdflink "https://arxiv.org/pdf/1810.04805.pdf" "Pre-training of Deep Bidirectional Transformers for Language Understanding" >}}

5. RoBERTa, following Liu et. al. {{< pdflink "https://arxiv.org/pdf/1907.11692.pdf" "A Robustly Optimized BERT Pretraining Approach" >}}

6. GPT-3: Few shot learners, 2020 OpenAI {{< pdflink "https://arxiv.org/pdf/2005.14165.pdf" "Language Models are Few-Shot Learners" >}}

7. PaLM: following Chowdhery et al. 2022 {{< pdflink "https://arxiv.org/pdf/2204.02311.pdf" "Scaling Language Modeling with Pathways" >}}

8. Maybe: MACAW-LLM, following Lyu et al. 2023 {{< pdflink "https://arxiv.org/pdf/2306.09093.pdf" "MULTI-MODAL LANGUAGE MODELING" >}}
## Paper

Transformers, following Vaswani et al. 2017 Google  {{< pdflink "https://arxiv.org/pdf/1706.03762.pdf" "Attention is all you need" >}}

To understand the transformers paper, let's understand the building blocks that make up the transformer architecture.

# Building Block 1: Attention

In its essence, attention allows the model to look-back on the previous inputs, based on the current-state, in an efficient manner.

### Softmax attention

Softmax attention is the simplest form, where we take advantage of 3 facts:-

1. Softmax outputs a _probability distribution / weights_ over a set of inputs
2. Softmax _amplifies_ larger values in the input
3. Weighted average is a soft-selection that is _differentiable_

Using this fact, we can take a vector q, and compute softmax weights over an input set U, followed by weighted average as an output.

![Softmax attention](/softmax_attention.png)

In the paper, attention takes the more complicated form with queries, keys, and values

> An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.

### Origins of attention

- Originally, additive attention was described previously in the paper by {{< pdflink "https://arxiv.org/pdf/1409.0473.pdf" "Dzmitry" >}}

![image.png](/image_1688788567033_0.png)

Refer to the diagram above from the paper by {{< pdflink "https://arxiv.org/pdf/1409.0473.pdf" "Dzmitry" >}}. In it, attention is realised by creating a context vector C that is generated via an alignment model. The model creates weights \\(alpha[tx,ty]\\) that help in building a weighted sum on the input states \\(h[j]\\) of the encoded sentence. This helps provide an "attention" mechanism.

## Queries, Keys, and Values

Today, the attention mechanism is generalised further with 3 separate query, key, and value vectors that create the final output. All 3 of these are generated from the same input vector in an encoder-only model.

### Why create 3 vectors ?

Because they help the model learn different representations of the input that contribute to the output.

1. Query vectors, enable the model to learn the optimal way to _query previous state_.
2. Key vectors, enable the model to expose the input state in a way that optimises _similarity matching_ between query and input.
3. Value vectors, enable the model to learn the optimal way to _output the knowledge_ about the input that is most useful for generation.

Together, these 3 vectors, enable the model to learn the best way to query the input, match it, and carry forward the most important knowledge that will help produce the right output.

The following diagram is helpful to understand how the query, key and value vectors interact to produce the outputs Y.

1. Note that Q, K, and V are all coming from the original inputs \\(X_i\\)
2. We compute similarity between Q and K, to produce alignment matrix \\(E_ik\\)
   - When \\(K_j\\) is more relevanto \\(Q_i\\), then \\(E_{ij}\\) will be higher
3. Softmax is used to create probability distribution from \\(E_{ij}\\) -> \\(A_{ij}\\)
4. Finally, the \\(V_i\\) are weighted summed based on the \\(A_{ij}\\) to produce the \\(Y_i\\)
   ![attention_calculations.png](/static/attention_calculations.png)

**In summary, attention mechanism allows the model to transform the input space into 3 separate spaces, and creates a way for the model to learn to dynamically _attend_ the most relevant parts of the historical input state.**

Coming back to the paper, the authors hint that they prefer this multiplicative attention mechanism, due to its computational effeciencies - even though historically, the additive attention was proven to work better back then.

The multiplicative attention was introduced [here](https://arxiv.org/pdf/1508.04025.pdf) by Luong et al.

The authors hypothize that the multiplicative attention had underperformed as it moves the logits into extreme ends where the gradients are close to 0. So they choose to scale down the logits before passing them to the softmax.

> We compute the dot products of the query with all keys, divide each by √dk, and apply a softmax function to obtain the weights on the values

##### Mathematically

![image.png](/image_1688789158876_0.png)


##### Visually

![image.png](/image_1688789296684_0.png)

# Building Block 2: Multi-Head & Self Attention

Further, the authors propose to do multi-head attention. This is essentially a way to parallelise the attention process on multiple heads instead of a single head.

So instead of doing a single attention with \\( d_{model}  \\) dimensions. They, parallely run N attention models with \\( d_{model}/N \\) dimensions each.

The reason for doing this?

> Multi-head attention allows the model to jointly attend to information from different representation  subspaces at different positions. With a single attention head, averaging inhibits this.

#### Mathematically:

![image.png](/image_1688828209600_0.png)

### Self Attention

In the paper, the overall model is a encoder-decoder model, and self-attention is applied only on the decoder side of the stack. However, for generative AI, the encoder only stack with self attention is the popular mechanism.
Self attention, is essentially where the attention is given to itself rather than a separate encoder model.

In a self-attention layer all of the keys, values and queries come from the same input space, in this case, the output of the previous layer in the encoder. 
Each position in the decoder can attend to all positions in the previous layer of the decoder.

The core of this goes back to the original intention described towards the beginning of the paper.

As a reminder

> This inherently sequential nature (of RNNs) precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching across examples  
> The authors detail this by comparing the complexity of different layers, and also how traversing the path for finding long-range dependencies is easy with attention, but relatively complex in other forms.

Below is the tabular version for comparison

![image.png](/image_1689489195442_0.png)

#### Few key points

**Comparison with convolutions**

- In convolutions, long range dependencies would require a stack of N/k convolutional layers. Traversing such a path, hence takes Log_k(n). They are generally more expensive then recurrent layers by a factor of k in terms of complexity

**Comparison with recurrence**

- The core win here, is that recurrent connections require n sequential operations, which becomes O(1) with self attention

**Attention is also more interpretable**

- The authors are able to build attention distributions on the model, to realise that the model is relatively easier to reason about the relationship between positions and tokens.

![image.png](/image_1689489530758_0.png)
# Building Block 3:  Positional Encoding

Since the authors completely got rid of the recurrence, or convolutional parts in the network - they need to provide the model with the positional information to compensate for this missing and crucial context.

To that effect, they chose to create positional embeddings (with the same dim size as the text embeddings).

But, they chose to not make them learnable parameters - and that makes sense to me.

They create the positional embeddings with the following logic

![image.png](/image_1689402565327_0.png)

> We  
> chose this function because we hypothesized it would allow the model to easily learn to attend by  
> relative positions, since for any fixed offset k, P Epos+k can be represented as a linear function of  
> P Epos.  
> The d2l.ai book has the best [explanation](https://d2l.ai/chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.html#positional-encoding) to this that I could find.

![image.png](/image_1689407591006_0.png)

If we plot different columns, we can see that one can easily be transformed into the other, via linear transformations.

Even after this though, I don't think I fully understand this part well. For now, I've marked this as a TODO, and will come back to it later.
# Final Architecture: Transformers

## Paper

Transformers, following Vaswani et al. 2017 Google  {{< pdflink "https://arxiv.org/pdf/1706.03762.pdf" "Attention is all you need" >}}

## The problem its solving

> This inherently sequential nature (of RNNs) precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching across examples

## Intention

> In this work we propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output.

## Architecture

Since we now understand attention, multi-head attention, and positional encodings - these building blocks can be put together to build the final architecture as shown in the paper.
![image.png](/image_1688744668247_0.png)


# Conclusion

The paper packs a ton of things into it. Its brilliant, but also probably takes a few iterations to absorb all the content well.

I intend to dive into the code that is available in [tensor2tensor](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor) and update this post with more understanding and learnings from the code.

In the next [post](/2023/10/01/understanding-gpt-1-2-3), I intend to cover GPT-1 and 2 and work our way towards the GPT-3 and other state-of-the-art model architectures and additions.
