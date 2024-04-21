
---
Category: AI
Title: Exploring Code LLMs
Layout: post
Name: Code LLMs
date: 2024-04-14
banner: "exploring_code_llms.png"
cover:
  image: "exploring_code_llms.png"
tags: [machine-learning, AI]
keywords: [machine-learning, AI]
---
# Introduction

The goal of this post is to deep-dive into LLM's that are **specialised in code generation tasks**, and see if we can use them to write code.

Note: Unlike copilot, we'll focus on *locally running LLM's*. This should be appealing to any developers working in enterprises that have data privacy and sharing concerns, but still want to improve their developer productivity with locally running models.

To test our understanding, we'll perform a few simple coding tasks, and compare the various methods in achieving the desired results and also show the shortcomings.

## The goal - A few simple coding task

1. Test 1: Generate a higher-order-component / decorator that enables logging on a react component

2. Test 2: Write a test plan, and implement the test cases

3. Test 3: Parse an uploaded excel file in the browser.

# How the rest of the post is structured

We're going to cover some theory, explain how to setup a locally running LLM model, and then finally conclude with the test results.

**Part 1: Quick theory**

Instead of explaining the concepts in painful detail, I'll refer to papers and quote specific interesting points that provide a summary. For a detailed reading, refer to the papers and links I've attached.

1. *Instruction Fine-tuning*:  Why instruction fine-tuning leads to much smaller models that can perform quite well on specific tasks, compared to much larger models

2. *Open source models available*: A quick intro on mistral, and deepseek-coder and their comparison.

3. *Model Quantization*: How we can significantly improve model inference costs, by improving memory footprint via using less precision weights.

**If you know all of the above, you may want to skip to [Part 2](#part-2-setting-up-the-environment-ollama-on-m1)**

**Part 2: Local LLM Setup**

Using Ollama and setting up my VSCode extension

VSCode Extension available here: https://github.com/Kshitij-Banerjee/kb-ollama-coder

**Part 3: Test Results**

Showing results on all 3 tasks outlines above.

# [Part 1] Understanding Instruction Finetuning

Before we venture into our evaluation of coding efficient LLMs. Let's quickly discuss what "Instruction Fine-tuning" really means.

We refer to this paper:   {{< pdflink "https://arxiv.org/pdf/2203.02155.pdf" "Training language models to follow instructions with human feedback" >}}

### Why instruction fine-tuning ?

> predicting the next token on a webpage from the internet—is different from the objective “follow the user’s instructions helpfully and safely”
### Performance Implications

> In human evaluations on our prompt distribution, outputs from the 1.3B parameter InstructGPT model are preferred to outputs from the 175B GPT-3, despite having 100x fewer paramete
### How they did it?

> Speciﬁcally, we use reinforcement learning from human feedback (RLHF; Christiano et al., 2017; Stiennon et al., 2020) to ﬁne-tune GPT-3 to follow a broad class of written instructions. This technique uses human preferences as a reward signal to ﬁne-tune our models. We ﬁrst hire a team of 40 contractors to label our data, based on their performance on a screening tes
> We then collect a dataset of human-written demonstrations of the desired output behavior on (mostly English) prompts submitted to the OpenAI API3 and some labeler-written prompts, and use this to train our supervised learning baselines. Next, we collect a dataset of human-labeled comparisons between outputs from our models on a larger set of API prompts. We then train a reward model (RM) on this dataset to predict which model output our labelers would prefer. Finally, we use this RM as a reward function and ﬁne-tune our supervised learning baseline to maximize this reward using the PPO algorithm
### Paper Results

> We call the resulting models InstructGPT.
![image.png](/image_1711792916760_0.png){:height 636, :width 1038}

> On the TruthfulQA benchmark, InstructGPT generates truthful and informative answers about twice as often as GPT-3
> During RLHF ﬁne-tuning, we observe performance regressions compared to GPT-3
> We can greatly reduce the performance regressions on these datasets by mixing PPO updates with updates that increase the log likelihood of the pretraining distribution (PPO-ptx), without compromising labeler preference scores.
> InstructGPT still makes simple mistakes. For example, InstructGPT can still fail to follow instructions, make up facts, give long hedging answers to simple questions, or fail to detect instructions with false premises$$
### Some notes on RLHF

#### Step 1: Supervised Fine Tuning:

> We ﬁne-tune GPT-3 on our labeler demonstrations using supervised learning. We trained for 16 epochs, using a cosine learning rate decay, and residual dropout of 0.2
#### Step 2  : Reward model

> Starting from the SFT model with the ﬁnal unembedding layer removed, we trained a model to take in a prompt and response, and output a scalar reward
> The underlying goal is to get a model or system that takes in a sequence of text, and returns a scalar reward which should numerically represent the human preference.
These reward models are themselves pretty huge. 6B parameters in Open AI case

#### Step 3: Fine-tuning with RL using PPO

##### PPO : {{< pdflink "https://arxiv.org/pdf/1707.06347.pdf" "Proximal Policy Optimization" >}}

> Given the prompt and response, it produces a reward determined by the reward model and ends the episode. In addition, we add a per-token KL penalty from the SFT model at each token to mitigate overoptimization of the reward model. The value function is initialized from the RM. We call these models “PPO.”
###### From : https://huggingface.co/blog/rlhf

> "Let's first formulate this fine-tuning task as a RL problem. First, the **policy** is a language model that takes in a prompt and returns a sequence of text (or just probability distributions over text). The **action space** of this policy is all the tokens corresponding to the vocabulary of the language model (often on the order of 50k tokens) and the **observation space** is the distribution of possible input token *sequences*, which is also quite large given previous uses of RL (the dimension is approximately the size of vocabulary ^ length of the input token sequence). The **reward function** is a combination of the preference model and a constraint on policy shift."
> Concatenated with the original prompt, that text is passed to the preference model, which returns a scalar notion of “preferability”, r*θ*​. In addition, per-token probability distributions from the RL policy are compared to the ones from the initial model to compute a penalty on the difference between them. In multiple papers from OpenAI, Anthropic, and DeepMind, this penalty has been designed as a scaled version of the Kullback–Leibler [(KL) divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) between these sequences of distributions over tokens, r_kl. The KL divergence term penalizes the RL policy from moving substantially away from the initial pretrained model with each training batch, which can be useful to make sure the model outputs reasonably coherent text snippets.
> Finally, the **update rule** is the parameter update from PPO that maximizes the reward metrics in the current batch of data (PPO is on-policy, which means the parameters are only updated with the current batch of prompt-generation pairs). PPO is a trust region optimization algorithm that uses constraints on the gradient to ensure the update step does not destabilize the learning process.
#### Helpful schematic showing the RL fine-tune process

![RL Process](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/rlhf/rlhf.png)

### Final Thoughts

InstructGPT outputs are more appropriate in the context of a customer assistant, more often follow explicit constraints deﬁned in the instruction (e.g. “Write your answer in 2 paragraphs or less.”),

Are less likely to fail to follow the correct instruction entirely,

Are less likely to make up facts (‘hallucinate’) less often in closed-domain tasks. These results suggest that InstructGPT models are more reliable and easier to control than GPT-3

#### Comparison of GPT vs Instruct GPT

![image.png](/image_1711810033442_0.png)

# [Part 1] Deep dive into Mistral Models

## Brief introduction to Mistral models, their architecture, and key features

We refer to the paper: {{< pdflink "https://arxiv.org/pdf/2310.06825.pdf" "Mistral 7b" >}}

### Objective:

> The search for balanced models delivering both high-level performance and efficiency
### Key Results:

> Mistral 7B outperforms the previous best 13B model (Llama 2, [ 26]) across all tested benchmarks, and surpasses the best 34B model (LLaMa 34B, [ 25 ]) in mathematics and code generation. Furthermore, Mistral 7B approaches the coding performance of Code-Llama 7B [ 20 ], without sacrificing performance on non-code related benchmarks
### Key Insights:

> Mistral 7B leverages grouped-query attention (GQA) [ 1 ], and sliding window attention (SWA) [6, 3]. GQA significantly accelerates the inference speed, and also reduces the memory requirement during decoding, allowing for higher batch sizes hence higher throughput, a crucial factor for real-time applications. In addition, SWA is designed to handle longer sequences more effectively at a reduced computational cost
### Sliding Window Attention

#### Why?

> The number of operations in vanilla attention is quadratic in the sequence length, and the memory increases linearly with the number of tokens. At inference time, this incurs higher latency and smaller throughput due to reduced cache availability. To alleviate this issue, we use sliding window attention: each token can attend to at most W tokens from the previous layer
![image.png](/image_1711811647174_0.png)

> Note that tokens outside the sliding window still influence next word prediction. At each attention layer, information can move forward by W tokens. Hence, after k attention layers, information can move forward by up to k × W tokens
> SWA exploits the stacked layers of a transformer to attend information beyond the window size W . The hidden state in position i of the layer k, hi, attends to all hidden states from the previous layer with positions between i − W and i. Recursively, hi can access tokens from the input layer at a distance of up to W × k tokens, as illustrated in Figure 1. At the last layer, using a window size of W = 4096, we have a theoretical attention span of approximately131K tokens. In practice, for a sequence length of 16K and W = 4096, changes made to FlashAttention [ 11 ] and xFormers [18 ] yield a 2x speed improvement over a vanilla attention baseline.
This fixed attention span, means we can implement a rolling buffer cache.  After W size, the cache starts overwriting the from the beginning. This also allows some pre-filling based optimizations.

### Comparison with Llama

![image.png](/image_1711812059350_0.png)

### Instruction Finetuning

> To evaluate the generalization capabilities of Mistral 7B, we fine-tuned it on instruction datasets publicly available on the Hugging Face repository. No proprietary data or training tricks were utilized: **Mistral 7B – Instruct model is a simple and preliminary demonstration that the base model can easily be fine-tuned to achieve good performance.**
### System prompt

> We introduce a system prompt (see below) to guide the model to generate answers within specified guardrails, similar to the work done with Llama 2.
The prompt: "Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity."

## How to select various model sizes, a Thumbrule from Mistral AI.

Refer to this article: [Model Selection](https://docs.mistral.ai/guides/model-selection/)

*TL;DR*

Small Tasks (Custom support, classification) => use **Mistral Small**

Medium Tasks (Data Extraction, Summarizing Documents, Writing emails.. ) => **Mistral medium**

Complex Tasks (Code Generation, RAG) => **Mistral Large**

![drawing](https://docs.mistral.ai/img/guides/modelselection1.png)

Benchmark on coding :

![image.png](/image_1711862922548_0.png)

| Model | MMLU | hellaswag (10-shot) | winograde (5-shot) | arc challenge (25-shot) | TriviaQA (5-shot) | TruthfulQA |
|---|---|---|---|---|---|---|
| Mistral 7B | 62.5% | 83.1% | 78.0% | 78.1% | 68.8% | 42.35% |
| Mixtral 8x7B | 70.6% | 86.7% | 81.2% | 85.8% | 78.38% | 47.5% |
| Mistral Small | 72.2% | 86.9% | 84.7% | 86.9% | 79.5% | 51.7% |
| Mistral Medium | 75.3% | 88.0% | 88% | 89.9% | 81.1% | 47% |
| Mistral Large | 81.2% | 89.2% | 86.7% | 94.0% | 82.7% | 50.6% |

# [Part 1] Deepseek Coder, an upgrade?

## Overview of Eval metrics

Before we understand and compare deepseeks performance, here's a quick overview on how models are measured on code specific tasks.

Leaderboard is provided here : https://evalplus.github.io/leaderboard.html

### What is HumanEval ?

https://github.com/openai/human-eval

HumanEval consists of 164 hand-written Python problems that are validated using test cases to assess the code generated by a Code LLM in a zero-shot setting,

### What is MBPP ?

https://huggingface.co/datasets/mbpp

While the MBPP benchmark includes 500 problems in a few-shot setting.

### DS-1000: More practical programming tasks, compared to Human Eval

DS-1000 benchmark, as introduced in the work by Lai et al. (2023), offers a comprehensive collection of 1,000 **practical** and realistic data science workflows across seven different libraries

## Deepseek coder

### Summary

> Each model in the series has been trained from scratch on 2 trillion tokens sourced from 87 programming languages, ensuring a comprehensive understanding of coding languages and syntax.
Refer to the paper from DeepSeek coder: {{< pdflink "https://arxiv.org/pdf/2401.14196.pdf" "DeepSeek Code" >}}

Useful links:

https://deepseekcoder.github.io/

https://ollama.com/library/deepseek-coder/tags

### How it's built

### Repository Context in Pre-training

> Besides, we attempt to organize the pretraining data at the repository level to enhance the pre-trained model’s understanding capability within the context of cross-files within a repository
They do this, by doing a topological sort on the dependent files and appending them into the context window of the LLM. More details below.

> We find that it can significantly boost the capability of cross-file code generation
### Next token prediction + Fill-in-the middle  (like BERT)

> In addition to employing the next token prediction loss during pre-training, we have also incorporated the Fill-In-Middle (FIM) approach.
### 16K context window (Mistral models have 4K sliding window attention)

> To meet the requirements of handling longer code inputs, we have extended the context length to 16K. This adjustment allows our models to handle more complex and extensive coding tasks, thereby increasing their versatility and applicability in various coding scenarios
### Data Preparation

![image.png](/image_1711865913142_0.png)

### Filtering Rule:

*TL;DR:* Remove non-code related, or data heavy files

1. Remove files with avg line length > 100, OR, maximum line length > 1000 characters.

2. Remove files with fewer than 25% alphabetic characters

3. remove <?xml version files

4. JSON/YAML files - keep fields that have character counts ranging from 50 -> 5000 . This removes data-heavy files.

### Dependency Parsing

Instead of simply passing in the current file, the dependent files within repository are parsed.

Parse Dependency between files, then arrange files in order that ensures context of each file is *before* the code of the current file. By aligning files based on dependencies, it accurately represents real coding practices and structures.

> This enhanced alignment not only makes our dataset more relevant but also potentially increases the practicality and applicability of the model in handling project-level code scenarios
> It’s worth noting that we only consider the invocation relationships between files and use regular expressions to extract them, such as"import" in Python, "using" in C#, and "include" in C.
A topological sort algorithm for doing this is provided in the paper.

> To incorporate file path information, a comment indicating the file’s path is added at the beginning of each file.
### Model Architecture

> Each model is a decoder-only Transformer, incorporating Rotary Position Embedding (RoPE)
> Notably, the DeepSeek 33B model integrates Grouped-Query-Attention (GQA) as described by Su et al. (2023), with a group size of 8, enhancing both training and inference efficiency. Additionally, we employ FlashAttention v2 (Dao, 2023) to expedite the computation involved in the attention mechanism
> we use AdamW (Loshchilov and Hutter, 2019) as the optimizer with 𝛽1 and 𝛽2 values of 0.9 and 0.95.
> he learning rate at each stage is scaled down to√︃ 110 of the preceding stage’s rate
Context Length:

> Theoretically, these modifications enable our model to process up to 64K tokens in context. However, empirical observations suggest that the model delivers its most reliable outputs within a 16K token range.\
### Instruction Tuning

> This data comprises helpful and impartial human instructions, structured by the Alpaca Instruction format. To demarcate each dialogue turn, we employed a unique delimiter token <|EOT|>
### Performance

Surpasses GPT3.5, and within reach of GPT4

![image.png](/image_1711864589567_0.png)

> To evaluate the model’s multilingual capabilities, we expanded the Python problems of Humaneval Benchmark to seven additional commonly used programming languages, namely C++, Java, PHP, TypeScript (TS), C#, Bash, and JavaScript (JS) (Cassano et al.,2023). For both benchmarks, We adopted a greedy search approach and re-implemented the baseline results using the same script and environment for fair comparison.
![image.png](/image_1711869248083_0.png)

### Interesting Notes

Chain of thought prompting

> Our analysis indicates that the implementation of Chain-of-Thought (CoT) prompting notably enhances the capabilities of DeepSeek-Coder-Instruct models. This improvement becomes particularly evident in the more challenging subsets of tasks. By adding the directive, "You need first to write a step-by-step outline and then write the code." following the initial prompt, we have observed enhancements in performance.
> This observation leads us to believe that the process of first crafting detailed code descriptions assists the model in more effectively understanding and addressing the intricacies of logic and dependencies in coding tasks, particularly those of higher complexity. Therefore, we strongly recommend employing CoT prompting strategies when utilizing DeepSeek-Coder-Instruct models for complex coding challenges.
# [Part 1] Model Quantization

Along with instruction fine-tuning, another neat technique that makes LLM's more performant (in terms of memory and resources), is model quantization

Model quantization enables one to reduce the memory footprint, and improve inference speed - with a tradeoff against the accuracy.

In short, Quantization is a process from **moving the weights of the model, from a high-information type like fp32 to a low-information but performant data-type like int8**

Reference: Huggingface guide on quantization - https://huggingface.co/docs/optimum/en/concept_guides/quantization

The two most common quantization cases are `float32 -> float16` and `float32 -> int8`.

Some schematics that explain the concept.

![Model Quantization 1: Basic Concepts | by Florian June | Medium](https://miro.medium.com/v2/resize:fit:516/1*Jlq_cyLvRdmp_K5jCd3LkA.png)

![Model Quantization: single precision, half precision, 8-bit integer](https://deci.ai/wp-content/uploads/2023/02/deci-quantization-blog-1b.png){:height 362, :width 719}

## Quantization to Int8

Let’s consider a float `x` in `[a, b]`, then we can write the following quantization scheme, also called the *affine quantization scheme*:
```
x = S * (x_q - Z)
```

`x_q` is the quantized `int8` value associated to `x`

`S` is the scale, and is a positive `float32`

`Z` is called the zero-point, it is the `int8` value corresponding to the value `0` in the `float32` realm.

```
x_q = round(x/S + Z)
```

```
x_q = clip(round(x/S + Z), round(a/S + Z), round(b/S + Z))
```

In effect, this means that we clip the ends, and perform a scaling computation in the middle. The clip-off obviously will lose to accuracy of information, and so will the rounding.

## Calibration

An example, explaining calibration to optimise clipping vs rounding error

![Model Quantization: Calibration](https://deci.ai/wp-content/uploads/2023/02/deci-quantization-blog-2a.jpg){:height 187, :width 314}

To ensure that we have a good balance of clipping vs rounding errors, based on the range [a, b] that we select. Some techniques are available

**Use per-channel granularity for weights and per-tensor for activations**

**Quantize residual connections separately by replacing blocks**

**Identify sensitive layers and skip them from quantization**

https://huggingface.co/blog/4bit-transformers-bitsandbytes

Model quantization + instruct  = *Quite Good* results

Good reference reading on the topic: https://deci.ai/quantization-and-quantization-aware-training

# [Part 2] Setting Up the Environment: Ollama on M1

## Option 1: Hosting the model

To host the models, I chose the ollama project: https://ollama.com/

Ollama is essentially, docker for LLM models and allows us to quickly run various LLM's and host them over standard completion APIs locally.

The website and documentation is pretty self-explanatory, so I wont go into the details of setting it up.

## Option 2: My machine is not strong enough, but I'd like to experiment

If your machine doesn't support these LLM's well (unless you have an M1 and above, you're in this category), then there is the following alternative solution I've found.

You can rent machines relatively cheaply (~0.4$ / hour) for inference methods, using [vast.ai](https://vast.ai/)

Once you've setup an account, added your billing methods, and have copied your API key from settings.

Clone the [llm-deploy repo](https://github.com/g1ibby/llm-deploy), and follow the instructions.

This repo figures out the cheapest available machine and hosts the ollama model as a docker image on it.

From 1 and 2, you should now have a hosted LLM model running. Now we need VSCode to call into these models and produce code.

## VSCode Extension Calling into the Model

Given the above best practices on how to provide the model its context, and the prompt engineering techniques that the authors suggested have positive outcomes on result. I created a VSCode plugin that implements these techniques, and is able to interact with Ollama running locally.

The source code for this plugin is available here:

https://github.com/Kshitij-Banerjee/kb-ollama-coder

This plugin achieves the following:-

It provides the LLM context on project/repository relevant files.

The plugin not only pulls the current file, but also loads all the currently open files in Vscode into the LLM context. 

It then trims the context to the last 16000/24000 characters (configurable)

This is an approximation, as deepseek coder enables 16K tokens, and approximate that each token is 1.5 tokens. In practice, I believe this can be much higher - so setting a higher value in the configuration should also work.

It adds a header prompt, based on the guidance from the paper. (Configurable) Example:-

"You need to first write a step-by-step outline and then write the code. The following is a complete {LANG} file named {FILE_NAME} in the project {PROJECT_NAME}. Anything NOT code is written as a CODE COMMENT."

## Source code

The source code for this plugin is available here:

https://github.com/Kshitij-Banerjee/kb-ollama-coder

For completeness, there are other VSCode plugins that might also solve a similar problem - and I am yet to try them

https://github.com/rjmacarthy/twinny

https://github.com/smallcloudai/refact

https://github.com/TabbyML/tabby

# [Part 3] Testing DeepseekCoder

## General Observations

For the most part, the 7b instruct model was quite useless and produces mostly error and incomplete responses.

The 33b models can do quite a few things correct. However, its probably at the brink of usefuleness

The model doesn't really understand writing test cases at all.

## Test Setting

For all the tests, I used a temperature setting of 0, as the guideline is to have lower temperatures for tasks like coding, instead of creative generation tasks (I find coding a pretty creative activity though, but just going with the guidelines for now)

## Test 1: Mount/Unmount logging using hooks and HOC

In this task, I wanted to test its ability to understand basic react concepts of useEffect and hooks / HOC

### 1.1 Writing custom hook with mount/unmount logging

*PROMPT*

```typescript
// A react custom hook that logs the component name on mount and unmount
```

*CODE OUTPUT:*

```typescript
import { useEffect } from 'react';
export default function withLoggingHook(componentName: string) {
  return () => {
    useEffect(() => {
      console.log(`${componentName} mounted`);
    return () => {
        console.log(`${componentName} unmounted`);
      };
    }, []);
  };
}
```

**Things to note:**
- It doesn't know when to stop, and kindof keep spitting code at times - so I had to manually stop it

### 1.2 Writing an HOC that augments the decorates the logging instead of using the hook.
*PROMPT:*
```tyepscript
// A react HOC, that wraps a given component and logs the component name
// on mount and unmount.
```
*CODE OUTPUT (Attempt 1):*

```typescript

import React from 'react';

export function withLoggingHook<T extends {}>(
	WrappedComponent: React.ComponentType<T>,
) {
	return class WithLoggingHoc extends React.PureComponent<T, {}> {
		constructor(props: T) {
			super(props);
			console.log('Mounted ' + WrappedComponent.name);
		}

		componentWillUnmount() {
			console.log('Unmounted ' + WrappedComponent.name);
		}

		render() {
			return <WrappedComponent {...this.props} />;
		}
	};
}

// ./src/generated/withLoggingHook.test.tsx
import React from 'react';
import { render, fireEvent } from '@testing-library/react';
import { withLoggingHook } from './withLoggingHook';

const TestComponent = () => <div>Test Component</div>;

describe('withLoggingHook', () => {
	beforeEach(() => {
		jest.spyOn(console, 'log');
	});

	afterEach(() => {
		jest.resetAllMocks();
	});

	it('should log when mounted and unmounted', () => {
		const ComponentWithLogging = withLoggingHook(TestComponent);

		const { rerender, unmount } = render(<ComponentWithLogging />);

		expect(console.log).toHaveBeenCalledWith('Mounted TestComponent');

		rerender(<div></div>);

		expect(console.log).not.toHaveBeenLastCalledWith('Unmounted TestComponent');

		unmount();

		expect(console.log).toHaveBeenLastCalledWith('Unmounted TestComponent');
	});
});

```

#### Mistakes:

- It used a constructor, instead of the componentDidMount method. 
- added a fireEvent import, but didn't use it later.

#### Retry:

I retried a couple more times. In the next attempt, it jumbled the output and got things completely wrong.

On attempt 3, it did pretty well though
*CODE OUTPUT (Attempt 3):*
```typescript

// A react HOC, that wraps a given component and logs the component name
// on mount and unmount.
import React from 'react';
import { useEffect } from 'react';
import { ComponentType } from 'react';

export function withLoggingHook<TProps>(
	WrappedComponent: ComponentType<TProps>,
	componentName: string,
) {
	const WithLogging = (props: TProps) => {
		useEffect(() => {
			console.log(`${componentName} is mounted`);
			return () => {
				console.log(`${componentName} is unmounted`);
			};
		}, []);
		return <WrappedComponent {...props} />;
	};
	WithLogging.displayName = `WithLogging(${getDisplayName(WrappedComponent)})`;
	return WithLogging;
}

function getDisplayName<TProps>(WrappedComponent: ComponentType<TProps>) {
	return WrappedComponent.displayName || WrappedComponent.name || 'Component';
}

```

I also noticed, that asking it specifically what I want, returned the right output in the first attempt

*CODE OUTPUT (Attempt 4): asking explicitly what I want*
```typescript

// A react HOC, that wraps a given component and logs the component name
// on mount and unmount. Use effects, instead of componentDidMount / unmount
import React from 'react';

export function withLoggingHook<P extends object>(
	Component: React.ComponentType<P>,
) {
	return (props: P): JSX.Element => {
		console.log('[withLoggingHook]', Component.name, 'mounted');

		React.useEffect(() => {
			return () =>
				console.log('[withLoggingHook]', Component.name, 'un-mounted');
		}, []);

		return <Component {...props} />;
	};
}
```

### Verdict:

- It's able to write basic hooks, but makes some errors with HOCs

- Sometimes, it automatically wrote the test case for the code it generated, which was a pleasant surprise!

- Retrying a few times leads to automatically producing a better answer.

- Component.name, is not defined, so that seems like a bug.

#### Potential future experiments

- Experimenting with [n, top_p ,and top_k parameters](https://platform.openai.com/docs/api-reference/chat/create#chat-create-n), and enabling the LLM to try multiple completions

- Using a 'Verifier' agent, that checks the completion for compilation and validity and retries

- Giving it concrete examples, that it can follow.

## Test 2: Write a test plan, and implement the test cases

Next, let's test its test case writing abilities

### Test 2.1 - Writing tests for withLogginHook

In the previous test, it wrote a withLoggingHook, let's use that as its context

As mentioned before, I used the vscode plugin I created previously, to give it context of the withLoggingHooks code

*PROMPT:*

```typescript
// [FILE-NAME] ./src/generated/withLoggingHook.tsx
// A react HOC, that wraps a given component and logs the component name
// on mount and unmount. Use effects, instead of componentDidMount / unmount
... code from withLogginHook<P> ...


// [FILE-NAME] generatedApps/DoctorFinderApp/doctor-finder-app/src/generated/withLoggingHook.spec.tsx
// Test cases for withLoggingHook

```

*CODE OUTPUT:*

```typescript
import { render } from '@testing-library/react';
import { withLoggingHook } from './withLoggingHook';

describe('withLoggingHook', () => {
	beforeEach(() => {
		// Mock console.log to prevent unwanted output during testing
		jest.spyOn(console, 'log').mockImplementation(() => {});
	});

	afterEach(() => {
		// Restore original console.log after each test
		(console.log as jest.Mock).mockRestore();
	});

	it('should log the component name on mount', () => {
		const TestComponent = withLoggingHook((): null => null);
		render(<TestComponent />);
		expect(console.log).toHaveBeenCalledWith(
			'[withLoggingHook]',
			'TestComponent',
			'mounted',
		);
	});

	it('should log the component name on unmount', () => {
		const TestComponent = withLoggingHook((): null => null);
		const { unmount } = render(<TestComponent />);
		unmount();
		expect(console.log).toHaveBeenCalledWith(
			'[withLoggingHook]',
			'TestComponent',
			'un-mounted',
		);
	});
});
```

### Mistakes:

- The test case in itself is fine, the issue with the withLoggingHook causes the tests to fail

- However, I did realise that multiple attempts on the same test case did not always lead to promising results.

### Verdict

- For simple test cases, it works quite well, but just barely.

- Anything more complex, it kinda makes too many bugs to be productively useful.

## Task 3: Excel Upload Component

Let's ask it to make a slightly more complicated component with some functionality

*PROMPT*

```typescript
// A react component, that allows the user to upload excel files.
// The excel file has one sheet, and that sheet has two columns
// Columns headers are foo, and bar, and both should have number values.
// When the file is uploaded, it is parsed using XLSX.
// The final parsed values of foo and bar, are given back as an array to the callback.
```

*CODE OUTPUT:*

```typescript
import React from 'react';
import XLSX from 'xlsx';

interface ExcelUploadComponentProps {
	onFileLoad: (data: { foo: number[]; bar: number[] }) => void;
}

const ExcelUploadComponent: React.FC<ExcelUploadComponentProps> = ({
	onFileLoad,
}: ExcelUploadComponentProps) => {
	const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
		if (!e.target.files || e.target.files.length === 0) return;

		const file = e.target.files[0];
		const reader = new FileReader();

		reader.onload = (event: ProgressEvent<FileReader>) => {
			if (!event.target || !event.target.result) return;

			const data = event.target.result as string;
			const workbook = XLSX.read(data, { type: 'binary' });
			const sheetName = workbook.SheetNames[0];
			const worksheet = workbook.Sheets[sheetName];

			// Convert the excel data to JSON
			const jsonData = XLSX.utils.sheet_to_json(worksheet, { header: 1 });

			// Extract the foo and bar values from the JSON data
			const fooValues = [];
			const barValues = [];

			for (let i = 0; i < jsonData.length; i++) {
				if (!jsonData[i]) continue;
				const row = jsonData[i];
				fooValues.push(row[0]);
				barValues.push(row[1]);
			}

			onFileLoad({ foo: fooValues, bar: barValues });
		};

		reader.readAsBinaryString(file);
	};

	return <input type="file" accept=".xls,.xlsx,.csv" onChange={handleChange} />;
};

export default ExcelUploadComponent;

```

### Verdict

- Now that was *pretty* good. I have no idea how to use the xlsx utility, and have not looked at its documentation

- Even though, I had to correct some typos and some other minor edits - this gave me a component that does exactly what I needed.

- I'd say this save me atleast 10-15 minutes of time googling for the api documentation and fumbling till I got it right.

# Conclusion

All in all, I think having locally running LLMs that can help us with code is possibly *very* near

These current models, while don't really get things correct always, *do* provide a pretty handy tool and in situations where new territory / new apps are being made, I think they can make significant progress.

Something to note, is that once I provide more longer contexts, the model seems to make a lot more errors. This is potentially only model specific, so future experimentation is needed here.

# What's next

There were quite a few things I didn't explore here. I will cover those in future posts.

- Here's a list of a few things I'm going to experiment next

- Providing more examples of *good* code, instead of trying to explicitly mention every detail we want

- Comparing other models on similar exercises. Possibly making a benchmark test suite to compare them against.

- Trying multi-agent setups. I having another LLM that can correct the first ones mistakes, or enter into a dialogue where two minds reach a better outcome is totally possible.

- A hint on this, is that once it gets something wrong, and I add the mistake to the prompt - the next iteration of the output is usually much better.
