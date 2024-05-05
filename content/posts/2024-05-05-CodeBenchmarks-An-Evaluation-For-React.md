
---
Category: AI, Machine-Learning
Title: Evaluating LLM Benchmarks for React
date: 2024-05-04
Layout: post
Name: Evaluating LLM Benchmarks for React
banner: "react-benchmark-eval.png"
cover:
  image: "react-benchmark-eval.png"
tags: [Machine-Learning, AI]
keywords: [Machine-Learning, AI]

---

# Introduction

I previously wrote about [writing react code with Deepseek-coder 33b model](https://kshitij-banerjee.github.io/2024/04/15/deepseek-coder-can-it-code-in-react/), and whether we could improve some of these shortcomings with the [latest research in the LLM space](https://kshitij-banerjee.github.io/2024/04/30/can-llms-produce-better-code/)

But to really measure and mark progress, it would require the build of a benchmark to test various hypothesis around it.

So in this post, I'm going to evaluate existing benchmarks that specifically measures LLM capabilities on coding capabilities.

My goal is to be able to build a benchmark that can test their React/Typescript coding capabilities.

## What we need

### Unit Test Evaluations

In this method, we'll require that the LLM write the code, and then we will run unit tests to measure the outcome.

We then will evaluate pass@1, pass@k, and strict-accuracy metrics.

### Visual verification

In this method, we want to test style replication and ask the LLM to produce a component with some given specifications.

We'll then verify it's output against a known ground-truth of correct visual output.

### Ease of writing, and similarity to real-life

I'd also want this to be similar to how we write code practically.

A file where some code is written, and a corresponding .test file that imports the code and runs a set of evaluations.

# How the rest of the post is structured

### Review of existing benchmarks and how they are setup

1. OpenAI Evals

2. [APPS benchmark](https://arxiv.org/pdf/2105.09938.pdf)

3. HumanEval

4. CanAiCode

5. MultiPL-E

6. RepoBench

### In a future post, I intent to cover

Details on Test based method

Details on Visual verification

Benchmark Results for 3 open source LLM models.

# 1) OpenAI Evals

This is probably the most renowned of all evaluation frameworks. https://github.com/openai/evals

However, they don't accept "Custom code" Evals. Meaning, only simple matches (Exact, Includes, Fuzzy Match) are possible test evaluations to run.

Even though OpenAI doesn't accept these evals. It's worth noting that we can simply fork the repo and write our own custom evals

The framework allows to [build a custom eval](https://github.com/openai/evals/blob/main/docs/custom-eval.md), as well as a [custom completion function](https://github.com/openai/evals/blob/main/docs/completion-fns.md). It also comes with a nice [cookbook tutorial](https://cookbook.openai.com/examples/evaluation/getting_started_with_openai_evals).

### Pros

1. Mature framework.

2. A ton of existing sample benchmarks. Once this is set up, it will allow one to find results on other interesting benchmarks.

3. Enables custom evals and custom completions

### Cons

1. Doesn't accept new custom evals.

2. It's a bit heavy to setup, with git LFS and lots of dependencies that are added over time

3. Doesn't have many code related benchmarks

### Verdict

👍 - This could work for building a react benchmark. It might be a bit hard to get off the ground though, and may limit customization.

# 2) APPS

Paper: [Measuring Coding Challenge Competence With APPS](https://arxiv.org/pdf/2105.09938.pdf)

Repository: https://github.com/hendrycks/apps

10,000 code generation problems of varying difficulties. Covers simple introductory problems, interview-level problems, and coding competition challenges

## Pros

1. Simple code base. See evaluation guide [here](https://github.com/hendrycks/apps/blob/main/eval/README.md)

2. A ton of **Coding specific** evaluations, with multiple difficulty levels.

## Cons

1. Most of the code benchmarks are *python.* So it may not work too well for other languages.

2. Isn't written with extensibility in mind, and mostly coded for testing python codebases.

## Verdict

- 👎 - Not something to use for custom real world "app" related benchmarking

# 3) HumanEval

From OpenAI again, hand-written set of evaluations

Repo: https://github.com/openai/human-eval

Paper: [Evaluating LLMs](https://arxiv.org/pdf/2107.03374.pdf)

> We evaluate functional correctness on a set of 164 handwritten programming problems, which we call the HumanEval dataset. Each problem includes a function signature, docstring, body, and several unit tests, with an average of 7.7 tests per problem

## Pros

1. Pretty simple codebase, and good examples

## Cons

1. Mostly python evaluations

## Verdict

If not testing python, this one is a 👎

# 4) CanAiCode

Repo: https://github.com/the-crypt-keeper/can-ai-code/blob/main/prompts/codellama-input-v2.txt

Leaderboard: https://huggingface.co/spaces/mike-ravkine/can-ai-code-results

## Pros

1. Supports Javascript, and not just python test cases.

2. Template based generation of test cases. See [template prompt](https://github.com/the-crypt-keeper/can-ai-code/blob/main/prompts/starcoder-fim-input.txt) for starcoder

```
{% if language == "python" %}<fim_prefix>def {{Signature}}:
    '''a function {{Input}} that returns {{Output}}{% if Fact %} given {{Fact}}{% endif %}'''
    <fim_suffix>

# another function{% endif %}
{% if language == "javascript" %}<fim_prefix>// a function {{Input}} that returns {{Output}}{% if Fact %} given {{Fact}}{% endif %}
function {{Signature}} {
<fim_suffix>
}

// another function{% endif %}<fim_middle>
```

3. Combined with yaml for tests

```yaml
.Checks: &Checks

FactorialZeroShot:
    Signature: "factorial(n)"
    Input: "with input n"
    Output: "the factorial of n using iteration"
    Description: "See if the model can implement a well known function"
    Checks:
      one_argument:
            assert: "len(f.args)"
            eq: 1
      returns_list:
            assert: "isinstance(f.call(1),int)"
            eq: true
      value_0:
            assert: "f.call(1)"
            eq: 1
      value_5:
            assert: "f.call(5)"
            eq: 120
```

## Cons

1 - Unfortunately, it is not customizable beyond simple input-output testing.

# 5) MultiPL-E

Meant to tests code LLMs on multiple programming languages.

> A system for translating unit test-driven neural code generation benchmarks to new languages. We have used MultiPL-E to translate two popular Python benchmarks (HumanEval and MBPP) to 18 other programming languages.

Examples shown here: https://nuprl.github.io/MultiPL-E/

Paper: [MultiPL-E](https://arxiv.org/pdf/2208.08227.pdf)

Repo: https://github.com/nuprl/MultiPL-E

# Pros

1. Examples on running JS tests: https://github.com/nuprl/MultiPL-E/blob/main/prompts/humaneval-js-keep.jsonl

2. Enables writing tests as a function, so not just simple input output comparisons.

3. Adding new tests seems simple: See https://nuprl.github.io/MultiPL-E/new_benchmark.html

# Cons

1. While the tutorial makes it sound like writing the test cases is really simple. This doesn't seem to be the case

2. https://github.com/nuprl/MultiPL-E/blob/main/prompts/humaneval-r-remove.jsonl

3. Each of the test case needs to be decoded in a particular jsonl format, with escape characters fixed etc.

## Verdict

👍 - This could work for building a react benchmark. But may not be easy to add new test cases to.

# 6) RepoBench

Paper: [RepoBench](https://arxiv.org/pdf/2306.03091.pdf)

Repo: https://github.com/Leolty/repobench

Validates LLM on 3 tasks

1 - Retrieval Task: Ability to retrieve the right contextual files.

2 - Completion Task: Ability to complete next line, given the context files.

3 - Combined Task: Retrieval + Completion

Some interesting points noted in the paper:

> Python Retrieval Shows Higher Accuracy Than Java: The language-specific results show that Python tasks typically show higher accuracy than
Java across all retrieval methods. This discrepancy might be attributed to Python’s simpler syntax and
less verbose nature, potentially reducing the variability of similar code snippets.

> Pronounced Performance Differences in Java for RepoBenchC-2k: The evaluation on Java showcases a marked differentiation in model performance: Codex
notably stands out as the superior model, followed by StarCoder, while CodeGen largely lags behind.

*While there are some intuitive reasons cited, this clearly shows that benchmarks on Python may not directly apply to React / Typescript codebases.*

### Interesting bits

The project is easy to read and some interesting files are

1 - Metrics: ExactMatch, Similarity, and Accuracy@K https://github.com/Leolty/repobench/blob/main/evaluation/metrics.py. Note: their accuracy@k is not a probabilistic calculation like the pass@k metric introduced in HumanEval, and refers to the number of accurate codes retrieved out of correct codes.

2 - Retriever: https://github.com/Leolty/repobench/blob/main/retriever/retriever.py

3 - Similarity (Jaccard, Edit, Cosine): https://github.com/Leolty/repobench/blob/main/retriever/similarity.py

4 - Promp constructor: https://github.com/Leolty/repobench/blob/main/data/utils.py

## Pros

1 - Easy to understand

2 - Repo level context understanding.

3 - Usage of Google drive for dataset.

4 - Multiple languages supported with various similarity metrics on next line.

## Cons

The question this benchmark is trying to answer is different from what we need.

We require unit-test and visual accuracy, assuming the right context is already given.

## Verdict

Not applicable.

# Conclusion

So far, the only ones that meet what I'm looking for are the open-ai evals, and the MultiPL-E benchmark.

Ideally, if these benchmarks were easier to prepare and mimicked the way we actually write code / test cases, then it would be much easier to extend.

So after this research, I believe the best answer is to build a new "ReactBench" - a benchmark that mimics how React code is structured and is geared towards accuracy on Typescript / React with unit-testing and snapshotting.
