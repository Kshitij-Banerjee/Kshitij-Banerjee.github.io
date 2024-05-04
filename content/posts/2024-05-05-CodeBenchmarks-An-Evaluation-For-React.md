
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

- OpenAI Evals

- [APPS benchmark](https://arxiv.org/pdf/2105.09938.pdf)

- HumanEval

- CanAiCode

### In a future post, I intent to cover

Details on Test based method

Details on Visual verification

Benchmark Results for 3 open source LLM models.

# OpenAI Evals

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

👍 - This could work for building a react benchmark.

# APPS

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

# HumanEval

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

# CanAiCode

Repo: https://github.com/the-crypt-keeper/can-ai-code/blob/main/prompts/codellama-input-v2.txt

Leaderboard: https://huggingface.co/spaces/mike-ravkine/can-ai-code-results

## Pros

Supports Javascript, and not just python test cases.

Template based generation of test cases. See [template prompt](https://github.com/the-crypt-keeper/can-ai-code/blob/main/prompts/starcoder-fim-input.txt) for starcoder

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

Combined with yaml for tests

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

Not customisable beyond simple input-output testing

# MultiPL-E

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

# Conclusion

So far, the only ones that meet what I'm looking for are the open-ai evals, and the MultiPL-E benchmark.

What would've been great, is if these benchmarks were easier to prepare, and mimicked the way we write files and test cases naturally.

I will possibly experiment with writing a simple benchmark myself, and also extending the openai evals framework.
