
---
Category: AI
Title: Deepseek-coder - Can it code in React?
Layout: post
Name: Deepseek-coder writes react
date: 2024-04-15
banner: "exploring_code_llms.png"
cover:
 image: "exploring_code_llms.png"
tags: [Machine-Learning, AI]
keywords: [Machine-Learning, AI]
---

# Introduction

The goal of this post is to deep-dive into LLM's that are **specialised in code generation tasks**, and see if we can use them to write code.

Note: Unlike copilot, we'll focus on *locally running LLM's*. This should be appealing to any developers working in enterprises that have data privacy and sharing concerns, but still want to improve their developer productivity with locally running models.

To test our understanding, we'll perform a few simple coding tasks, and compare the various methods in achieving the desired results and also show the shortcomings.

In [Part-1](https://kshitij-banerjee.github.io/2024/04/14/exploring-code-llms/), I covered some papers around instruction fine-tuning, GQA and Model Quantization - All of which make running LLM's locally possible.

In this Part-2 of the series, I'd like to focus on how to setup an M1 to run deepseek-coder, and the verdict on its coding capabilities in react on a few tests

## The tasks

1. Test 1: Generate a higher-order-component / decorator that enables logging on a react component

2. Test 2: Write a test plan, and implement the test cases

3. Test 3: Parse an uploaded excel file in the browser.

# Setting Up the Environment: Ollama on M1

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

# Testing DeepseekCoder

## General Observations

For the most part, the 7b instruct model was quite useless and produces mostly error and incomplete responses.

The 33b models can do quite a few things correct. However, its probably at the brink of usefuleness

The model doesn't really understand writing test cases at all.

## Test Setting

For all the tests, I used a temperature setting of 0, as the guideline is to have lower temperatures for tasks like coding, instead of creative generation tasks (I find coding a pretty creative activity though, but just going with the guidelines for now)

## Test 1: Mount/Unmount logging using hooks and HOC

In this task, I wanted to test its ability to understand basic react concepts of useEffect and hooks / HOC

### 1.1 Writing custom hook with mount/unmount logging

PROMPT

```typescript
// A react custom hook that logs the component name on mount and unmount
```

CODE OUTPUT:

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

It doesn't know when to stop, and kindof keep spitting code at times - so I had to manually stop it

### Verdict

- 👍 - Works as expected

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

It used a constructor, instead of the componentDidMount method. 

added a fireEvent import, but didn't use it later.

#### Retry:

I retried a couple more times. In next attempt, it jumbled the output and got things completely wrong.

Attempt 3 is did pretty well though

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

### Verdict: 👍

It's able to write basic hooks, but makes some errors with HOCs

Sometimes, it automatically wrote the test case for the code it generated, which was a pleasant surprise!

Retrying a few times leads to automatically producing a better answer.

Component.name, is not defined, so that seems like a bug.

#### Potential future experiments

Experimenting with [n, top_p ,and top_k parameters](https://platform.openai.com/docs/api-reference/chat/create#chat-create-n), and enabling the LLM to try multiple completions

Using a 'Verifier' agent, that checks the completion for compilation and validity and retries

Giving it concrete examples, that it can follow.

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

The test case in itself is fine, the issue with the withLoggingHook causes the tests to fail

However, I did realise that multiple attempts on the same test case did not always lead to promising results.

### Verdict: 👎

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

### Verdict: 👍 👍

- Now that, was *pretty* good. I have no idea how to use the xlsx utility, and have not looked at its documentation

- Even though, I had to correct some typos and some other minor edits - this gave me a component that does exactly what I needed.

- I'd say this save me atleast 10-15 minutes of time googling for the api documentation and fumbling till I got it right.

# Conclusion

All in all, I think having locally running LLMs that can help us with code is possibly *very* near

These current models, while don't really get things correct always, *do* provide a pretty handy tool and in situations where new territory / new apps are being made, I think they can make significant progress.

Something to note, is that once I provide more longer contexts, the model seems to make a lot more errors. This is potentially only model specific, so future experimentation is needed here.

# What's next

There were quite a few things I didn't explore here. I will cover those in future posts.

Here's a list of a few things I'm going to experiment next

- Providing more examples of *good* code, instead of trying to explicitly mention every detail we want

- Comparing other models on similar exercises. Possibly making a benchmark test suite to compare them against.

- Trying multi-agent setups. I having another LLM that can correct the first ones mistakes, or enter into a dialogue where two minds reach a better outcome is totally possible.

-- A hint on this, is that once it gets something wrong, and I add the mistake to the prompt - the next iteration of the output is usually much better.
