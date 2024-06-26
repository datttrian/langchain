# Quickstart

In this quickstart we'll show you how to:

- Get setup with LangChain, LangSmith, and LangServe
- Use the most basic and common components of LangChain: prompt templates, models, and output parsers
- Use LangChain Expression Language, the protocol that LangChain is built on and which facilitates component chaining
- Build a simple application with LangChain
- Trace your application with LangSmith
- Serve your application with LangServe

That's a fair amount to cover! Let's dive in.

## Setup

### Jupyter Notebook

This guide (and most of the other guides in the documentation) uses [Jupyter notebooks](https://jupyter.org/) and assumes the reader is as well. Jupyter notebooks are perfect for learning how to work with LLM systems because oftentimes things can go wrong (unexpected output, API down, etc.) and going through guides in an interactive environment is a great way to better understand them.

You do not NEED to go through the guide in a Jupyter Notebook, but it is recommended. See [here](https://jupyter.org/install) for instructions on how to install.

### Installation

To install LangChain run:

#### Pip

```sh
pip install langchain
```

#### Conda

```sh
conda install langchain -c conda-forge
```

For more details, see our [Installation guide](https://python.langchain.com/v0.1/docs/get_started/installation/).

### LangSmith

Many of the applications you build with LangChain will contain multiple steps with multiple invocations of LLM calls. As these applications get more and more complex, it becomes crucial to be able to inspect what exactly is going on inside your chain or agent. The best way to do this is with [LangSmith](https://smith.langchain.com).

Note that LangSmith is not needed, but it is helpful. If you do want to use LangSmith, after you sign up at the link above, make sure to set your environment variables to start logging traces:

```sh
export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_API_KEY="..."
```

## Building with LangChain

LangChain enables building applications that connect external sources of data and computation to LLMs. In this quickstart, we will walk through a few different ways of doing that. We will start with a simple LLM chain, which just relies on information in the prompt template to respond. Next, we will build a retrieval chain, which fetches data from a separate database and passes that into the prompt template. We will then add in chat history, to create a conversation retrieval chain. This allows you to interact in a chat manner with this LLM, so it remembers previous questions. Finally, we will build an agent - which utilizes an LLM to determine whether or not it needs to fetch data to answer questions. We will cover these at a high level, but there are a lot of details to all of these! We will link to relevant docs.

## LLM Chain

We'll show how to use models available via API, like OpenAI, and local open source models, using integrations like Ollama.

### OpenAI

First we'll need to import the LangChain x OpenAI integration package.

```python
pip install --quiet langchain-openai
```

    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m A new release of pip is available: [0m[31;49m23.0.1[0m[39;49m -> [0m[32;49m24.1[0m
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m To update, run: [0m[32;49mpip install --upgrade pip[0m
    Note: you may need to restart the kernel to use updated packages.

Accessing the API requires an API key, which you can get by creating an account and heading [here](https://platform.openai.com/account/api-keys). Once we have a key we'll want to set it as an environment variable by running:

```sh
export OPENAI_API_KEY="..."
```

We can then initialize the model:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI()
```

#### API Reference

- [ChatOpenAI](https://api.python.langchain.com/en/latest/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html)

If you'd prefer not to set an environment variable you can pass the key in directly via the `api_key` named parameter when initiating the OpenAI LLM class:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(api_key="...")
```

#### API Reference

- [ChatOpenAI](https://api.python.langchain.com/en/latest/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html)

### Local (using Ollama)

[Ollama](https://ollama.ai/) allows you to run open-source large language models, such as Llama 2, locally.

First, follow [these instructions](https://github.com/jmorganca/ollama) to set up and run a local Ollama instance:

- [Download](https://ollama.ai/download)
- Fetch a model via `ollama pull llama2`

Then, make sure the Ollama server is running. After that, you can do:

```python
from langchain_community.llms import Ollama
llm = Ollama(model="llama2")
```

#### API Reference

- [Ollama](https://api.python.langchain.com/en/latest/llms/langchain_community.llms.ollama.Ollama.html)

### Anthropic

First we'll need to import the LangChain x Anthropic package.

```sh
pip install langchain-anthropic
```

Accessing the API requires an API key, which you can get by creating an account [here](https://claude.ai/login). Once we have a key we'll want to set it as an environment variable by running:

```sh
export ANTHROPIC_API_KEY="..."
```

We can then initialize the model:

```python
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0.2, max_tokens=1024)
```

#### API Reference

- [ChatAnthropic](https://api.python.langchain.com/en/latest/chat_models/langchain_anthropic.chat_models.ChatAnthropic.html)

If you'd prefer not to set an environment variable you can pass the key in directly via the `api_key` named parameter when initiating the Anthropic Chat Model class:

```python
llm = ChatAnthropic(api_key="...")
```

### Cohere

First we'll need to import the Cohere SDK package.

```sh
pip install langchain-cohere
```

Accessing the API requires an API key, which you can get by creating an account and heading [here](https://dashboard.cohere.com/api-keys). Once we have a key we'll want to set it as an environment variable by running:

```sh
export COHERE_API_KEY="..."
```

We can then initialize the model:

```python
from langchain_cohere import ChatCohere

llm = ChatCohere()
```

#### API Reference

- [ChatCohere](https://api.python.langchain.com/en/latest/chat_models/langchain_cohere.chat_models.ChatCohere.html)

If you'd prefer not to set an environment variable you can pass the key in directly via the `cohere_api_key` named parameter when initiating the Cohere LLM class:

```python
from langchain_cohere import ChatCohere

llm = ChatCohere(cohere_api_key="...")
```

#### API Reference

- [ChatCohere](https://api.python.langchain.com/en/latest/chat_models/langchain_cohere.chat_models.ChatCohere.html)

Once you've installed and initialized the LLM of your choice, we can try using it! Let's ask it what LangSmith is - this is something that wasn't present in the training data so it shouldn't have a very good response.

```python
llm.invoke("how can langsmith help with testing?")
```

    AIMessage(content='Langsmith can help with testing by providing automated test scripts that can be run to test the functionality and performance of software applications. These test scripts can be customized to simulate various user interactions and scenarios, helping to identify bugs and errors in the software. Langsmith can also help with setting up continuous integration and continuous deployment pipelines to streamline the testing process and ensure that any issues are detected and resolved quickly. Additionally, Langsmith can provide insights and recommendations for improving test coverage and effectiveness, ultimately leading to a more reliable and stable software product.', response_metadata={'token_usage': {'completion_tokens': 106, 'prompt_tokens': 15, 'total_tokens': 121}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-084f5829-a8df-49a6-8c6d-07fb125788c3-0', usage_metadata={'input_tokens': 15, 'output_tokens': 106, 'total_tokens': 121})

We can also guide its response with a prompt template. Prompt templates convert raw user input to better input to the LLM.

```python
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a world class technical documentation writer."),
    ("user", "{input}")
])
```

#### API Reference

- [ChatPromptTemplate](https://api.python.langchain.com/en/latest/prompts/langchain_core.prompts.chat.ChatPromptTemplate.html)

We can now combine these into a simple LLM chain:

```python
chain = prompt | llm 
```

We can now invoke it and ask the same question. It still won't know the answer, but it should respond in a more proper tone for a technical writer!

```python
chain.invoke({"input": "how can langsmith help with testing?"})
```

    AIMessage(content='Langsmith can help with testing in several ways:\n\n1. Automated Testing: Langsmith can generate test cases based on the grammar and language rules defined in the language specification. These test cases can be used to automate testing procedures, ensuring comprehensive coverage and consistency in testing.\n\n2. Test Data Generation: Langsmith can generate realistic and diverse test data for testing different scenarios and edge cases, helping to improve the quality of testing and uncover potential defects in the software.\n\n3. Test Scenario Generation: By defining specific test scenarios in the language specification, Langsmith can assist in generating test scripts or scenarios for different use cases, making it easier to conduct thorough testing of the software.\n\n4. Integration Testing: Langsmith can help in integrating different components or systems by providing a common language and grammar for communication, facilitating better coordination and collaboration in testing activities.\n\nOverall, Langsmith can streamline the testing process, enhance test coverage, and improve the overall quality of software testing by leveraging its language generation capabilities.', response_metadata={'token_usage': {'completion_tokens': 196, 'prompt_tokens': 28, 'total_tokens': 224}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-e0a0f74b-c922-4c1d-949a-bff67be46745-0', usage_metadata={'input_tokens': 28, 'output_tokens': 196, 'total_tokens': 224})

The output of a ChatModel (and therefore, of this chain) is a message. However, it's often much more convenient to work with strings. Let's add a simple output parser to convert the chat message to a string.

```python
from langchain_core.output_parsers import StrOutputParser

output_parser = StrOutputParser()
```

#### API Reference

- [StrOutputParser](https://api.python.langchain.com/en/latest/output_parsers/langchain_core.output_parsers.string.StrOutputParser.html)

We can now add this to the previous chain:

```python
chain = prompt | llm | output_parser
```

We can now invoke it and ask the same question. The answer will now be a string (rather than a ChatMessage).

```python
chain.invoke({"input": "how can langsmith help with testing?"})
```

    'Langsmith can help with testing in various ways through its natural language processing capabilities. Here are some ways Langsmith can assist with testing:\n\n1. **Test Case Generation**: Langsmith can be used to automatically generate test cases based on natural language requirements or specifications. By analyzing the textual requirements, Langsmith can assist in creating comprehensive test cases that cover different scenarios and edge cases.\n\n2. **Test Data Generation**: Langsmith can also help in generating test data for testing purposes. By understanding the context and requirements specified in natural language, Langsmith can create relevant and diverse test data sets to ensure thorough testing coverage.\n\n3. **Automated Testing**: Langsmith can be integrated with testing frameworks and tools to automate testing processes. By leveraging its natural language processing capabilities, Langsmith can interpret test scripts written in natural language and execute them as automated tests.\n\n4. **Defect Analysis**: Langsmith can assist in analyzing defect reports and identifying patterns or common issues in the reported defects. By processing and analyzing defect descriptions in natural language, Langsmith can help in identifying root causes and potential areas for improvement.\n\n5. **Test Reporting and Analysis**: Langsmith can generate reports and perform analysis on test results using natural language processing techniques. It can help in summarizing test outcomes, identifying trends, and providing insights to stakeholders in a clear and understandable format.\n\nOverall, Langsmith can streamline the testing process, improve test coverage, and enhance the efficiency of testing activities by leveraging its natural language processing capabilities.'

### Diving Deeper

We've now successfully set up a basic LLM chain. We only touched on the basics of prompts, models, and output parsers - for a deeper dive into everything mentioned here, see [this section of documentation](https://python.langchain.com/v0.1/docs/modules/model_io/).

## Retrieval Chain

To properly answer the original question ("how can langsmith help with testing?"), we need to provide additional context to the LLM. We can do this via *retrieval*. Retrieval is useful when you have **too much data** to pass to the LLM directly. You can then use a retriever to fetch only the most relevant pieces and pass those in.

In this process, we will look up relevant documents from a *Retriever* and then pass them into the prompt. A Retriever can be backed by anything - a SQL table, the internet, etc - but in this instance, we will populate a vector store and use that as a retriever. For more information on vectorstores, see [this documentation](https://python.langchain.com/v0.1/docs/modules/data_connection/vectorstores/).

First, we need to load the data that we want to index. To do this, we will use the WebBaseLoader. This requires installing [BeautifulSoup](https://beautiful-soup-4.readthedocs.io/en/latest/):

```python
pip install --quiet beautifulsoup4
```

    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m A new release of pip is available: [0m[31;49m23.0.1[0m[39;49m -> [0m[32;49m24.1[0m
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m To update, run: [0m[32;49mpip install --upgrade pip[0m
    Note: you may need to restart the kernel to use updated packages.

After that, we can import and use WebBaseLoader.

```python
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")

docs = loader.load()
```

#### API Reference

- [WebBaseLoader](https://api.python.langchain.com/en/latest/document_loaders/langchain_community.document_loaders.web_base.WebBaseLoader.html)

Next, we need to index it into a vectorstore. This requires a few components, namely an [embedding model](https://python.langchain.com/v0.1/docs/modules/data_connection/text_embedding/) and a [vectorstore](https://python.langchain.com/v0.1/docs/modules/data_connection/vectorstores/).

For embedding models, we once again provide examples for accessing via API or by running local models.

### OpenAI (API)

Make sure you have the `langchain_openai` package installed and the appropriate environment variables set (these are the same as needed for the LLM).

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
```

#### API Reference

- [OpenAIEmbeddings](https://api.python.langchain.com/en/latest/embeddings/langchain_openai.embeddings.base.OpenAIEmbeddings.html)

### Local (using Ollama)

Make sure you have Ollama running (same set up as with the LLM).

```python
from langchain_community.embeddings import OllamaEmbeddings

embeddings = OllamaEmbeddings()
```

#### API Reference

- [OllamaEmbeddings](https://api.python.langchain.com/en/latest/embeddings/langchain_community.embeddings.ollama.OllamaEmbeddings.html)

### Cohere (API)

Make sure you have the `cohere` package installed and the appropriate environment variables set (these are the same as needed for the LLM).

```python
from langchain_cohere.embeddings import CohereEmbeddings

embeddings = CohereEmbeddings()
```

#### API Reference

- [CohereEmbeddings](https://api.python.langchain.com/en/latest/embeddings/langchain_cohere.embeddings.CohereEmbeddings.html)

Now, we can use this embedding model to ingest documents into a vectorstore. We will use a simple local vectorstore, [FAISS](https://python.langchain.com/v0.1/docs/integrations/vectorstores/faiss/), for simplicity's sake.

First, we need to install the required packages for that:

```python
pip install --quiet faiss-cpu
```

    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m A new release of pip is available: [0m[31;49m23.0.1[0m[39;49m -> [0m[32;49m24.1[0m
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m To update, run: [0m[32;49mpip install --upgrade pip[0m
    Note: you may need to restart the kernel to use updated packages.

Then we can build our index:

```python
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)
```

#### API Reference

- [FAISS](https://api.python.langchain.com/en/latest/vectorstores/langchain_community.vectorstores.faiss.FAISS.html)
- [RecursiveCharacterTextSplitter](https://api.python.langchain.com/en/latest/character/langchain_text_splitters.character.RecursiveCharacterTextSplitter.html)

Now that we have this data indexed in a vectorstore, we will create a retrieval chain. This chain will take an incoming question, look up relevant documents, then pass those documents along with the original question into an LLM and ask it to answer the original question.

First, let's set up the chain that takes a question and the retrieved documents and generates an answer.

```python
from langchain.chains.combine_documents import create_stuff_documents_chain

prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)
```

#### API Reference

- [create_stuff_documents_chain](https://api.python.langchain.com/en/latest/chains/langchain.chains.combine_documents.stuff.create_stuff_documents_chain.html)

If we wanted to, we could run this ourselves by passing in documents directly:

```python
from langchain_core.documents import Document

document_chain.invoke({
    "input": "how can langsmith help with testing?",
    "context": [Document(page_content="langsmith can let you visualize test results")]
})
```

    'Langsmith can help visualize test results.'

#### API Reference

- [Document](https://api.python.langchain.com/en/latest/documents/langchain_core.documents.base.Document.html)

However, we want the documents to first come from the retriever we just set up. That way, we can use the retriever to dynamically select the most relevant documents and pass those in for a given question.

```python
from langchain.chains import create_retrieval_chain

retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)
```

#### API Reference

- [create_retrieval_chain](https://api.python.langchain.com/en/latest/chains/langchain.chains.retrieval.create_retrieval_chain.html)

We can now invoke this chain. This returns a dictionary - the response from the LLM is in the `answer` key:

```python
response = retrieval_chain.invoke({"input": "how can langsmith help with testing?"})
print(response["answer"])

# LangSmith offers several features that can help with testing:...
```

    LangSmith can help with testing by allowing developers to create datasets, run tests on their LLM applications using these datasets, and evaluate the test results. Developers can upload test cases in bulk, create them on the fly, or export them from application traces. LangSmith also supports running custom evaluations (LLM and heuristic based) to score test results, making it easier to track and diagnose regressions in test scores across multiple revisions of an application.

This answer should be much more accurate!

### Diving Deeper

We've now successfully set up a basic retrieval chain. We only touched on the basics of retrieval - for a deeper dive into everything mentioned here, see [this section of documentation](https://python.langchain.com/v0.1/docs/modules/data_connection/).

## Conversation Retrieval Chain

The chain we've created so far can only answer single questions. One of the main types of LLM applications that people are building are chat bots. So how do we turn this chain into one that can answer follow-up questions?

We can still use the `create_retrieval_chain` function, but we need to change two things:

1. The retrieval method should now not just work on the most recent input, but rather should take the whole history into account.
2. The final LLM chain should likewise take the whole history into account.

**Updating Retrieval**

In order to update retrieval, we will create a new chain. This chain will take in the most recent input (`input`) and the conversation history (`chat_history`) and use an LLM to generate a search query.

```python
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder

# First we need a prompt that we can pass into an LLM to generate this search query

prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "Given the above conversation, generate a search query to look up to get information relevant to the conversation")
])
retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
```

#### API Reference

- [create_history_aware_retriever](https://api.python.langchain.com/en/latest/chains/langchain.chains.history_aware_retriever.create_history_aware_retriever.html)
- [MessagesPlaceholder](https://api.python.langchain.com/en/latest/prompts/langchain_core.prompts.chat.MessagesPlaceholder.html)

We can test this out by passing in an instance where the user asks a follow-up question.

```python
from langchain_core.messages import HumanMessage, AIMessage

chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
retriever_chain.invoke({
    "chat_history": chat_history,
    "input": "Tell me how"
})
```

    [Document(page_content='LangSmith User Guide | 🦜️🛠️ LangSmith', metadata={'source': 'https://docs.smith.langchain.com/user_guide', 'title': 'LangSmith User Guide | 🦜️🛠️ LangSmith', 'description': 'LangSmith is a platform for LLM application development, monitoring, and testing. In this guide, we’ll highlight the breadth of workflows LangSmith supports and how they fit into each stage of the application development lifecycle. We hope this will inform users how to best utilize this powerful platform or give them something to consider if they’re just starting their journey.', 'language': 'en'}),
     Document(page_content='Skip to main contentGo to API DocsSearchGo to AppQuick StartUser GuideTracingEvaluationProduction Monitoring & AutomationsPrompt HubProxyPricingSelf-HostingCookbookThis is outdated documentation for 🦜️🛠️ LangSmith, which is no longer actively maintained.For up-to-date documentation, see the latest version.User GuideOn this pageLangSmith User GuideLangSmith is a platform for LLM application development, monitoring, and testing. In this guide, we’ll highlight the breadth of workflows LangSmith supports and how they fit into each stage of the application development lifecycle. We hope this will inform users how to best utilize this powerful platform or give them something to consider if they’re just starting their journey.Prototyping\u200bPrototyping LLM applications often involves quick experimentation between prompts, model types, retrieval strategy and other parameters.\nThe ability to rapidly understand how the model is performing — and debug where it is failing — is incredibly important for this phase.Debugging\u200bWhen developing new LLM applications, we suggest having LangSmith tracing enabled by default.\nOftentimes, it isn’t necessary to look at every single trace. However, when things go wrong (an unexpected end result, infinite agent loop, slower than expected execution, higher than expected token usage), it’s extremely helpful to debug by looking through the application traces. LangSmith gives clear visibility and debugging information at each step of an LLM sequence, making it much easier to identify and root-cause issues.\nWe provide native rendering of chat messages, functions, and retrieve documents.Initial Test Set\u200bWhile many developers still ship an initial version of their application based on “vibe checks”, we’ve seen an increasing number of engineering teams start to adopt a more test driven approach. LangSmith allows developers to create datasets, which are collections of inputs and reference outputs, and use these to run tests on their LLM applications.\nThese test cases can be uploaded in bulk, created on the fly, or exported from application traces. LangSmith also makes it easy to run custom evaluations (both LLM and heuristic based) to score test results.Comparison View\u200bWhen prototyping different versions of your applications and making changes, it’s important to see whether or not you’ve regressed with respect to your initial test cases.\nOftentimes, changes in the prompt, retrieval strategy, or model choice can have huge implications in responses produced by your application.\nIn order to get a sense for which variant is performing better, it’s useful to be able to view results for different configurations on the same datapoints side-by-side. We’ve invested heavily in a user-friendly comparison view for test runs to track and diagnose regressions in test scores across multiple revisions of your application.Playground\u200bLangSmith provides a playground environment for rapid iteration and experimentation.\nThis allows you to quickly test out different prompts and models. You can open the playground from any prompt or model run in your trace.', metadata={'source': 'https://docs.smith.langchain.com/user_guide', 'title': 'LangSmith User Guide | 🦜️🛠️ LangSmith', 'description': 'LangSmith is a platform for LLM application development, monitoring, and testing. In this guide, we’ll highlight the breadth of workflows LangSmith supports and how they fit into each stage of the application development lifecycle. We hope this will inform users how to best utilize this powerful platform or give them something to consider if they’re just starting their journey.', 'language': 'en'}),
     Document(page_content='meaning that they involve a series of interactions between the user and the application. LangSmith provides a threads view that groups traces from a single conversation together, making it easier to track the performance of and annotate your application across multiple turns.Was this page helpful?You can leave detailed feedback on GitHub.PreviousQuick StartNextOverviewPrototypingBeta TestingProductionCommunityDiscordTwitterGitHubDocs CodeLangSmith SDKPythonJS/TSMoreHomepageBlogLangChain Python DocsLangChain JS/TS DocsCopyright © 2024 LangChain, Inc.', metadata={'source': 'https://docs.smith.langchain.com/user_guide', 'title': 'LangSmith User Guide | 🦜️🛠️ LangSmith', 'description': 'LangSmith is a platform for LLM application development, monitoring, and testing. In this guide, we’ll highlight the breadth of workflows LangSmith supports and how they fit into each stage of the application development lifecycle. We hope this will inform users how to best utilize this powerful platform or give them something to consider if they’re just starting their journey.', 'language': 'en'}),
     Document(page_content="Every playground run is logged in the system and can be used to create test cases or compare with other runs.Beta Testing\u200bBeta testing allows developers to collect more data on how their LLM applications are performing in real-world scenarios. In this phase, it’s important to develop an understanding for the types of inputs the app is performing well or poorly on and how exactly it’s breaking down in those cases. Both feedback collection and run annotation are critical for this workflow. This will help in curation of test cases that can help track regressions/improvements and development of automatic evaluations.Capturing Feedback\u200bWhen launching your application to an initial set of users, it’s important to gather human feedback on the responses it’s producing. This helps draw attention to the most interesting runs and highlight edge cases that are causing problematic responses. LangSmith allows you to attach feedback scores to logged traces (oftentimes, this is hooked up to a feedback button in your app), then filter on traces that have a specific feedback tag and score. A common workflow is to filter on traces that receive a poor user feedback score, then drill down into problematic points using the detailed trace view.Annotating Traces\u200bLangSmith also supports sending runs to annotation queues, which allow annotators to closely inspect interesting traces and annotate them with respect to different criteria. Annotators can be PMs, engineers, or even subject matter experts. This allows users to catch regressions across important evaluation criteria.Adding Runs to a Dataset\u200bAs your application progresses through the beta testing phase, it's essential to continue collecting data to refine and improve its performance. LangSmith enables you to add runs as examples to datasets (from both the project page and within an annotation queue), expanding your test coverage on real-world scenarios. This is a key benefit in having your logging system and your evaluation/testing system in the same platform.Production\u200bClosely inspecting key data points, growing benchmarking datasets, annotating traces, and drilling down into important data in trace view are workflows you’ll also want to do once your app hits production.However, especially at the production stage, it’s crucial to get a high-level overview of application performance with respect to latency, cost, and feedback scores. This ensures that it's delivering desirable results at scale.Online evaluations and automations allow you to process and score production traces in near real-time.Additionally, threads provide a seamless way to group traces from a single conversation, making it easier to track the performance of your application across multiple turns.Monitoring and A/B Testing\u200bLangSmith provides monitoring charts that allow you to track key metrics over time. You can expand to view metrics for a given period and drill down into a specific data point to get a trace table for that time period — this is especially handy for debugging production issues.LangSmith also allows for tag and metadata grouping, which allows users to mark different versions of their applications with different identifiers and view how they are performing side-by-side within each chart. This is helpful for A/B testing changes in prompt, model, or retrieval strategy.Automations\u200bAutomations are a powerful feature in LangSmith that allow you to perform actions on traces in near real-time. This can be used to automatically score traces, send them to annotation queues, or send them to datasets.To define an automation, simply provide a filter condition, a sampling rate, and an action to perform. Automations are particularly helpful for processing traces at production scale.Threads\u200bMany LLM applications are multi-turn, meaning that they involve a series of interactions between the user and the application. LangSmith provides a threads view that groups traces from a single conversation together, making it easier to", metadata={'source': 'https://docs.smith.langchain.com/user_guide', 'title': 'LangSmith User Guide | 🦜️🛠️ LangSmith', 'description': 'LangSmith is a platform for LLM application development, monitoring, and testing. In this guide, we’ll highlight the breadth of workflows LangSmith supports and how they fit into each stage of the application development lifecycle. We hope this will inform users how to best utilize this powerful platform or give them something to consider if they’re just starting their journey.', 'language': 'en'})]

#### API Reference

- [HumanMessage](https://api.python.langchain.com/en/latest/messages/langchain_core.messages.human.HumanMessage.html)
- [AIMessage](https://api.python.langchain.com/en/latest/messages/langchain_core.messages.ai.AIMessage.html)

You should see that this returns documents about testing in LangSmith. This is because the LLM generated a new query, combining the chat history with the follow-up question.

Now that we have this new retriever, we can create a new chain to continue the conversation with these retrieved documents in mind.

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's questions based on the below context:\n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])
document_chain = create_stuff_documents_chain(llm, prompt)

retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)
```

We can now test this out end-to-end:

```python
chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
retrieval_chain.invoke({
    "chat_history": chat_history,
    "input": "Tell me how"
})
```

    {'chat_history': [HumanMessage(content='Can LangSmith help test my LLM applications?'),
      AIMessage(content='Yes!')],
     'input': 'Tell me how',
     'context': [Document(page_content='Skip to main contentGo to API DocsSearchGo to AppQuick StartUser GuideTracingEvaluationProduction Monitoring & AutomationsPrompt HubProxyPricingSelf-HostingCookbookThis is outdated documentation for 🦜️🛠️ LangSmith, which is no longer actively maintained.For up-to-date documentation, see the latest version.User GuideOn this pageLangSmith User GuideLangSmith is a platform for LLM application development, monitoring, and testing. In this guide, we’ll highlight the breadth of workflows LangSmith supports and how they fit into each stage of the application development lifecycle. We hope this will inform users how to best utilize this powerful platform or give them something to consider if they’re just starting their journey.Prototyping\u200bPrototyping LLM applications often involves quick experimentation between prompts, model types, retrieval strategy and other parameters.\nThe ability to rapidly understand how the model is performing — and debug where it is failing — is incredibly important for this phase.Debugging\u200bWhen developing new LLM applications, we suggest having LangSmith tracing enabled by default.\nOftentimes, it isn’t necessary to look at every single trace. However, when things go wrong (an unexpected end result, infinite agent loop, slower than expected execution, higher than expected token usage), it’s extremely helpful to debug by looking through the application traces. LangSmith gives clear visibility and debugging information at each step of an LLM sequence, making it much easier to identify and root-cause issues.\nWe provide native rendering of chat messages, functions, and retrieve documents.Initial Test Set\u200bWhile many developers still ship an initial version of their application based on “vibe checks”, we’ve seen an increasing number of engineering teams start to adopt a more test driven approach. LangSmith allows developers to create datasets, which are collections of inputs and reference outputs, and use these to run tests on their LLM applications.\nThese test cases can be uploaded in bulk, created on the fly, or exported from application traces. LangSmith also makes it easy to run custom evaluations (both LLM and heuristic based) to score test results.Comparison View\u200bWhen prototyping different versions of your applications and making changes, it’s important to see whether or not you’ve regressed with respect to your initial test cases.\nOftentimes, changes in the prompt, retrieval strategy, or model choice can have huge implications in responses produced by your application.\nIn order to get a sense for which variant is performing better, it’s useful to be able to view results for different configurations on the same datapoints side-by-side. We’ve invested heavily in a user-friendly comparison view for test runs to track and diagnose regressions in test scores across multiple revisions of your application.Playground\u200bLangSmith provides a playground environment for rapid iteration and experimentation.\nThis allows you to quickly test out different prompts and models. You can open the playground from any prompt or model run in your trace.', metadata={'source': 'https://docs.smith.langchain.com/user_guide', 'title': 'LangSmith User Guide | 🦜️🛠️ LangSmith', 'description': 'LangSmith is a platform for LLM application development, monitoring, and testing. In this guide, we’ll highlight the breadth of workflows LangSmith supports and how they fit into each stage of the application development lifecycle. We hope this will inform users how to best utilize this powerful platform or give them something to consider if they’re just starting their journey.', 'language': 'en'}),
      Document(page_content='LangSmith User Guide | 🦜️🛠️ LangSmith', metadata={'source': 'https://docs.smith.langchain.com/user_guide', 'title': 'LangSmith User Guide | 🦜️🛠️ LangSmith', 'description': 'LangSmith is a platform for LLM application development, monitoring, and testing. In this guide, we’ll highlight the breadth of workflows LangSmith supports and how they fit into each stage of the application development lifecycle. We hope this will inform users how to best utilize this powerful platform or give them something to consider if they’re just starting their journey.', 'language': 'en'}),
      Document(page_content="Every playground run is logged in the system and can be used to create test cases or compare with other runs.Beta Testing\u200bBeta testing allows developers to collect more data on how their LLM applications are performing in real-world scenarios. In this phase, it’s important to develop an understanding for the types of inputs the app is performing well or poorly on and how exactly it’s breaking down in those cases. Both feedback collection and run annotation are critical for this workflow. This will help in curation of test cases that can help track regressions/improvements and development of automatic evaluations.Capturing Feedback\u200bWhen launching your application to an initial set of users, it’s important to gather human feedback on the responses it’s producing. This helps draw attention to the most interesting runs and highlight edge cases that are causing problematic responses. LangSmith allows you to attach feedback scores to logged traces (oftentimes, this is hooked up to a feedback button in your app), then filter on traces that have a specific feedback tag and score. A common workflow is to filter on traces that receive a poor user feedback score, then drill down into problematic points using the detailed trace view.Annotating Traces\u200bLangSmith also supports sending runs to annotation queues, which allow annotators to closely inspect interesting traces and annotate them with respect to different criteria. Annotators can be PMs, engineers, or even subject matter experts. This allows users to catch regressions across important evaluation criteria.Adding Runs to a Dataset\u200bAs your application progresses through the beta testing phase, it's essential to continue collecting data to refine and improve its performance. LangSmith enables you to add runs as examples to datasets (from both the project page and within an annotation queue), expanding your test coverage on real-world scenarios. This is a key benefit in having your logging system and your evaluation/testing system in the same platform.Production\u200bClosely inspecting key data points, growing benchmarking datasets, annotating traces, and drilling down into important data in trace view are workflows you’ll also want to do once your app hits production.However, especially at the production stage, it’s crucial to get a high-level overview of application performance with respect to latency, cost, and feedback scores. This ensures that it's delivering desirable results at scale.Online evaluations and automations allow you to process and score production traces in near real-time.Additionally, threads provide a seamless way to group traces from a single conversation, making it easier to track the performance of your application across multiple turns.Monitoring and A/B Testing\u200bLangSmith provides monitoring charts that allow you to track key metrics over time. You can expand to view metrics for a given period and drill down into a specific data point to get a trace table for that time period — this is especially handy for debugging production issues.LangSmith also allows for tag and metadata grouping, which allows users to mark different versions of their applications with different identifiers and view how they are performing side-by-side within each chart. This is helpful for A/B testing changes in prompt, model, or retrieval strategy.Automations\u200bAutomations are a powerful feature in LangSmith that allow you to perform actions on traces in near real-time. This can be used to automatically score traces, send them to annotation queues, or send them to datasets.To define an automation, simply provide a filter condition, a sampling rate, and an action to perform. Automations are particularly helpful for processing traces at production scale.Threads\u200bMany LLM applications are multi-turn, meaning that they involve a series of interactions between the user and the application. LangSmith provides a threads view that groups traces from a single conversation together, making it easier to", metadata={'source': 'https://docs.smith.langchain.com/user_guide', 'title': 'LangSmith User Guide | 🦜️🛠️ LangSmith', 'description': 'LangSmith is a platform for LLM application development, monitoring, and testing. In this guide, we’ll highlight the breadth of workflows LangSmith supports and how they fit into each stage of the application development lifecycle. We hope this will inform users how to best utilize this powerful platform or give them something to consider if they’re just starting their journey.', 'language': 'en'}),
      Document(page_content='meaning that they involve a series of interactions between the user and the application. LangSmith provides a threads view that groups traces from a single conversation together, making it easier to track the performance of and annotate your application across multiple turns.Was this page helpful?You can leave detailed feedback on GitHub.PreviousQuick StartNextOverviewPrototypingBeta TestingProductionCommunityDiscordTwitterGitHubDocs CodeLangSmith SDKPythonJS/TSMoreHomepageBlogLangChain Python DocsLangChain JS/TS DocsCopyright © 2024 LangChain, Inc.', metadata={'source': 'https://docs.smith.langchain.com/user_guide', 'title': 'LangSmith User Guide | 🦜️🛠️ LangSmith', 'description': 'LangSmith is a platform for LLM application development, monitoring, and testing. In this guide, we’ll highlight the breadth of workflows LangSmith supports and how they fit into each stage of the application development lifecycle. We hope this will inform users how to best utilize this powerful platform or give them something to consider if they’re just starting their journey.', 'language': 'en'})],
     'answer': "LangSmith can help you test your LLM applications by providing features such as the ability to create datasets for running tests, uploading test cases in bulk, exporting test cases from application traces, running custom evaluations, and comparing different versions of your applications to track performance. Additionally, LangSmith offers a playground environment for rapid iteration and experimentation, beta testing capabilities for collecting real-world performance data, capturing feedback from users, annotating traces for evaluation, and adding runs to datasets to refine and improve your application's performance. In the production stage, LangSmith allows for monitoring key metrics, A/B testing changes, and setting up automations to process traces in real-time."}

We can see that this gives a coherent answer - we've successfully turned our retrieval chain into a chatbot!

## Agent

We've so far created examples of chains - where each step is known ahead of time. The final thing we will create is an agent - where the LLM decides what steps to take.

**NOTE: for this example we will only show how to create an agent using OpenAI models, as local models are not reliable enough yet.**

One of the first things to do when building an agent is to decide what tools it should have access to. For this example, we will give the agent access to two tools:

1. The retriever we just created. This will let it easily answer questions about LangSmith.
2. A search tool. This will let it easily answer questions that require up-to-date information.

First, let's set up a tool for the retriever we just created:

```python
from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever,
    "langsmith_search",
    "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",
)
```

#### API Reference

- [create_retriever_tool](https://api.python.langchain.com/en/latest/tools/langchain_core.tools.create_retriever_tool.html)

The search tool that we will use is [Tavily](https://python.langchain.com/v0.1/docs/integrations/retrievers/tavily/). This will require an API key (they have a generous free tier). After creating it on their platform, you need to set it as an environment variable:

```bash
export TAVILY_API_KEY=...
```

If you do not want to set up an API key, you can skip creating this tool.

```python
from langchain_community.tools.tavily_search import TavilySearchResults

search = TavilySearchResults()
```

#### API Reference

- [TavilySearchResults](https://api.python.langchain.com/en/latest/tools/langchain_community.tools.tavily_search.tool.TavilySearchResults.html)

We can now create a list of the tools we want to work with:

```python
tools = [retriever_tool, search]
```

Now that we have the tools, we can create an agent to use them. We will go over this pretty quickly - for a deeper dive into what exactly is going on, check out the [Agent's Getting Started documentation](https://python.langchain.com/v0.1/docs/modules/agents/).

Install `langchainhub` first:

```python
pip install --quiet langchainhub
```

    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m A new release of pip is available: [0m[31;49m23.0.1[0m[39;49m -> [0m[32;49m24.1[0m
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m To update, run: [0m[32;49mpip install --upgrade pip[0m
    Note: you may need to restart the kernel to use updated packages.

Install the `langchain-openai` package to interact with OpenAI:

```python
pip install --quiet langchain-openai
```

    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m A new release of pip is available: [0m[31;49m23.0.1[0m[39;49m -> [0m[32;49m24.1[0m
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m To update, run: [0m[32;49mpip install --upgrade pip[0m
    Note: you may need to restart the kernel to use updated packages.

Now we can use it to get a predefined prompt:

```python
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/openai-functions-agent")

# You need to set OPENAI_API_KEY environment variable or pass it as argument `api_key`.
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
```

#### API Reference

- [ChatOpenAI](https://api.python.langchain.com/en/latest/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html)
- [create_openai_functions_agent](https://api.python.langchain.com/en/latest/agents/langchain.agents.openai_functions_agent.base.create_openai_functions_agent.html)
- [AgentExecutor](https://api.python.langchain.com/en/latest/agents/langchain.agents.agent.AgentExecutor.html)

We can now invoke the agent and see how it responds! We can ask it questions about LangSmith:

```python
agent_executor.invoke({"input": "how can langsmith help with testing?"})
```

    [1m> Entering new AgentExecutor chain...[0m
    [32;1m[1;3m
    Invoking: `langsmith_search` with `{'query': 'how can LangSmith help with testing'}`
    
    
    [0m[36;1m[1;3mSkip to main contentGo to API DocsSearchGo to AppQuick StartUser GuideTracingEvaluationProduction Monitoring & AutomationsPrompt HubProxyPricingSelf-HostingCookbookThis is outdated documentation for 🦜️🛠️ LangSmith, which is no longer actively maintained.For up-to-date documentation, see the latest version.User GuideOn this pageLangSmith User GuideLangSmith is a platform for LLM application development, monitoring, and testing. In this guide, we’ll highlight the breadth of workflows LangSmith supports and how they fit into each stage of the application development lifecycle. We hope this will inform users how to best utilize this powerful platform or give them something to consider if they’re just starting their journey.Prototyping​Prototyping LLM applications often involves quick experimentation between prompts, model types, retrieval strategy and other parameters.
    The ability to rapidly understand how the model is performing — and debug where it is failing — is incredibly important for this phase.Debugging​When developing new LLM applications, we suggest having LangSmith tracing enabled by default.
    Oftentimes, it isn’t necessary to look at every single trace. However, when things go wrong (an unexpected end result, infinite agent loop, slower than expected execution, higher than expected token usage), it’s extremely helpful to debug by looking through the application traces. LangSmith gives clear visibility and debugging information at each step of an LLM sequence, making it much easier to identify and root-cause issues.
    We provide native rendering of chat messages, functions, and retrieve documents.Initial Test Set​While many developers still ship an initial version of their application based on “vibe checks”, we’ve seen an increasing number of engineering teams start to adopt a more test driven approach. LangSmith allows developers to create datasets, which are collections of inputs and reference outputs, and use these to run tests on their LLM applications.
    These test cases can be uploaded in bulk, created on the fly, or exported from application traces. LangSmith also makes it easy to run custom evaluations (both LLM and heuristic based) to score test results.Comparison View​When prototyping different versions of your applications and making changes, it’s important to see whether or not you’ve regressed with respect to your initial test cases.
    Oftentimes, changes in the prompt, retrieval strategy, or model choice can have huge implications in responses produced by your application.
    In order to get a sense for which variant is performing better, it’s useful to be able to view results for different configurations on the same datapoints side-by-side. We’ve invested heavily in a user-friendly comparison view for test runs to track and diagnose regressions in test scores across multiple revisions of your application.Playground​LangSmith provides a playground environment for rapid iteration and experimentation.
    This allows you to quickly test out different prompts and models. You can open the playground from any prompt or model run in your trace.
    
    Every playground run is logged in the system and can be used to create test cases or compare with other runs.Beta Testing​Beta testing allows developers to collect more data on how their LLM applications are performing in real-world scenarios. In this phase, it’s important to develop an understanding for the types of inputs the app is performing well or poorly on and how exactly it’s breaking down in those cases. Both feedback collection and run annotation are critical for this workflow. This will help in curation of test cases that can help track regressions/improvements and development of automatic evaluations.Capturing Feedback​When launching your application to an initial set of users, it’s important to gather human feedback on the responses it’s producing. This helps draw attention to the most interesting runs and highlight edge cases that are causing problematic responses. LangSmith allows you to attach feedback scores to logged traces (oftentimes, this is hooked up to a feedback button in your app), then filter on traces that have a specific feedback tag and score. A common workflow is to filter on traces that receive a poor user feedback score, then drill down into problematic points using the detailed trace view.Annotating Traces​LangSmith also supports sending runs to annotation queues, which allow annotators to closely inspect interesting traces and annotate them with respect to different criteria. Annotators can be PMs, engineers, or even subject matter experts. This allows users to catch regressions across important evaluation criteria.Adding Runs to a Dataset​As your application progresses through the beta testing phase, it's essential to continue collecting data to refine and improve its performance. LangSmith enables you to add runs as examples to datasets (from both the project page and within an annotation queue), expanding your test coverage on real-world scenarios. This is a key benefit in having your logging system and your evaluation/testing system in the same platform.Production​Closely inspecting key data points, growing benchmarking datasets, annotating traces, and drilling down into important data in trace view are workflows you’ll also want to do once your app hits production.However, especially at the production stage, it’s crucial to get a high-level overview of application performance with respect to latency, cost, and feedback scores. This ensures that it's delivering desirable results at scale.Online evaluations and automations allow you to process and score production traces in near real-time.Additionally, threads provide a seamless way to group traces from a single conversation, making it easier to track the performance of your application across multiple turns.Monitoring and A/B Testing​LangSmith provides monitoring charts that allow you to track key metrics over time. You can expand to view metrics for a given period and drill down into a specific data point to get a trace table for that time period — this is especially handy for debugging production issues.LangSmith also allows for tag and metadata grouping, which allows users to mark different versions of their applications with different identifiers and view how they are performing side-by-side within each chart. This is helpful for A/B testing changes in prompt, model, or retrieval strategy.Automations​Automations are a powerful feature in LangSmith that allow you to perform actions on traces in near real-time. This can be used to automatically score traces, send them to annotation queues, or send them to datasets.To define an automation, simply provide a filter condition, a sampling rate, and an action to perform. Automations are particularly helpful for processing traces at production scale.Threads​Many LLM applications are multi-turn, meaning that they involve a series of interactions between the user and the application. LangSmith provides a threads view that groups traces from a single conversation together, making it easier to
    
    LangSmith User Guide | 🦜️🛠️ LangSmith
    
    meaning that they involve a series of interactions between the user and the application. LangSmith provides a threads view that groups traces from a single conversation together, making it easier to track the performance of and annotate your application across multiple turns.Was this page helpful?You can leave detailed feedback on GitHub.PreviousQuick StartNextOverviewPrototypingBeta TestingProductionCommunityDiscordTwitterGitHubDocs CodeLangSmith SDKPythonJS/TSMoreHomepageBlogLangChain Python DocsLangChain JS/TS DocsCopyright © 2024 LangChain, Inc.[0m[32;1m[1;3mLangSmith can help with testing in various ways throughout the application development lifecycle:
    
    1. **Prototyping**: LangSmith supports prototyping LLM applications by allowing quick experimentation between prompts, model types, retrieval strategies, and other parameters.
    
    2. **Debugging**: LangSmith provides tracing capabilities that enable developers to debug issues in LLM applications by offering clear visibility and debugging information at each step of the sequence.
    
    3. **Initial Test Set**: Developers can create datasets of inputs and reference outputs to run tests on LLM applications. LangSmith makes it easy to run custom evaluations to score test results.
    
    4. **Comparison View**: LangSmith offers a comparison view for test runs, allowing developers to track and diagnose regressions in test scores across multiple revisions of the application.
    
    5. **Playground**: LangSmith provides a playground environment for rapid iteration and experimentation with different prompts and models.
    
    6. **Beta Testing**: Developers can use LangSmith for beta testing to collect data on how LLM applications perform in real-world scenarios and gather feedback for improvements.
    
    7. **Capturing Feedback**: LangSmith allows developers to gather human feedback on the responses produced by the application and attach feedback scores to logged traces.
    
    8. **Annotating Traces**: LangSmith supports sending runs to annotation queues for annotators to inspect and annotate interesting traces with different criteria.
    
    9. **Adding Runs to a Dataset**: Developers can add runs as examples to datasets to expand test coverage on real-world scenarios and refine the application's performance.
    
    10. **Production**: In the production stage, LangSmith helps in closely inspecting key data points, monitoring application performance metrics, and processing production traces in near real-time.
    
    11. **Monitoring and A/B Testing**: LangSmith provides monitoring charts to track key metrics over time and allows for A/B testing changes in prompt, model, or retrieval strategy.
    
    12. **Automations**: Automations in LangSmith enable developers to perform actions on traces in near real-time, such as scoring traces, sending them to annotation queues, or datasets.
    
    13. **Threads**: LangSmith provides a threads view to group traces from a single conversation together, making it easier to track the performance of the application across multiple turns.
    
    These features in LangSmith can significantly aid in testing and improving the performance of LLM applications during development and production stages.[0m
    
    [1m> Finished chain.[0m





    {'input': 'how can langsmith help with testing?',
     'output': "LangSmith can help with testing in various ways throughout the application development lifecycle:\n\n1. **Prototyping**: LangSmith supports prototyping LLM applications by allowing quick experimentation between prompts, model types, retrieval strategies, and other parameters.\n\n2. **Debugging**: LangSmith provides tracing capabilities that enable developers to debug issues in LLM applications by offering clear visibility and debugging information at each step of the sequence.\n\n3. **Initial Test Set**: Developers can create datasets of inputs and reference outputs to run tests on LLM applications. LangSmith makes it easy to run custom evaluations to score test results.\n\n4. **Comparison View**: LangSmith offers a comparison view for test runs, allowing developers to track and diagnose regressions in test scores across multiple revisions of the application.\n\n5. **Playground**: LangSmith provides a playground environment for rapid iteration and experimentation with different prompts and models.\n\n6. **Beta Testing**: Developers can use LangSmith for beta testing to collect data on how LLM applications perform in real-world scenarios and gather feedback for improvements.\n\n7. **Capturing Feedback**: LangSmith allows developers to gather human feedback on the responses produced by the application and attach feedback scores to logged traces.\n\n8. **Annotating Traces**: LangSmith supports sending runs to annotation queues for annotators to inspect and annotate interesting traces with different criteria.\n\n9. **Adding Runs to a Dataset**: Developers can add runs as examples to datasets to expand test coverage on real-world scenarios and refine the application's performance.\n\n10. **Production**: In the production stage, LangSmith helps in closely inspecting key data points, monitoring application performance metrics, and processing production traces in near real-time.\n\n11. **Monitoring and A/B Testing**: LangSmith provides monitoring charts to track key metrics over time and allows for A/B testing changes in prompt, model, or retrieval strategy.\n\n12. **Automations**: Automations in LangSmith enable developers to perform actions on traces in near real-time, such as scoring traces, sending them to annotation queues, or datasets.\n\n13. **Threads**: LangSmith provides a threads view to group traces from a single conversation together, making it easier to track the performance of the application across multiple turns.\n\nThese features in LangSmith can significantly aid in testing and improving the performance of LLM applications during development and production stages."}

We can ask it about the weather:

```python
agent_executor.invoke({"input": "what is the weather in SF?"})
```

    [1m> Entering new AgentExecutor chain...[0m
    [32;1m[1;3m
    Invoking: `tavily_search_results_json` with `{'query': 'weather in San Francisco'}`
    
    
    [0m[33;1m[1;3m[{'url': 'https://www.accuweather.com/en/us/san-francisco/94103/weather-forecast/347629', 'content': 'Get the current and future weather conditions for San Francisco, CA, including temperature, precipitation, wind, air quality and more. See the hourly and 10-day outlook, radar maps, alerts and allergy information.'}, {'url': 'https://weather.com/weather/tenday/l/San Francisco CA USCA0987:1:US', 'content': "Comfy & Cozy\nThat's Not What Was Expected\nOutside\n'No-Name Storms' In Florida\nGifts From On High\nWhat To Do For Wheezing\nSurviving The Season\nStay Safe\nAir Quality Index\nAir quality is considered satisfactory, and air pollution poses little or no risk.\n Health & Activities\nSeasonal Allergies and Pollen Count Forecast\nNo pollen detected in your area\nCold & Flu Forecast\nFlu risk is low in your area\nWe recognize our responsibility to use data and technology for good. recents\nSpecialty Forecasts\n10 Day Weather-San Francisco, CA\nToday\nMon 18 | Day\nConsiderable cloudiness. Tue 19\nTue 19 | Day\nLight rain early...then remaining cloudy with showers in the afternoon. Wed 27\nWed 27 | Day\nOvercast with rain showers at times."}, {'url': 'https://forecast.weather.gov/zipcity.php?inputstring=San+Francisco,CA', 'content': 'Get the current and extended weather conditions for San Francisco, including temperature, humidity, wind, and visibility. See the detailed forecast for the next week, with sunny, partly cloudy, and mostly clear days.'}, {'url': 'https://www.accuweather.com/en/us/san-francisco/94103/current-weather/347629', 'content': 'Current weather in San Francisco, CA. Check current conditions in San Francisco, CA with radar, hourly, and more.'}, {'url': 'https://www.accuweather.com/en/us/san-francisco/94103/weather-forecast/6-347629_1_al', 'content': 'Get the current weather, air quality, and health and activities information for San Francisco, CA. See the hourly, daily, and monthly forecasts, as well as radar maps and severe weather alerts.'}][0m[32;1m[1;3mYou can check the current and future weather conditions for San Francisco, CA on [AccuWeather](https://www.accuweather.com/en/us/san-francisco/94103/weather-forecast/347629). This includes information on temperature, precipitation, wind, air quality, and more.[0m
    
    [1m> Finished chain.[0m





    {'input': 'what is the weather in SF?',
     'output': 'You can check the current and future weather conditions for San Francisco, CA on [AccuWeather](https://www.accuweather.com/en/us/san-francisco/94103/weather-forecast/347629). This includes information on temperature, precipitation, wind, air quality, and more.'}

We can have conversations with it:

```python
chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
agent_executor.invoke({
    "chat_history": chat_history,
    "input": "Tell me how"
})
```

    [1m> Entering new AgentExecutor chain...[0m
    [32;1m[1;3m
    Invoking: `langsmith_search` with `{'query': 'LangSmith LLM application testing services'}`
    
    
    [0m[36;1m[1;3mLangSmith User Guide | 🦜️🛠️ LangSmith
    
    Skip to main contentGo to API DocsSearchGo to AppQuick StartUser GuideTracingEvaluationProduction Monitoring & AutomationsPrompt HubProxyPricingSelf-HostingCookbookThis is outdated documentation for 🦜️🛠️ LangSmith, which is no longer actively maintained.For up-to-date documentation, see the latest version.User GuideOn this pageLangSmith User GuideLangSmith is a platform for LLM application development, monitoring, and testing. In this guide, we’ll highlight the breadth of workflows LangSmith supports and how they fit into each stage of the application development lifecycle. We hope this will inform users how to best utilize this powerful platform or give them something to consider if they’re just starting their journey.Prototyping​Prototyping LLM applications often involves quick experimentation between prompts, model types, retrieval strategy and other parameters.
    The ability to rapidly understand how the model is performing — and debug where it is failing — is incredibly important for this phase.Debugging​When developing new LLM applications, we suggest having LangSmith tracing enabled by default.
    Oftentimes, it isn’t necessary to look at every single trace. However, when things go wrong (an unexpected end result, infinite agent loop, slower than expected execution, higher than expected token usage), it’s extremely helpful to debug by looking through the application traces. LangSmith gives clear visibility and debugging information at each step of an LLM sequence, making it much easier to identify and root-cause issues.
    We provide native rendering of chat messages, functions, and retrieve documents.Initial Test Set​While many developers still ship an initial version of their application based on “vibe checks”, we’ve seen an increasing number of engineering teams start to adopt a more test driven approach. LangSmith allows developers to create datasets, which are collections of inputs and reference outputs, and use these to run tests on their LLM applications.
    These test cases can be uploaded in bulk, created on the fly, or exported from application traces. LangSmith also makes it easy to run custom evaluations (both LLM and heuristic based) to score test results.Comparison View​When prototyping different versions of your applications and making changes, it’s important to see whether or not you’ve regressed with respect to your initial test cases.
    Oftentimes, changes in the prompt, retrieval strategy, or model choice can have huge implications in responses produced by your application.
    In order to get a sense for which variant is performing better, it’s useful to be able to view results for different configurations on the same datapoints side-by-side. We’ve invested heavily in a user-friendly comparison view for test runs to track and diagnose regressions in test scores across multiple revisions of your application.Playground​LangSmith provides a playground environment for rapid iteration and experimentation.
    This allows you to quickly test out different prompts and models. You can open the playground from any prompt or model run in your trace.
    
    Every playground run is logged in the system and can be used to create test cases or compare with other runs.Beta Testing​Beta testing allows developers to collect more data on how their LLM applications are performing in real-world scenarios. In this phase, it’s important to develop an understanding for the types of inputs the app is performing well or poorly on and how exactly it’s breaking down in those cases. Both feedback collection and run annotation are critical for this workflow. This will help in curation of test cases that can help track regressions/improvements and development of automatic evaluations.Capturing Feedback​When launching your application to an initial set of users, it’s important to gather human feedback on the responses it’s producing. This helps draw attention to the most interesting runs and highlight edge cases that are causing problematic responses. LangSmith allows you to attach feedback scores to logged traces (oftentimes, this is hooked up to a feedback button in your app), then filter on traces that have a specific feedback tag and score. A common workflow is to filter on traces that receive a poor user feedback score, then drill down into problematic points using the detailed trace view.Annotating Traces​LangSmith also supports sending runs to annotation queues, which allow annotators to closely inspect interesting traces and annotate them with respect to different criteria. Annotators can be PMs, engineers, or even subject matter experts. This allows users to catch regressions across important evaluation criteria.Adding Runs to a Dataset​As your application progresses through the beta testing phase, it's essential to continue collecting data to refine and improve its performance. LangSmith enables you to add runs as examples to datasets (from both the project page and within an annotation queue), expanding your test coverage on real-world scenarios. This is a key benefit in having your logging system and your evaluation/testing system in the same platform.Production​Closely inspecting key data points, growing benchmarking datasets, annotating traces, and drilling down into important data in trace view are workflows you’ll also want to do once your app hits production.However, especially at the production stage, it’s crucial to get a high-level overview of application performance with respect to latency, cost, and feedback scores. This ensures that it's delivering desirable results at scale.Online evaluations and automations allow you to process and score production traces in near real-time.Additionally, threads provide a seamless way to group traces from a single conversation, making it easier to track the performance of your application across multiple turns.Monitoring and A/B Testing​LangSmith provides monitoring charts that allow you to track key metrics over time. You can expand to view metrics for a given period and drill down into a specific data point to get a trace table for that time period — this is especially handy for debugging production issues.LangSmith also allows for tag and metadata grouping, which allows users to mark different versions of their applications with different identifiers and view how they are performing side-by-side within each chart. This is helpful for A/B testing changes in prompt, model, or retrieval strategy.Automations​Automations are a powerful feature in LangSmith that allow you to perform actions on traces in near real-time. This can be used to automatically score traces, send them to annotation queues, or send them to datasets.To define an automation, simply provide a filter condition, a sampling rate, and an action to perform. Automations are particularly helpful for processing traces at production scale.Threads​Many LLM applications are multi-turn, meaning that they involve a series of interactions between the user and the application. LangSmith provides a threads view that groups traces from a single conversation together, making it easier to
    
    meaning that they involve a series of interactions between the user and the application. LangSmith provides a threads view that groups traces from a single conversation together, making it easier to track the performance of and annotate your application across multiple turns.Was this page helpful?You can leave detailed feedback on GitHub.PreviousQuick StartNextOverviewPrototypingBeta TestingProductionCommunityDiscordTwitterGitHubDocs CodeLangSmith SDKPythonJS/TSMoreHomepageBlogLangChain Python DocsLangChain JS/TS DocsCopyright © 2024 LangChain, Inc.[0m[32;1m[1;3mLangSmith is a platform for LLM application development, monitoring, and testing. It supports various workflows that can help you test your LLM applications effectively. Here are some key features and workflows supported by LangSmith:
    
    1. Prototyping: Allows quick experimentation between prompts, model types, retrieval strategy, and other parameters.
    2. Debugging: Provides clear visibility and debugging information at each step of an LLM sequence to identify and root-cause issues.
    3. Initial Test Set: Enables developers to create datasets of inputs and reference outputs to run tests on LLM applications.
    4. Comparison View: Helps track and diagnose regressions in test scores across multiple revisions of your application.
    5. Playground: Provides a rapid iteration and experimentation environment to test different prompts and models.
    6. Beta Testing: Collects data on how LLM applications perform in real-world scenarios and helps in refining the application.
    7. Capturing Feedback: Allows gathering human feedback on the responses produced by the application.
    8. Annotating Traces: Supports sending runs to annotation queues for close inspection and annotation.
    9. Production Monitoring: Provides a high-level overview of application performance with respect to latency, cost, and feedback scores.
    10. Monitoring and A/B Testing: Allows tracking key metrics over time and A/B testing changes in prompt, model, or retrieval strategy.
    11. Automations: Enables performing actions on traces in near real-time, such as scoring traces, sending them to annotation queues, or datasets.
    12. Threads: Groups traces from a single conversation together to track the performance of the application across multiple turns.
    
    These features and workflows in LangSmith can assist you in testing your LLM applications effectively.[0m
    
    [1m> Finished chain.[0m





    {'chat_history': [HumanMessage(content='Can LangSmith help test my LLM applications?'),
      AIMessage(content='Yes!')],
     'input': 'Tell me how',
     'output': 'LangSmith is a platform for LLM application development, monitoring, and testing. It supports various workflows that can help you test your LLM applications effectively. Here are some key features and workflows supported by LangSmith:\n\n1. Prototyping: Allows quick experimentation between prompts, model types, retrieval strategy, and other parameters.\n2. Debugging: Provides clear visibility and debugging information at each step of an LLM sequence to identify and root-cause issues.\n3. Initial Test Set: Enables developers to create datasets of inputs and reference outputs to run tests on LLM applications.\n4. Comparison View: Helps track and diagnose regressions in test scores across multiple revisions of your application.\n5. Playground: Provides a rapid iteration and experimentation environment to test different prompts and models.\n6. Beta Testing: Collects data on how LLM applications perform in real-world scenarios and helps in refining the application.\n7. Capturing Feedback: Allows gathering human feedback on the responses produced by the application.\n8. Annotating Traces: Supports sending runs to annotation queues for close inspection and annotation.\n9. Production Monitoring: Provides a high-level overview of application performance with respect to latency, cost, and feedback scores.\n10. Monitoring and A/B Testing: Allows tracking key metrics over time and A/B testing changes in prompt, model, or retrieval strategy.\n11. Automations: Enables performing actions on traces in near real-time, such as scoring traces, sending them to annotation queues, or datasets.\n12. Threads: Groups traces from a single conversation together to track the performance of the application across multiple turns.\n\nThese features and workflows in LangSmith can assist you in testing your LLM applications effectively.'}

### Diving Deeper

We've now successfully set up a basic agent. We only touched on the basics of agents - for a deeper dive into everything mentioned here, see [this section of documentation](https://python.langchain.com/v0.1/docs/modules/agents/).

## Serving with LangServe

Now that we've built an application, we need to serve it. That's where LangServe comes in. LangServe helps developers deploy LangChain chains as a REST API. You do not need to use LangServe to use LangChain, but in this guide, we'll show how you can deploy your app with LangServe.

While the first part of this guide was intended to be run in a Jupyter Notebook, we will now move out of that. We will be creating a Python file and then interacting with it from the command line.

Install with:

```python
pip install --quiet "langserve[all]"
```

    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m A new release of pip is available: [0m[31;49m23.0.1[0m[39;49m -> [0m[32;49m24.1[0m
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m To update, run: [0m[32;49mpip install --upgrade pip[0m
    Note: you may need to restart the kernel to use updated packages.

### Server

To create a server for our application we'll make a `serve.py` file. This will contain our logic for serving our application. It consists of three things:

1. The definition of our chain that we just built above.
2. Our FastAPI app.
3. A definition of a route from which to serve the chain, which is done with `langserve.add_routes`.

```python
#!/usr/bin/env python
from typing import List

from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.messages import BaseMessage
from langserve import add_routes

# 1. Load Retriever
loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
embeddings = OpenAIEmbeddings()
vector = FAISS.from_documents(documents, embeddings)
retriever = vector.as_retriever()

# 2. Create Tools
retriever_tool = create_retriever_tool(
    retriever,
    "langsmith_search",
    "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",
)
search = TavilySearchResults()
tools = [retriever_tool, search]

# 3. Create Agent
prompt = hub.pull("hwchase17/openai-functions-agent")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 4. App definition
app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple API server using LangChain's Runnable interfaces",
)

# 5. Adding chain route

# We need to add these input/output schemas because the current AgentExecutor
# is lacking in schemas.

class Input(BaseModel):
    input: str
    chat_history: List[BaseMessage] = Field(
        ...,
        extra={"widget": {"type": "chat", "input": "location"}},
    )


class Output(BaseModel):
    output: str

add_routes(
    app,
    agent_executor.with_types(input_type=Input, output_type=Output),
    path="/agent",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
```

#### API Reference

- [ChatPromptTemplate](https://api.python.langchain.com/en/latest/prompts/langchain_core.prompts.chat.ChatPromptTemplate.html)
- [ChatOpenAI](https://api.python.langchain.com/en/latest/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html)
- [WebBaseLoader](https://api.python.langchain.com/en/latest/document_loaders/langchain_community.document_loaders.web_base.WebBaseLoader.html)
- [OpenAIEmbeddings](https://api.python.langchain.com/en/latest/embeddings/langchain_openai.embeddings.base.OpenAIEmbeddings.html)
- [FAISS](https://api.python.langchain.com/en/latest/vectorstores/langchain_community.vectorstores.faiss.FAISS.html)
- [RecursiveCharacterTextSplitter](https://api.python.langchain.com/en/latest/character/langchain_text_splitters.character.RecursiveCharacterTextSplitter.html)
- [create_retriever_tool](https://api.python.langchain.com/en/latest/tools/langchain_core.tools.create_retriever_tool.html)
- [TavilySearchResults](https://api.python.langchain.com/en/latest/tools/langchain_community.tools.tavily_search.tool.TavilySearchResults.html)

- [create_openai_functions_agent](https://api.python.langchain.com/en/latest/agents/langchain.agents.openai_functions_agent.base.create_openai_functions_agent.html)
- [AgentExecutor](https://api.python.langchain.com/en/latest/agents/langchain.agents.agent.AgentExecutor.html)
- [BaseMessage](https://api.python.langchain.com/en/latest/messages/langchain_core.messages.base.BaseMessage.html)

And that's it! If we execute this file:

```bash
python serve.py
```

we should see our chain being served at localhost:8000.

### Playground

Every LangServe service comes with a simple built-in UI for configuring and invoking the application with streaming output and visibility into intermediate steps. Head to [http://localhost:8000/agent/playground/](http://localhost:8000/agent/playground/) to try it out! Pass in the same question as before - "how can langsmith help with testing?" - and it should respond the same as before.

### Client

Now let's set up a client for programmatically interacting with our service. We can easily do this with the [langserve.RemoteRunnable](https://python.langchain.com/v0.2/docs/langserve#client). Using this, we can interact with the served chain as if it were running client-side.

```python
from langserve import RemoteRunnable

remote_chain = RemoteRunnable("http://localhost:8000/agent/")
remote_chain.invoke({
    "input": "how can langsmith help with testing?",
    "chat_history": []  # Providing an empty list as this is the first call
})
```

To learn more about the many other features of LangServe, [head here](https://python.langchain.com/v0.1/docs/langserve/).

## Next steps

We've touched on how to build an application with LangChain, how to trace it with LangSmith, and how to serve it with LangServe. There are a lot more features in all three of these than we can cover here. To continue on your journey, we recommend you read the following (in order):

- All of these features are backed by [LangChain Expression Language (LCEL)](https://python.langchain.com/v0.1/docs/expression_language/) - a way to chain these components together. Check out that documentation to better understand how to create custom chains.
- [Model IO](https://python.langchain.com/v0.1/docs/modules/model_io/) covers more details of prompts, LLMs, and output parsers.
- [Retrieval](https://python.langchain.com/v0.1/docs/modules/data_connection/) covers more details of everything related to retrieval.
- [Agents](https://python.langchain.com/v0.1/docs/modules/agents/) covers details of everything related to agents.
- Explore common [end-to-end use cases](https://python.langchain.com/v0.1/docs/use_cases/) and [template applications](https://python.langchain.com/v0.1/docs/templates/).
- Read up on [LangSmith](https://python.langchain.com/v0.1/docs/langsmith/), the platform for debugging, testing, monitoring and more.
- Learn more about serving your applications with [LangServe](https://python.langchain.com/v0.1/docs/langserve/).
