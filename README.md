# LangChain

## Basics

### Build a Simple LLM Application with LCEL

```python
#!/usr/bin/env python
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langserve import add_routes

# 0. Load environment variables from a .env file
load_dotenv()

# 1. Create prompt template
system_template = "Translate the following into {language}:"
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

# 2. Create model
model = ChatOpenAI()

# 3. Create parser
parser = StrOutputParser()

# 4. Create chain
chain = prompt_template | model | parser

# 5. App definition
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server using LangChain's Runnable interfaces",
)

# 6. Adding chain route
add_routes(
    app,
    chain,
    path="/chain",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
```

```python
from langserve import RemoteRunnable

remote_chain = RemoteRunnable("http://localhost:8000/chain/")
remote_chain.invoke({"language": "italian", "text": "hi"})
```

    'Ciao'

### Build a Chatbot

```python
from operator import itemgetter

from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

# Load environment variables from a .env file
load_dotenv()

# Initialize the OpenAI model with the gpt-3.5-turbo model
model = ChatOpenAI(model="gpt-3.5-turbo")

# Initialize a dictionary to store session histories
store = {}


# Function to retrieve or create a session history
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:

        store[session_id] = ChatMessageHistory()

    return store[session_id]


# Define a chat prompt template with system and placeholder messages
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Define a trimmer to trim messages to a maximum token count
trimmer = trim_messages(
    max_tokens=65,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

# Define the initial set of messages
messages = [
    SystemMessage(content="you're a good assistant"),
    HumanMessage(content="hi! I'm bob"),
    AIMessage(content="hi!"),
    HumanMessage(content="I like vanilla ice cream"),
    AIMessage(content="nice"),
    HumanMessage(content="whats 2 + 2"),
    AIMessage(content="4"),
    HumanMessage(content="thanks"),
    AIMessage(content="no problem!"),
    HumanMessage(content="having fun?"),
    AIMessage(content="yes!"),
    HumanMessage(content="hi! I'm bob"),
]

# Create a runnable chain that processes messages
chain = (
    RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer)
    | prompt
    | model
)

# Create a runnable with message history, binding the chain with the session history function
with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages",
)

# Configuration dictionary for the session
config = {"configurable": {"session_id": "abc16"}}

# Stream responses by passing messages and language configuration to the runnable with message history
for r in with_message_history.stream(
    {
        "messages": messages + [HumanMessage(content="whats my name?")],
        "language": "English",
    },
    config=config,
):
    print(r.content, end="|")
```

    |Your| name| is| Bob|!||

### Build vector stores and retrievers

```python
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Load environment variables from a .env file
load_dotenv()

# Create a list of documents, each with content and metadata
documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Goldfish are popular pets for beginners, requiring relatively simple care.",
        metadata={"source": "fish-pets-doc"},
    ),
    Document(
        page_content="Parrots are intelligent birds capable of mimicking human speech.",
        metadata={"source": "bird-pets-doc"},
    ),
    Document(
        page_content="Rabbits are social animals that need plenty of space to hop around.",
        metadata={"source": "mammal-pets-doc"},
    ),
]

# Create a Chroma vector store from the documents, using OpenAI embeddings
vectorstore = Chroma.from_documents(
    documents,
    embedding=OpenAIEmbeddings(),
)

# Create a retriever from the vector store for similarity-based search
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},  # Retrieve the top 1 similar document
)

# Initialize the OpenAI model with the specified version
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

# Define a message template for the chat prompt
message = """
Answer this question using the provided context only.

{question}

Context:
{context}
"""

# Create a chat prompt template from the message
prompt = ChatPromptTemplate.from_messages([("human", message)])

# Define the RAG (Retrieval-Augmented Generation) chain with context retriever and question passthrough
rag_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm

# Invoke the RAG chain with a question about cats
response = rag_chain.invoke("tell me about cats")

# Print the response content
print(response.content)
```

    Cats are independent pets that often enjoy their own space.

### Build an Agent

```python
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import create_react_agent

# Load environment variables from a .env file
load_dotenv()

# Initialize the OpenAI chat model
model = ChatOpenAI()

# Initialize the search tool with a limit of 2 results per query
search = TavilySearchResults(max_results=2)
tools = [search]

# Initialize an in-memory SQLite database for saving agent state
memory = SqliteSaver.from_conn_string(":memory:")

# Create a reactive agent executor with the model, tools, and memory
agent_executor = create_react_agent(model, tools, checkpointer=memory)

# Configuration for the agent's execution, including a thread ID
config = {"configurable": {"thread_id": "abc123"}}

# Execute the agent with a greeting message and print the response chunks
for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="hi im bob! and i live in sf")]}, config
):
    print(chunk)
    print("----")

# Execute the agent with a conversational memory query and print the response chunks
for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="whats the weather where I live?")]}, config
):
    print(chunk)
    print("----")
```

    {'agent': {'messages': [AIMessage(content='Hello Bob! How can I assist you today regarding San Francisco?', response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 90, 'total_tokens': 104}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-e1908de0-59d3-476e-b7d7-80b0ac9057a7-0', usage_metadata={'input_tokens': 90, 'output_tokens': 14, 'total_tokens': 104})]}}
    ----
    {'agent': {'messages': [AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_Skjm3emip3mCJmAWIriG5g8t', 'function': {'arguments': '{"query":"current weather in San Francisco"}', 'name': 'tavily_search_results_json'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 22, 'prompt_tokens': 119, 'total_tokens': 141}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-1053d174-ba3a-45a0-8445-696e109f8fee-0', tool_calls=[{'name': 'tavily_search_results_json', 'args': {'query': 'current weather in San Francisco'}, 'id': 'call_Skjm3emip3mCJmAWIriG5g8t'}], usage_metadata={'input_tokens': 119, 'output_tokens': 22, 'total_tokens': 141})]}}
    ----
    {'tools': {'messages': [ToolMessage(content='[{"url": "https://www.weatherapi.com/", "content": "{\'location\': {\'name\': \'San Francisco\', \'region\': \'California\', \'country\': \'United States of America\', \'lat\': 37.78, \'lon\': -122.42, \'tz_id\': \'America/Los_Angeles\', \'localtime_epoch\': 1719435980, \'localtime\': \'2024-06-26 14:06\'}, \'current\': {\'last_updated_epoch\': 1719435600, \'last_updated\': \'2024-06-26 14:00\', \'temp_c\': 17.8, \'temp_f\': 64.0, \'is_day\': 1, \'condition\': {\'text\': \'Partly cloudy\', \'icon\': \'//cdn.weatherapi.com/weather/64x64/day/116.png\', \'code\': 1003}, \'wind_mph\': 17.4, \'wind_kph\': 28.1, \'wind_degree\': 290, \'wind_dir\': \'WNW\', \'pressure_mb\': 1017.0, \'pressure_in\': 30.04, \'precip_mm\': 0.0, \'precip_in\': 0.0, \'humidity\': 63, \'cloud\': 25, \'feelslike_c\': 17.8, \'feelslike_f\': 64.0, \'windchill_c\': 16.4, \'windchill_f\': 61.5, \'heatindex_c\': 16.4, \'heatindex_f\': 61.6, \'dewpoint_c\': 10.5, \'dewpoint_f\': 50.9, \'vis_km\': 16.0, \'vis_miles\': 9.0, \'uv\': 5.0, \'gust_mph\': 17.9, \'gust_kph\': 28.9}}"}, {"url": "https://www.wunderground.com/hourly/us/ca/san-francisco/date/2024-6-26", "content": "Current Weather for Popular Cities . San Francisco, CA 58 \\u00b0 F Fair; Manhattan, NY warning 73 \\u00b0 F Clear; Schiller Park, IL (60176) warning 76 \\u00b0 F Mostly Cloudy; Boston, MA 65 \\u00b0 F Cloudy ..."}]', name='tavily_search_results_json', tool_call_id='call_Skjm3emip3mCJmAWIriG5g8t')]}}
    ----
    {'agent': {'messages': [AIMessage(content="The current weather in San Francisco is as follows:\n- Temperature: 64.0°F (17.8°C)\n- Condition: Partly cloudy\n- Wind: 28.1 km/h from WNW\n- Humidity: 63%\n- Visibility: 9.0 miles\n- UV Index: 5.0\n\nIf you'd like more detailed information, you can visit [Weather API](https://www.weatherapi.com/).", response_metadata={'token_usage': {'completion_tokens': 93, 'prompt_tokens': 668, 'total_tokens': 761}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-708e2d35-47c9-49f7-9615-218ec233eede-0', usage_metadata={'input_tokens': 668, 'output_tokens': 93, 'total_tokens': 761})]}}
    ----

## Working with external knowledge

### Build a Retrieval Augmented Generation (RAG) Application

```python
import bs4
from dotenv import load_dotenv
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables from a .env file
load_dotenv()

# Initialize the OpenAI model with the specified version
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

# Load, chunk, and index the contents of the blog

# Define a web base loader to load the contents of the specified blog URL
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=(
                "post-content",
                "post-title",
                "post-header",
            )  # Specify the classes to parse
        )
    ),
)

# Load the documents from the web page
docs = loader.load()

# Define a text splitter to chunk the documents into smaller pieces
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Split the loaded documents into chunks
splits = text_splitter.split_documents(docs)

# Create a Chroma vector store from the document chunks, using OpenAI embeddings
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# Retrieve and generate using the relevant snippets of the blog

# Create a retriever from the vector store for similarity-based search
retriever = vectorstore.as_retriever()

# Pull a predefined prompt template from the hub
prompt = hub.pull("rlm/rag-prompt")


# Function to format the documents into a single string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Define the RAG (Retrieval-Augmented Generation) chain with context retriever, question passthrough, prompt, and LLM
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Invoke the RAG chain with a question about Task Decomposition and print the response
print(rag_chain.invoke("What is Task Decomposition?"))
```

    USER_AGENT environment variable not set, consider setting it to identify your requests.


    Task decomposition is a technique used to break down complex tasks into smaller and simpler steps. This method allows for better planning and execution of tasks by transforming big tasks into more manageable ones. It can be done using prompting techniques like Chain of Thought and Tree of Thoughts to guide the model in decomposing tasks effectively.

### Build a Conversational RAG Application

#### Chains

```python
import bs4
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables from a .env file
load_dotenv()

# Initialize the OpenAI model with the specified version and temperature
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Construct retriever

# Define a web base loader to load the contents of the specified blog URL
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=(
                "post-content",
                "post-title",
                "post-header",
            )  # Specify the classes to parse
        )
    ),
)

# Load the documents from the web page
docs = loader.load()

# Define a text splitter to chunk the documents into smaller pieces
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Split the loaded documents into chunks
splits = text_splitter.split_documents(docs)

# Create a Chroma vector store from the document chunks, using OpenAI embeddings
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# Create a retriever from the vector store for similarity-based search
retriever = vectorstore.as_retriever()

# Contextualize question

# Define a system prompt for contextualizing the question
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

# Create a chat prompt template for contextualizing the question
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create a history-aware retriever using the LLM and the contextualize question prompt
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Answer question

# Define a system prompt for answering the question
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

# Create a chat prompt template for answering the question
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create a chain to combine retrieved documents and the question-answering task
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Create a retrieval-augmented generation (RAG) chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Statefully manage chat history

# Initialize a dictionary to store session histories
store = {}


# Function to retrieve or create a session history
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# Create a runnable with message history for the RAG chain
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# Invoke the conversational RAG chain with the first query
print(
    conversational_rag_chain.invoke(
        {"input": "What is Task Decomposition?"},
        config={
            "configurable": {"session_id": "abc123"}
        },  # Constructs a key "abc123" in `store`.
    )["answer"]
)

# Invoke the conversational RAG chain with the second query
print(
    conversational_rag_chain.invoke(
        {"input": "What are common ways of doing it?"},
        config={"configurable": {"session_id": "abc123"}},
    )["answer"]
)
```

    USER_AGENT environment variable not set, consider setting it to identify your requests.


    Task decomposition is a technique used to break down complex tasks into smaller and simpler steps. This approach helps agents or models handle difficult tasks by dividing them into more manageable subtasks. Task decomposition can be achieved through methods like Chain of Thought (CoT) or Tree of Thoughts, which guide the model in thinking step by step or exploring multiple reasoning possibilities at each step.
    Task decomposition can be done in common ways such as using Language Model (LLM) with simple prompting like "Steps for XYZ" or "What are the subgoals for achieving XYZ?", providing task-specific instructions like "Write a story outline" for writing a novel, or incorporating human inputs to guide the decomposition process. These methods help in breaking down complex tasks into smaller, more manageable steps for better handling and understanding.

#### Agents

```python
import bs4
from dotenv import load_dotenv
from langchain.tools.retriever import create_retriever_tool
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import create_react_agent

# Load environment variables from a .env file
load_dotenv()

# Initialize an in-memory SQLite saver for storing checkpoints
memory = SqliteSaver.from_conn_string(":memory:")

# Initialize the OpenAI model with the specified version and temperature
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Load, chunk, and index the contents of the blog

# Define a web base loader to load the contents of the specified blog URL
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=(
                "post-content",
                "post-title",
                "post-header",
            )  # Specify the classes to parse
        )
    ),
)

# Load the documents from the web page
docs = loader.load()

# Define a text splitter to chunk the documents into smaller pieces
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Split the loaded documents into chunks
splits = text_splitter.split_documents(docs)

# Create a Chroma vector store from the document chunks, using OpenAI embeddings
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# Create a retriever from the vector store for similarity-based search
retriever = vectorstore.as_retriever()

# Build retriever tool
tool = create_retriever_tool(
    retriever,
    "blog_post_retriever",  # Name of the retriever tool
    "Searches and returns excerpts from the Autonomous Agents blog post.",  # Description of the retriever tool
)
tools = [tool]

# Create an agent executor with the LLM, tools, and checkpoint saver
agent_executor = create_react_agent(llm, tools, checkpointer=memory)

# Configuration dictionary for the session
config = {"configurable": {"thread_id": "abc123"}}

# Define the first query
query = "What is Task Decomposition?"

# Stream responses for the query using the agent executor
for s in agent_executor.stream(
    {"messages": [HumanMessage(content=query)]}, config=config
):
    # Print each response and a separator
    print(s)
    print("----")

# Define the second query
query = "What according to the blog post are common ways of doing it? redo the search"

# Stream responses for the conversational query using the agent executor
for s in agent_executor.stream(
    {"messages": [HumanMessage(content=query)]}, config=config
):
    # Print each response and a separator
    print(s)
    print("----")
```

    USER_AGENT environment variable not set, consider setting it to identify your requests.


    {'agent': {'messages': [AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_gK0PUUXkWMr3yXzVSqS7y9A7', 'function': {'arguments': '{"query":"Task Decomposition"}', 'name': 'blog_post_retriever'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 19, 'prompt_tokens': 68, 'total_tokens': 87}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-cbeb78ad-63c6-4415-bc4d-85cc877856e0-0', tool_calls=[{'name': 'blog_post_retriever', 'args': {'query': 'Task Decomposition'}, 'id': 'call_gK0PUUXkWMr3yXzVSqS7y9A7'}], usage_metadata={'input_tokens': 68, 'output_tokens': 19, 'total_tokens': 87})]}}
    ----
    {'tools': {'messages': [ToolMessage(content='Fig. 1. Overview of a LLM-powered autonomous agent system.\nComponent One: Planning#\nA complicated task usually involves many steps. An agent needs to know what they are and plan ahead.\nTask Decomposition#\nChain of thought (CoT; Wei et al. 2022) has become a standard prompting technique for enhancing model performance on complex tasks. The model is instructed to “think step by step” to utilize more test-time computation to decompose hard tasks into smaller and simpler steps. CoT transforms big tasks into multiple manageable tasks and shed lights into an interpretation of the model’s thinking process.\n\nTree of Thoughts (Yao et al. 2023) extends CoT by exploring multiple reasoning possibilities at each step. It first decomposes the problem into multiple thought steps and generates multiple thoughts per step, creating a tree structure. The search process can be BFS (breadth-first search) or DFS (depth-first search) with each state evaluated by a classifier (via a prompt) or majority vote.\nTask decomposition can be done (1) by LLM with simple prompting like "Steps for XYZ.\\n1.", "What are the subgoals for achieving XYZ?", (2) by using task-specific instructions; e.g. "Write a story outline." for writing a novel, or (3) with human inputs.\n\n(3) Task execution: Expert models execute on the specific tasks and log results.\nInstruction:\n\nWith the input and the inference results, the AI assistant needs to describe the process and results. The previous stages can be formed as - User Input: {{ User Input }}, Task Planning: {{ Tasks }}, Model Selection: {{ Model Assignment }}, Task Execution: {{ Predictions }}. You must first answer the user\'s request in a straightforward manner. Then describe the task process and show your analysis and model inference results to the user in the first person. If inference results contain a file path, must tell the user the complete file path.\n\nFig. 11. Illustration of how HuggingGPT works. (Image source: Shen et al. 2023)\nThe system comprises of 4 stages:\n(1) Task planning: LLM works as the brain and parses the user requests into multiple tasks. There are four attributes associated with each task: task type, ID, dependencies, and arguments. They use few-shot examples to guide LLM to do task parsing and planning.\nInstruction:', name='blog_post_retriever', tool_call_id='call_gK0PUUXkWMr3yXzVSqS7y9A7')]}}
    ----
    {'agent': {'messages': [AIMessage(content='Task decomposition is a technique used to break down complex tasks into smaller and simpler steps. This approach helps autonomous agents in planning and executing tasks more effectively. One common method for task decomposition is the Chain of Thought (CoT) technique, which prompts the model to think step by step and decompose hard tasks into manageable steps. Another extension of CoT is the Tree of Thoughts, which explores multiple reasoning possibilities at each step by creating a tree structure of thought steps.\n\nTask decomposition can be achieved through various methods, such as using language models with simple prompting, task-specific instructions, or human inputs. By breaking down tasks into smaller components, autonomous agents can better plan and execute tasks efficiently.\n\nIf you would like more detailed information or examples related to task decomposition, feel free to ask!', response_metadata={'token_usage': {'completion_tokens': 157, 'prompt_tokens': 588, 'total_tokens': 745}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-9b513333-2e2a-4505-b2b6-b520be87d72b-0', usage_metadata={'input_tokens': 588, 'output_tokens': 157, 'total_tokens': 745})]}}
    ----
    {'agent': {'messages': [AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_Ob7hhwE6tC1JW6TbfYnvfxD8', 'function': {'arguments': '{"query":"common ways of task decomposition"}', 'name': 'blog_post_retriever'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 21, 'prompt_tokens': 768, 'total_tokens': 789}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-c9d6866e-b9d6-42a1-8ab8-ac4d2b43a820-0', tool_calls=[{'name': 'blog_post_retriever', 'args': {'query': 'common ways of task decomposition'}, 'id': 'call_Ob7hhwE6tC1JW6TbfYnvfxD8'}], usage_metadata={'input_tokens': 768, 'output_tokens': 21, 'total_tokens': 789})]}}
    ----
    {'tools': {'messages': [ToolMessage(content='Tree of Thoughts (Yao et al. 2023) extends CoT by exploring multiple reasoning possibilities at each step. It first decomposes the problem into multiple thought steps and generates multiple thoughts per step, creating a tree structure. The search process can be BFS (breadth-first search) or DFS (depth-first search) with each state evaluated by a classifier (via a prompt) or majority vote.\nTask decomposition can be done (1) by LLM with simple prompting like "Steps for XYZ.\\n1.", "What are the subgoals for achieving XYZ?", (2) by using task-specific instructions; e.g. "Write a story outline." for writing a novel, or (3) with human inputs.\n\nFig. 1. Overview of a LLM-powered autonomous agent system.\nComponent One: Planning#\nA complicated task usually involves many steps. An agent needs to know what they are and plan ahead.\nTask Decomposition#\nChain of thought (CoT; Wei et al. 2022) has become a standard prompting technique for enhancing model performance on complex tasks. The model is instructed to “think step by step” to utilize more test-time computation to decompose hard tasks into smaller and simpler steps. CoT transforms big tasks into multiple manageable tasks and shed lights into an interpretation of the model’s thinking process.\n\nResources:\n1. Internet access for searches and information gathering.\n2. Long Term memory management.\n3. GPT-3.5 powered Agents for delegation of simple tasks.\n4. File output.\n\nPerformance Evaluation:\n1. Continuously review and analyze your actions to ensure you are performing to the best of your abilities.\n2. Constructively self-criticize your big-picture behavior constantly.\n3. Reflect on past decisions and strategies to refine your approach.\n4. Every command has a cost, so be smart and efficient. Aim to complete tasks in the least number of steps.\n\n(3) Task execution: Expert models execute on the specific tasks and log results.\nInstruction:\n\nWith the input and the inference results, the AI assistant needs to describe the process and results. The previous stages can be formed as - User Input: {{ User Input }}, Task Planning: {{ Tasks }}, Model Selection: {{ Model Assignment }}, Task Execution: {{ Predictions }}. You must first answer the user\'s request in a straightforward manner. Then describe the task process and show your analysis and model inference results to the user in the first person. If inference results contain a file path, must tell the user the complete file path.', name='blog_post_retriever', tool_call_id='call_Ob7hhwE6tC1JW6TbfYnvfxD8')]}}
    ----
    {'agent': {'messages': [AIMessage(content='Common ways of task decomposition, as mentioned in the blog post, include:\n\n1. Using Language Models (LLM) with Simple Prompting: Language models can be utilized with simple prompts like "Steps for XYZ" or "What are the subgoals for achieving XYZ" to break down tasks into smaller components.\n\n2. Task-Specific Instructions: Task decomposition can also be achieved by providing task-specific instructions. For example, using instructions like "Write a story outline" for tasks such as writing a novel.\n\n3. Human Inputs: Another method of task decomposition involves human inputs, where individuals provide input to break down complex tasks into manageable steps.\n\nThese approaches help in breaking down complex tasks into smaller and simpler steps, enabling autonomous agents to plan and execute tasks more effectively.', response_metadata={'token_usage': {'completion_tokens': 154, 'prompt_tokens': 1313, 'total_tokens': 1467}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-da3a7949-df80-4df5-ac1f-4787693f76c6-0', usage_metadata={'input_tokens': 1313, 'output_tokens': 154, 'total_tokens': 1467})]}}
    ----

### Build a Question/Answering system over SQL data

#### Chains

```python
from operator import itemgetter

from dotenv import load_dotenv
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

# Load environment variables from a .env file
load_dotenv()

# Initialize the OpenAI model with the specified version
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

# Create a connection to the SQL database
db = SQLDatabase.from_uri("sqlite:///Chinook.db")

# Create an initial SQL query chain with the LLM and the database
chain = create_sql_query_chain(llm, db)

# Create a tool to execute SQL queries on the database
execute_query = QuerySQLDataBaseTool(db=db)

# Create another SQL query chain for writing queries
write_query = create_sql_query_chain(llm, db)

# Combine the write query chain and the execute query tool into a single chain
chain = write_query | execute_query

# Define a prompt template to generate an answer based on the user question, SQL query, and SQL result
answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)

# Combine the query writing, execution, and answering into a single chain
chain = (
    RunnablePassthrough.assign(
        query=write_query
    ).assign(  # Assign the query writing chain
        result=itemgetter("query") | execute_query
    )  # Assign the result execution chain
    | answer_prompt  # Use the answer prompt to format the response
    | llm  # Use the LLM to generate the final answer
    | StrOutputParser()  # Parse the output as a string
)

# Invoke the chain with a user question
answer = chain.invoke({"question": "How many employees are there"})

# Print the answer
print(answer)
```

    There are a total of 8 employees.

#### Agents

```python
import ast
import re

from dotenv import load_dotenv
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.prebuilt import create_react_agent

# Load environment variables from a .env file
load_dotenv()

# Initialize the OpenAI model with the specified version
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

# Create a connection to the SQL database
db = SQLDatabase.from_uri("sqlite:///Chinook.db")

# Create tools from a toolkit for interacting with the SQL database using the LLM
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()


# Define a function to run a query on the database and process the results
def query_as_list(db, query):
    # Run the query on the database
    res = db.run(query)
    # Parse the result into a list of values
    res = [el for sub in ast.literal_eval(res) for el in sub if el]
    # Remove numeric values and strip whitespace
    res = [re.sub(r"\b\d+\b", "", string).strip() for string in res]
    # Return unique values as a list
    return list(set(res))


# Query the list of artist names and album titles from the database
artists = query_as_list(db, "SELECT Name FROM Artist")
albums = query_as_list(db, "SELECT Title FROM Album")

# Create a FAISS vector store from the combined list of artists and albums, using OpenAI embeddings
vector_db = FAISS.from_texts(artists + albums, OpenAIEmbeddings())

# Create a retriever from the vector store for similarity-based search
retriever = vector_db.as_retriever(search_kwargs={"k": 5})

# Define a description for the retriever tool
description = """Use to look up values to filter on. Input is an approximate spelling of the proper noun, output is \
valid proper nouns. Use the noun most similar to the search."""

# Create a retriever tool with the defined retriever and description
retriever_tool = create_retriever_tool(
    retriever,
    name="search_proper_nouns",
    description=description,
)

# Define a system message with instructions for the agent
system = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct SQLite query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the given tools. Only use the information returned by the tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

You have access to the following tables: {table_names}

If you need to filter on a proper noun, you must ALWAYS first look up the filter value using the "search_proper_nouns" tool!
Do not try to guess at the proper name - use this function to find similar ones.""".format(
    table_names=db.get_usable_table_names()
)

# Create a system message with the defined instructions
system_message = SystemMessage(content=system)

# Append the retriever tool to the list of tools
tools.append(retriever_tool)

# Create an agent with the LLM, tools, and system message
agent = create_react_agent(llm, tools, messages_modifier=system_message)

# Define the first query
query = "How many albums does alis in chain have?"

# Stream responses for the first query using the agent
for s in agent.stream({"messages": [HumanMessage(content=query)]}):
    # Print each response and a separator
    print(s)
    print("----")
```

    {'agent': {'messages': [AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_QlXRSyobLzujJsu5oMu1fFHr', 'function': {'arguments': '{"query":"alis in chain"}', 'name': 'search_proper_nouns'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 19, 'prompt_tokens': 674, 'total_tokens': 693}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-85437a2e-bda2-480e-9901-074b172bbeac-0', tool_calls=[{'name': 'search_proper_nouns', 'args': {'query': 'alis in chain'}, 'id': 'call_QlXRSyobLzujJsu5oMu1fFHr'}], usage_metadata={'input_tokens': 674, 'output_tokens': 19, 'total_tokens': 693})]}}
    ----
    {'tools': {'messages': [ToolMessage(content='Alice In Chains\n\nAisha Duo\n\nXis\n\nDa Lama Ao Caos\n\nA-Sides', name='search_proper_nouns', tool_call_id='call_QlXRSyobLzujJsu5oMu1fFHr')]}}
    ----
    {'agent': {'messages': [AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_bSApsWZMwxyysqEz2FHwoavg', 'function': {'arguments': '{"query":"SELECT COUNT(Album.AlbumId) AS TotalAlbums FROM Album JOIN Artist ON Album.ArtistId = Artist.ArtistId WHERE Artist.Name = \'Alice In Chains\'"}', 'name': 'sql_db_query'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 49, 'prompt_tokens': 724, 'total_tokens': 773}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-e9e74020-eb99-4cea-988a-ad8cd4ab3878-0', tool_calls=[{'name': 'sql_db_query', 'args': {'query': "SELECT COUNT(Album.AlbumId) AS TotalAlbums FROM Album JOIN Artist ON Album.ArtistId = Artist.ArtistId WHERE Artist.Name = 'Alice In Chains'"}, 'id': 'call_bSApsWZMwxyysqEz2FHwoavg'}], usage_metadata={'input_tokens': 724, 'output_tokens': 49, 'total_tokens': 773})]}}
    ----
    {'tools': {'messages': [ToolMessage(content='[(1,)]', name='sql_db_query', tool_call_id='call_bSApsWZMwxyysqEz2FHwoavg')]}}
    ----
    {'agent': {'messages': [AIMessage(content='Alice In Chains has 1 album.', response_metadata={'token_usage': {'completion_tokens': 9, 'prompt_tokens': 786, 'total_tokens': 795}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-8a3fa8ca-001c-49a8-880c-9ab2fa4756df-0', usage_metadata={'input_tokens': 786, 'output_tokens': 9, 'total_tokens': 795})]}}
    ----

### Build a Query Analysis System

```python
import datetime
from typing import List, Optional

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import YoutubeLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables from a .env file
load_dotenv()

# Initialize the OpenAI model with the specified version and temperature
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

# List of YouTube video URLs to process
urls = [
    "https://www.youtube.com/watch?v=HAn9vnJy6S4",
    "https://www.youtube.com/watch?v=dA1cHGACXCo",
    "https://www.youtube.com/watch?v=ZcEMLz27sL4",
    "https://www.youtube.com/watch?v=hvAPnpSfSGo",
    "https://www.youtube.com/watch?v=EhlPDL4QrWY",
    "https://www.youtube.com/watch?v=mmBo8nlu2j0",
    "https://www.youtube.com/watch?v=rQdibOsL1ps",
    "https://www.youtube.com/watch?v=28lC4fqukoc",
    "https://www.youtube.com/watch?v=es-9MgxB-uc",
    "https://www.youtube.com/watch?v=wLRHwKuKvOE",
    "https://www.youtube.com/watch?v=ObIltMaRJvY",
    "https://www.youtube.com/watch?v=DjuXACWYkkU",
    "https://www.youtube.com/watch?v=o7C9ld6Ln-M",
]

# Load the documents from the YouTube videos
docs = []
for url in urls:
    docs.extend(YoutubeLoader.from_youtube_url(url, add_video_info=True).load())

# Add additional metadata: the year the video was published
for doc in docs:
    doc.metadata["publish_year"] = int(
        datetime.datetime.strptime(
            doc.metadata["publish_date"], "%Y-%m-%d %H:%M:%S"
        ).strftime("%Y")
    )

# Split the documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
chunked_docs = text_splitter.split_documents(docs)

# Create embeddings for the document chunks
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(
    chunked_docs,
    embeddings,
)


# Define a Pydantic model for the search
class Search(BaseModel):
    """Search over a database of tutorial videos about a software library."""

    query: str = Field(
        ...,
        description="Similarity search query applied to video transcripts.",
    )
    publish_year: Optional[int] = Field(None, description="Year video was published")


# Define the system prompt for converting user questions into database queries
system = """You are an expert at converting user questions into database queries. \
You have access to a database of tutorial videos about a software library for building LLM-powered applications. \
Given a question, return a list of database queries optimized to retrieve the most relevant results.

If there are acronyms or words you are not familiar with, do not try to rephrase them."""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

# Create a structured LLM that outputs a Search object
structured_llm = llm.with_structured_output(Search)
query_analyzer = {"question": RunnablePassthrough()} | prompt | structured_llm


# Define a retrieval function that searches the vector store based on the search criteria
def retrieval(search: Search) -> List[Document]:
    if search.publish_year is not None:
        # Apply a filter for the publish year if specified
        _filter = {"publish_year": {"$eq": search.publish_year}}
    else:
        _filter = None
    # Perform a similarity search on the vector store
    return vectorstore.similarity_search(search.query, filter=_filter)


# Create a retrieval chain combining query analysis and retrieval
retrieval_chain = query_analyzer | retrieval

# Invoke the retrieval chain with a query
results = retrieval_chain.invoke("RAG tutorial published in 2023")

# Print the titles and publish dates of the retrieved documents
print([(doc.metadata["title"], doc.metadata["publish_date"]) for doc in results])
```

    [('Getting Started with Multi-Modal LLMs', '2023-12-20 00:00:00'), ('LangServe and LangChain Templates Webinar', '2023-11-02 00:00:00'), ('Getting Started with Multi-Modal LLMs', '2023-12-20 00:00:00'), ('Building a Research Assistant from Scratch', '2023-11-16 00:00:00')]

### Build a Local RAG Application

```python
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.llms.llamafile import Llamafile
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load the data from a web page
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
data = loader.load()

# Split the loaded data into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

# Create a Chroma vector store from the document chunks, using HuggingFace embeddings
vectorstore = Chroma.from_documents(
    documents=all_splits,
    embedding=HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    ),
)

# Initialize the Llamafile language model
llm = Llamafile()


# Define a function to format documents into a single string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Pull a predefined RAG (Retrieval-Augmented Generation) prompt template from the hub
rag_prompt = hub.pull("rlm/rag-prompt")

# Define the question to be asked
question = "What are the approaches to Task Decomposition?"

# Create a retriever from the vector store for similarity-based search
retriever = vectorstore.as_retriever()

# Create a question-answering chain using the retriever, RAG prompt, LLM, and output parser
qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

# Invoke the question-answering chain with the specified question and print the result
print(qa_chain.invoke(question))
```

    Long-term planning is an essential step in task decomposition, as it helps determine the most effective way to complete a task. LLMs struggle to plan effectively over long periods due to their limitations in reasoning and memory. However, when human intervention can be utilized, the benefits of using LLMs for this process become apparent. Long-term planning involves identifying the steps required to achieve a goal, which is a complex task. This requires understanding the problem domain and its requirements. The complexity of the problem makes it challenging for LLMs to make accurate plans without human intervention. In contrast, humans can learn from trial and error, adapting their plans as they go along. It’s essential to remember that even with human-assisted tasks, planning still requires careful attention to detail, as it is necessary to avoid any mistakes in execution that could negatively impact the outcome.
    Challenges in long-term planning and task decomposition: Planning over a lengthy history and effectively exploring the solution space remain challenging. LLMs struggle to adjust plans when faced with unexpected errors, making them less robust compared to humans who learn from trial and error. Long-term planning involves identifying the steps required to achieve a goal, which is a complex task. This requires understanding the problem domain and its requirements. The complexity of the problem makes it challenging for LLMs to make accurate plans without human intervention. In contrast, humans can learn from trial and error, adapting their plans as they go along. It’s essential to remember that even with human-assisted tasks, planning still requires careful attention to detail, as it is necessary to avoid any mistakes in execution that could negatively impact the outcome.</s>

### Build a PDF ingestion and Question/Answering system

```python
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables from a .env file
load_dotenv()

# Initialize the OpenAI model with the specified version
llm = ChatOpenAI(model="gpt-4o")

# Define the path to the PDF file
file_path = "nke-10k-2023.pdf"

# Load the documents from the PDF file
loader = PyPDFLoader(file_path)
docs = loader.load()

# Split the loaded documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Create a Chroma vector store from the document chunks, using OpenAI embeddings
vectorstore = Chroma.from_documents(
    documents=splits, embedding=OpenAIEmbeddings())

# Create a retriever from the vector store for similarity-based search
retriever = vectorstore.as_retriever()

# Define a system prompt for answering questions
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

# Create a chat prompt template for answering questions
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Create a chain to combine retrieved documents and the question-answering task
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# Create a retrieval-augmented generation (RAG) chain
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Invoke the RAG chain with a question and retrieve the results
results = rag_chain.invoke({"input": "What was Nike's revenue in 2023?"})

# Print the results
print(results)
```

    {'input': "What was Nike's revenue in 2023?", 'context': [Document(page_content='Table of Contents\nFISCAL 2023 NIKE BRAND REVENUE HIGHLIGHTS\nThe following tables present NIKE Brand revenues disaggregated by reportable operating segment, distribution channel and major product line:\nFISCAL 2023 COMPARED TO FISCAL 2022\n•NIKE, Inc. Revenues were $51.2 billion in fiscal 2023, which increased 10% and 16% compared to fiscal 2022 on a reported and currency-neutral basis, respectively.\nThe increase was due to higher revenues in North America, Europe, Middle East & Africa ("EMEA"), APLA and Greater China, which contributed approximately 7, 6,\n2 and 1 percentage points to NIKE, Inc. Revenues, respectively.\n•NIKE Brand revenues, which represented over 90% of NIKE, Inc. Revenues, increased 10% and 16% on a reported and currency-neutral basis, respectively. This\nincrease was primarily due to higher revenues in Men\'s, the Jordan Brand, Women\'s and Kids\' which grew 17%, 35%,11% and 10%, respectively, on a wholesale\nequivalent basis.', metadata={'page': 35, 'source': 'nke-10k-2023.pdf'}), Document(page_content='Enterprise Resource Planning Platform, data and analytics, demand sensing, insight gathering, and other areas to create an end-to-end technology foundation, which we\nbelieve will further accelerate our digital transformation. We believe this unified approach will accelerate growth and unlock more efficiency for our business, while driving\nspeed and responsiveness as we serve consumers globally.\nFINANCIAL HIGHLIGHTS\n•In fiscal 2023, NIKE, Inc. achieved record Revenues of $51.2 billion, which increased 10% and 16% on a reported and currency-neutral basis, respectively\n•NIKE Direct revenues grew 14% from $18.7 billion in fiscal 2022 to $21.3 billion in fiscal 2023, and represented approximately 44% of total NIKE Brand revenues for\nfiscal 2023\n•Gross margin for the fiscal year decreased 250 basis points to 43.5% primarily driven by higher product costs, higher markdowns and unfavorable changes in foreign\ncurrency exchange rates, partially offset by strategic pricing actions', metadata={'page': 30, 'source': 'nke-10k-2023.pdf'}), Document(page_content="Table of Contents\nNORTH AMERICA\n(Dollars in millions) FISCAL 2023FISCAL 2022 % CHANGE% CHANGE\nEXCLUDING\nCURRENCY\nCHANGESFISCAL 2021 % CHANGE% CHANGE\nEXCLUDING\nCURRENCY\nCHANGES\nRevenues by:\nFootwear $ 14,897 $ 12,228 22 % 22 %$ 11,644 5 % 5 %\nApparel 5,947 5,492 8 % 9 % 5,028 9 % 9 %\nEquipment 764 633 21 % 21 % 507 25 % 25 %\nTOTAL REVENUES $ 21,608 $ 18,353 18 % 18 %$ 17,179 7 % 7 %\nRevenues by:    \nSales to Wholesale Customers $ 11,273 $ 9,621 17 % 18 %$ 10,186 -6 % -6 %\nSales through NIKE Direct 10,335 8,732 18 % 18 % 6,993 25 % 25 %\nTOTAL REVENUES $ 21,608 $ 18,353 18 % 18 %$ 17,179 7 % 7 %\nEARNINGS BEFORE INTEREST AND TAXES $ 5,454 $ 5,114 7 % $ 5,089 0 %\nFISCAL 2023 COMPARED TO FISCAL 2022\n•North America revenues increased 18% on a currency-neutral basis, primarily due to higher revenues in Men's and the Jordan Brand. NIKE Direct revenues\nincreased 18%, driven by strong digital sales growth of 23%, comparable store sales growth of 9% and the addition of new stores.", metadata={'page': 39, 'source': 'nke-10k-2023.pdf'}), Document(page_content="Table of Contents\nEUROPE, MIDDLE EAST & AFRICA\n(Dollars in millions) FISCAL 2023FISCAL 2022 % CHANGE% CHANGE\nEXCLUDING\nCURRENCY\nCHANGESFISCAL 2021 % CHANGE% CHANGE\nEXCLUDING\nCURRENCY\nCHANGES\nRevenues by:\nFootwear $ 8,260 $ 7,388 12 % 25 %$ 6,970 6 % 9 %\nApparel 4,566 4,527 1 % 14 % 3,996 13 % 16 %\nEquipment 592 564 5 % 18 % 490 15 % 17 %\nTOTAL REVENUES $ 13,418 $ 12,479 8 % 21 %$ 11,456 9 % 12 %\nRevenues by:    \nSales to Wholesale Customers $ 8,522 $ 8,377 2 % 15 %$ 7,812 7 % 10 %\nSales through NIKE Direct 4,896 4,102 19 % 33 % 3,644 13 % 15 %\nTOTAL REVENUES $ 13,418 $ 12,479 8 % 21 %$ 11,456 9 % 12 %\nEARNINGS BEFORE INTEREST AND TAXES $ 3,531 $ 3,293 7 % $ 2,435 35 % \nFISCAL 2023 COMPARED TO FISCAL 2022\n•EMEA revenues increased 21% on a currency-neutral basis, due to higher revenues in Men's, the Jordan Brand, Women's and Kids'. NIKE Direct revenues\nincreased 33%, driven primarily by strong digital sales growth of 43% and comparable store sales growth of 22%.", metadata={'page': 40, 'source': 'nke-10k-2023.pdf'})], 'answer': "Nike's revenue in fiscal 2023 was $51.2 billion."}

## Specialized tasks

### Build an Extraction Chain

```python
from typing import List, Optional

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

# Load environment variables from a .env file
load_dotenv()

# Initialize the OpenAI model with temperature set to 0 for deterministic output
llm = ChatOpenAI(temperature=0)

# Define a custom prompt to provide instructions and any additional context
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm. "
            "Only extract relevant information from the text. "
            "If you do not know the value of an attribute asked to extract, "
            "return null for the attribute's value.",
        ),
        # The 'human' part of the prompt will be populated with the input text
        ("human", "{text}"),
    ]
)


# Define a Pydantic model for extracting information about a person
class Person(BaseModel):
    """Information about a person."""

    # The name of the person
    name: Optional[str] = Field(default=None, description="The name of the person")
    # The color of the person's hair if known
    hair_color: Optional[str] = Field(
        default=None, description="The color of the person's hair if known"
    )
    # Height measured in meters
    height_in_meters: Optional[str] = Field(
        default=None, description="Height measured in meters"
    )


# Define a Pydantic model for extracting data about multiple people
class Data(BaseModel):
    """Extracted data about people."""

    # A list of Person objects
    people: List[Person]


# Create a runnable pipeline that combines the prompt and the LLM with structured output
runnable = prompt | llm.with_structured_output(schema=Data)

# Define the input text for extraction
text = "My name is Jeff, my hair is black and i am 6 feet tall. Anna has the same color hair as me."

# Invoke the pipeline with the input text and print the result
result = runnable.invoke({"text": text})
print(result)
```

    people=[Person(name='Jeff', hair_color='black', height_in_meters='1.83'), Person(name='Anna', hair_color='black', height_in_meters=None)]

### Classify text into labels

```python
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

# Load environment variables from a .env file
load_dotenv()


# Define a Pydantic model for classification
class Classification(BaseModel):
    sentiment: str = Field(..., enum=["happy", "neutral", "sad"])
    aggressiveness: int = Field(
        ...,
        description="Describes how aggressive the statement is, the higher the number the more aggressive",
        enum=[1, 2, 3, 4, 5],
    )
    language: str = Field(
        ..., enum=["spanish", "english", "french", "german", "italian"]
    )


# Initialize the OpenAI model with specified temperature and model, and configure it to use structured output
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0125").with_structured_output(
    Classification
)

# Define a prompt template for extracting desired information
tagging_prompt = ChatPromptTemplate.from_template(
    """
Extract the desired information from the following passage.

Only extract the properties mentioned in the 'Classification' function.

Passage:
{input}
"""
)

# Create a chain that combines the prompt and the LLM
chain = tagging_prompt | llm

# Define the input text for classification
inp = "Estoy increiblemente contento de haberte conocido! Creo que seremos muy buenos amigos!"

# Invoke the chain with the input text and print the result
print(chain.invoke({"input": inp}))
```

    sentiment='happy' aggressiveness=1 language='spanish'

```python

```
