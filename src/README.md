# LangChain

## Build a Simple LLM Application with LCEL

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

## Build a Chatbot

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

    |Your| name| is| Bob|.||

## Build vector stores and retrievers

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

## Build an Agent

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

    {'agent': {'messages': [AIMessage(content='Hello Bob! How can I assist you today regarding San Francisco?', response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 90, 'total_tokens': 104}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-8848f240-6fc6-47d3-b97c-686f65677ed8-0', usage_metadata={'input_tokens': 90, 'output_tokens': 14, 'total_tokens': 104})]}}
    ----
    {'agent': {'messages': [AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_fIDiygPZgTJJpv0X6C6VcSlJ', 'function': {'arguments': '{"query":"weather in San Francisco"}', 'name': 'tavily_search_results_json'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 21, 'prompt_tokens': 119, 'total_tokens': 140}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-51431379-a950-4d1b-a996-cc44154543f4-0', tool_calls=[{'name': 'tavily_search_results_json', 'args': {'query': 'weather in San Francisco'}, 'id': 'call_fIDiygPZgTJJpv0X6C6VcSlJ'}], usage_metadata={'input_tokens': 119, 'output_tokens': 21, 'total_tokens': 140})]}}
    ----
    {'tools': {'messages': [ToolMessage(content='[{"url": "https://www.weatherapi.com/", "content": "{\'location\': {\'name\': \'San Francisco\', \'region\': \'California\', \'country\': \'United States of America\', \'lat\': 37.78, \'lon\': -122.42, \'tz_id\': \'America/Los_Angeles\', \'localtime_epoch\': 1719430643, \'localtime\': \'2024-06-26 12:37\'}, \'current\': {\'last_updated_epoch\': 1719430200, \'last_updated\': \'2024-06-26 12:30\', \'temp_c\': 17.2, \'temp_f\': 63.0, \'is_day\': 1, \'condition\': {\'text\': \'Partly cloudy\', \'icon\': \'//cdn.weatherapi.com/weather/64x64/day/116.png\', \'code\': 1003}, \'wind_mph\': 10.5, \'wind_kph\': 16.9, \'wind_degree\': 260, \'wind_dir\': \'W\', \'pressure_mb\': 1018.0, \'pressure_in\': 30.05, \'precip_mm\': 0.0, \'precip_in\': 0.0, \'humidity\': 70, \'cloud\': 25, \'feelslike_c\': 17.2, \'feelslike_f\': 63.0, \'windchill_c\': 15.7, \'windchill_f\': 60.3, \'heatindex_c\': 15.8, \'heatindex_f\': 60.5, \'dewpoint_c\': 10.9, \'dewpoint_f\': 51.7, \'vis_km\': 16.0, \'vis_miles\': 9.0, \'uv\': 5.0, \'gust_mph\': 13.8, \'gust_kph\': 22.2}}"}, {"url": "https://world-weather.info/forecast/usa/san_francisco/june-2024/", "content": "Extended weather forecast in San Francisco. Hourly Week 10 days 14 days 30 days Year. Detailed \\u26a1 San Francisco Weather Forecast for June 2024 - day/night \\ud83c\\udf21\\ufe0f temperatures, precipitations - World-Weather.info."}]', name='tavily_search_results_json', tool_call_id='call_fIDiygPZgTJJpv0X6C6VcSlJ')]}}
    ----
    {'agent': {'messages': [AIMessage(content="The current weather in San Francisco is partly cloudy with a temperature of 63.0°F (17.2°C). The wind is blowing from the west at 16.9 km/h. If you'd like more detailed information or an extended forecast, you can visit [this link](https://www.weatherapi.com/).", response_metadata={'token_usage': {'completion_tokens': 67, 'prompt_tokens': 652, 'total_tokens': 719}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-0482994e-6006-447e-82ba-31415d3cd8b5-0', usage_metadata={'input_tokens': 652, 'output_tokens': 67, 'total_tokens': 719})]}}
    ----

```python

```
