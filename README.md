# LangChain

## Specialized tasks

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

## Working with external knowledge

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
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Create a Chroma vector store from the document chunks, using OpenAI embeddings
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

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

### Build a local RAG application

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

### Build a Question/Answering system over SQL data

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
def query_as_list(database, sql_query):
    # Run the query on the database
    res = database.run(sql_query)
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

### Build a Conversational RAG Application

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
    bs_kwargs={
        "parse_only": bs4.SoupStrainer(
            class_=(
                "post-content",
                "post-title",
                "post-header",
            )  # Specify the classes to parse
        )
    },
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
    bs_kwargs={
        "parse_only": bs4.SoupStrainer(
            class_=(
                "post-content",
                "post-title",
                "post-header",
            )  # Specify the classes to parse
        )
    },
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
    bs_kwargs={
        "parse_only": bs4.SoupStrainer(
            class_=(
                "post-content",
                "post-title",
                "post-header",
            )  # Specify the classes to parse
        )
    },
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
def format_docs(documents):
    return "\n\n".join(doc.page_content for doc in documents)


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
    bs_kwargs={
        "parse_only": bs4.SoupStrainer(
            class_=(
                "post-content",
                "post-title",
                "post-header",
            )  # Specify the classes to parse
        )
    },
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

## Basics

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

### Build a Simple LLM Application with LCEL

```python
#!/usr/bin/env python
import uvicorn
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
    uvicorn.run(app, host="localhost", port=8000)
```

```python
from langserve import RemoteRunnable

remote_chain = RemoteRunnable("http://localhost:8000/chain/")
remote_chain.invoke({"language": "italian", "text": "hi"})
```

    'Ciao'

```python

```
