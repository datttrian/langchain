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
