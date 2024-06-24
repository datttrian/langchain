import os

import openai
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables from a .env file
load_dotenv()

# Set OpenAI API key from environment variables
openai.api_key = os.environ["OPENAI_API_KEY"]

# Initialize the embeddings model
embeddings = OpenAIEmbeddings()

# Load documents from a specified web page
loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
docs = loader.load()

# Split the loaded documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)

# Create a vector store (FAISS) from the document embeddings
vector = FAISS.from_documents(documents, embeddings)

# Set up a retriever using the vector store to fetch relevant documents based on the query
retriever = vector.as_retriever()

# Set up a tool for the retriever
retriever_tool = create_retriever_tool(
    retriever,
    "langsmith_search",
    "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",
)

# Set up a search tool
search = TavilySearchResults()

# Create a list of the tools
tools = [retriever_tool, search]

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/openai-functions-agent")

# Initialize the language model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Create an agent to use the tools
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Ask a question about LangSmith
agent_executor.invoke({"input": "how can langsmith help with testing?"})

# Ask a question about the weather
agent_executor.invoke({"input": "what is the weather in SF?"})

# Have conversations about LangSmith helping testing LLM applications
chat_history = [
    HumanMessage(content="Can LangSmith help test my LLM applications?"),
    AIMessage(content="Yes!"),
]
agent_executor.invoke({"chat_history": chat_history, "input": "Tell me how"})
