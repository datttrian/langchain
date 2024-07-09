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
