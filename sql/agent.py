import os
import openai
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent

# Load environment variables from .env file
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

# Initialize database connection
db = SQLDatabase.from_uri("sqlite:///Chinook.db")

# Initialize the language model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Create the SQL agent executor
agent_executor = create_sql_agent(
    llm, db=db, agent_type="openai-tools", verbose=True)

# Invoke the agent with the input and capture the result
result = agent_executor.invoke(
    {
        "input": "List the total sales per country. Which country's customers spent the most?"
    }
)

# Print the result
print(result)
