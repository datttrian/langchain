import os

import openai
from dotenv import load_dotenv
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI

from langchain.chains import create_sql_query_chain

# Load environment variables
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

# Initialize database and language model
db = SQLDatabase.from_uri("sqlite:///Chinook.db")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Create the query tools
execute_query = QuerySQLDataBaseTool(db=db)
write_query = create_sql_query_chain(llm, db)

# Combine the chains
chain = write_query | execute_query

# Invoke the chain and print the result
result = chain.invoke({"question": "How many employees are there"})
print(result)
