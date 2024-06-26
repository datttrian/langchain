import os
from operator import itemgetter

import openai
from dotenv import load_dotenv
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

# Initialize SQL Database connection
db = SQLDatabase.from_uri("sqlite:///Chinook.db")

# Initialize the language model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Define the tools and chains
execute_query = QuerySQLDataBaseTool(db=db)
write_query = create_sql_query_chain(llm, db)

# Define the answer prompt template
answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)

# Define the answer chain
answer = answer_prompt | llm | StrOutputParser()

# Define the complete chain
chain = (
    RunnablePassthrough.assign(query=write_query).assign(
        result=itemgetter("query") | execute_query
    )
    | answer
)

# Invoke the chain with a sample question and print the result
result = chain.invoke({"question": "How many employees are there?"})
print(result)
