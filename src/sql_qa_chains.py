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
