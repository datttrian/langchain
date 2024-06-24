import os

import openai
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI

from langchain_community.agent_toolkits import create_sql_agent


load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]


db = SQLDatabase.from_uri("sqlite:///Chinook.db")


llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

agent_executor = create_sql_agent(
    llm, db=db, agent_type="openai-tools", verbose=True)

agent_executor.invoke(
    {
        "input": "Describe the playlisttrack table"
    }
)
