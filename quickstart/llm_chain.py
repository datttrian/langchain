import os
from langchain_openai import ChatOpenAI
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]
llm = ChatOpenAI()
result = llm.invoke("how can langsmith help with testing?")
print(result)
