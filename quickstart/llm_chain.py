import os
from langchain_openai import ChatOpenAI
import openai
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]
llm = ChatOpenAI()
result = llm.invoke("how can langsmith help with testing?")
print(result)


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a world class technical documentation writer."),
        ("user", "{input}"),
    ]
)
