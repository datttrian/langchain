import os

import openai
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]


llm = ChatOpenAI()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a world class technical documentation writer."),
        ("user", "{input}"),
    ]
)


output_parser = StrOutputParser()
chain = prompt | llm | output_parser
result = chain.invoke({"input": "how can langsmith help with testing?"})
print(result)
