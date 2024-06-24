import os
from langchain_openai import ChatOpenAI
import openai
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

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

chain = prompt | llm

chain.invoke({"input": "how can langsmith help with testing?"})


output_parser = StrOutputParser()
chain = prompt | llm | output_parser
chain.invoke({"input": "how can langsmith help with testing?"})
