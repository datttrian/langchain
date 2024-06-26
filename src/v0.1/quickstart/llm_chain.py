import os

import openai
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

# Initialize the language model
llm = ChatOpenAI()

# Create the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a world class technical documentation writer."),
        ("user", "{input}"),
    ]
)

# Initialize the output parser
output_parser = StrOutputParser()

# Create the chain
chain = prompt | llm | output_parser

# Invoke the chain with the input and capture the output
result = chain.invoke({"input": "how can langsmith help with testing?"})

# Print the result
print(result)
