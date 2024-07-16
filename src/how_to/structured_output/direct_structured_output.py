from typing import List

from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

# Load environment variables from a .env file
load_dotenv()

# Initialize the OpenAI model
llm = ChatOpenAI()


# Define a Pydantic model for a person
class Person(BaseModel):
    """Information about a person."""

    name: str = Field(..., description="The name of the person")
    height_in_meters: float = Field(
        ..., description="The height of the person expressed in meters."
    )


# Define a Pydantic model for a list of people
class People(BaseModel):
    """Identifying information about all people in a text."""

    people: List[Person]


# Create a Pydantic output parser for the People model
parser = PydanticOutputParser(pydantic_object=People)

# Define a chat prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the user query. Wrap the output in `json` tags\n{format_instructions}",
        ),
        ("human", "{query}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# Define the query
query = "Anna is 23 years old and she is 6 feet tall"

# Create a chain by combining the prompt, the LLM, and the parser
chain = prompt | llm | parser

# Invoke the chain with the query and print the result
print(chain.invoke({"query": query}))
