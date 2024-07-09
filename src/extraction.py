from typing import List, Optional

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

# Load environment variables from a .env file
load_dotenv()

# Initialize the OpenAI model with temperature set to 0 for deterministic output
llm = ChatOpenAI(temperature=0)

# Define a custom prompt to provide instructions and any additional context
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm. "
            "Only extract relevant information from the text. "
            "If you do not know the value of an attribute asked to extract, "
            "return null for the attribute's value.",
        ),
        # The 'human' part of the prompt will be populated with the input text
        ("human", "{text}"),
    ]
)


# Define a Pydantic model for extracting information about a person
class Person(BaseModel):
    """Information about a person."""

    # The name of the person
    name: Optional[str] = Field(default=None, description="The name of the person")
    # The color of the person's hair if known
    hair_color: Optional[str] = Field(
        default=None, description="The color of the person's hair if known"
    )
    # Height measured in meters
    height_in_meters: Optional[str] = Field(
        default=None, description="Height measured in meters"
    )


# Define a Pydantic model for extracting data about multiple people
class Data(BaseModel):
    """Extracted data about people."""

    # A list of Person objects
    people: List[Person]


# Create a runnable pipeline that combines the prompt and the LLM with structured output
runnable = prompt | llm.with_structured_output(schema=Data)

# Define the input text for extraction
text = "My name is Jeff, my hair is black and i am 6 feet tall. Anna has the same color hair as me."

# Invoke the pipeline with the input text and print the result
result = runnable.invoke({"text": text})
print(result)
