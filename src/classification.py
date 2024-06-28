from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

# Load environment variables from a .env file
load_dotenv()


# Define a Pydantic model for classification
class Classification(BaseModel):
    sentiment: str = Field(..., enum=["happy", "neutral", "sad"])
    aggressiveness: int = Field(
        ...,
        description="Describes how aggressive the statement is, the higher the number the more aggressive",
        enum=[1, 2, 3, 4, 5],
    )
    language: str = Field(
        ..., enum=["spanish", "english", "french", "german", "italian"]
    )


# Initialize the OpenAI model with specified temperature and model, and configure it to use structured output
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0125").with_structured_output(
    Classification
)

# Define a prompt template for extracting desired information
tagging_prompt = ChatPromptTemplate.from_template(
    """
Extract the desired information from the following passage.

Only extract the properties mentioned in the 'Classification' function.

Passage:
{input}
"""
)

# Create a chain that combines the prompt and the LLM
chain = tagging_prompt | llm

# Define the input text for classification
inp = "Estoy increiblemente contento de haberte conocido! Creo que seremos muy buenos amigos!"

# Invoke the chain with the input text and print the result
print(chain.invoke({"input": inp}))
