from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticToolsParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

# Load environment variables from a .env file
load_dotenv()

# Initialize the OpenAI model with the specified model name and temperature
llm = ChatOpenAI()


# Define a Pydantic model for addition
class Add(BaseModel):
    """Add two integers together."""

    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")


# Define a Pydantic model for multiplication
class Multiply(BaseModel):
    """Multiply two integers together."""

    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")


# Define the Add and Multiply models tools
tools = [Add, Multiply]

# Bind the tools to the LLM
llm_with_tools = llm.bind_tools(tools)

# Define the query to be processed by the LLM
query = "What is 3 * 12? Also, what is 11 + 49?"

# Invoke the LLM with tools and print the tool calls
print(llm_with_tools.invoke(query).tool_calls)

# Create a chain by combining the LLM with tools and the Pydantic tools parser
chain = llm_with_tools | PydanticToolsParser(tools=[Multiply, Add])

# Invoke the chain with the query and print the result
print(chain.invoke(query))
