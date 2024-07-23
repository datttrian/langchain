from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticToolsParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)


# Note that the docstrings here are crucial, as they will be passed along
# to the model along with the class name.
class Add(BaseModel):
    """Add two integers together."""

    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")


class Multiply(BaseModel):
    """Multiply two integers together."""

    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")


tools = [Add, Multiply]


llm_with_tools = llm.bind_tools(tools)

query = "What is 3 * 12? Also, what is 11 + 49?"

llm_with_tools.invoke(query).tool_calls


chain = llm_with_tools | PydanticToolsParser(tools=[Multiply, Add])
print(chain.invoke(query))
