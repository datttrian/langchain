# LangChain

## How to return structured data from a model

### Custom structured output

```python
import json
import re
from typing import List

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

# Load environment variables from a .env file
load_dotenv()

# Initialize the OpenAI model with the specified model name
llm = ChatOpenAI(model="gpt-4")


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


# Define a chat prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the user query. Output your answer as JSON that "
            "matches the given schema: ```json\n{schema}\n```. "
            "Make sure to wrap the answer in ```json and ``` tags",
        ),
        ("human", "{query}"),
    ]
).partial(schema=People.schema())


# Custom parser to extract JSON content from a string where JSON is embedded between ```json and ``` tags
def extract_json(message: AIMessage) -> List[dict]:
    """Extracts JSON content from a string where JSON is embedded between ```json and ``` tags.

    Parameters:
        text (str): The text containing the JSON content.

    Returns:
        list: A list of extracted JSON strings.
    """
    text = message.content
    # Define the regular expression pattern to match JSON blocks
    pattern = r"```json(.*?)```"

    # Find all non-overlapping matches of the pattern in the string
    matches = re.findall(pattern, text, re.DOTALL)

    # Return the list of matched JSON strings, stripping any leading or trailing whitespace
    try:
        return [json.loads(match.strip()) for match in matches]
    except Exception as exc:
        raise ValueError(f"Failed to parse: {message}") from exc


# Define the query
query = "Anna is 23 years old and she is 6 feet tall"

# Create a chain by combining the prompt, the LLM, and the custom parser
chain = prompt | llm | extract_json

# Invoke the chain with the query and print the result
print(chain.invoke({"query": query}))
```

    [{'people': [{'name': 'Anna', 'height_in_meters': 1.8288}]}]

### Direct structured output

```python
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
```

    people=[Person(name='Anna', height_in_meters=1.83)]

### Few-shot prompting

```python
from typing import Optional, Union

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

# Load environment variables from a .env file
load_dotenv()

# Initialize the OpenAI model with the specified version and temperature
llm = ChatOpenAI(model="gpt-4-0125-preview", temperature=0)


# Define a Pydantic model for a joke
class Joke(BaseModel):
    """Joke to tell user."""

    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")
    rating: Optional[int] = Field(description="How funny the joke is, from 1 to 10")


# Define a Pydantic model for a conversational response
class ConversationalResponse(BaseModel):
    """Respond in a conversational manner. Be kind and helpful."""

    response: str = Field(description="A conversational response to the user's query")


# Define a Pydantic model for the response which can be either a joke or a conversational response
class Response(BaseModel):
    output: Union[Joke, ConversationalResponse]


# Create a structured LLM that outputs a Response and includes raw output
structured_llm = llm.with_structured_output(Response, include_raw=True)

# Define example interactions to improve extraction quality
examples = [
    HumanMessage("Tell me a joke about planes", name="example_user"),
    AIMessage(
        "",
        name="example_assistant",
        tool_calls=[
            {
                "name": "joke",
                "args": {
                    "setup": "Why don't planes ever get tired?",
                    "punchline": "Because they have rest wings!",
                    "rating": 2,
                },
                "id": "1",
            }
        ],
    ),
    ToolMessage("", tool_call_id="1"),
    HumanMessage("Tell me another joke about planes", name="example_user"),
    AIMessage(
        "",
        name="example_assistant",
        tool_calls=[
            {
                "name": "joke",
                "args": {
                    "setup": "Cargo",
                    "punchline": "Cargo 'vroom vroom', but planes go 'zoom zoom'!",
                    "rating": 10,
                },
                "id": "2",
            }
        ],
    ),
    ToolMessage("", tool_call_id="2"),
    HumanMessage("Now about caterpillars", name="example_user"),
    AIMessage(
        "",
        tool_calls=[
            {
                "name": "joke",
                "args": {
                    "setup": "Caterpillar",
                    "punchline": "Caterpillar really slow, but watch me turn into a butterfly and steal the show!",
                    "rating": 5,
                },
                "id": "3",
            }
        ],
    ),
    ToolMessage("", tool_call_id="3"),
]

# Define a system prompt for the LLM
system = """You are a hilarious comedian. Your specialty is knock-knock jokes. \
Return a joke which has the setup (the response to "Who's there?") \
and the final punchline (the response to "<setup> who?")."""

# Create a chat prompt template with the system prompt and placeholders for examples and user input
prompt = ChatPromptTemplate.from_messages(
    [("system", system), ("placeholder", "{examples}"), ("human", "{input}")]
)

# Combine the prompt template and the structured LLM into a single chain
few_shot_structured_llm = prompt | structured_llm

# Invoke the chain with the input and examples, and print the result
result = few_shot_structured_llm.invoke({"input": "crocodiles", "examples": examples})
print(result)
```

    {'raw': AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_J2TcBVzyFMue6TO51fjbcWlg', 'function': {'arguments': '{"output":{"setup":"Croco","punchline":"Croco-dial my number, because I\'m about to snap!","rating":7}}', 'name': 'Response'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 31, 'prompt_tokens': 392, 'total_tokens': 423}, 'model_name': 'gpt-4-0125-preview', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-aea5c6e9-53f2-4aed-87bd-bd0d55645476-0', tool_calls=[{'name': 'Response', 'args': {'output': {'setup': 'Croco', 'punchline': "Croco-dial my number, because I'm about to snap!", 'rating': 7}}, 'id': 'call_J2TcBVzyFMue6TO51fjbcWlg', 'type': 'tool_call'}], usage_metadata={'input_tokens': 392, 'output_tokens': 31, 'total_tokens': 423}), 'parsed': Response(output=Joke(setup='Croco', punchline="Croco-dial my number, because I'm about to snap!", rating=7)), 'parsing_error': None}

## Summarize text

### Refine

```python
from dotenv import load_dotenv
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter

# Load environment variables from a .env file
load_dotenv()

# Load the data from a web page
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
docs = loader.load()

# Initialize the OpenAI model with the default temperature
llm = ChatOpenAI(temperature=0)

# Split the documents into smaller chunks
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000, chunk_overlap=0
)
split_docs = text_splitter.split_documents(docs)

# Load the summarize chain with the LLM and chain type "refine"
chain = load_summarize_chain(llm, chain_type="refine")

# Invoke the chain with the split documents and print the result
result = chain.invoke(split_docs)
print(result["output_text"])

# Define a prompt template for summarizing the text
prompt_template = """Write a concise summary of the following:
{text}
CONCISE SUMMARY:"""
prompt = PromptTemplate.from_template(prompt_template)

# Define a refine prompt template for refining the summary with additional context
refine_template = (
    "Your job is to produce a final summary\n"
    "We have provided an existing summary up to a certain point: {existing_answer}\n"
    "We have the opportunity to refine the existing summary"
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{text}\n"
    "------------\n"
    "Given the new context, refine the original summary in Italian"
    "If the context isn't useful, return the original summary."
)
refine_prompt = PromptTemplate.from_template(refine_template)

# Load the summarize chain with the LLM, chain type "refine", question prompt, and refine prompt
chain = load_summarize_chain(
    llm=llm,
    chain_type="refine",
    question_prompt=prompt,
    refine_prompt=refine_prompt,
    return_intermediate_steps=True,
    input_key="input_documents",
    output_key="output_text",
)

# Invoke the chain with the split documents and print the first three intermediate steps
result = chain.invoke({"input_documents": split_docs}, return_only_outputs=True)
print("\n\n".join(result["intermediate_steps"][:3]))
```

    Created a chunk of size 1003, which is longer than the specified 1000


    The refined summary now includes additional context on the challenges faced in long-term planning, task decomposition, reliability of natural language interfaces, and the limitations of LLMs in adjusting plans and handling unexpected errors. The design of the autonomous agent system is highlighted, emphasizing the structured code architecture for building agents powered by Large Language Models. Each component, including the Agent, LanguageModel, and Environment classes, is defined in separate files with specific functionalities. The main file orchestrates the interaction between these components to simulate the autonomous agent's behavior. The summary also includes a citation for further reference. Additional references are provided for further exploration into the advancements in LLM-powered autonomous agents and related research in the field.
    The article discusses the concept of building autonomous agents powered by LLM (large language model) as the core controller. It explores the components of planning, memory, and tool use in such agents, highlighting their potential for problem-solving and showcasing proof-of-concept examples. The use of LLM extends beyond generating content to being a powerful general problem solver.
    
    L'articolo discute il concetto di costruire agenti autonomi alimentati da LLM (large language model) come controller principale. Esplora i componenti di pianificazione, memoria e utilizzo degli strumenti in tali agenti, evidenziando il loro potenziale per la risoluzione dei problemi e mostrando esempi di proof-of-concept. L'uso di LLM si estende oltre la generazione di contenuti per diventare un potente risolutore generale di problemi. Inoltre, vengono presentati approcci come Task Decomposition, Self-Reflection e Reflexion per migliorare le capacità di ragionamento e di auto-miglioramento degli agenti autonomi.
    
    L'articolo discute il concetto di costruire agenti autonomi alimentati da LLM (large language model) come controller principale. Esplora i componenti di pianificazione, memoria e utilizzo degli strumenti in tali agenti, evidenziando il loro potenziale per la risoluzione dei problemi e mostrando esempi di proof-of-concept. L'uso di LLM si estende oltre la generazione di contenuti per diventare un potente risolutore generale di problemi. Vengono presentati approcci come Task Decomposition, Self-Reflection e Reflexion per migliorare le capacità di ragionamento e di auto-miglioramento degli agenti autonomi. Inoltre, vengono introdotti nuovi concetti come Chain of Hindsight (CoH) e Algorithm Distillation (AD) che mirano a migliorare le capacità di auto-miglioramento e di apprendimento sequenziale degli agenti autonomi, portando a risultati promettenti nella risoluzione di problemi complessi.

### Map-Reduce

```python
from dotenv import load_dotenv
from langchain import hub
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter

# Load environment variables from a .env file
load_dotenv()

# Load the data from a web page
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
docs = loader.load()

# Initialize the OpenAI model with the default temperature
llm = ChatOpenAI(temperature=0)

# Pull the map and reduce prompts from the hub
map_prompt = hub.pull("rlm/map-prompt")
map_chain = LLMChain(llm=llm, prompt=map_prompt)

reduce_prompt = hub.pull("rlm/reduce-prompt")
reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

# Define a StuffDocumentsChain to process the documents
combine_documents_chain = StuffDocumentsChain(
    llm_chain=reduce_chain, document_variable_name="doc_summaries"
)

# Define a ReduceDocumentsChain to combine and iteratively reduce the mapped documents
reduce_documents_chain = ReduceDocumentsChain(
    combine_documents_chain=combine_documents_chain,
    collapse_documents_chain=combine_documents_chain,
    token_max=4000,
)

# Define a MapReduceDocumentsChain to combine documents by mapping a chain over them and reducing the results
map_reduce_chain = MapReduceDocumentsChain(
    llm_chain=map_chain,
    reduce_documents_chain=reduce_documents_chain,
    document_variable_name="docs",
    return_intermediate_steps=False,
)

# Split the documents into smaller chunks
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000, chunk_overlap=0
)
split_docs = text_splitter.split_documents(docs)

# Invoke the MapReduceDocumentsChain with the split documents and print the result
result = map_reduce_chain.invoke(split_docs)
print(result["output_text"])
```

    Created a chunk of size 1003, which is longer than the specified 1000


    The main themes identified in the set of documents revolve around the utilization of large language models (LLMs) in building autonomous agents with capabilities such as planning, memory, tool use, and self-reflection. The documents discuss challenges, approaches, and examples of LLM-powered agents like AutoGPT, GPT-Engineer, and BabyAGI. Additionally, topics include algorithm distillation for reinforcement learning, memory types, Maximum Inner Product Search (MIPS), similarity search algorithms, neuro-symbolic architecture, task planning, and execution in AI assistants, as well as real-world applications and case studies of LLMs in various domains. Overall, the documents emphasize the importance of efficient task delegation, automation, performance evaluation, and continuous improvement in utilizing LLMs for diverse applications.

### Stuff

```python
from dotenv import load_dotenv
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Load environment variables from a .env file
load_dotenv()

# Load the data from a web page
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
docs = loader.load()

# Define the prompt template for summarizing the text
prompt_template = """Write a concise summary of the following:
"{text}"
CONCISE SUMMARY:"""
prompt = PromptTemplate.from_template(prompt_template)

# Initialize the OpenAI model with specified temperature and model name
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")

# Create an LLM chain with the model and prompt
llm_chain = LLMChain(llm=llm, prompt=prompt)

# Define a StuffDocumentsChain to process the documents
stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")

# Load the documents again (this line seems redundant since docs are already loaded)
docs = loader.load()

# Invoke the chain with the loaded documents and print the result
print(stuff_chain.invoke(docs)["output_text"])
```

    The article discusses the concept of building autonomous agents powered by large language models (LLMs). It explores the components of such agents, including planning, memory, and tool use. The article provides case studies and examples of proof-of-concept demos, highlighting the challenges and limitations of LLM-powered agents. It also includes citations and references for further reading.

```python
from dotenv import load_dotenv
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI

# Load environment variables from a .env file
load_dotenv()

# Load the data from a web page
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
docs = loader.load()

# Initialize the OpenAI model with specified temperature and model name
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-1106")

# Load the summarize chain with the LLM and chain type "stuff"
chain = load_summarize_chain(llm, chain_type="stuff")

# Invoke the chain with the loaded documents and get the result
result = chain.invoke(docs)

# Print the summary output
print(result["output_text"])
```

    The article discusses the concept of LLM-powered autonomous agents, which use large language models as their core controllers. It covers the components of these agents, including planning, memory, and tool use, as well as case studies and proof-of-concept examples. The challenges and limitations of using natural language interfaces for these agents are also addressed. The article provides citations and references for further reading.

## Classify text into labels

```python
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
```

    sentiment='happy' aggressiveness=1 language='spanish'

## Build an Extraction Chain

```python
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
```

    people=[Person(name='Jeff', hair_color='black', height_in_meters='1.83'), Person(name='Anna', hair_color='black', height_in_meters=None)]

## Build a PDF ingestion and Question/Answering system

```python
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables from a .env file
load_dotenv()

# Initialize the OpenAI model with the specified version
llm = ChatOpenAI(model="gpt-4o")

# Define the path to the PDF file
file_path = "nke-10k-2023.pdf"

# Load the documents from the PDF file
loader = PyPDFLoader(file_path)
docs = loader.load()

# Split the loaded documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Create a Chroma vector store from the document chunks, using OpenAI embeddings
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# Create a retriever from the vector store for similarity-based search
retriever = vectorstore.as_retriever()

# Define a system prompt for answering questions
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

# Create a chat prompt template for answering questions
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Create a chain to combine retrieved documents and the question-answering task
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# Create a retrieval-augmented generation (RAG) chain
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Invoke the RAG chain with a question and retrieve the results
results = rag_chain.invoke({"input": "What was Nike's revenue in 2023?"})

# Print the results
print(results)
```

    {'input': "What was Nike's revenue in 2023?", 'context': [Document(page_content='Table of Contents\nFISCAL 2023 NIKE BRAND REVENUE HIGHLIGHTS\nThe following tables present NIKE Brand revenues disaggregated by reportable operating segment, distribution channel and major product line:\nFISCAL 2023 COMPARED TO FISCAL 2022\n•NIKE, Inc. Revenues were $51.2 billion in fiscal 2023, which increased 10% and 16% compared to fiscal 2022 on a reported and currency-neutral basis, respectively.\nThe increase was due to higher revenues in North America, Europe, Middle East & Africa ("EMEA"), APLA and Greater China, which contributed approximately 7, 6,\n2 and 1 percentage points to NIKE, Inc. Revenues, respectively.\n•NIKE Brand revenues, which represented over 90% of NIKE, Inc. Revenues, increased 10% and 16% on a reported and currency-neutral basis, respectively. This\nincrease was primarily due to higher revenues in Men\'s, the Jordan Brand, Women\'s and Kids\' which grew 17%, 35%,11% and 10%, respectively, on a wholesale\nequivalent basis.', metadata={'page': 35, 'source': 'nke-10k-2023.pdf'}), Document(page_content='Enterprise Resource Planning Platform, data and analytics, demand sensing, insight gathering, and other areas to create an end-to-end technology foundation, which we\nbelieve will further accelerate our digital transformation. We believe this unified approach will accelerate growth and unlock more efficiency for our business, while driving\nspeed and responsiveness as we serve consumers globally.\nFINANCIAL HIGHLIGHTS\n•In fiscal 2023, NIKE, Inc. achieved record Revenues of $51.2 billion, which increased 10% and 16% on a reported and currency-neutral basis, respectively\n•NIKE Direct revenues grew 14% from $18.7 billion in fiscal 2022 to $21.3 billion in fiscal 2023, and represented approximately 44% of total NIKE Brand revenues for\nfiscal 2023\n•Gross margin for the fiscal year decreased 250 basis points to 43.5% primarily driven by higher product costs, higher markdowns and unfavorable changes in foreign\ncurrency exchange rates, partially offset by strategic pricing actions', metadata={'page': 30, 'source': 'nke-10k-2023.pdf'}), Document(page_content="Table of Contents\nNORTH AMERICA\n(Dollars in millions) FISCAL 2023FISCAL 2022 % CHANGE% CHANGE\nEXCLUDING\nCURRENCY\nCHANGESFISCAL 2021 % CHANGE% CHANGE\nEXCLUDING\nCURRENCY\nCHANGES\nRevenues by:\nFootwear $ 14,897 $ 12,228 22 % 22 %$ 11,644 5 % 5 %\nApparel 5,947 5,492 8 % 9 % 5,028 9 % 9 %\nEquipment 764 633 21 % 21 % 507 25 % 25 %\nTOTAL REVENUES $ 21,608 $ 18,353 18 % 18 %$ 17,179 7 % 7 %\nRevenues by:    \nSales to Wholesale Customers $ 11,273 $ 9,621 17 % 18 %$ 10,186 -6 % -6 %\nSales through NIKE Direct 10,335 8,732 18 % 18 % 6,993 25 % 25 %\nTOTAL REVENUES $ 21,608 $ 18,353 18 % 18 %$ 17,179 7 % 7 %\nEARNINGS BEFORE INTEREST AND TAXES $ 5,454 $ 5,114 7 % $ 5,089 0 %\nFISCAL 2023 COMPARED TO FISCAL 2022\n•North America revenues increased 18% on a currency-neutral basis, primarily due to higher revenues in Men's and the Jordan Brand. NIKE Direct revenues\nincreased 18%, driven by strong digital sales growth of 23%, comparable store sales growth of 9% and the addition of new stores.", metadata={'page': 39, 'source': 'nke-10k-2023.pdf'}), Document(page_content="Table of Contents\nEUROPE, MIDDLE EAST & AFRICA\n(Dollars in millions) FISCAL 2023FISCAL 2022 % CHANGE% CHANGE\nEXCLUDING\nCURRENCY\nCHANGESFISCAL 2021 % CHANGE% CHANGE\nEXCLUDING\nCURRENCY\nCHANGES\nRevenues by:\nFootwear $ 8,260 $ 7,388 12 % 25 %$ 6,970 6 % 9 %\nApparel 4,566 4,527 1 % 14 % 3,996 13 % 16 %\nEquipment 592 564 5 % 18 % 490 15 % 17 %\nTOTAL REVENUES $ 13,418 $ 12,479 8 % 21 %$ 11,456 9 % 12 %\nRevenues by:    \nSales to Wholesale Customers $ 8,522 $ 8,377 2 % 15 %$ 7,812 7 % 10 %\nSales through NIKE Direct 4,896 4,102 19 % 33 % 3,644 13 % 15 %\nTOTAL REVENUES $ 13,418 $ 12,479 8 % 21 %$ 11,456 9 % 12 %\nEARNINGS BEFORE INTEREST AND TAXES $ 3,531 $ 3,293 7 % $ 2,435 35 % \nFISCAL 2023 COMPARED TO FISCAL 2022\n•EMEA revenues increased 21% on a currency-neutral basis, due to higher revenues in Men's, the Jordan Brand, Women's and Kids'. NIKE Direct revenues\nincreased 33%, driven primarily by strong digital sales growth of 43% and comparable store sales growth of 22%.", metadata={'page': 40, 'source': 'nke-10k-2023.pdf'})], 'answer': "Nike's revenue in fiscal 2023 was $51.2 billion."}

## Build a Question Answering application over a Graph Database

```python
from dotenv import load_dotenv
from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI

# Load environment variables from a .env file
load_dotenv()

# Initialize a connection to the Neo4j graph database
graph = Neo4jGraph()

# Define a query to import movie information into the Neo4j graph database
movies_query = """
LOAD CSV WITH HEADERS FROM 
'https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/movies/movies_small.csv'
AS row
MERGE (m:Movie {id:row.movieId})
SET m.released = date(row.released),
    m.title = row.title,
    m.imdbRating = toFloat(row.imdbRating)
FOREACH (director in split(row.director, '|') | 
    MERGE (p:Person {name:trim(director)})
    MERGE (p)-[:DIRECTED]->(m))
FOREACH (actor in split(row.actors, '|') | 
    MERGE (p:Person {name:trim(actor)})
    MERGE (p)-[:ACTED_IN]->(m))
FOREACH (genre in split(row.genres, '|') | 
    MERGE (g:Genre {name:trim(genre)})
    MERGE (m)-[:IN_GENRE]->(g))
"""

# Execute the query to import the movie information
graph.query(movies_query)

# Refresh the schema to ensure it is up-to-date
graph.refresh_schema()

# Initialize the OpenAI model with specified temperature and model name
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Create a GraphCypherQAChain with the graph and LLM, enabling verbose mode and Cypher query validation
chain = GraphCypherQAChain.from_llm(
    graph=graph, llm=llm, verbose=True, validate_cypher=True
)

# Invoke the chain with a query to get the cast of the movie "Casino"
response = chain.invoke({"query": "What was the cast of the Casino?"})
print(response)
```

    [1m> Entering new GraphCypherQAChain chain...[0m
    Generated Cypher:
    [32;1m[1;3mMATCH (:Movie {title: 'Casino'})<-[:ACTED_IN]-(actor:Person)
    RETURN actor.name[0m
    Full Context:
    [32;1m[1;3m[{'actor.name': 'Robert De Niro'}, {'actor.name': 'Joe Pesci'}, {'actor.name': 'Sharon Stone'}, {'actor.name': 'James Woods'}][0m
    
    [1m> Finished chain.[0m
    {'query': 'What was the cast of the Casino?', 'result': 'The cast of Casino included Robert De Niro, Joe Pesci, Sharon Stone, and James Woods.'}

## Build a local RAG application

```python
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.llms.llamafile import Llamafile
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load the data from a web page
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
data = loader.load()

# Split the loaded data into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

# Create a Chroma vector store from the document chunks, using HuggingFace embeddings
vectorstore = Chroma.from_documents(
    documents=all_splits,
    embedding=HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    ),
)

# Initialize the Llamafile language model
llm = Llamafile()


# Define a function to format documents into a single string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Pull a predefined RAG (Retrieval-Augmented Generation) prompt template from the hub
rag_prompt = hub.pull("rlm/rag-prompt")

# Define the question to be asked
question = "What are the approaches to Task Decomposition?"

# Create a retriever from the vector store for similarity-based search
retriever = vectorstore.as_retriever()

# Create a question-answering chain using the retriever, RAG prompt, LLM, and output parser
qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

# Invoke the question-answering chain with the specified question and print the result
print(qa_chain.invoke(question))
```

    Step 1: Determine the task: Task decomposition is used when a given task has multiple subgoals or goals. In this case, it's best to identify the most important goal or goals and then separate them into their respective tasks. For example, "Write a story outline" might be a separate task from "Write a novel."
    
    Step 2: Develop prompts for each task: Determine how you will guide the assistant towards achieving each subgoal or goal. Depending on the context, this can involve asking questions, providing instructions, or suggesting solutions. For example, if you're working on writing a novel and your assistant wants to know what the subgoals are, you might give them "Write a story outline."
    
    Step 3: Provide feedback and guidance: As the assistant works through each task, provide feedback and guidance as necessary to help them succeed. For example, if they're working on writing a story outline, you may ask them questions about their approach or suggest improvements based on your prior experience with this type of work.
    
    Step 4: Celebrate successes: When the assistant completes each task successfully, celebrate their achievement! This can help reinforce and build confidence in their abilities.
    
    Overall, using task decomposition is a great way to break down complex tasks into smaller, more manageable parts. By providing prompts, feedback, and guidance as needed, you can help your assistant succeed and achieve their goals.</s>

## Build a Query Analysis System

```python
import datetime
from typing import List, Optional

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import YoutubeLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables from a .env file
load_dotenv()

# Initialize the OpenAI model with the specified version and temperature
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

# List of YouTube video URLs to process
urls = [
    "https://www.youtube.com/watch?v=HAn9vnJy6S4",
    "https://www.youtube.com/watch?v=dA1cHGACXCo",
    "https://www.youtube.com/watch?v=ZcEMLz27sL4",
    "https://www.youtube.com/watch?v=hvAPnpSfSGo",
    "https://www.youtube.com/watch?v=EhlPDL4QrWY",
    "https://www.youtube.com/watch?v=mmBo8nlu2j0",
    "https://www.youtube.com/watch?v=rQdibOsL1ps",
    "https://www.youtube.com/watch?v=28lC4fqukoc",
    "https://www.youtube.com/watch?v=es-9MgxB-uc",
    "https://www.youtube.com/watch?v=wLRHwKuKvOE",
    "https://www.youtube.com/watch?v=ObIltMaRJvY",
    "https://www.youtube.com/watch?v=DjuXACWYkkU",
    "https://www.youtube.com/watch?v=o7C9ld6Ln-M",
]

# Load the documents from the YouTube videos
docs = []
for url in urls:
    docs.extend(YoutubeLoader.from_youtube_url(url, add_video_info=True).load())

# Add additional metadata: the year the video was published
for doc in docs:
    doc.metadata["publish_year"] = int(
        datetime.datetime.strptime(
            doc.metadata["publish_date"], "%Y-%m-%d %H:%M:%S"
        ).strftime("%Y")
    )

# Split the documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
chunked_docs = text_splitter.split_documents(docs)

# Create embeddings for the document chunks
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(
    chunked_docs,
    embeddings,
)


# Define a Pydantic model for the search
class Search(BaseModel):
    """Search over a database of tutorial videos about a software library."""

    query: str = Field(
        ...,
        description="Similarity search query applied to video transcripts.",
    )
    publish_year: Optional[int] = Field(None, description="Year video was published")


# Define the system prompt for converting user questions into database queries
system = """You are an expert at converting user questions into database queries. \
You have access to a database of tutorial videos about a software library for building LLM-powered applications. \
Given a question, return a list of database queries optimized to retrieve the most relevant results.

If there are acronyms or words you are not familiar with, do not try to rephrase them."""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

# Create a structured LLM that outputs a Search object
structured_llm = llm.with_structured_output(Search)
query_analyzer = {"question": RunnablePassthrough()} | prompt | structured_llm


# Define a retrieval function that searches the vector store based on the search criteria
def retrieval(search: Search) -> List[Document]:
    if search.publish_year is not None:
        # Apply a filter for the publish year if specified
        _filter = {"publish_year": {"$eq": search.publish_year}}
    else:
        _filter = None
    # Perform a similarity search on the vector store
    return vectorstore.similarity_search(search.query, filter=_filter)


# Create a retrieval chain combining query analysis and retrieval
retrieval_chain = query_analyzer | retrieval

# Invoke the retrieval chain with a query
results = retrieval_chain.invoke("RAG tutorial published in 2023")

# Print the titles and publish dates of the retrieved documents
print([(doc.metadata["title"], doc.metadata["publish_date"]) for doc in results])
```

    [('Getting Started with Multi-Modal LLMs', '2023-12-20 00:00:00'), ('LangServe and LangChain Templates Webinar', '2023-11-02 00:00:00'), ('Getting Started with Multi-Modal LLMs', '2023-12-20 00:00:00'), ('Building a Research Assistant from Scratch', '2023-11-16 00:00:00')]

## Build a Question/Answering system over SQL data

### Agents

```python
import ast
import re

from dotenv import load_dotenv
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.prebuilt import create_react_agent

# Load environment variables from a .env file
load_dotenv()

# Initialize the OpenAI model with the specified version
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

# Create a connection to the SQL database
db = SQLDatabase.from_uri("sqlite:///Chinook.db")

# Create tools from a toolkit for interacting with the SQL database using the LLM
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()


# Define a function to run a query on the database and process the results
def query_as_list(database, sql_query):
    # Run the query on the database
    res = database.run(sql_query)
    # Parse the result into a list of values
    res = [el for sub in ast.literal_eval(res) for el in sub if el]
    # Remove numeric values and strip whitespace
    res = [re.sub(r"\b\d+\b", "", string).strip() for string in res]
    # Return unique values as a list
    return list(set(res))


# Query the list of artist names and album titles from the database
artists = query_as_list(db, "SELECT Name FROM Artist")
albums = query_as_list(db, "SELECT Title FROM Album")

# Create a FAISS vector store from the combined list of artists and albums, using OpenAI embeddings
vector_db = FAISS.from_texts(artists + albums, OpenAIEmbeddings())

# Create a retriever from the vector store for similarity-based search
retriever = vector_db.as_retriever(search_kwargs={"k": 5})

# Define a description for the retriever tool
description = """Use to look up values to filter on. Input is an approximate spelling of the proper noun, output is \
valid proper nouns. Use the noun most similar to the search."""

# Create a retriever tool with the defined retriever and description
retriever_tool = create_retriever_tool(
    retriever,
    name="search_proper_nouns",
    description=description,
)

# Define a system message with instructions for the agent
system = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct SQLite query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the given tools. Only use the information returned by the tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

You have access to the following tables: {table_names}

If you need to filter on a proper noun, you must ALWAYS first look up the filter value using the "search_proper_nouns" tool!
Do not try to guess at the proper name - use this function to find similar ones.""".format(
    table_names=db.get_usable_table_names()
)

# Create a system message with the defined instructions
system_message = SystemMessage(content=system)

# Append the retriever tool to the list of tools
tools.append(retriever_tool)

# Create an agent with the LLM, tools, and system message
agent = create_react_agent(llm, tools, messages_modifier=system_message)

# Define the first query
query = "How many albums does alis in chain have?"

# Stream responses for the first query using the agent
for s in agent.stream({"messages": [HumanMessage(content=query)]}):
    # Print each response and a separator
    print(s)
    print("----")
```

    {'agent': {'messages': [AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_nfil0RLn0K4yO8hLJXK7xcY5', 'function': {'arguments': '{"query":"alis in chain"}', 'name': 'search_proper_nouns'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 19, 'prompt_tokens': 674, 'total_tokens': 693}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-0c8c598b-d28d-4186-9d82-7228f1618f76-0', tool_calls=[{'name': 'search_proper_nouns', 'args': {'query': 'alis in chain'}, 'id': 'call_nfil0RLn0K4yO8hLJXK7xcY5'}], usage_metadata={'input_tokens': 674, 'output_tokens': 19, 'total_tokens': 693})]}}
    ----
    {'tools': {'messages': [ToolMessage(content='Alice In Chains\n\nAisha Duo\n\nXis\n\nDa Lama Ao Caos\n\nA-Sides', name='search_proper_nouns', tool_call_id='call_nfil0RLn0K4yO8hLJXK7xcY5')]}}
    ----
    {'agent': {'messages': [AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_Y1jpNuia5BXlZvzreUbsv2uO', 'function': {'arguments': '{"query":"SELECT COUNT(*) AS album_count FROM Album WHERE ArtistId IN (SELECT ArtistId FROM Artist WHERE Name = \'Alice In Chains\')"}', 'name': 'sql_db_query'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 40, 'prompt_tokens': 724, 'total_tokens': 764}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-30a2eb63-6d03-498e-b9bf-efefb0a77fae-0', tool_calls=[{'name': 'sql_db_query', 'args': {'query': "SELECT COUNT(*) AS album_count FROM Album WHERE ArtistId IN (SELECT ArtistId FROM Artist WHERE Name = 'Alice In Chains')"}, 'id': 'call_Y1jpNuia5BXlZvzreUbsv2uO'}], usage_metadata={'input_tokens': 724, 'output_tokens': 40, 'total_tokens': 764})]}}
    ----
    {'tools': {'messages': [ToolMessage(content='[(1,)]', name='sql_db_query', tool_call_id='call_Y1jpNuia5BXlZvzreUbsv2uO')]}}
    ----
    {'agent': {'messages': [AIMessage(content='Alice In Chains has 1 album.', response_metadata={'token_usage': {'completion_tokens': 9, 'prompt_tokens': 777, 'total_tokens': 786}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-ea762f3d-a900-47a8-98da-875bb19851ef-0', usage_metadata={'input_tokens': 777, 'output_tokens': 9, 'total_tokens': 786})]}}
    ----

### Chains

```python
from operator import itemgetter

from dotenv import load_dotenv
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

# Load environment variables from a .env file
load_dotenv()

# Initialize the OpenAI model with the specified version
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

# Create a connection to the SQL database
db = SQLDatabase.from_uri("sqlite:///Chinook.db")

# Create an initial SQL query chain with the LLM and the database
chain = create_sql_query_chain(llm, db)

# Create a tool to execute SQL queries on the database
execute_query = QuerySQLDataBaseTool(db=db)

# Create another SQL query chain for writing queries
write_query = create_sql_query_chain(llm, db)

# Combine the write query chain and the execute query tool into a single chain
chain = write_query | execute_query

# Define a prompt template to generate an answer based on the user question, SQL query, and SQL result
answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)

# Combine the query writing, execution, and answering into a single chain
chain = (
    RunnablePassthrough.assign(
        query=write_query
    ).assign(  # Assign the query writing chain
        result=itemgetter("query") | execute_query
    )  # Assign the result execution chain
    | answer_prompt  # Use the answer prompt to format the response
    | llm  # Use the LLM to generate the final answer
    | StrOutputParser()  # Parse the output as a string
)

# Invoke the chain with a user question
answer = chain.invoke({"question": "How many employees are there"})

# Print the answer
print(answer)
```

    There are 8 employees.

## Build a Conversational RAG Application

### Agents

```python
import bs4
from dotenv import load_dotenv
from langchain.tools.retriever import create_retriever_tool
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import create_react_agent

# Load environment variables from a .env file
load_dotenv()

# Initialize an in-memory SQLite saver for storing checkpoints
memory = SqliteSaver.from_conn_string(":memory:")

# Initialize the OpenAI model with the specified version and temperature
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Load, chunk, and index the contents of the blog

# Define a web base loader to load the contents of the specified blog URL
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={
        "parse_only": bs4.SoupStrainer(
            class_=(
                "post-content",
                "post-title",
                "post-header",
            )  # Specify the classes to parse
        )
    },
)

# Load the documents from the web page
docs = loader.load()

# Define a text splitter to chunk the documents into smaller pieces
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Split the loaded documents into chunks
splits = text_splitter.split_documents(docs)

# Create a Chroma vector store from the document chunks, using OpenAI embeddings
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# Create a retriever from the vector store for similarity-based search
retriever = vectorstore.as_retriever()

# Build retriever tool
tool = create_retriever_tool(
    retriever,
    "blog_post_retriever",  # Name of the retriever tool
    "Searches and returns excerpts from the Autonomous Agents blog post.",  # Description of the retriever tool
)
tools = [tool]

# Create an agent executor with the LLM, tools, and checkpoint saver
agent_executor = create_react_agent(llm, tools, checkpointer=memory)

# Configuration dictionary for the session
config = {"configurable": {"thread_id": "abc123"}}

# Define the first query
query = "What is Task Decomposition?"

# Stream responses for the query using the agent executor
for s in agent_executor.stream(
    {"messages": [HumanMessage(content=query)]}, config=config
):
    # Print each response and a separator
    print(s)
    print("----")

# Define the second query
query = "What according to the blog post are common ways of doing it? redo the search"

# Stream responses for the conversational query using the agent executor
for s in agent_executor.stream(
    {"messages": [HumanMessage(content=query)]}, config=config
):
    # Print each response and a separator
    print(s)
    print("----")
```

    {'agent': {'messages': [AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_KasVENthsc22RFkLShcQO1ro', 'function': {'arguments': '{"query":"Task Decomposition"}', 'name': 'blog_post_retriever'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 19, 'prompt_tokens': 68, 'total_tokens': 87}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-d8f6069c-a1e5-4e4d-bec3-0e64fabff736-0', tool_calls=[{'name': 'blog_post_retriever', 'args': {'query': 'Task Decomposition'}, 'id': 'call_KasVENthsc22RFkLShcQO1ro'}], usage_metadata={'input_tokens': 68, 'output_tokens': 19, 'total_tokens': 87})]}}
    ----
    {'tools': {'messages': [ToolMessage(content='Fig. 1. Overview of a LLM-powered autonomous agent system.\nComponent One: Planning#\nA complicated task usually involves many steps. An agent needs to know what they are and plan ahead.\nTask Decomposition#\nChain of thought (CoT; Wei et al. 2022) has become a standard prompting technique for enhancing model performance on complex tasks. The model is instructed to “think step by step” to utilize more test-time computation to decompose hard tasks into smaller and simpler steps. CoT transforms big tasks into multiple manageable tasks and shed lights into an interpretation of the model’s thinking process.\n\nFig. 1. Overview of a LLM-powered autonomous agent system.\nComponent One: Planning#\nA complicated task usually involves many steps. An agent needs to know what they are and plan ahead.\nTask Decomposition#\nChain of thought (CoT; Wei et al. 2022) has become a standard prompting technique for enhancing model performance on complex tasks. The model is instructed to “think step by step” to utilize more test-time computation to decompose hard tasks into smaller and simpler steps. CoT transforms big tasks into multiple manageable tasks and shed lights into an interpretation of the model’s thinking process.\n\nFig. 1. Overview of a LLM-powered autonomous agent system.\nComponent One: Planning#\nA complicated task usually involves many steps. An agent needs to know what they are and plan ahead.\nTask Decomposition#\nChain of thought (CoT; Wei et al. 2022) has become a standard prompting technique for enhancing model performance on complex tasks. The model is instructed to “think step by step” to utilize more test-time computation to decompose hard tasks into smaller and simpler steps. CoT transforms big tasks into multiple manageable tasks and shed lights into an interpretation of the model’s thinking process.\n\nFig. 1. Overview of a LLM-powered autonomous agent system.\nComponent One: Planning#\nA complicated task usually involves many steps. An agent needs to know what they are and plan ahead.\nTask Decomposition#\nChain of thought (CoT; Wei et al. 2022) has become a standard prompting technique for enhancing model performance on complex tasks. The model is instructed to “think step by step” to utilize more test-time computation to decompose hard tasks into smaller and simpler steps. CoT transforms big tasks into multiple manageable tasks and shed lights into an interpretation of the model’s thinking process.', name='blog_post_retriever', tool_call_id='call_KasVENthsc22RFkLShcQO1ro')]}}
    ----
    {'agent': {'messages': [AIMessage(content='Task decomposition is a technique used to break down complex tasks into smaller and simpler steps. This approach involves transforming big tasks into multiple manageable tasks, allowing for a more systematic and structured approach to problem-solving. By decomposing tasks, agents can better understand the steps involved and plan ahead effectively. One common method for task decomposition is the Chain of Thought (CoT) technique, which instructs models to "think step by step" and decompose hard tasks into smaller components. This technique enhances model performance on complex tasks by utilizing more test-time computation.', response_metadata={'token_usage': {'completion_tokens': 110, 'prompt_tokens': 590, 'total_tokens': 700}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-76e138bb-0372-47a9-9890-4fde4fbe155a-0', usage_metadata={'input_tokens': 590, 'output_tokens': 110, 'total_tokens': 700})]}}
    ----
    {'agent': {'messages': [AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_xHD7lAimugv43jycSvZrbSZ8', 'function': {'arguments': '{"query":"common ways of task decomposition"}', 'name': 'blog_post_retriever'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 21, 'prompt_tokens': 723, 'total_tokens': 744}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-78e75628-1c19-4ce8-804a-7b0487f1d419-0', tool_calls=[{'name': 'blog_post_retriever', 'args': {'query': 'common ways of task decomposition'}, 'id': 'call_xHD7lAimugv43jycSvZrbSZ8'}], usage_metadata={'input_tokens': 723, 'output_tokens': 21, 'total_tokens': 744})]}}
    ----
    {'tools': {'messages': [ToolMessage(content='Tree of Thoughts (Yao et al. 2023) extends CoT by exploring multiple reasoning possibilities at each step. It first decomposes the problem into multiple thought steps and generates multiple thoughts per step, creating a tree structure. The search process can be BFS (breadth-first search) or DFS (depth-first search) with each state evaluated by a classifier (via a prompt) or majority vote.\nTask decomposition can be done (1) by LLM with simple prompting like "Steps for XYZ.\\n1.", "What are the subgoals for achieving XYZ?", (2) by using task-specific instructions; e.g. "Write a story outline." for writing a novel, or (3) with human inputs.\n\nTree of Thoughts (Yao et al. 2023) extends CoT by exploring multiple reasoning possibilities at each step. It first decomposes the problem into multiple thought steps and generates multiple thoughts per step, creating a tree structure. The search process can be BFS (breadth-first search) or DFS (depth-first search) with each state evaluated by a classifier (via a prompt) or majority vote.\nTask decomposition can be done (1) by LLM with simple prompting like "Steps for XYZ.\\n1.", "What are the subgoals for achieving XYZ?", (2) by using task-specific instructions; e.g. "Write a story outline." for writing a novel, or (3) with human inputs.\n\nTree of Thoughts (Yao et al. 2023) extends CoT by exploring multiple reasoning possibilities at each step. It first decomposes the problem into multiple thought steps and generates multiple thoughts per step, creating a tree structure. The search process can be BFS (breadth-first search) or DFS (depth-first search) with each state evaluated by a classifier (via a prompt) or majority vote.\nTask decomposition can be done (1) by LLM with simple prompting like "Steps for XYZ.\\n1.", "What are the subgoals for achieving XYZ?", (2) by using task-specific instructions; e.g. "Write a story outline." for writing a novel, or (3) with human inputs.\n\nTree of Thoughts (Yao et al. 2023) extends CoT by exploring multiple reasoning possibilities at each step. It first decomposes the problem into multiple thought steps and generates multiple thoughts per step, creating a tree structure. The search process can be BFS (breadth-first search) or DFS (depth-first search) with each state evaluated by a classifier (via a prompt) or majority vote.\nTask decomposition can be done (1) by LLM with simple prompting like "Steps for XYZ.\\n1.", "What are the subgoals for achieving XYZ?", (2) by using task-specific instructions; e.g. "Write a story outline." for writing a novel, or (3) with human inputs.', name='blog_post_retriever', tool_call_id='call_xHD7lAimugv43jycSvZrbSZ8')]}}
    ----
    {'agent': {'messages': [AIMessage(content='According to the blog post, common ways of task decomposition include:\n\n1. Using LLM with simple prompting, such as "Steps for XYZ" or "What are the subgoals for achieving XYZ?"\n2. Using task-specific instructions, for example, "Write a story outline" for writing a novel.\n3. Involving human inputs in the task decomposition process.\n\nAdditionally, the Tree of Thoughts technique extends the Chain of Thought (CoT) method by exploring multiple reasoning possibilities at each step. It decomposes the problem into multiple thought steps and generates multiple thoughts per step, creating a tree structure. The search process can be BFS (breadth-first search) or DFS (depth-first search) with each state evaluated by a classifier or majority vote.', response_metadata={'token_usage': {'completion_tokens': 152, 'prompt_tokens': 1339, 'total_tokens': 1491}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-84da0fe7-80b7-47dc-a0ee-3393856f3b3e-0', usage_metadata={'input_tokens': 1339, 'output_tokens': 152, 'total_tokens': 1491})]}}
    ----

### Chains

```python
import bs4
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables from a .env file
load_dotenv()

# Initialize the OpenAI model with the specified version and temperature
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Construct retriever

# Define a web base loader to load the contents of the specified blog URL
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={
        "parse_only": bs4.SoupStrainer(
            class_=(
                "post-content",
                "post-title",
                "post-header",
            )  # Specify the classes to parse
        )
    },
)

# Load the documents from the web page
docs = loader.load()

# Define a text splitter to chunk the documents into smaller pieces
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Split the loaded documents into chunks
splits = text_splitter.split_documents(docs)

# Create a Chroma vector store from the document chunks, using OpenAI embeddings
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# Create a retriever from the vector store for similarity-based search
retriever = vectorstore.as_retriever()

# Contextualize question

# Define a system prompt for contextualizing the question
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

# Create a chat prompt template for contextualizing the question
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create a history-aware retriever using the LLM and the contextualize question prompt
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Answer question

# Define a system prompt for answering the question
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

# Create a chat prompt template for answering the question
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create a chain to combine retrieved documents and the question-answering task
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Create a retrieval-augmented generation (RAG) chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Statefully manage chat history

# Initialize a dictionary to store session histories
store = {}


# Function to retrieve or create a session history
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# Create a runnable with message history for the RAG chain
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# Invoke the conversational RAG chain with the first query
print(
    conversational_rag_chain.invoke(
        {"input": "What is Task Decomposition?"},
        config={
            "configurable": {"session_id": "abc123"}
        },  # Constructs a key "abc123" in `store`.
    )["answer"]
)

# Invoke the conversational RAG chain with the second query
print(
    conversational_rag_chain.invoke(
        {"input": "What are common ways of doing it?"},
        config={"configurable": {"session_id": "abc123"}},
    )["answer"]
)
```

    Task Decomposition is a technique used to break down complex tasks into smaller and simpler steps. It involves transforming big tasks into multiple manageable tasks to make them easier to accomplish. This approach helps agents or models to better understand and interpret the thinking process involved in completing a task.
    One common way of task decomposition is through the use of techniques like Chain of Thought (CoT), where models are instructed to "think step by step" to break down hard tasks. Another approach is to divide the task into smaller subtasks that can be tackled individually. These methods help in enhancing model performance on complex tasks by simplifying the overall process.

## Build a Retrieval Augmented Generation (RAG) Application

```python
import bs4
from dotenv import load_dotenv
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables from a .env file
load_dotenv()

# Initialize the OpenAI model with the specified version
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

# Load, chunk, and index the contents of the blog

# Define a web base loader to load the contents of the specified blog URL
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={
        "parse_only": bs4.SoupStrainer(
            class_=(
                "post-content",
                "post-title",
                "post-header",
            )  # Specify the classes to parse
        )
    },
)

# Load the documents from the web page
docs = loader.load()

# Define a text splitter to chunk the documents into smaller pieces
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Split the loaded documents into chunks
splits = text_splitter.split_documents(docs)

# Create a Chroma vector store from the document chunks, using OpenAI embeddings
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# Retrieve and generate using the relevant snippets of the blog

# Create a retriever from the vector store for similarity-based search
retriever = vectorstore.as_retriever()

# Pull a predefined prompt template from the hub
prompt = hub.pull("rlm/rag-prompt")


# Function to format the documents into a single string
def format_docs(documents):
    return "\n\n".join(doc.page_content for doc in documents)


# Define the RAG (Retrieval-Augmented Generation) chain with context retriever, question passthrough, prompt, and LLM
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Invoke the RAG chain with a question about Task Decomposition and print the response
print(rag_chain.invoke("What is Task Decomposition?"))
```

    Task Decomposition involves breaking down complex tasks into smaller and simpler steps. This technique, such as Chain of Thought, helps enhance model performance by transforming big tasks into multiple manageable tasks through a step-by-step approach. It allows for a clearer interpretation of the model's thinking process.

## Build an Agent

```python
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import create_react_agent

# Load environment variables from a .env file
load_dotenv()

# Initialize the OpenAI chat model
model = ChatOpenAI()

# Initialize the search tool with a limit of 2 results per query
search = TavilySearchResults(max_results=2)
tools = [search]

# Initialize an in-memory SQLite database for saving agent state
memory = SqliteSaver.from_conn_string(":memory:")

# Create a reactive agent executor with the model, tools, and memory
agent_executor = create_react_agent(model, tools, checkpointer=memory)

# Configuration for the agent's execution, including a thread ID
config = {"configurable": {"thread_id": "abc123"}}

# Execute the agent with a greeting message and print the response chunks
for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="hi im bob! and i live in sf")]}, config
):
    print(chunk)
    print("----")

# Execute the agent with a conversational memory query and print the response chunks
for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="whats the weather where I live?")]}, config
):
    print(chunk)
    print("----")
```

    {'agent': {'messages': [AIMessage(content='Hello Bob! How can I assist you today regarding San Francisco?', response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 90, 'total_tokens': 104}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-c2aaf030-867b-48a8-95bb-0bba95ac8913-0', usage_metadata={'input_tokens': 90, 'output_tokens': 14, 'total_tokens': 104})]}}
    ----
    {'agent': {'messages': [AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_5Hz5aZjPY0XmnJBvvihCl0Wj', 'function': {'arguments': '{"query":"San Francisco weather"}', 'name': 'tavily_search_results_json'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 20, 'prompt_tokens': 119, 'total_tokens': 139}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-544d5b48-e9ca-42b6-b5a8-c3771c972c67-0', tool_calls=[{'name': 'tavily_search_results_json', 'args': {'query': 'San Francisco weather'}, 'id': 'call_5Hz5aZjPY0XmnJBvvihCl0Wj'}], usage_metadata={'input_tokens': 119, 'output_tokens': 20, 'total_tokens': 139})]}}
    ----
    {'tools': {'messages': [ToolMessage(content='[{"url": "https://www.accuweather.com/en/us/san-francisco/94103/july-weather/347629", "content": "Get the monthly weather forecast for San Francisco, CA, including daily high/low, historical averages, to help you plan ahead.Missing:  09/07/2024"}, {"url": "https://www.weather25.com/north-america/usa/california/san-francisco?page=month&month=September", "content": "The temperatures in San Francisco in September are comfortable with low of 57\\u00b0F and and high up to 77\\u00b0F. There is little to no rain in San Francisco during\\u00a0..."}]', name='tavily_search_results_json', tool_call_id='call_5Hz5aZjPY0XmnJBvvihCl0Wj')]}}
    ----
    {'agent': {'messages': [AIMessage(content='The weather in San Francisco for September typically ranges from a low of 57°F to a high of 77°F. There is usually little to no rain during this time. If you would like more detailed information, you can visit [AccuWeather](https://www.accuweather.com/en/us/san-francisco/94103/september-weather/347629) or [Weather25](https://www.weather25.com/north-america/usa/california/san-francisco?page=month&month=September).', response_metadata={'token_usage': {'completion_tokens': 111, 'prompt_tokens': 307, 'total_tokens': 418}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-648f201c-9285-43f3-a157-8de9c00278c1-0', usage_metadata={'input_tokens': 307, 'output_tokens': 111, 'total_tokens': 418})]}}
    ----

## Build vector stores and retrievers

```python
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Load environment variables from a .env file
load_dotenv()

# Create a list of documents, each with content and metadata
documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Goldfish are popular pets for beginners, requiring relatively simple care.",
        metadata={"source": "fish-pets-doc"},
    ),
    Document(
        page_content="Parrots are intelligent birds capable of mimicking human speech.",
        metadata={"source": "bird-pets-doc"},
    ),
    Document(
        page_content="Rabbits are social animals that need plenty of space to hop around.",
        metadata={"source": "mammal-pets-doc"},
    ),
]

# Create a Chroma vector store from the documents, using OpenAI embeddings
vectorstore = Chroma.from_documents(
    documents,
    embedding=OpenAIEmbeddings(),
)

# Create a retriever from the vector store for similarity-based search
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},  # Retrieve the top 1 similar document
)

# Initialize the OpenAI model with the specified version
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

# Define a message template for the chat prompt
message = """
Answer this question using the provided context only.

{question}

Context:
{context}
"""

# Create a chat prompt template from the message
prompt = ChatPromptTemplate.from_messages([("human", message)])

# Define the RAG (Retrieval-Augmented Generation) chain with context retriever and question passthrough
rag_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm

# Invoke the RAG chain with a question about cats
response = rag_chain.invoke("tell me about cats")

# Print the response content
print(response.content)
```

    Cats are independent pets that often enjoy their own space.

## Build a Chatbot

```python
from operator import itemgetter

from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

# Load environment variables from a .env file
load_dotenv()

# Initialize the OpenAI model with the gpt-3.5-turbo model
model = ChatOpenAI(model="gpt-3.5-turbo")

# Initialize a dictionary to store session histories
store = {}


# Function to retrieve or create a session history
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:

        store[session_id] = ChatMessageHistory()

    return store[session_id]


# Define a chat prompt template with system and placeholder messages
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Define a trimmer to trim messages to a maximum token count
trimmer = trim_messages(
    max_tokens=65,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

# Define the initial set of messages
messages = [
    SystemMessage(content="you're a good assistant"),
    HumanMessage(content="hi! I'm bob"),
    AIMessage(content="hi!"),
    HumanMessage(content="I like vanilla ice cream"),
    AIMessage(content="nice"),
    HumanMessage(content="whats 2 + 2"),
    AIMessage(content="4"),
    HumanMessage(content="thanks"),
    AIMessage(content="no problem!"),
    HumanMessage(content="having fun?"),
    AIMessage(content="yes!"),
    HumanMessage(content="hi! I'm bob"),
]

# Create a runnable chain that processes messages
chain = (
    RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer)
    | prompt
    | model
)

# Create a runnable with message history, binding the chain with the session history function
with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages",
)

# Configuration dictionary for the session
config = {"configurable": {"session_id": "abc16"}}

# Stream responses by passing messages and language configuration to the runnable with message history
for r in with_message_history.stream(
    {
        "messages": messages + [HumanMessage(content="whats my name?")],
        "language": "English",
    },
    config=config,
):
    print(r.content, end="|")
```

    |Your| name| is| Bob|.||

## Build a Simple LLM Application with LCEL

```python
#!/usr/bin/env python
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langserve import add_routes

# 0. Load environment variables from a .env file
load_dotenv()

# 1. Create prompt template
system_template = "Translate the following into {language}:"
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

# 2. Create model
model = ChatOpenAI()

# 3. Create parser
parser = StrOutputParser()

# 4. Create chain
chain = prompt_template | model | parser

# 5. App definition
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server using LangChain's Runnable interfaces",
)

# 6. Adding chain route
add_routes(
    app,
    chain,
    path="/chain",
)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
```

```python
from langserve import RemoteRunnable

remote_chain = RemoteRunnable("http://localhost:8000/chain/")
remote_chain.invoke({"language": "italian", "text": "hi"})
```

    'Ciao'

```python

```
