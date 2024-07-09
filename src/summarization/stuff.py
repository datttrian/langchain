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
