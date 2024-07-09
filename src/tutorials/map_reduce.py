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
