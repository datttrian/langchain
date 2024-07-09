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
