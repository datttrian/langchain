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
