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
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=(
                "post-content",
                "post-title",
                "post-header",
            )  # Specify the classes to parse
        )
    ),
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
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Define the RAG (Retrieval-Augmented Generation) chain with context retriever, question passthrough, prompt, and LLM
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Invoke the RAG chain with a question about Task Decomposition and print the response
print(rag_chain.invoke("What is Task Decomposition?"))
