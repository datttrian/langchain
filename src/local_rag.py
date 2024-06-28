from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.llms.llamafile import Llamafile
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
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

# Define a prompt template for summarizing the main themes in the retrieved documents
prompt = PromptTemplate.from_template(
    "Summarize the main themes in these retrieved docs: {docs}"
)


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
