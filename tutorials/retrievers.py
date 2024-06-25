import os
import openai
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

# Load environment variables from the .env file
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

# Create sample documents with content and metadata
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

# Initialize Chroma vector store with OpenAI embeddings
vectorstore = Chroma.from_documents(
    documents,
    embedding=OpenAIEmbeddings(),
)

# Create a retriever using the vector store's as_retriever method
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)

# Set up a minimal retrieval-augmented generation (RAG) application
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Define the message template for RAG
message = """
Answer this question using the provided context only.

{question}

Context:
{context}
"""

# Create a chat prompt template from the message
prompt = ChatPromptTemplate.from_messages([("human", message)])

# Define the RAG chain using context retriever, passthrough question, prompt, and language model
rag_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm

# Invoke the RAG chain with a query
response = rag_chain.invoke("tell me about cats")

# Print the response content
print(response.content)
