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
