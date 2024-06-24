import os

import openai
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables from a .env file
load_dotenv()

# Set OpenAI API key from environment variables
openai.api_key = os.environ["OPENAI_API_KEY"]

# Initialize the language model
llm = ChatOpenAI()

# Initialize the embeddings model
embeddings = OpenAIEmbeddings()

# Load documents from a specified web page
loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
docs = loader.load()

# Split the loaded documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)

# Create a vector store (FAISS) from the document embeddings
vector = FAISS.from_documents(documents, embeddings)

# Set up a retriever using the vector store to fetch relevant documents based on the query
retriever = vector.as_retriever()

# Define the prompt template for the language model to generate the search query
prompt = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        (
            "user",
            "Given the above conversation, generate a search query to look up to get information relevant to the conversation",
        ),
    ]
)

# Create a document chain that will take the documents and the prompt to generate answers
document_chain = create_stuff_documents_chain(llm, prompt)

# Create a retrieval chain that combines the retriever and the document chain
retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

chat_history = [
    HumanMessage(content="Can LangSmith help test my LLM applications?"),
    AIMessage(content="Yes!"),
]
retriever_chain.invoke({"chat_history": chat_history, "input": "Tell me how"})
