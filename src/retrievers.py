from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

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


vectorstore = Chroma.from_documents(
    documents,
    embedding=OpenAIEmbeddings(),
)
vectorstore.similarity_search("cat")

vectorstore.similarity_search_with_score("cat")

embedding = OpenAIEmbeddings().embed_query("cat")

vectorstore.similarity_search_by_vector(embedding)


retriever = RunnableLambda(vectorstore.similarity_search).bind(k=1)  # select top result

retriever.batch(["cat", "shark"])

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)

retriever.batch(["cat", "shark"])


load_dotenv()


llm = ChatOpenAI(model="gpt-3.5-turbo-0125")


message = """
Answer this question using the provided context only.

{question}

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([("human", message)])

rag_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm

response = rag_chain.invoke("tell me about cats")

print(response.content)
