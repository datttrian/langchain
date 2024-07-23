from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Load environment variables from a .env file
load_dotenv()

# Initialize the OpenAI model with specified model name and temperature
model = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.0)

# Define the examples for few-shot learning
examples = [
    {"input": "2 🦜 2", "output": "4"},
    {"input": "2 🦜 3", "output": "5"},
    {"input": "2 🦜 4", "output": "6"},
    {"input": "What did the cow say to the moon?", "output": "nothing at all"},
    {
        "input": "Write me a poem about the moon",
        "output": "One for the moon, and one for me, who are we to talk about the moon?",
    },
]

# Concatenate input and output values for each example to vectorize
to_vectorize = [" ".join(example.values()) for example in examples]

# Create embeddings and a vector store from the examples
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=examples)

# Create a semantic similarity example selector
example_selector = SemanticSimilarityExampleSelector(
    vectorstore=vectorstore,
    k=2,  # Number of examples to select
)

# Select examples similar to the input "horse"
example_selector.select_examples({"input": "horse"})

# Define the few-shot prompt
few_shot_prompt = FewShotChatMessagePromptTemplate(
    input_variables=["input"],  # Input variables to pass to the example_selector
    example_selector=example_selector,
    example_prompt=ChatPromptTemplate.from_messages(
        [("human", "{input}"), ("ai", "{output}")]
    ),  # Define how each example will be formatted
)

# Define the final prompt template
final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a wondrous wizard of math."),
        few_shot_prompt,  # Include the few-shot prompt
        ("human", "{input}"),
    ]
)

# Create a chain by combining the final prompt template and the model
chain = final_prompt | model

# Invoke the chain with the input and print the result
print(chain.invoke({"input": "What's 3 🦜 3?"}))
