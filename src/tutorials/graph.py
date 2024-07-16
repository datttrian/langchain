from dotenv import load_dotenv
from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI

# Load environment variables from a .env file
load_dotenv()

# Initialize a connection to the Neo4j graph database
graph = Neo4jGraph()

# Define a query to import movie information into the Neo4j graph database
movies_query = """
LOAD CSV WITH HEADERS FROM 
'https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/movies/movies_small.csv'
AS row
MERGE (m:Movie {id:row.movieId})
SET m.released = date(row.released),
    m.title = row.title,
    m.imdbRating = toFloat(row.imdbRating)
FOREACH (director in split(row.director, '|') | 
    MERGE (p:Person {name:trim(director)})
    MERGE (p)-[:DIRECTED]->(m))
FOREACH (actor in split(row.actors, '|') | 
    MERGE (p:Person {name:trim(actor)})
    MERGE (p)-[:ACTED_IN]->(m))
FOREACH (genre in split(row.genres, '|') | 
    MERGE (g:Genre {name:trim(genre)})
    MERGE (m)-[:IN_GENRE]->(g))
"""

# Execute the query to import the movie information
graph.query(movies_query)

# Refresh the schema to ensure it is up-to-date
graph.refresh_schema()

# Initialize the OpenAI model with specified temperature and model name
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Create a GraphCypherQAChain with the graph and LLM, enabling verbose mode and Cypher query validation
chain = GraphCypherQAChain.from_llm(
    graph=graph, llm=llm, verbose=True, validate_cypher=True
)

# Invoke the chain with a query to get the cast of the movie "Casino"
response = chain.invoke({"query": "What was the cast of the Casino?"})
print(response)
