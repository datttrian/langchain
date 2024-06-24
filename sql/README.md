# Quickstart

In this guide we'll go over the basic ways to create a Q&A chain and
agent over a SQL database. These systems will allow us to ask a question
about the data in a SQL database and get back a natural language answer.
The main difference between the two is that our agent can query the
database in a loop as many time as it needs to answer the question.

## ⚠️ Security note ⚠️

Building Q&A systems of SQL databases requires executing model-generated
SQL queries. There are inherent risks in doing this. Make sure that your
database connection permissions are always scoped as narrowly as
possible for your chain/agent's needs. This will mitigate though not
eliminate the risks of building a model-driven system. For more on
general security best practices, [see here](/v0.1/docs/security/).

## Architecture

At a high-level, the steps of any SQL chain and agent are:

1. **Convert question to SQL query**: Model converts user input to a
    SQL query.
2. **Execute SQL query**: Execute the SQL query.
3. **Answer the question**: Model responds to user input using the
    query results.

<img
src="https://python.langchain.com/v0.1/assets/images/sql_usecase-d432701261f05ab69b38576093718cf3.png"
class="img_ev3q" loading="lazy" width="1571" height="470"
alt="sql_usecase.png" />

## Setup

First, get required packages and set environment variables:

```python
%pip install --upgrade --quiet  langchain langchain-community
```

    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m A new release of pip is available: [0m[31;49m23.0.1[0m[39;49m -> [0m[32;49m24.1[0m
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m To update, run: [0m[32;49mpip install --upgrade pip[0m
    Note: you may need to restart the kernel to use updated packages.

We will use an OpenAI model in this guide.

```python
import getpass
import os

os.environ["OPENAI_API_KEY"] = getpass.getpass()

# Uncomment the below to use LangSmith. Not required.
# os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
```

The below example will use a SQLite connection with Chinook database.
Follow <a href="https://database.guide/2-sample-databases-sqlite/"
target="_blank" rel="noopener noreferrer">these installation steps</a>
to create `Chinook.db` in the same directory as this notebook:

- Save <a
    href="https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"
    target="_blank" rel="noopener noreferrer">this file</a> as
    `Chinook_Sqlite.sql`
- Run `sqlite3 Chinook.db`
- Run `.read Chinook_Sqlite.sql`
- Test `SELECT * FROM Artist LIMIT 10;`

Now, `Chinhook.db` is in our directory and we can interface with it
using the SQLAlchemy-driven `SQLDatabase` class:

```python
from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri("sqlite:///Chinook.db")
print(db.dialect)
print(db.get_usable_table_names())
db.run("SELECT * FROM Artist LIMIT 10;")
```

    sqlite
    ['Album', 'Artist', 'Customer', 'Employee', 'Genre', 'Invoice', 'InvoiceLine', 'MediaType', 'Playlist', 'PlaylistTrack', 'Track']





    "[(1, 'AC/DC'), (2, 'Accept'), (3, 'Aerosmith'), (4, 'Alanis Morissette'), (5, 'Alice In Chains'), (6, 'Antônio Carlos Jobim'), (7, 'Apocalyptica'), (8, 'Audioslave'), (9, 'BackBeat'), (10, 'Billy Cobham')]"

#### API Reference

- [SQLDatabase](https://api.python.langchain.com/en/latest/utilities/langchain_community.utilities.sql_database.SQLDatabase.html)

Great! We've got a SQL database that we can query. Now let's try hooking
it up to an LLM.

## Chain

Let's create a simple chain that takes a question, turns it into a SQL
query, executes the query, and uses the result to answer the original
question.

### Convert question to SQL query

The first step in a SQL chain or agent is to take the user input and
convert it to a SQL query. LangChain comes with a built-in chain for
this: <a
href="https://api.python.langchain.com/en/latest/chains/langchain.chains.sql_database.query.create_sql_query_chain.html"
target="_blank" rel="noopener noreferrer">create_sql_query_chain</a>.

```python
pip install --quiet langchain-openai
```

    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m A new release of pip is available: [0m[31;49m23.0.1[0m[39;49m -> [0m[32;49m24.1[0m
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m To update, run: [0m[32;49mpip install --upgrade pip[0m
    Note: you may need to restart the kernel to use updated packages.

```python
from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
chain = create_sql_query_chain(llm, db)
response = chain.invoke({"question": "How many employees are there"})
response
```

    'SELECT COUNT("EmployeeId") AS "TotalEmployees" FROM "Employee"'

#### API Reference

- [create_sql_query_chain](https://api.python.langchain.com/en/latest/chains/langchain.chains.sql_database.query.create_sql_query_chain.html)
- [ChatOpenAI](https://api.python.langchain.com/en/latest/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html)

We can execute the query to make sure it's valid:

```python
db.run(response)
```

    '[(8,)]'

We can look at the <a
href="https://smith.langchain.com/public/c8fa52ea-be46-4829-bde2-52894970b830/r"
target="_blank" rel="noopener noreferrer">LangSmith trace</a> to get a
better understanding of what this chain is doing. We can also inspect
the chain directly for its prompts. Looking at the prompt (below), we
can see that it is:

- Dialect-specific. In this case it references SQLite explicitly.
- Has definitions for all the available tables.
- Has three examples rows for each table.

This technique is inspired by papers like
<a href="https://arxiv.org/pdf/2204.00498.pdf" target="_blank"
rel="noopener noreferrer">this</a>, which suggest showing examples rows
and being explicit about tables improves performance. We can also
inspect the full prompt like so:

```python
chain.get_prompts()[0].pretty_print()
```

    You are a SQLite expert. Given an input question, first create a syntactically correct SQLite query to run, then look at the results of the query and return the answer to the input question.
    Unless the user specifies in the question a specific number of examples to obtain, query for at most 5 results using the LIMIT clause as per SQLite. You can order the results to return the most informative data in the database.
    Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in double quotes (") to denote them as delimited identifiers.
    Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
    Pay attention to use date('now') function to get the current date, if the question involves "today".
    
    Use the following format:
    
    Question: Question here
    SQLQuery: SQL Query to run
    SQLResult: Result of the SQLQuery
    Answer: Final answer here
    
    Only use the following tables:
    [33;1m[1;3m{table_info}[0m
    
    Question: [33;1m[1;3m{input}[0m

### Execute SQL query

Now that we've generated a SQL query, we'll want to execute it. **This
is the most dangerous part of creating a SQL chain.** Consider carefully
if it is OK to run automated queries over your data. Minimize the
database connection permissions as much as possible. Consider adding a
human approval step to you chains before query execution (see below).

We can use the `QuerySQLDatabaseTool` to easily add query execution to
our chain:

```python
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

execute_query = QuerySQLDataBaseTool(db=db)
write_query = create_sql_query_chain(llm, db)
chain = write_query | execute_query
chain.invoke({"question": "How many employees are there"})
```

    '[(8,)]'

#### API Reference

- [QuerySQLDataBaseTool](https://api.python.langchain.com/en/latest/tools/langchain_community.tools.sql_database.tool.QuerySQLDataBaseTool.html)

### Answer the question

Now that we've got a way to automatically generate and execute queries,
we just need to combine the original question and SQL query result to
generate a final answer. We can do this by passing question and result
to the LLM once more:

```python
from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)

answer = answer_prompt | llm | StrOutputParser()
chain = (
    RunnablePassthrough.assign(query=write_query).assign(
        result=itemgetter("query") | execute_query
    )
    | answer
)

chain.invoke({"question": "How many employees are there"})
```

    'There are a total of 8 employees.'

#### API Reference

- [StrOutputParser](https://api.python.langchain.com/en/latest/output_parsers/langchain_core.output_parsers.string.StrOutputParser.html)
- [PromptTemplate](https://api.python.langchain.com/en/latest/prompts/langchain_core.prompts.prompt.PromptTemplate.html)
- [RunnablePassthrough](https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.passthrough.RunnablePassthrough.html)

### Next steps

For more complex query-generation, we may want to create few-shot
prompts or add query-checking steps. For advanced techniques like this
and more check out:

- [Prompting strategies](https://python.langchain.com/v0.1/docs/use_cases/sql/prompting/):
    Advanced prompt engineering techniques.
- [Query checking](https://python.langchain.com//v0.1/docs/use_cases/sql/query_checking/): Add
    query validation and error handling.
- [Large databses](https://python.langchain.com/v0.1/docs/use_cases/sql/large_db/): Techniques for
    working with large databases.

## Agents

LangChain has an SQL Agent which provides a more flexible way of
interacting with SQL databases. The main advantages of using the SQL
Agent are:

- It can answer questions based on the databases' schema as well as on
    the databases' content (like describing a specific table).
- It can recover from errors by running a generated query, catching
    the traceback and regenerating it correctly.
- It can answer questions that require multiple dependent queries.
- It will save tokens by only considering the schema from relevant
    tables.

To initialize the agent, we use `create_sql_agent` function. This agent
contains the `SQLDatabaseToolkit` which contains tools to:

- Create and execute queries
- Check query syntax
- Retrieve table descriptions
- ... and more

### Initializing agent

```python
from langchain_community.agent_toolkits import create_sql_agent

agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)
```

#### API Reference

- [create_sql_agent](https://api.python.langchain.com/en/latest/agent_toolkits/langchain_community.agent_toolkits.sql.base.create_sql_agent.html)

```python
agent_executor.invoke(
    {
        "input": "List the total sales per country. Which country's customers spent the most?"
    }
)
```

    [1m> Entering new SQL Agent Executor chain...[0m
    [32;1m[1;3m
    Invoking: `sql_db_list_tables` with `{}`
    
    
    [0m[38;5;200m[1;3mAlbum, Artist, Customer, Employee, Genre, Invoice, InvoiceLine, MediaType, Playlist, PlaylistTrack, Track[0m[32;1m[1;3m
    Invoking: `sql_db_schema` with `{'table_names': 'Customer, Invoice, InvoiceLine'}`
    
    
    [0m[33;1m[1;3m
    CREATE TABLE "Customer" (
     "CustomerId" INTEGER NOT NULL, 
     "FirstName" NVARCHAR(40) NOT NULL, 
     "LastName" NVARCHAR(20) NOT NULL, 
     "Company" NVARCHAR(80), 
     "Address" NVARCHAR(70), 
     "City" NVARCHAR(40), 
     "State" NVARCHAR(40), 
     "Country" NVARCHAR(40), 
     "PostalCode" NVARCHAR(10), 
     "Phone" NVARCHAR(24), 
     "Fax" NVARCHAR(24), 
     "Email" NVARCHAR(60) NOT NULL, 
     "SupportRepId" INTEGER, 
     PRIMARY KEY ("CustomerId"), 
     FOREIGN KEY("SupportRepId") REFERENCES "Employee" ("EmployeeId")
    )
    
    /*
    3 rows from Customer table:
    CustomerId FirstName LastName Company Address City State Country PostalCode Phone Fax Email SupportRepId
    1 Luís Gonçalves Embraer - Empresa Brasileira de Aeronáutica S.A. Av. Brigadeiro Faria Lima, 2170 São José dos Campos SP Brazil 12227-000 +55 (12) 3923-5555 +55 (12) 3923-5566 luisg@embraer.com.br 3
    2 Leonie Köhler None Theodor-Heuss-Straße 34 Stuttgart None Germany 70174 +49 0711 2842222 None leonekohler@surfeu.de 5
    3 François Tremblay None 1498 rue Bélanger Montréal QC Canada H2G 1A7 +1 (514) 721-4711 None ftremblay@gmail.com 3
    */
    
    
    CREATE TABLE "Invoice" (
     "InvoiceId" INTEGER NOT NULL, 
     "CustomerId" INTEGER NOT NULL, 
     "InvoiceDate" DATETIME NOT NULL, 
     "BillingAddress" NVARCHAR(70), 
     "BillingCity" NVARCHAR(40), 
     "BillingState" NVARCHAR(40), 
     "BillingCountry" NVARCHAR(40), 
     "BillingPostalCode" NVARCHAR(10), 
     "Total" NUMERIC(10, 2) NOT NULL, 
     PRIMARY KEY ("InvoiceId"), 
     FOREIGN KEY("CustomerId") REFERENCES "Customer" ("CustomerId")
    )
    
    /*
    3 rows from Invoice table:
    InvoiceId CustomerId InvoiceDate BillingAddress BillingCity BillingState BillingCountry BillingPostalCode Total
    1 2 2021-01-01 00:00:00 Theodor-Heuss-Straße 34 Stuttgart None Germany 70174 1.98
    2 4 2021-01-02 00:00:00 Ullevålsveien 14 Oslo None Norway 0171 3.96
    3 8 2021-01-03 00:00:00 Grétrystraat 63 Brussels None Belgium 1000 5.94
    */
    
    
    CREATE TABLE "InvoiceLine" (
     "InvoiceLineId" INTEGER NOT NULL, 
     "InvoiceId" INTEGER NOT NULL, 
     "TrackId" INTEGER NOT NULL, 
     "UnitPrice" NUMERIC(10, 2) NOT NULL, 
     "Quantity" INTEGER NOT NULL, 
     PRIMARY KEY ("InvoiceLineId"), 
     FOREIGN KEY("TrackId") REFERENCES "Track" ("TrackId"), 
     FOREIGN KEY("InvoiceId") REFERENCES "Invoice" ("InvoiceId")
    )
    
    /*
    3 rows from InvoiceLine table:
    InvoiceLineId InvoiceId TrackId UnitPrice Quantity
    1 1 2 0.99 1
    2 1 4 0.99 1
    3 2 6 0.99 1
    */[0m[32;1m[1;3m
    Invoking: `sql_db_query` with `{'query': 'SELECT BillingCountry AS Country, SUM(Total) AS TotalSales FROM Invoice GROUP BY BillingCountry ORDER BY TotalSales DESC'}`
    
    
    [0m[36;1m[1;3m[('USA', 523.0600000000003), ('Canada', 303.9599999999999), ('France', 195.09999999999994), ('Brazil', 190.09999999999997), ('Germany', 156.48), ('United Kingdom', 112.85999999999999), ('Czech Republic', 90.24000000000001), ('Portugal', 77.23999999999998), ('India', 75.25999999999999), ('Chile', 46.62), ('Ireland', 45.62), ('Hungary', 45.62), ('Austria', 42.62), ('Finland', 41.620000000000005), ('Netherlands', 40.62), ('Norway', 39.62), ('Sweden', 38.620000000000005), ('Poland', 37.620000000000005), ('Italy', 37.620000000000005), ('Denmark', 37.620000000000005), ('Australia', 37.620000000000005), ('Argentina', 37.620000000000005), ('Spain', 37.62), ('Belgium', 37.62)][0m[32;1m[1;3mThe total sales per country are as follows:
    1. USA: $523.06
    2. Canada: $303.96
    3. France: $195.10
    4. Brazil: $190.10
    5. Germany: $156.48
    
    The country whose customers spent the most is the USA with a total sales amount of $523.06.[0m
    
    [1m> Finished chain.[0m





    {'input': "List the total sales per country. Which country's customers spent the most?",
     'output': 'The total sales per country are as follows:\n1. USA: $523.06\n2. Canada: $303.96\n3. France: $195.10\n4. Brazil: $190.10\n5. Germany: $156.48\n\nThe country whose customers spent the most is the USA with a total sales amount of $523.06.'}

```python
agent_executor.invoke({"input": "Describe the playlisttrack table"})
```

    [1m> Entering new SQL Agent Executor chain...[0m
    [32;1m[1;3m
    Invoking: `sql_db_list_tables` with `{}`
    
    
    [0m[38;5;200m[1;3mAlbum, Artist, Customer, Employee, Genre, Invoice, InvoiceLine, MediaType, Playlist, PlaylistTrack, Track[0m[32;1m[1;3m
    Invoking: `sql_db_schema` with `{'table_names': 'PlaylistTrack'}`
    
    
    [0m[33;1m[1;3m
    CREATE TABLE "PlaylistTrack" (
     "PlaylistId" INTEGER NOT NULL, 
     "TrackId" INTEGER NOT NULL, 
     PRIMARY KEY ("PlaylistId", "TrackId"), 
     FOREIGN KEY("TrackId") REFERENCES "Track" ("TrackId"), 
     FOREIGN KEY("PlaylistId") REFERENCES "Playlist" ("PlaylistId")
    )
    
    /*
    3 rows from PlaylistTrack table:
    PlaylistId TrackId
    1 3402
    1 3389
    1 3390
    */[0m[32;1m[1;3mThe `PlaylistTrack` table has the following columns:
    - PlaylistId (INTEGER, NOT NULL)
    - TrackId (INTEGER, NOT NULL)
    
    It has a composite primary key on the columns PlaylistId and TrackId. Additionally, there are foreign key constraints on TrackId referencing the Track table and PlaylistId referencing the Playlist table.
    
    Here are 3 sample rows from the `PlaylistTrack` table:
    - PlaylistId: 1, TrackId: 3402
    - PlaylistId: 1, TrackId: 3389
    - PlaylistId: 1, TrackId: 3390[0m
    
    [1m> Finished chain.[0m





    {'input': 'Describe the playlisttrack table',
     'output': 'The `PlaylistTrack` table has the following columns:\n- PlaylistId (INTEGER, NOT NULL)\n- TrackId (INTEGER, NOT NULL)\n\nIt has a composite primary key on the columns PlaylistId and TrackId. Additionally, there are foreign key constraints on TrackId referencing the Track table and PlaylistId referencing the Playlist table.\n\nHere are 3 sample rows from the `PlaylistTrack` table:\n- PlaylistId: 1, TrackId: 3402\n- PlaylistId: 1, TrackId: 3389\n- PlaylistId: 1, TrackId: 3390'}

# Agents

LangChain has a SQL Agent which provides a more flexible way of
interacting with SQL Databases than a chain. The main advantages of
using the SQL Agent are:

- It can answer questions based on the databases' schema as well as on
    the databases' content (like describing a specific table).
- It can recover from errors by running a generated query, catching
    the traceback and regenerating it correctly.
- It can query the database as many times as needed to answer the user
    question.
- It will save tokens by only retrieving the schema from relevant
    tables.

To initialize the agent we'll use the <a
href="https://api.python.langchain.com/en/latest/agent_toolkits/langchain_community.agent_toolkits.sql.base.create_sql_agent.html"
target="_blank" rel="noopener noreferrer">create_sql_agent</a>
constructor. This agent uses the `SQLDatabaseToolkit` which contains
tools to:

- Create and execute queries
- Check query syntax
- Retrieve table descriptions
- ... and more

## Setup

First, get required packages and set environment variables:

``` prism-code
%pip install --upgrade --quiet  langchain langchain-community langchain-openai
```

<span class="copyButtonIcons_eSgA" aria-hidden="true"><img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJjb3B5QnV0dG9uSWNvbl95OTdOIj48cGF0aCBmaWxsPSJjdXJyZW50Q29sb3IiIGQ9Ik0xOSwyMUg4VjdIMTlNMTksNUg4QTIsMiAwIDAsMCA2LDdWMjFBMiwyIDAgMCwwIDgsMjNIMTlBMiwyIDAgMCwwIDIxLDIxVjdBMiwyIDAgMCwwIDE5LDVNMTYsMUg0QTIsMiAwIDAsMCAyLDNWMTdINFYzSDE2VjFaIiAvPjwvc3ZnPg=="
class="copyButtonIcon_y97N" /><img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJjb3B5QnV0dG9uU3VjY2Vzc0ljb25fTGpkUyI+PHBhdGggZmlsbD0iY3VycmVudENvbG9yIiBkPSJNMjEsN0w5LDE5TDMuNSwxMy41TDQuOTEsMTIuMDlMOSwxNi4xN0wxOS41OSw1LjU5TDIxLDdaIiAvPjwvc3ZnPg=="
class="copyButtonSuccessIcon_LjdS" /></span>

We default to OpenAI models in this guide, but you can swap them out for
the model provider of your choice.

``` prism-code
import getpass
import os

os.environ["OPENAI_API_KEY"] = getpass.getpass()

# Uncomment the below to use LangSmith. Not required.
# os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
```

<span class="copyButtonIcons_eSgA" aria-hidden="true"><img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJjb3B5QnV0dG9uSWNvbl95OTdOIj48cGF0aCBmaWxsPSJjdXJyZW50Q29sb3IiIGQ9Ik0xOSwyMUg4VjdIMTlNMTksNUg4QTIsMiAwIDAsMCA2LDdWMjFBMiwyIDAgMCwwIDgsMjNIMTlBMiwyIDAgMCwwIDIxLDIxVjdBMiwyIDAgMCwwIDE5LDVNMTYsMUg0QTIsMiAwIDAsMCAyLDNWMTdINFYzSDE2VjFaIiAvPjwvc3ZnPg=="
class="copyButtonIcon_y97N" /><img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJjb3B5QnV0dG9uU3VjY2Vzc0ljb25fTGpkUyI+PHBhdGggZmlsbD0iY3VycmVudENvbG9yIiBkPSJNMjEsN0w5LDE5TDMuNSwxMy41TDQuOTEsMTIuMDlMOSwxNi4xN0wxOS41OSw1LjU5TDIxLDdaIiAvPjwvc3ZnPg=="
class="copyButtonSuccessIcon_LjdS" /></span>

The below example will use a SQLite connection with Chinook database.
Follow <a href="https://database.guide/2-sample-databases-sqlite/"
target="_blank" rel="noopener noreferrer">these installation steps</a>
to create `Chinook.db` in the same directory as this notebook:

- Save <a
    href="https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"
    target="_blank" rel="noopener noreferrer">this file</a> as
    `Chinook_Sqlite.sql`
- Run `sqlite3 Chinook.db`
- Run `.read Chinook_Sqlite.sql`
- Test `SELECT * FROM Artist LIMIT 10;`

Now, `Chinhook.db` is in our directory and we can interface with it
using the SQLAlchemy-driven `SQLDatabase` class:

``` prism-code
from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri("sqlite:///Chinook.db")
print(db.dialect)
print(db.get_usable_table_names())
db.run("SELECT * FROM Artist LIMIT 10;")
```

<span class="copyButtonIcons_eSgA" aria-hidden="true"><img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJjb3B5QnV0dG9uSWNvbl95OTdOIj48cGF0aCBmaWxsPSJjdXJyZW50Q29sb3IiIGQ9Ik0xOSwyMUg4VjdIMTlNMTksNUg4QTIsMiAwIDAsMCA2LDdWMjFBMiwyIDAgMCwwIDgsMjNIMTlBMiwyIDAgMCwwIDIxLDIxVjdBMiwyIDAgMCwwIDE5LDVNMTYsMUg0QTIsMiAwIDAsMCAyLDNWMTdINFYzSDE2VjFaIiAvPjwvc3ZnPg=="
class="copyButtonIcon_y97N" /><img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJjb3B5QnV0dG9uU3VjY2Vzc0ljb25fTGpkUyI+PHBhdGggZmlsbD0iY3VycmVudENvbG9yIiBkPSJNMjEsN0w5LDE5TDMuNSwxMy41TDQuOTEsMTIuMDlMOSwxNi4xN0wxOS41OSw1LjU5TDIxLDdaIiAvPjwvc3ZnPg=="
class="copyButtonSuccessIcon_LjdS" /></span>

#### API Reference

- [SQLDatabase](https://api.python.langchain.com/en/latest/utilities/langchain_community.utilities.sql_database.SQLDatabase.html)

``` prism-code
sqlite
['Album', 'Artist', 'Customer', 'Employee', 'Genre', 'Invoice', 'InvoiceLine', 'MediaType', 'Playlist', 'PlaylistTrack', 'Track']
```

<img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJ3b3JkV3JhcEJ1dHRvbkljb25fQndtYSIgYXJpYS1oaWRkZW49InRydWUiPjxwYXRoIGZpbGw9ImN1cnJlbnRDb2xvciIgZD0iTTQgMTloNnYtMkg0djJ6TTIwIDVINHYyaDE2VjV6bS0zIDZINHYyaDEzLjI1YzEuMSAwIDIgLjkgMiAycy0uOSAyLTIgMkgxNXYtMmwtMyAzbDMgM3YtMmgyYzIuMjEgMCA0LTEuNzkgNC00cy0xLjc5LTQtNC00eiIgLz48L3N2Zz4="
class="wordWrapButtonIcon_Bwma" />

<span class="copyButtonIcons_eSgA" aria-hidden="true"><img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJjb3B5QnV0dG9uSWNvbl95OTdOIj48cGF0aCBmaWxsPSJjdXJyZW50Q29sb3IiIGQ9Ik0xOSwyMUg4VjdIMTlNMTksNUg4QTIsMiAwIDAsMCA2LDdWMjFBMiwyIDAgMCwwIDgsMjNIMTlBMiwyIDAgMCwwIDIxLDIxVjdBMiwyIDAgMCwwIDE5LDVNMTYsMUg0QTIsMiAwIDAsMCAyLDNWMTdINFYzSDE2VjFaIiAvPjwvc3ZnPg=="
class="copyButtonIcon_y97N" /><img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJjb3B5QnV0dG9uU3VjY2Vzc0ljb25fTGpkUyI+PHBhdGggZmlsbD0iY3VycmVudENvbG9yIiBkPSJNMjEsN0w5LDE5TDMuNSwxMy41TDQuOTEsMTIuMDlMOSwxNi4xN0wxOS41OSw1LjU5TDIxLDdaIiAvPjwvc3ZnPg=="
class="copyButtonSuccessIcon_LjdS" /></span>

``` prism-code
"[(1, 'AC/DC'), (2, 'Accept'), (3, 'Aerosmith'), (4, 'Alanis Morissette'), (5, 'Alice In Chains'), (6, 'Antônio Carlos Jobim'), (7, 'Apocalyptica'), (8, 'Audioslave'), (9, 'BackBeat'), (10, 'Billy Cobham')]"
```

<img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJ3b3JkV3JhcEJ1dHRvbkljb25fQndtYSIgYXJpYS1oaWRkZW49InRydWUiPjxwYXRoIGZpbGw9ImN1cnJlbnRDb2xvciIgZD0iTTQgMTloNnYtMkg0djJ6TTIwIDVINHYyaDE2VjV6bS0zIDZINHYyaDEzLjI1YzEuMSAwIDIgLjkgMiAycy0uOSAyLTIgMkgxNXYtMmwtMyAzbDMgM3YtMmgyYzIuMjEgMCA0LTEuNzkgNC00cy0xLjc5LTQtNC00eiIgLz48L3N2Zz4="
class="wordWrapButtonIcon_Bwma" />

<span class="copyButtonIcons_eSgA" aria-hidden="true"><img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJjb3B5QnV0dG9uSWNvbl95OTdOIj48cGF0aCBmaWxsPSJjdXJyZW50Q29sb3IiIGQ9Ik0xOSwyMUg4VjdIMTlNMTksNUg4QTIsMiAwIDAsMCA2LDdWMjFBMiwyIDAgMCwwIDgsMjNIMTlBMiwyIDAgMCwwIDIxLDIxVjdBMiwyIDAgMCwwIDE5LDVNMTYsMUg0QTIsMiAwIDAsMCAyLDNWMTdINFYzSDE2VjFaIiAvPjwvc3ZnPg=="
class="copyButtonIcon_y97N" /><img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJjb3B5QnV0dG9uU3VjY2Vzc0ljb25fTGpkUyI+PHBhdGggZmlsbD0iY3VycmVudENvbG9yIiBkPSJNMjEsN0w5LDE5TDMuNSwxMy41TDQuOTEsMTIuMDlMOSwxNi4xN0wxOS41OSw1LjU5TDIxLDdaIiAvPjwvc3ZnPg=="
class="copyButtonSuccessIcon_LjdS" /></span>

## Agent<a href="#agent" class="hash-link" aria-label="Direct link to Agent"

title="Direct link to Agent">​</a>

We'll use an OpenAI chat model and an `"openai-tools"` agent, which will
use OpenAI's function-calling API to drive the agent's tool selection
and invocations.

As we can see, the agent will first choose which tables are relevant and
then add the schema for those tables and a few sample rows to the
prompt.

``` prism-code
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)
```

<span class="copyButtonIcons_eSgA" aria-hidden="true"><img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJjb3B5QnV0dG9uSWNvbl95OTdOIj48cGF0aCBmaWxsPSJjdXJyZW50Q29sb3IiIGQ9Ik0xOSwyMUg4VjdIMTlNMTksNUg4QTIsMiAwIDAsMCA2LDdWMjFBMiwyIDAgMCwwIDgsMjNIMTlBMiwyIDAgMCwwIDIxLDIxVjdBMiwyIDAgMCwwIDE5LDVNMTYsMUg0QTIsMiAwIDAsMCAyLDNWMTdINFYzSDE2VjFaIiAvPjwvc3ZnPg=="
class="copyButtonIcon_y97N" /><img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJjb3B5QnV0dG9uU3VjY2Vzc0ljb25fTGpkUyI+PHBhdGggZmlsbD0iY3VycmVudENvbG9yIiBkPSJNMjEsN0w5LDE5TDMuNSwxMy41TDQuOTEsMTIuMDlMOSwxNi4xN0wxOS41OSw1LjU5TDIxLDdaIiAvPjwvc3ZnPg=="
class="copyButtonSuccessIcon_LjdS" /></span>

#### API Reference

- [create_sql_agent](https://api.python.langchain.com/en/latest/agent_toolkits/langchain_community.agent_toolkits.sql.base.create_sql_agent.html)
- [ChatOpenAI](https://api.python.langchain.com/en/latest/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html)

``` prism-code
agent_executor.invoke(
    "List the total sales per country. Which country's customers spent the most?"
)
```

<span class="copyButtonIcons_eSgA" aria-hidden="true"><img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJjb3B5QnV0dG9uSWNvbl95OTdOIj48cGF0aCBmaWxsPSJjdXJyZW50Q29sb3IiIGQ9Ik0xOSwyMUg4VjdIMTlNMTksNUg4QTIsMiAwIDAsMCA2LDdWMjFBMiwyIDAgMCwwIDgsMjNIMTlBMiwyIDAgMCwwIDIxLDIxVjdBMiwyIDAgMCwwIDE5LDVNMTYsMUg0QTIsMiAwIDAsMCAyLDNWMTdINFYzSDE2VjFaIiAvPjwvc3ZnPg=="
class="copyButtonIcon_y97N" /><img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJjb3B5QnV0dG9uU3VjY2Vzc0ljb25fTGpkUyI+PHBhdGggZmlsbD0iY3VycmVudENvbG9yIiBkPSJNMjEsN0w5LDE5TDMuNSwxMy41TDQuOTEsMTIuMDlMOSwxNi4xN0wxOS41OSw1LjU5TDIxLDdaIiAvPjwvc3ZnPg=="
class="copyButtonSuccessIcon_LjdS" /></span>

``` prism-code

[1m> Entering new AgentExecutor chain...[0m
[32;1m[1;3m
Invoking: `sql_db_list_tables` with `{}`


[0m[38;5;200m[1;3mAlbum, Artist, Customer, Employee, Genre, Invoice, InvoiceLine, MediaType, Playlist, PlaylistTrack, Track[0m[32;1m[1;3m
Invoking: `sql_db_schema` with `Invoice,Customer`


[0m[33;1m[1;3m
CREATE TABLE "Customer" (
    "CustomerId" INTEGER NOT NULL, 
    "FirstName" NVARCHAR(40) NOT NULL, 
    "LastName" NVARCHAR(20) NOT NULL, 
    "Company" NVARCHAR(80), 
    "Address" NVARCHAR(70), 
    "City" NVARCHAR(40), 
    "State" NVARCHAR(40), 
    "Country" NVARCHAR(40), 
    "PostalCode" NVARCHAR(10), 
    "Phone" NVARCHAR(24), 
    "Fax" NVARCHAR(24), 
    "Email" NVARCHAR(60) NOT NULL, 
    "SupportRepId" INTEGER, 
    PRIMARY KEY ("CustomerId"), 
    FOREIGN KEY("SupportRepId") REFERENCES "Employee" ("EmployeeId")
)

/*
3 rows from Customer table:
CustomerId  FirstName   LastName    Company Address City    State   Country PostalCode  Phone   Fax Email   SupportRepId
1   Luís    Gonçalves   Embraer - Empresa Brasileira de Aeronáutica S.A.    Av. Brigadeiro Faria Lima, 2170 São José dos Campos SP  Brazil  12227-000   +55 (12) 3923-5555  +55 (12) 3923-5566  luisg@embraer.com.br    3
2   Leonie  Köhler  None    Theodor-Heuss-Straße 34 Stuttgart   None    Germany 70174   +49 0711 2842222    None    leonekohler@surfeu.de   5
3   François    Tremblay    None    1498 rue Bélanger   Montréal    QC  Canada  H2G 1A7 +1 (514) 721-4711   None    ftremblay@gmail.com 3
*/


CREATE TABLE "Invoice" (
    "InvoiceId" INTEGER NOT NULL, 
    "CustomerId" INTEGER NOT NULL, 
    "InvoiceDate" DATETIME NOT NULL, 
    "BillingAddress" NVARCHAR(70), 
    "BillingCity" NVARCHAR(40), 
    "BillingState" NVARCHAR(40), 
    "BillingCountry" NVARCHAR(40), 
    "BillingPostalCode" NVARCHAR(10), 
    "Total" NUMERIC(10, 2) NOT NULL, 
    PRIMARY KEY ("InvoiceId"), 
    FOREIGN KEY("CustomerId") REFERENCES "Customer" ("CustomerId")
)

/*
3 rows from Invoice table:
InvoiceId   CustomerId  InvoiceDate BillingAddress  BillingCity BillingState    BillingCountry  BillingPostalCode   Total
1   2   2009-01-01 00:00:00 Theodor-Heuss-Straße 34 Stuttgart   None    Germany 70174   1.98
2   4   2009-01-02 00:00:00 Ullevålsveien 14    Oslo    None    Norway  0171    3.96
3   8   2009-01-03 00:00:00 Grétrystraat 63 Brussels    None    Belgium 1000    5.94
*/[0m[32;1m[1;3m
Invoking: `sql_db_query` with `SELECT c.Country, SUM(i.Total) AS TotalSales FROM Invoice i JOIN Customer c ON i.CustomerId = c.CustomerId GROUP BY c.Country ORDER BY TotalSales DESC LIMIT 10;`
responded: To list the total sales per country, I can query the "Invoice" and "Customer" tables. I will join these tables on the "CustomerId" column and group the results by the "BillingCountry" column. Then, I will calculate the sum of the "Total" column to get the total sales per country. Finally, I will order the results in descending order of the total sales.

Here is the SQL query:

```sql
SELECT c.Country, SUM(i.Total) AS TotalSales
FROM Invoice i
JOIN Customer c ON i.CustomerId = c.CustomerId
GROUP BY c.Country
ORDER BY TotalSales DESC
LIMIT 10;
```

<img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJ3b3JkV3JhcEJ1dHRvbkljb25fQndtYSIgYXJpYS1oaWRkZW49InRydWUiPjxwYXRoIGZpbGw9ImN1cnJlbnRDb2xvciIgZD0iTTQgMTloNnYtMkg0djJ6TTIwIDVINHYyaDE2VjV6bS0zIDZINHYyaDEzLjI1YzEuMSAwIDIgLjkgMiAycy0uOSAyLTIgMkgxNXYtMmwtMyAzbDMgM3YtMmgyYzIuMjEgMCA0LTEuNzkgNC00cy0xLjc5LTQtNC00eiIgLz48L3N2Zz4="
class="wordWrapButtonIcon_Bwma" />

<span class="copyButtonIcons_eSgA" aria-hidden="true"><img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJjb3B5QnV0dG9uSWNvbl95OTdOIj48cGF0aCBmaWxsPSJjdXJyZW50Q29sb3IiIGQ9Ik0xOSwyMUg4VjdIMTlNMTksNUg4QTIsMiAwIDAsMCA2LDdWMjFBMiwyIDAgMCwwIDgsMjNIMTlBMiwyIDAgMCwwIDIxLDIxVjdBMiwyIDAgMCwwIDE5LDVNMTYsMUg0QTIsMiAwIDAsMCAyLDNWMTdINFYzSDE2VjFaIiAvPjwvc3ZnPg=="
class="copyButtonIcon_y97N" /><img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJjb3B5QnV0dG9uU3VjY2Vzc0ljb25fTGpkUyI+PHBhdGggZmlsbD0iY3VycmVudENvbG9yIiBkPSJNMjEsN0w5LDE5TDMuNSwxMy41TDQuOTEsMTIuMDlMOSwxNi4xN0wxOS41OSw1LjU5TDIxLDdaIiAvPjwvc3ZnPg=="
class="copyButtonSuccessIcon_LjdS" /></span>

Now, I will execute this query to get the total sales per country.

\[0m\[36;1m\[1;3m\[('USA', 523.0600000000003), ('Canada',
303.9599999999999), ('France', 195.09999999999994), ('Brazil',
190.09999999999997), ('Germany', 156.48), ('United Kingdom',
112.85999999999999), ('Czech Republic', 90.24000000000001), ('Portugal',
77.23999999999998), ('India', 75.25999999999999), ('Chile',
46.62)\]\[0m\[32;1m\[1;3mThe total sales per country are as follows:

1. USA: $523.06
2. Canada: $303.96
3. France: $195.10
4. Brazil: $190.10
5. Germany: $156.48
6. United Kingdom: $112.86
7. Czech Republic: $90.24
8. Portugal: $77.24
9. India: $75.26
10. Chile: $46.62

To answer the second question, the country whose customers spent the
most is the USA, with a total sales of $523.06.\[0m

\[1m\> Finished chain.\[0m

``` prism-code

```output
{'input': "List the total sales per country. Which country's customers spent the most?",
 'output': 'The total sales per country are as follows:\n\n1. USA: $523.06\n2. Canada: $303.96\n3. France: $195.10\n4. Brazil: $190.10\n5. Germany: $156.48\n6. United Kingdom: $112.86\n7. Czech Republic: $90.24\n8. Portugal: $77.24\n9. India: $75.26\n10. Chile: $46.62\n\nTo answer the second question, the country whose customers spent the most is the USA, with a total sales of $523.06.'}
```

<img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJ3b3JkV3JhcEJ1dHRvbkljb25fQndtYSIgYXJpYS1oaWRkZW49InRydWUiPjxwYXRoIGZpbGw9ImN1cnJlbnRDb2xvciIgZD0iTTQgMTloNnYtMkg0djJ6TTIwIDVINHYyaDE2VjV6bS0zIDZINHYyaDEzLjI1YzEuMSAwIDIgLjkgMiAycy0uOSAyLTIgMkgxNXYtMmwtMyAzbDMgM3YtMmgyYzIuMjEgMCA0LTEuNzkgNC00cy0xLjc5LTQtNC00eiIgLz48L3N2Zz4="
class="wordWrapButtonIcon_Bwma" />

<span class="copyButtonIcons_eSgA" aria-hidden="true"><img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJjb3B5QnV0dG9uSWNvbl95OTdOIj48cGF0aCBmaWxsPSJjdXJyZW50Q29sb3IiIGQ9Ik0xOSwyMUg4VjdIMTlNMTksNUg4QTIsMiAwIDAsMCA2LDdWMjFBMiwyIDAgMCwwIDgsMjNIMTlBMiwyIDAgMCwwIDIxLDIxVjdBMiwyIDAgMCwwIDE5LDVNMTYsMUg0QTIsMiAwIDAsMCAyLDNWMTdINFYzSDE2VjFaIiAvPjwvc3ZnPg=="
class="copyButtonIcon_y97N" /><img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJjb3B5QnV0dG9uU3VjY2Vzc0ljb25fTGpkUyI+PHBhdGggZmlsbD0iY3VycmVudENvbG9yIiBkPSJNMjEsN0w5LDE5TDMuNSwxMy41TDQuOTEsMTIuMDlMOSwxNi4xN0wxOS41OSw1LjU5TDIxLDdaIiAvPjwvc3ZnPg=="
class="copyButtonSuccessIcon_LjdS" /></span>

``` prism-code
agent_executor.invoke("Describe the playlisttrack table")
```

<span class="copyButtonIcons_eSgA" aria-hidden="true"><img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJjb3B5QnV0dG9uSWNvbl95OTdOIj48cGF0aCBmaWxsPSJjdXJyZW50Q29sb3IiIGQ9Ik0xOSwyMUg4VjdIMTlNMTksNUg4QTIsMiAwIDAsMCA2LDdWMjFBMiwyIDAgMCwwIDgsMjNIMTlBMiwyIDAgMCwwIDIxLDIxVjdBMiwyIDAgMCwwIDE5LDVNMTYsMUg0QTIsMiAwIDAsMCAyLDNWMTdINFYzSDE2VjFaIiAvPjwvc3ZnPg=="
class="copyButtonIcon_y97N" /><img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJjb3B5QnV0dG9uU3VjY2Vzc0ljb25fTGpkUyI+PHBhdGggZmlsbD0iY3VycmVudENvbG9yIiBkPSJNMjEsN0w5LDE5TDMuNSwxMy41TDQuOTEsMTIuMDlMOSwxNi4xN0wxOS41OSw1LjU5TDIxLDdaIiAvPjwvc3ZnPg=="
class="copyButtonSuccessIcon_LjdS" /></span>

``` prism-code

[1m> Entering new AgentExecutor chain...[0m
[32;1m[1;3m
Invoking: `sql_db_list_tables` with `{}`


[0m[38;5;200m[1;3mAlbum, Artist, Customer, Employee, Genre, Invoice, InvoiceLine, MediaType, Playlist, PlaylistTrack, Track[0m[32;1m[1;3m
Invoking: `sql_db_schema` with `PlaylistTrack`


[0m[33;1m[1;3m
CREATE TABLE "PlaylistTrack" (
    "PlaylistId" INTEGER NOT NULL, 
    "TrackId" INTEGER NOT NULL, 
    PRIMARY KEY ("PlaylistId", "TrackId"), 
    FOREIGN KEY("TrackId") REFERENCES "Track" ("TrackId"), 
    FOREIGN KEY("PlaylistId") REFERENCES "Playlist" ("PlaylistId")
)

/*
3 rows from PlaylistTrack table:
PlaylistId  TrackId
1   3402
1   3389
1   3390
*/[0m[32;1m[1;3mThe `PlaylistTrack` table has two columns: `PlaylistId` and `TrackId`. It is a junction table that represents the many-to-many relationship between playlists and tracks. 

Here is the schema of the `PlaylistTrack` table:
```

<img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJ3b3JkV3JhcEJ1dHRvbkljb25fQndtYSIgYXJpYS1oaWRkZW49InRydWUiPjxwYXRoIGZpbGw9ImN1cnJlbnRDb2xvciIgZD0iTTQgMTloNnYtMkg0djJ6TTIwIDVINHYyaDE2VjV6bS0zIDZINHYyaDEzLjI1YzEuMSAwIDIgLjkgMiAycy0uOSAyLTIgMkgxNXYtMmwtMyAzbDMgM3YtMmgyYzIuMjEgMCA0LTEuNzkgNC00cy0xLjc5LTQtNC00eiIgLz48L3N2Zz4="
class="wordWrapButtonIcon_Bwma" />

<span class="copyButtonIcons_eSgA" aria-hidden="true"><img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJjb3B5QnV0dG9uSWNvbl95OTdOIj48cGF0aCBmaWxsPSJjdXJyZW50Q29sb3IiIGQ9Ik0xOSwyMUg4VjdIMTlNMTksNUg4QTIsMiAwIDAsMCA2LDdWMjFBMiwyIDAgMCwwIDgsMjNIMTlBMiwyIDAgMCwwIDIxLDIxVjdBMiwyIDAgMCwwIDE5LDVNMTYsMUg0QTIsMiAwIDAsMCAyLDNWMTdINFYzSDE2VjFaIiAvPjwvc3ZnPg=="
class="copyButtonIcon_y97N" /><img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJjb3B5QnV0dG9uU3VjY2Vzc0ljb25fTGpkUyI+PHBhdGggZmlsbD0iY3VycmVudENvbG9yIiBkPSJNMjEsN0w5LDE5TDMuNSwxMy41TDQuOTEsMTIuMDlMOSwxNi4xN0wxOS41OSw1LjU5TDIxLDdaIiAvPjwvc3ZnPg=="
class="copyButtonSuccessIcon_LjdS" /></span>

CREATE TABLE "PlaylistTrack" ( "PlaylistId" INTEGER NOT NULL, "TrackId"
INTEGER NOT NULL, PRIMARY KEY ("PlaylistId", "TrackId"), FOREIGN
KEY("TrackId") REFERENCES "Track" ("TrackId"), FOREIGN KEY("PlaylistId")
REFERENCES "Playlist" ("PlaylistId") )

``` prism-code
The `PlaylistId` column is a foreign key referencing the `PlaylistId` column in the `Playlist` table. The `TrackId` column is a foreign key referencing the `TrackId` column in the `Track` table.

Here are three sample rows from the `PlaylistTrack` table:
```

<img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJ3b3JkV3JhcEJ1dHRvbkljb25fQndtYSIgYXJpYS1oaWRkZW49InRydWUiPjxwYXRoIGZpbGw9ImN1cnJlbnRDb2xvciIgZD0iTTQgMTloNnYtMkg0djJ6TTIwIDVINHYyaDE2VjV6bS0zIDZINHYyaDEzLjI1YzEuMSAwIDIgLjkgMiAycy0uOSAyLTIgMkgxNXYtMmwtMyAzbDMgM3YtMmgyYzIuMjEgMCA0LTEuNzkgNC00cy0xLjc5LTQtNC00eiIgLz48L3N2Zz4="
class="wordWrapButtonIcon_Bwma" />

<span class="copyButtonIcons_eSgA" aria-hidden="true"><img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJjb3B5QnV0dG9uSWNvbl95OTdOIj48cGF0aCBmaWxsPSJjdXJyZW50Q29sb3IiIGQ9Ik0xOSwyMUg4VjdIMTlNMTksNUg4QTIsMiAwIDAsMCA2LDdWMjFBMiwyIDAgMCwwIDgsMjNIMTlBMiwyIDAgMCwwIDIxLDIxVjdBMiwyIDAgMCwwIDE5LDVNMTYsMUg0QTIsMiAwIDAsMCAyLDNWMTdINFYzSDE2VjFaIiAvPjwvc3ZnPg=="
class="copyButtonIcon_y97N" /><img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJjb3B5QnV0dG9uU3VjY2Vzc0ljb25fTGpkUyI+PHBhdGggZmlsbD0iY3VycmVudENvbG9yIiBkPSJNMjEsN0w5LDE5TDMuNSwxMy41TDQuOTEsMTIuMDlMOSwxNi4xN0wxOS41OSw1LjU5TDIxLDdaIiAvPjwvc3ZnPg=="
class="copyButtonSuccessIcon_LjdS" /></span>

PlaylistId TrackId 1 3402 1 3389 1 3390

``` prism-code
Please let me know if there is anything else I can help with.[0m

[1m> Finished chain.[0m
```

<span class="copyButtonIcons_eSgA" aria-hidden="true"><img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJjb3B5QnV0dG9uSWNvbl95OTdOIj48cGF0aCBmaWxsPSJjdXJyZW50Q29sb3IiIGQ9Ik0xOSwyMUg4VjdIMTlNMTksNUg4QTIsMiAwIDAsMCA2LDdWMjFBMiwyIDAgMCwwIDgsMjNIMTlBMiwyIDAgMCwwIDIxLDIxVjdBMiwyIDAgMCwwIDE5LDVNMTYsMUg0QTIsMiAwIDAsMCAyLDNWMTdINFYzSDE2VjFaIiAvPjwvc3ZnPg=="
class="copyButtonIcon_y97N" /><img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJjb3B5QnV0dG9uU3VjY2Vzc0ljb25fTGpkUyI+PHBhdGggZmlsbD0iY3VycmVudENvbG9yIiBkPSJNMjEsN0w5LDE5TDMuNSwxMy41TDQuOTEsMTIuMDlMOSwxNi4xN0wxOS41OSw1LjU5TDIxLDdaIiAvPjwvc3ZnPg=="
class="copyButtonSuccessIcon_LjdS" /></span>

``` prism-code
{'input': 'Describe the playlisttrack table',
 'output': 'The `PlaylistTrack` table has two columns: `PlaylistId` and `TrackId`. It is a junction table that represents the many-to-many relationship between playlists and tracks. \n\nHere is the schema of the `PlaylistTrack` table:\n\n```\nCREATE TABLE "PlaylistTrack" (\n\t"PlaylistId" INTEGER NOT NULL, \n\t"TrackId" INTEGER NOT NULL, \n\tPRIMARY KEY ("PlaylistId", "TrackId"), \n\tFOREIGN KEY("TrackId") REFERENCES "Track" ("TrackId"), \n\tFOREIGN KEY("PlaylistId") REFERENCES "Playlist" ("PlaylistId")\n)\n```\n\nThe `PlaylistId` column is a foreign key referencing the `PlaylistId` column in the `Playlist` table. The `TrackId` column is a foreign key referencing the `TrackId` column in the `Track` table.\n\nHere are three sample rows from the `PlaylistTrack` table:\n\n```\nPlaylistId   TrackId\n1            3402\n1            3389\n1            3390\n```\n\nPlease let me know if there is anything else I can help with.'}
```

<img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJ3b3JkV3JhcEJ1dHRvbkljb25fQndtYSIgYXJpYS1oaWRkZW49InRydWUiPjxwYXRoIGZpbGw9ImN1cnJlbnRDb2xvciIgZD0iTTQgMTloNnYtMkg0djJ6TTIwIDVINHYyaDE2VjV6bS0zIDZINHYyaDEzLjI1YzEuMSAwIDIgLjkgMiAycy0uOSAyLTIgMkgxNXYtMmwtMyAzbDMgM3YtMmgyYzIuMjEgMCA0LTEuNzkgNC00cy0xLjc5LTQtNC00eiIgLz48L3N2Zz4="
class="wordWrapButtonIcon_Bwma" />

<span class="copyButtonIcons_eSgA" aria-hidden="true"><img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJjb3B5QnV0dG9uSWNvbl95OTdOIj48cGF0aCBmaWxsPSJjdXJyZW50Q29sb3IiIGQ9Ik0xOSwyMUg4VjdIMTlNMTksNUg4QTIsMiAwIDAsMCA2LDdWMjFBMiwyIDAgMCwwIDgsMjNIMTlBMiwyIDAgMCwwIDIxLDIxVjdBMiwyIDAgMCwwIDE5LDVNMTYsMUg0QTIsMiAwIDAsMCAyLDNWMTdINFYzSDE2VjFaIiAvPjwvc3ZnPg=="
class="copyButtonIcon_y97N" /><img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJjb3B5QnV0dG9uU3VjY2Vzc0ljb25fTGpkUyI+PHBhdGggZmlsbD0iY3VycmVudENvbG9yIiBkPSJNMjEsN0w5LDE5TDMuNSwxMy41TDQuOTEsMTIuMDlMOSwxNi4xN0wxOS41OSw1LjU5TDIxLDdaIiAvPjwvc3ZnPg=="
class="copyButtonSuccessIcon_LjdS" /></span>

## Using a dynamic few-shot prompt<a href="#using-a-dynamic-few-shot-prompt" class="hash-link"

aria-label="Direct link to Using a dynamic few-shot prompt"
title="Direct link to Using a dynamic few-shot prompt">​</a>

To optimize agent performance, we can provide a custom prompt with
domain-specific knowledge. In this case we'll create a few shot prompt
with an example selector, that will dynamically build the few shot
prompt based on the user input. This will help the model make better
queries by inserting relevant queries in the prompt that the model can
use as reference.

First we need some user input \\\> SQL query examples:

``` prism-code
examples = [
    {"input": "List all artists.", "query": "SELECT * FROM Artist;"},
    {
        "input": "Find all albums for the artist 'AC/DC'.",
        "query": "SELECT * FROM Album WHERE ArtistId = (SELECT ArtistId FROM Artist WHERE Name = 'AC/DC');",
    },
    {
        "input": "List all tracks in the 'Rock' genre.",
        "query": "SELECT * FROM Track WHERE GenreId = (SELECT GenreId FROM Genre WHERE Name = 'Rock');",
    },
    {
        "input": "Find the total duration of all tracks.",
        "query": "SELECT SUM(Milliseconds) FROM Track;",
    },
    {
        "input": "List all customers from Canada.",
        "query": "SELECT * FROM Customer WHERE Country = 'Canada';",
    },
    {
        "input": "How many tracks are there in the album with ID 5?",
        "query": "SELECT COUNT(*) FROM Track WHERE AlbumId = 5;",
    },
    {
        "input": "Find the total number of invoices.",
        "query": "SELECT COUNT(*) FROM Invoice;",
    },
    {
        "input": "List all tracks that are longer than 5 minutes.",
        "query": "SELECT * FROM Track WHERE Milliseconds > 300000;",
    },
    {
        "input": "Who are the top 5 customers by total purchase?",
        "query": "SELECT CustomerId, SUM(Total) AS TotalPurchase FROM Invoice GROUP BY CustomerId ORDER BY TotalPurchase DESC LIMIT 5;",
    },
    {
        "input": "Which albums are from the year 2000?",
        "query": "SELECT * FROM Album WHERE strftime('%Y', ReleaseDate) = '2000';",
    },
    {
        "input": "How many employees are there",
        "query": 'SELECT COUNT(*) FROM "Employee"',
    },
]
```

<img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJ3b3JkV3JhcEJ1dHRvbkljb25fQndtYSIgYXJpYS1oaWRkZW49InRydWUiPjxwYXRoIGZpbGw9ImN1cnJlbnRDb2xvciIgZD0iTTQgMTloNnYtMkg0djJ6TTIwIDVINHYyaDE2VjV6bS0zIDZINHYyaDEzLjI1YzEuMSAwIDIgLjkgMiAycy0uOSAyLTIgMkgxNXYtMmwtMyAzbDMgM3YtMmgyYzIuMjEgMCA0LTEuNzkgNC00cy0xLjc5LTQtNC00eiIgLz48L3N2Zz4="
class="wordWrapButtonIcon_Bwma" />

<span class="copyButtonIcons_eSgA" aria-hidden="true"><img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJjb3B5QnV0dG9uSWNvbl95OTdOIj48cGF0aCBmaWxsPSJjdXJyZW50Q29sb3IiIGQ9Ik0xOSwyMUg4VjdIMTlNMTksNUg4QTIsMiAwIDAsMCA2LDdWMjFBMiwyIDAgMCwwIDgsMjNIMTlBMiwyIDAgMCwwIDIxLDIxVjdBMiwyIDAgMCwwIDE5LDVNMTYsMUg0QTIsMiAwIDAsMCAyLDNWMTdINFYzSDE2VjFaIiAvPjwvc3ZnPg=="
class="copyButtonIcon_y97N" /><img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJjb3B5QnV0dG9uU3VjY2Vzc0ljb25fTGpkUyI+PHBhdGggZmlsbD0iY3VycmVudENvbG9yIiBkPSJNMjEsN0w5LDE5TDMuNSwxMy41TDQuOTEsMTIuMDlMOSwxNi4xN0wxOS41OSw1LjU5TDIxLDdaIiAvPjwvc3ZnPg=="
class="copyButtonSuccessIcon_LjdS" /></span>

Now we can create an example selector. This will take the actual user
input and select some number of examples to add to our few-shot prompt.
We'll use a SemanticSimilarityExampleSelector, which will perform a
semantic search using the embeddings and vector store we configure to
find the examples most similar to our input:

``` prism-code
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(),
    FAISS,
    k=5,
    input_keys=["input"],
)
```

<span class="copyButtonIcons_eSgA" aria-hidden="true"><img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJjb3B5QnV0dG9uSWNvbl95OTdOIj48cGF0aCBmaWxsPSJjdXJyZW50Q29sb3IiIGQ9Ik0xOSwyMUg4VjdIMTlNMTksNUg4QTIsMiAwIDAsMCA2LDdWMjFBMiwyIDAgMCwwIDgsMjNIMTlBMiwyIDAgMCwwIDIxLDIxVjdBMiwyIDAgMCwwIDE5LDVNMTYsMUg0QTIsMiAwIDAsMCAyLDNWMTdINFYzSDE2VjFaIiAvPjwvc3ZnPg=="
class="copyButtonIcon_y97N" /><img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJjb3B5QnV0dG9uU3VjY2Vzc0ljb25fTGpkUyI+PHBhdGggZmlsbD0iY3VycmVudENvbG9yIiBkPSJNMjEsN0w5LDE5TDMuNSwxMy41TDQuOTEsMTIuMDlMOSwxNi4xN0wxOS41OSw1LjU5TDIxLDdaIiAvPjwvc3ZnPg=="
class="copyButtonSuccessIcon_LjdS" /></span>

#### API Reference

- [FAISS](https://api.python.langchain.com/en/latest/vectorstores/langchain_community.vectorstores.faiss.FAISS.html)
- [SemanticSimilarityExampleSelector](https://api.python.langchain.com/en/latest/example_selectors/langchain_core.example_selectors.semantic_similarity.SemanticSimilarityExampleSelector.html)
- [OpenAIEmbeddings](https://api.python.langchain.com/en/latest/embeddings/langchain_openai.embeddings.base.OpenAIEmbeddings.html)

Now we can create our FewShotPromptTemplate, which takes our example
selector, an example prompt for formatting each example, and a string
prefix and suffix to put before and after our formatted examples:

``` prism-code
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)

system_prefix = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the given tools. Only use the information returned by the tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

If the question does not seem related to the database, just return "I don't know" as the answer.

Here are some examples of user inputs and their corresponding SQL queries:"""

few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=PromptTemplate.from_template(
        "User input: {input}\nSQL query: {query}"
    ),
    input_variables=["input", "dialect", "top_k"],
    prefix=system_prefix,
    suffix="",
)
```

<img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJ3b3JkV3JhcEJ1dHRvbkljb25fQndtYSIgYXJpYS1oaWRkZW49InRydWUiPjxwYXRoIGZpbGw9ImN1cnJlbnRDb2xvciIgZD0iTTQgMTloNnYtMkg0djJ6TTIwIDVINHYyaDE2VjV6bS0zIDZINHYyaDEzLjI1YzEuMSAwIDIgLjkgMiAycy0uOSAyLTIgMkgxNXYtMmwtMyAzbDMgM3YtMmgyYzIuMjEgMCA0LTEuNzkgNC00cy0xLjc5LTQtNC00eiIgLz48L3N2Zz4="
class="wordWrapButtonIcon_Bwma" />

<span class="copyButtonIcons_eSgA" aria-hidden="true"><img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJjb3B5QnV0dG9uSWNvbl95OTdOIj48cGF0aCBmaWxsPSJjdXJyZW50Q29sb3IiIGQ9Ik0xOSwyMUg4VjdIMTlNMTksNUg4QTIsMiAwIDAsMCA2LDdWMjFBMiwyIDAgMCwwIDgsMjNIMTlBMiwyIDAgMCwwIDIxLDIxVjdBMiwyIDAgMCwwIDE5LDVNMTYsMUg0QTIsMiAwIDAsMCAyLDNWMTdINFYzSDE2VjFaIiAvPjwvc3ZnPg=="
class="copyButtonIcon_y97N" /><img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJjb3B5QnV0dG9uU3VjY2Vzc0ljb25fTGpkUyI+PHBhdGggZmlsbD0iY3VycmVudENvbG9yIiBkPSJNMjEsN0w5LDE5TDMuNSwxMy41TDQuOTEsMTIuMDlMOSwxNi4xN0wxOS41OSw1LjU5TDIxLDdaIiAvPjwvc3ZnPg=="
class="copyButtonSuccessIcon_LjdS" /></span>

#### API Reference

- [ChatPromptTemplate](https://api.python.langchain.com/en/latest/prompts/langchain_core.prompts.chat.ChatPromptTemplate.html)
- [FewShotPromptTemplate](https://api.python.langchain.com/en/latest/prompts/langchain_core.prompts.few_shot.FewShotPromptTemplate.html)
- [MessagesPlaceholder](https://api.python.langchain.com/en/latest/prompts/langchain_core.prompts.chat.MessagesPlaceholder.html)
- [PromptTemplate](https://api.python.langchain.com/en/latest/prompts/langchain_core.prompts.prompt.PromptTemplate.html)
- [SystemMessagePromptTemplate](https://api.python.langchain.com/en/latest/prompts/langchain_core.prompts.chat.SystemMessagePromptTemplate.html)

Since our underlying agent is an [OpenAI tools
agent](/v0.1/docs/modules/agents/agent_types/openai_tools/), which uses
OpenAI function calling, our full prompt should be a chat prompt with a
human message template and an agent_scratchpad `MessagesPlaceholder`.
The few-shot prompt will be used for our system message:

``` prism-code
full_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate(prompt=few_shot_prompt),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)
```

<span class="copyButtonIcons_eSgA" aria-hidden="true"><img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJjb3B5QnV0dG9uSWNvbl95OTdOIj48cGF0aCBmaWxsPSJjdXJyZW50Q29sb3IiIGQ9Ik0xOSwyMUg4VjdIMTlNMTksNUg4QTIsMiAwIDAsMCA2LDdWMjFBMiwyIDAgMCwwIDgsMjNIMTlBMiwyIDAgMCwwIDIxLDIxVjdBMiwyIDAgMCwwIDE5LDVNMTYsMUg0QTIsMiAwIDAsMCAyLDNWMTdINFYzSDE2VjFaIiAvPjwvc3ZnPg=="
class="copyButtonIcon_y97N" /><img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJjb3B5QnV0dG9uU3VjY2Vzc0ljb25fTGpkUyI+PHBhdGggZmlsbD0iY3VycmVudENvbG9yIiBkPSJNMjEsN0w5LDE5TDMuNSwxMy41TDQuOTEsMTIuMDlMOSwxNi4xN0wxOS41OSw1LjU5TDIxLDdaIiAvPjwvc3ZnPg=="
class="copyButtonSuccessIcon_LjdS" /></span>

``` prism-code
# Example formatted prompt
prompt_val = full_prompt.invoke(
    {
        "input": "How many arists are there",
        "top_k": 5,
        "dialect": "SQLite",
        "agent_scratchpad": [],
    }
)
print(prompt_val.to_string())
```

<span class="copyButtonIcons_eSgA" aria-hidden="true"><img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJjb3B5QnV0dG9uSWNvbl95OTdOIj48cGF0aCBmaWxsPSJjdXJyZW50Q29sb3IiIGQ9Ik0xOSwyMUg4VjdIMTlNMTksNUg4QTIsMiAwIDAsMCA2LDdWMjFBMiwyIDAgMCwwIDgsMjNIMTlBMiwyIDAgMCwwIDIxLDIxVjdBMiwyIDAgMCwwIDE5LDVNMTYsMUg0QTIsMiAwIDAsMCAyLDNWMTdINFYzSDE2VjFaIiAvPjwvc3ZnPg=="
class="copyButtonIcon_y97N" /><img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJjb3B5QnV0dG9uU3VjY2Vzc0ljb25fTGpkUyI+PHBhdGggZmlsbD0iY3VycmVudENvbG9yIiBkPSJNMjEsN0w5LDE5TDMuNSwxMy41TDQuOTEsMTIuMDlMOSwxNi4xN0wxOS41OSw1LjU5TDIxLDdaIiAvPjwvc3ZnPg=="
class="copyButtonSuccessIcon_LjdS" /></span>

``` prism-code
System: You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct SQLite query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the given tools. Only use the information returned by the tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

If the question does not seem related to the database, just return "I don't know" as the answer.

Here are some examples of user inputs and their corresponding SQL queries:

User input: List all artists.
SQL query: SELECT * FROM Artist;

User input: How many employees are there
SQL query: SELECT COUNT(*) FROM "Employee"

User input: How many tracks are there in the album with ID 5?
SQL query: SELECT COUNT(*) FROM Track WHERE AlbumId = 5;

User input: List all tracks in the 'Rock' genre.
SQL query: SELECT * FROM Track WHERE GenreId = (SELECT GenreId FROM Genre WHERE Name = 'Rock');

User input: Which albums are from the year 2000?
SQL query: SELECT * FROM Album WHERE strftime('%Y', ReleaseDate) = '2000';
Human: How many arists are there
```

<img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJ3b3JkV3JhcEJ1dHRvbkljb25fQndtYSIgYXJpYS1oaWRkZW49InRydWUiPjxwYXRoIGZpbGw9ImN1cnJlbnRDb2xvciIgZD0iTTQgMTloNnYtMkg0djJ6TTIwIDVINHYyaDE2VjV6bS0zIDZINHYyaDEzLjI1YzEuMSAwIDIgLjkgMiAycy0uOSAyLTIgMkgxNXYtMmwtMyAzbDMgM3YtMmgyYzIuMjEgMCA0LTEuNzkgNC00cy0xLjc5LTQtNC00eiIgLz48L3N2Zz4="
class="wordWrapButtonIcon_Bwma" />

<span class="copyButtonIcons_eSgA" aria-hidden="true"><img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJjb3B5QnV0dG9uSWNvbl95OTdOIj48cGF0aCBmaWxsPSJjdXJyZW50Q29sb3IiIGQ9Ik0xOSwyMUg4VjdIMTlNMTksNUg4QTIsMiAwIDAsMCA2LDdWMjFBMiwyIDAgMCwwIDgsMjNIMTlBMiwyIDAgMCwwIDIxLDIxVjdBMiwyIDAgMCwwIDE5LDVNMTYsMUg0QTIsMiAwIDAsMCAyLDNWMTdINFYzSDE2VjFaIiAvPjwvc3ZnPg=="
class="copyButtonIcon_y97N" /><img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJjb3B5QnV0dG9uU3VjY2Vzc0ljb25fTGpkUyI+PHBhdGggZmlsbD0iY3VycmVudENvbG9yIiBkPSJNMjEsN0w5LDE5TDMuNSwxMy41TDQuOTEsMTIuMDlMOSwxNi4xN0wxOS41OSw1LjU5TDIxLDdaIiAvPjwvc3ZnPg=="
class="copyButtonSuccessIcon_LjdS" /></span>

And now we can create our agent with our custom prompt:

``` prism-code
agent = create_sql_agent(
    llm=llm,
    db=db,
    prompt=full_prompt,
    verbose=True,
    agent_type="openai-tools",
)
```

<span class="copyButtonIcons_eSgA" aria-hidden="true"><img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJjb3B5QnV0dG9uSWNvbl95OTdOIj48cGF0aCBmaWxsPSJjdXJyZW50Q29sb3IiIGQ9Ik0xOSwyMUg4VjdIMTlNMTksNUg4QTIsMiAwIDAsMCA2LDdWMjFBMiwyIDAgMCwwIDgsMjNIMTlBMiwyIDAgMCwwIDIxLDIxVjdBMiwyIDAgMCwwIDE5LDVNMTYsMUg0QTIsMiAwIDAsMCAyLDNWMTdINFYzSDE2VjFaIiAvPjwvc3ZnPg=="
class="copyButtonIcon_y97N" /><img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJjb3B5QnV0dG9uU3VjY2Vzc0ljb25fTGpkUyI+PHBhdGggZmlsbD0iY3VycmVudENvbG9yIiBkPSJNMjEsN0w5LDE5TDMuNSwxMy41TDQuOTEsMTIuMDlMOSwxNi4xN0wxOS41OSw1LjU5TDIxLDdaIiAvPjwvc3ZnPg=="
class="copyButtonSuccessIcon_LjdS" /></span>

Let's try it out:

``` prism-code
agent.invoke({"input": "How many artists are there?"})
```

<span class="copyButtonIcons_eSgA" aria-hidden="true"><img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJjb3B5QnV0dG9uSWNvbl95OTdOIj48cGF0aCBmaWxsPSJjdXJyZW50Q29sb3IiIGQ9Ik0xOSwyMUg4VjdIMTlNMTksNUg4QTIsMiAwIDAsMCA2LDdWMjFBMiwyIDAgMCwwIDgsMjNIMTlBMiwyIDAgMCwwIDIxLDIxVjdBMiwyIDAgMCwwIDE5LDVNMTYsMUg0QTIsMiAwIDAsMCAyLDNWMTdINFYzSDE2VjFaIiAvPjwvc3ZnPg=="
class="copyButtonIcon_y97N" /><img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJjb3B5QnV0dG9uU3VjY2Vzc0ljb25fTGpkUyI+PHBhdGggZmlsbD0iY3VycmVudENvbG9yIiBkPSJNMjEsN0w5LDE5TDMuNSwxMy41TDQuOTEsMTIuMDlMOSwxNi4xN0wxOS41OSw1LjU5TDIxLDdaIiAvPjwvc3ZnPg=="
class="copyButtonSuccessIcon_LjdS" /></span>

``` prism-code

[1m> Entering new AgentExecutor chain...[0m
[32;1m[1;3m
Invoking: `sql_db_query` with `{'query': 'SELECT COUNT(*) FROM Artist'}`


[0m[36;1m[1;3m[(275,)][0m[32;1m[1;3mThere are 275 artists in the database.[0m

[1m> Finished chain.[0m
```

<span class="copyButtonIcons_eSgA" aria-hidden="true"><img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJjb3B5QnV0dG9uSWNvbl95OTdOIj48cGF0aCBmaWxsPSJjdXJyZW50Q29sb3IiIGQ9Ik0xOSwyMUg4VjdIMTlNMTksNUg4QTIsMiAwIDAsMCA2LDdWMjFBMiwyIDAgMCwwIDgsMjNIMTlBMiwyIDAgMCwwIDIxLDIxVjdBMiwyIDAgMCwwIDE5LDVNMTYsMUg0QTIsMiAwIDAsMCAyLDNWMTdINFYzSDE2VjFaIiAvPjwvc3ZnPg=="
class="copyButtonIcon_y97N" /><img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJjb3B5QnV0dG9uU3VjY2Vzc0ljb25fTGpkUyI+PHBhdGggZmlsbD0iY3VycmVudENvbG9yIiBkPSJNMjEsN0w5LDE5TDMuNSwxMy41TDQuOTEsMTIuMDlMOSwxNi4xN0wxOS41OSw1LjU5TDIxLDdaIiAvPjwvc3ZnPg=="
class="copyButtonSuccessIcon_LjdS" /></span>

``` prism-code
{'input': 'How many artists are there?',
 'output': 'There are 275 artists in the database.'}
```

<span class="copyButtonIcons_eSgA" aria-hidden="true"><img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJjb3B5QnV0dG9uSWNvbl95OTdOIj48cGF0aCBmaWxsPSJjdXJyZW50Q29sb3IiIGQ9Ik0xOSwyMUg4VjdIMTlNMTksNUg4QTIsMiAwIDAsMCA2LDdWMjFBMiwyIDAgMCwwIDgsMjNIMTlBMiwyIDAgMCwwIDIxLDIxVjdBMiwyIDAgMCwwIDE5LDVNMTYsMUg0QTIsMiAwIDAsMCAyLDNWMTdINFYzSDE2VjFaIiAvPjwvc3ZnPg=="
class="copyButtonIcon_y97N" /><img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJjb3B5QnV0dG9uU3VjY2Vzc0ljb25fTGpkUyI+PHBhdGggZmlsbD0iY3VycmVudENvbG9yIiBkPSJNMjEsN0w5LDE5TDMuNSwxMy41TDQuOTEsMTIuMDlMOSwxNi4xN0wxOS41OSw1LjU5TDIxLDdaIiAvPjwvc3ZnPg=="
class="copyButtonSuccessIcon_LjdS" /></span>

## Dealing with high-cardinality columns<a href="#dealing-with-high-cardinality-columns" class="hash-link"

aria-label="Direct link to Dealing with high-cardinality columns"
title="Direct link to Dealing with high-cardinality columns">​</a>

In order to filter columns that contain proper nouns such as addresses,
song names or artists, we first need to double-check the spelling in
order to filter the data correctly.

We can achieve this by creating a vector store with all the distinct
proper nouns that exist in the database. We can then have the agent
query that vector store each time the user includes a proper noun in
their question, to find the correct spelling for that word. In this way,
the agent can make sure it understands which entity the user is
referring to before building the target query.

First we need the unique values for each entity we want, for which we
define a function that parses the result into a list of elements:

``` prism-code
import ast
import re


def query_as_list(db, query):
    res = db.run(query)
    res = [el for sub in ast.literal_eval(res) for el in sub if el]
    res = [re.sub(r"\b\d+\b", "", string).strip() for string in res]
    return list(set(res))


artists = query_as_list(db, "SELECT Name FROM Artist")
albums = query_as_list(db, "SELECT Title FROM Album")
albums[:5]
```

<span class="copyButtonIcons_eSgA" aria-hidden="true"><img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJjb3B5QnV0dG9uSWNvbl95OTdOIj48cGF0aCBmaWxsPSJjdXJyZW50Q29sb3IiIGQ9Ik0xOSwyMUg4VjdIMTlNMTksNUg4QTIsMiAwIDAsMCA2LDdWMjFBMiwyIDAgMCwwIDgsMjNIMTlBMiwyIDAgMCwwIDIxLDIxVjdBMiwyIDAgMCwwIDE5LDVNMTYsMUg0QTIsMiAwIDAsMCAyLDNWMTdINFYzSDE2VjFaIiAvPjwvc3ZnPg=="
class="copyButtonIcon_y97N" /><img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJjb3B5QnV0dG9uU3VjY2Vzc0ljb25fTGpkUyI+PHBhdGggZmlsbD0iY3VycmVudENvbG9yIiBkPSJNMjEsN0w5LDE5TDMuNSwxMy41TDQuOTEsMTIuMDlMOSwxNi4xN0wxOS41OSw1LjU5TDIxLDdaIiAvPjwvc3ZnPg=="
class="copyButtonSuccessIcon_LjdS" /></span>

``` prism-code
['Os Cães Ladram Mas A Caravana Não Pára',
 'War',
 'Mais Do Mesmo',
 "Up An' Atom",
 'Riot Act']
```

<span class="copyButtonIcons_eSgA" aria-hidden="true"><img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJjb3B5QnV0dG9uSWNvbl95OTdOIj48cGF0aCBmaWxsPSJjdXJyZW50Q29sb3IiIGQ9Ik0xOSwyMUg4VjdIMTlNMTksNUg4QTIsMiAwIDAsMCA2LDdWMjFBMiwyIDAgMCwwIDgsMjNIMTlBMiwyIDAgMCwwIDIxLDIxVjdBMiwyIDAgMCwwIDE5LDVNMTYsMUg0QTIsMiAwIDAsMCAyLDNWMTdINFYzSDE2VjFaIiAvPjwvc3ZnPg=="
class="copyButtonIcon_y97N" /><img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJjb3B5QnV0dG9uU3VjY2Vzc0ljb25fTGpkUyI+PHBhdGggZmlsbD0iY3VycmVudENvbG9yIiBkPSJNMjEsN0w5LDE5TDMuNSwxMy41TDQuOTEsMTIuMDlMOSwxNi4xN0wxOS41OSw1LjU5TDIxLDdaIiAvPjwvc3ZnPg=="
class="copyButtonSuccessIcon_LjdS" /></span>

Now we can proceed with creating the custom **retriever tool** and the
final agent:

``` prism-code
from langchain.agents.agent_toolkits import create_retriever_tool

vector_db = FAISS.from_texts(artists + albums, OpenAIEmbeddings())
retriever = vector_db.as_retriever(search_kwargs={"k": 5})
description = """Use to look up values to filter on. Input is an approximate spelling of the proper noun, output is \
valid proper nouns. Use the noun most similar to the search."""
retriever_tool = create_retriever_tool(
    retriever,
    name="search_proper_nouns",
    description=description,
)
```

<img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJ3b3JkV3JhcEJ1dHRvbkljb25fQndtYSIgYXJpYS1oaWRkZW49InRydWUiPjxwYXRoIGZpbGw9ImN1cnJlbnRDb2xvciIgZD0iTTQgMTloNnYtMkg0djJ6TTIwIDVINHYyaDE2VjV6bS0zIDZINHYyaDEzLjI1YzEuMSAwIDIgLjkgMiAycy0uOSAyLTIgMkgxNXYtMmwtMyAzbDMgM3YtMmgyYzIuMjEgMCA0LTEuNzkgNC00cy0xLjc5LTQtNC00eiIgLz48L3N2Zz4="
class="wordWrapButtonIcon_Bwma" />

<span class="copyButtonIcons_eSgA" aria-hidden="true"><img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJjb3B5QnV0dG9uSWNvbl95OTdOIj48cGF0aCBmaWxsPSJjdXJyZW50Q29sb3IiIGQ9Ik0xOSwyMUg4VjdIMTlNMTksNUg4QTIsMiAwIDAsMCA2LDdWMjFBMiwyIDAgMCwwIDgsMjNIMTlBMiwyIDAgMCwwIDIxLDIxVjdBMiwyIDAgMCwwIDE5LDVNMTYsMUg0QTIsMiAwIDAsMCAyLDNWMTdINFYzSDE2VjFaIiAvPjwvc3ZnPg=="
class="copyButtonIcon_y97N" /><img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJjb3B5QnV0dG9uU3VjY2Vzc0ljb25fTGpkUyI+PHBhdGggZmlsbD0iY3VycmVudENvbG9yIiBkPSJNMjEsN0w5LDE5TDMuNSwxMy41TDQuOTEsMTIuMDlMOSwxNi4xN0wxOS41OSw1LjU5TDIxLDdaIiAvPjwvc3ZnPg=="
class="copyButtonSuccessIcon_LjdS" /></span>

#### API Reference

- [create_retriever_tool](https://api.python.langchain.com/en/latest/tools/langchain_core.tools.create_retriever_tool.html)

``` prism-code
system = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the given tools. Only use the information returned by the tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

If you need to filter on a proper noun, you must ALWAYS first look up the filter value using the "search_proper_nouns" tool! 

You have access to the following tables: {table_names}

If the question does not seem related to the database, just return "I don't know" as the answer."""

prompt = ChatPromptTemplate.from_messages(
    [("system", system), ("human", "{input}"), MessagesPlaceholder("agent_scratchpad")]
)
agent = create_sql_agent(
    llm=llm,
    db=db,
    extra_tools=[retriever_tool],
    prompt=prompt,
    agent_type="openai-tools",
    verbose=True,
)
```

<img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJ3b3JkV3JhcEJ1dHRvbkljb25fQndtYSIgYXJpYS1oaWRkZW49InRydWUiPjxwYXRoIGZpbGw9ImN1cnJlbnRDb2xvciIgZD0iTTQgMTloNnYtMkg0djJ6TTIwIDVINHYyaDE2VjV6bS0zIDZINHYyaDEzLjI1YzEuMSAwIDIgLjkgMiAycy0uOSAyLTIgMkgxNXYtMmwtMyAzbDMgM3YtMmgyYzIuMjEgMCA0LTEuNzkgNC00cy0xLjc5LTQtNC00eiIgLz48L3N2Zz4="
class="wordWrapButtonIcon_Bwma" />

<span class="copyButtonIcons_eSgA" aria-hidden="true"><img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJjb3B5QnV0dG9uSWNvbl95OTdOIj48cGF0aCBmaWxsPSJjdXJyZW50Q29sb3IiIGQ9Ik0xOSwyMUg4VjdIMTlNMTksNUg4QTIsMiAwIDAsMCA2LDdWMjFBMiwyIDAgMCwwIDgsMjNIMTlBMiwyIDAgMCwwIDIxLDIxVjdBMiwyIDAgMCwwIDE5LDVNMTYsMUg0QTIsMiAwIDAsMCAyLDNWMTdINFYzSDE2VjFaIiAvPjwvc3ZnPg=="
class="copyButtonIcon_y97N" /><img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJjb3B5QnV0dG9uU3VjY2Vzc0ljb25fTGpkUyI+PHBhdGggZmlsbD0iY3VycmVudENvbG9yIiBkPSJNMjEsN0w5LDE5TDMuNSwxMy41TDQuOTEsMTIuMDlMOSwxNi4xN0wxOS41OSw1LjU5TDIxLDdaIiAvPjwvc3ZnPg=="
class="copyButtonSuccessIcon_LjdS" /></span>

``` prism-code
agent.invoke({"input": "How many albums does alis in chain have?"})
```

<span class="copyButtonIcons_eSgA" aria-hidden="true"><img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJjb3B5QnV0dG9uSWNvbl95OTdOIj48cGF0aCBmaWxsPSJjdXJyZW50Q29sb3IiIGQ9Ik0xOSwyMUg4VjdIMTlNMTksNUg4QTIsMiAwIDAsMCA2LDdWMjFBMiwyIDAgMCwwIDgsMjNIMTlBMiwyIDAgMCwwIDIxLDIxVjdBMiwyIDAgMCwwIDE5LDVNMTYsMUg0QTIsMiAwIDAsMCAyLDNWMTdINFYzSDE2VjFaIiAvPjwvc3ZnPg=="
class="copyButtonIcon_y97N" /><img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJjb3B5QnV0dG9uU3VjY2Vzc0ljb25fTGpkUyI+PHBhdGggZmlsbD0iY3VycmVudENvbG9yIiBkPSJNMjEsN0w5LDE5TDMuNSwxMy41TDQuOTEsMTIuMDlMOSwxNi4xN0wxOS41OSw1LjU5TDIxLDdaIiAvPjwvc3ZnPg=="
class="copyButtonSuccessIcon_LjdS" /></span>

``` prism-code

[1m> Entering new AgentExecutor chain...[0m
[32;1m[1;3m
Invoking: `search_proper_nouns` with `{'query': 'alis in chain'}`


[0m[36;1m[1;3mAlice In Chains

Aisha Duo

Xis

Da Lama Ao Caos

A-Sides[0m[32;1m[1;3m
Invoking: `sql_db_query` with `SELECT COUNT(*) FROM Album WHERE ArtistId = (SELECT ArtistId FROM Artist WHERE Name = 'Alice In Chains')`


[0m[36;1m[1;3m[(1,)][0m[32;1m[1;3mAlice In Chains has 1 album.[0m

[1m> Finished chain.[0m
```

<img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJ3b3JkV3JhcEJ1dHRvbkljb25fQndtYSIgYXJpYS1oaWRkZW49InRydWUiPjxwYXRoIGZpbGw9ImN1cnJlbnRDb2xvciIgZD0iTTQgMTloNnYtMkg0djJ6TTIwIDVINHYyaDE2VjV6bS0zIDZINHYyaDEzLjI1YzEuMSAwIDIgLjkgMiAycy0uOSAyLTIgMkgxNXYtMmwtMyAzbDMgM3YtMmgyYzIuMjEgMCA0LTEuNzkgNC00cy0xLjc5LTQtNC00eiIgLz48L3N2Zz4="
class="wordWrapButtonIcon_Bwma" />

<span class="copyButtonIcons_eSgA" aria-hidden="true"><img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJjb3B5QnV0dG9uSWNvbl95OTdOIj48cGF0aCBmaWxsPSJjdXJyZW50Q29sb3IiIGQ9Ik0xOSwyMUg4VjdIMTlNMTksNUg4QTIsMiAwIDAsMCA2LDdWMjFBMiwyIDAgMCwwIDgsMjNIMTlBMiwyIDAgMCwwIDIxLDIxVjdBMiwyIDAgMCwwIDE5LDVNMTYsMUg0QTIsMiAwIDAsMCAyLDNWMTdINFYzSDE2VjFaIiAvPjwvc3ZnPg=="
class="copyButtonIcon_y97N" /><img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJjb3B5QnV0dG9uU3VjY2Vzc0ljb25fTGpkUyI+PHBhdGggZmlsbD0iY3VycmVudENvbG9yIiBkPSJNMjEsN0w5LDE5TDMuNSwxMy41TDQuOTEsMTIuMDlMOSwxNi4xN0wxOS41OSw1LjU5TDIxLDdaIiAvPjwvc3ZnPg=="
class="copyButtonSuccessIcon_LjdS" /></span>

``` prism-code
{'input': 'How many albums does alis in chain have?',
 'output': 'Alice In Chains has 1 album.'}
```

<span class="copyButtonIcons_eSgA" aria-hidden="true"><img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJjb3B5QnV0dG9uSWNvbl95OTdOIj48cGF0aCBmaWxsPSJjdXJyZW50Q29sb3IiIGQ9Ik0xOSwyMUg4VjdIMTlNMTksNUg4QTIsMiAwIDAsMCA2LDdWMjFBMiwyIDAgMCwwIDgsMjNIMTlBMiwyIDAgMCwwIDIxLDIxVjdBMiwyIDAgMCwwIDE5LDVNMTYsMUg0QTIsMiAwIDAsMCAyLDNWMTdINFYzSDE2VjFaIiAvPjwvc3ZnPg=="
class="copyButtonIcon_y97N" /><img
src="data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjQgMjQiIGNsYXNzPSJjb3B5QnV0dG9uU3VjY2Vzc0ljb25fTGpkUyI+PHBhdGggZmlsbD0iY3VycmVudENvbG9yIiBkPSJNMjEsN0w5LDE5TDMuNSwxMy41TDQuOTEsMTIuMDlMOSwxNi4xN0wxOS41OSw1LjU5TDIxLDdaIiAvPjwvc3ZnPg=="
class="copyButtonSuccessIcon_LjdS" /></span>

As we can see, the agent used the `search_proper_nouns` tool in order to
check how to correctly query the database for this specific artist.

## Next steps<a href="#next-steps" class="hash-link"

aria-label="Direct link to Next steps"
title="Direct link to Next steps">​</a>

Under the hood, `create_sql_agent` is just passing in SQL tools to more
generic agent constructors. To learn more about the built-in generic
agent types as well as how to build custom agents, head to the [Agents
Modules](/v0.1/docs/modules/agents/).

The built-in `AgentExecutor` runs a simple Agent action -\> Tool call
-\> Agent action... loop. To build more complex agent runtimes, head to
the <a href="https://langchain-ai.github.io/langgraph/" target="_blank"
rel="noopener noreferrer">LangGraph section</a>.
