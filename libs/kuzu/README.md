# 🦜️🔗 LangChain Kùzu

This package contains the LangChain integration with Kùzu, an embeddable property graph database.

## 📦 Installation

```bash
pip install -U langchain-kuzu
```


## Usage

Kùzu's integration with LangChain makes it convenient to create and update graphs from unstructured text, and also to query graphs via a Text2Cypher pipeline that utilizes the
power of LangChain's LLM chains. In this section, we'll cover
an example of how to use the `KuzuGraph` and `KuzuQAChain` classes to create a graph, add nodes and relationships, and query the graph in conjunction with OpenAI's LLMs.

Let's install some additional dependencies:

```bash
pip install -U langchain-openai langchain-experimental
```

### Creating a graph

```py
text = "Tim Cook is the CEO of Apple. Apple has its headquarters in California."
```

First define the LLM to use for graph extraction, and the schema for the graph.

```py
# Define schema
allowed_nodes = ["Person", "Company", "Location"]
allowed_relationships = [
    ("Person", "IS_CEO_OF", "Company"),
    ("Company", "HAS_HEADQUARTERS_IN", "Location"),
]
```

The `LLMGraphTransformer` class is used to convert the text into a graph document.

```py
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI

# Define the LLMGraphTransformer
llm_transformer = LLMGraphTransformer(
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY),
    allowed_nodes=allowed_nodes,
    allowed_relationships=allowed_relationships,
)
```


```py
from langchain_core.documents import Document

# Convert the given text into graph documents
documents = [Document(page_content=text)]
graph_documents = llm_transformer.convert_to_graph_documents(documents)
```

To ingest the graph documents directly into a Kùzu database, we can use the `KuzuGraph` class.

```py
import kuzu
from langchain_kuzu.graphs.kuzu_graph import KuzuGraph

db = kuzu.Database("test_db")

# Create a graph object
graph = KuzuGraph(db, allow_dangerous_requests=True)

# Add the graph document to the graph
graph.add_graph_documents(
    graph_documents,
    include_source=True,
)
```

### Query the graph

To query the graph, we can define a `KuzuQAChain` object. Then, we can invoke the chain with a query by connecting to the existing database that's stored in the `test_db` directory as per the
previous step.

```py
import kuzu
from langchain_kuzu.graphs.kuzu_graph import KuzuGraph
from langchain_kuzu.chains.graph_qa.kuzu import KuzuQAChain

db = kuzu.Database("test_db")
graph = KuzuGraph(db, allow_dangerous_requests=True)

# Create the KuzuQAChain with verbosity enabled to see the generated Cypher queries
chain = KuzuQAChain.from_llm(
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.3, api_key=OPENAI_API_KEY),
    graph=graph,
    verbose=True,
    allow_dangerous_requests=True,
)

# Query the graph
queries = [
    "Who is the CEO of Apple?",
    "Where is Apple headquartered?",
]

for query in queries:
    result = chain.invoke(query)
    print(f"Query: {query}\nResult: {result}\n")
```

The following is the output:

```
> Entering new KuzuQAChain chain...
Generated Cypher:
MATCH (p:Person)-[:IS_CEO_OF]->(c:Company) WHERE c.id = 'Apple' RETURN p.id
Full Context:
[{'p.id': 'Tim Cook'}]

> Finished chain.
Query: Who is the CEO of Apple?
Result: {'query': 'Who is the CEO of Apple?', 'result': 'Tim Cook is the CEO of Apple.'}



> Entering new KuzuQAChain chain...
Generated Cypher:
MATCH (c:Company {id: "Apple"})-[:HAS_HEADQUARTERS_IN]->(l:Location) RETURN l.id
Full Context:
[{'l.id': 'California'}]

> Finished chain.
Query: Where is Apple headquartered?
Result: {'query': 'Where is Apple headquartered?', 'result': 'Apple is headquartered in California.'}
```

### Updating the graph

You can update or mutate the graph's state by connecting to the existing database and running your
own Cypher queries.

```py
import kuzu

db = kuzu.Database("test_db")
conn = kuzu.Connection(db)

# Create a new relationship table
conn.execute("CREATE REL TABLE IS_COO_OF(FROM Person TO Company)")

# Add a new person-company relationship for Jeff Williams, the COO of Apple
conn.execute("CREATE (p:Person {id: 'Jeff Williams'})")
conn.execute(
    """
    MATCH (c:Company {id: 'Apple'}), (p:Person {id: 'Jeff Williams'})
    CREATE (p)-[:IS_COO_OF]->(c)
    """
)
```