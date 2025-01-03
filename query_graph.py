import os
import sys

import kuzu
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

sys.path.append(os.path.abspath("./libs/kuzu"))
from langchain_kuzu.chains.graph_qa.kuzu import KuzuQAChain
from langchain_kuzu.graphs.kuzu_graph import KuzuGraph


def create_qa_chain(graph, seed=42):
    generation_llm = ChatOpenAI(
        temperature=0.3,
        model="gpt-4",
        api_key=os.environ.get("OPENAI_API_KEY"),
        seed=seed,
    )
    return KuzuQAChain.from_llm(
        llm=generation_llm,
        graph=graph,
        verbose=True,
        allow_dangerous_requests=True,
    )


def query_graph(chain, query):
    result = chain.invoke(query)
    print(f"Query: {query}\nResult: {result}\n")
    return result


if __name__ == "__main__":
    load_dotenv()
    db = kuzu.Database("test_db")
    conn = kuzu.Connection(db)
    graph = KuzuGraph(db, allow_dangerous_requests=True)
    chain = create_qa_chain(graph)

    queries = [
        "Who did Pierre Curie work with?",
        "What Nobel Prize did Marie Curie and Pierre Curie win together?",
    ]

    for query in queries:
        query_graph(chain, query)
