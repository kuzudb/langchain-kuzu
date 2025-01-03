from typing import List, Dict, Any
import os
import shutil
import sys

import kuzu
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI

sys.path.append(os.path.abspath("./libs/kuzu"))
from langchain_kuzu.graphs.kuzu_graph import KuzuGraph

load_dotenv()
SEED = 42
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


def init_database(db_name: str = "test_db", overwrite=False) -> kuzu.Database:
    if overwrite:
        shutil.rmtree(db_name, ignore_errors=True)
    db = kuzu.Database(db_name)
    return db


def load_text(filepath: str) -> List[Document]:
    with open(filepath, "r") as f:
        text = f.read()
    assert text, "Text is empty: Please check the input file and try again."
    return [Document(page_content=text)]


def get_graph_schema() -> Dict[str, List[Any]]:
    return {
        "allowed_nodes": ["Person", "NobelPrize", "Discovery"],
        "allowed_relationships": [
            ("Person", "WORKED_WITH", "Person"),
            ("Person", "IS_MARRIED_TO", "Person"),
            ("Person", "DISCOVERED", "Discovery"),
            ("Person", "WON", "NobelPrize"),
        ],
    }


def extract_graph_data(documents: List[Document]) -> List[Any]:
    extraction_llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-4o-mini",
        api_key=OPENAI_API_KEY,
        seed=SEED,
    )

    schema = get_graph_schema()
    llm_transformer = LLMGraphTransformer(
        llm=extraction_llm,
        allowed_nodes=schema["allowed_nodes"],
        allowed_relationships=schema["allowed_relationships"],
    )

    return llm_transformer.convert_to_graph_documents(documents)


def create_graph(graph_documents: List[Any], db: kuzu.Database) -> KuzuGraph:
    schema = get_graph_schema()
    graph = KuzuGraph(db, allow_dangerous_requests=True)

    graph.add_graph_documents(
        graph_documents,
        allowed_relationships=schema["allowed_relationships"],
        include_source=True,
    )
    return graph


if __name__ == "__main__":
    documents = load_text("/Users/prrao/code/kuzu-rag/curie/curie.txt")
    graph_documents = extract_graph_data(documents)

    db = init_database("test_db", overwrite=True)

    print(f"Nodes:{graph_documents[0].nodes}")
    print(f"Relationships:{graph_documents[0].relationships}")
    print("Finished extracting nodes and relationships.")

    graph = create_graph(graph_documents, db)