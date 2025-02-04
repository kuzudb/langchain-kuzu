from typing import Any, Generator, Optional
from unittest.mock import Mock, patch

import pytest

from langchain_kuzu.graphs.graph_document import GraphDocument, Node
from langchain_kuzu.graphs.kuzu_graph import KuzuGraph


class MockCursor:
    def __init__(self, results: Optional[list[Any]] = None) -> None:
        self.results = results or []
        self.current_index: int = 0
        self.column_names: list[str] = ["column1"]  # default

    def has_next(self) -> bool:
        return self.current_index < len(self.results)

    def get_next(self) -> Any:
        if self.has_next():
            result = self.results[self.current_index]
            self.current_index += 1
            return result
        return None

    def get_column_names(self) -> list[str]:
        return self.column_names


@pytest.fixture
def mock_kuzu_connection() -> Generator[Mock, None, None]:
    with patch("kuzu.Connection", autospec=True) as mock_conn:
        # Mock schema-related methods
        connection = mock_conn.return_value
        connection._get_node_table_names.return_value = [
            "Person",
            "Company",
        ]
        connection._get_node_property_names.return_value = {
            "name": {"type": "STRING", "dimension": 0}
        }
        connection._get_rel_table_names.return_value = [
            {"src": "Person", "name": "WORKS_AT", "dst": "Company"}
        ]
        connection.execute = Mock(
            return_value=Mock(
                has_next=lambda: False,
                get_column_names=lambda: ["column1"],
                call_args_list=[],
                reset_mock=lambda: None,
                call_count=0,
            )
        )
        yield connection


@pytest.fixture
def kuzu_graph(mock_kuzu_connection: Mock) -> KuzuGraph:
    db = Mock()

    # Create a concrete subclass that implements the abstract method
    class ConcreteKuzuGraph(KuzuGraph):
        @property
        def get_structured_schema(self) -> dict:
            return {}

    return ConcreteKuzuGraph(db, allow_dangerous_requests=True)


def test_init_without_dangerous_requests() -> None:
    db = Mock()

    # Create concrete subclass for testing
    class ConcreteKuzuGraph(KuzuGraph):
        @property
        def get_structured_schema(self) -> dict:
            return {}

    with pytest.raises(ValueError, match="powerful tool"):
        ConcreteKuzuGraph(db, allow_dangerous_requests=False)


def test_query(kuzu_graph: KuzuGraph, mock_kuzu_connection: Mock) -> None:
    mock_kuzu_connection.execute.return_value = MockCursor(
        results=[["Rhea Seacrest", 33]]
    )
    mock_kuzu_connection.execute.return_value.column_names = ["name", "age"]

    result = kuzu_graph.query("MATCH (p:Person) RETURN p.name, p.age")
    assert result == [{"name": "Rhea Seacrest", "age": 33}]


def test_refresh_schema(kuzu_graph: KuzuGraph, mock_kuzu_connection: Mock) -> None:
    kuzu_graph.refresh_schema()
    assert "Node properties" in kuzu_graph.schema
    assert "Relationships properties" in kuzu_graph.schema
    assert "Relationships" in kuzu_graph.schema


def test_query_with_params(kuzu_graph: KuzuGraph, mock_kuzu_connection: Mock) -> None:
    mock_kuzu_connection.execute.return_value = MockCursor(results=[["value1"]])

    result = kuzu_graph.query("MATCH (n) WHERE n.id = $id RETURN n", {"id": "123"})
    assert result == [{"column1": "value1"}]


def test_refresh_schema_with_array_properties(
    kuzu_graph: KuzuGraph, mock_kuzu_connection: Mock
) -> None:
    mock_kuzu_connection._get_node_property_names.return_value = {
        "vector": {"type": "FLOAT", "dimension": 2, "shape": [768]},
        "tags": {"type": "STRING", "dimension": 1},
    }

    kuzu_graph.refresh_schema()
    assert "FLOAT[768]" in kuzu_graph.schema
    assert "STRING[]" in kuzu_graph.schema


def test_add_graph_documents_with_source(kuzu_graph: KuzuGraph) -> None:
    from langchain_core.documents import Document

    # Create test data with source document
    node1 = Node(id="1", type="Person")
    source_doc = Document(page_content="Test content", metadata={})
    doc = GraphDocument(nodes=[node1], relationships=[], source=source_doc)

    kuzu_graph.add_graph_documents([doc], include_source=True)

    # Verify Chunk table and MENTIONS relationship were created
    expected_queries = [
        "CREATE NODE TABLE IF NOT EXISTS Chunk",
        "MERGE (c:Chunk {id: $id})",
        "CREATE REL TABLE GROUP IF NOT EXISTS MENTIONS",
        "MERGE (c)-[m:MENTIONS]->(e)",
    ]

    for query in expected_queries:
        assert any(
            query in call.args[0]
            for call in kuzu_graph.conn.execute.call_args_list  # type: ignore[attr-defined]
        )


def test_add_graph_documents_with_existing_source_id(kuzu_graph: KuzuGraph) -> None:
    from langchain_core.documents import Document

    node1 = Node(id="1", type="Person")
    source_doc = Document(page_content="Test content", metadata={"id": "existing_id"})
    doc = GraphDocument(nodes=[node1], relationships=[], source=source_doc)

    kuzu_graph.add_graph_documents([doc], include_source=True)

    # Verify the existing ID was used
    assert any(
        "'existing_id'" in str(call)
        for call in kuzu_graph.conn.execute.call_args_list  # type: ignore[attr-defined]
    )


def test_get_schema_property(kuzu_graph: KuzuGraph) -> None:
    test_schema = "test schema"
    kuzu_graph.schema = test_schema
    assert kuzu_graph.get_schema == test_schema
