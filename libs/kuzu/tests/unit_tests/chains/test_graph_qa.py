from typing import Any, Dict, List

import pytest
from langchain_core.prompts import PromptTemplate
from llms.fake_llm import FakeLLM

from langchain_kuzu.chains.graph_qa.kuzu import (
    KuzuQAChain,
    extract_cypher,
    remove_prefix,
)
from langchain_kuzu.chains.graph_qa.prompts import (
    CYPHER_QA_PROMPT,
    KUZU_GENERATION_PROMPT,
)
from langchain_kuzu.graphs.graph_document import GraphDocument
from langchain_kuzu.graphs.graph_store import GraphStore


class FakeGraphStore(GraphStore):
    @property
    def get_schema(self) -> str:
        """Returns the schema of the Graph database"""
        return ""

    @property
    def get_structured_schema(self) -> Dict[str, Any]:
        """Returns the schema of the Graph database"""
        return {}

    def query(self, query: str, params: dict = {}) -> List[Dict[str, Any]]:
        """Query the graph."""
        return []

    def refresh_schema(self) -> None:
        """Refreshes the graph schema information."""
        pass

    def add_graph_documents(
        self, graph_documents: List[GraphDocument], include_source: bool = False
    ) -> None:
        """Take GraphDocument as input as uses it to construct a graph."""
        pass


def test_cypher_generation_failure() -> None:
    """Test the chain doesn't fail if the Cypher query fails to be generated."""
    qa_prompt_template = "QA Prompt"
    cypher_prompt_template = "Cypher Prompt"
    qa_prompt = PromptTemplate(template=qa_prompt_template, input_variables=[])
    cypher_prompt = PromptTemplate(template=cypher_prompt_template, input_variables=[])
    llm = FakeLLM()
    chain = KuzuQAChain.from_llm(
        llm=llm,
        graph=FakeGraphStore(),
        allow_dangerous_requests=True,
        qa_prompt=qa_prompt,
        cypher_prompt=cypher_prompt,
    )
    response = chain.invoke({"query": ""})
    assert response["result"] == "foo"


def test_graph_cypher_qa_chain_prompt_selection_4() -> None:
    qa_prompt_template = "QA Prompt"
    cypher_prompt_template = "Cypher Prompt"
    qa_prompt = PromptTemplate(template=qa_prompt_template, input_variables=[])
    cypher_prompt = PromptTemplate(template=cypher_prompt_template, input_variables=[])
    chain = KuzuQAChain.from_llm(
        llm=FakeLLM(),
        graph=FakeGraphStore(),
        allow_dangerous_requests=True,
        qa_prompt=qa_prompt,
        cypher_prompt=cypher_prompt,
    )
    assert chain.qa_chain.prompt == qa_prompt
    assert chain.cypher_generation_chain.prompt == cypher_prompt


def test_remove_prefix() -> None:
    # Test with matching prefix
    assert remove_prefix("cypherMATCH (n)", "cypher") == "MATCH (n)"
    # Test with no prefix
    assert remove_prefix("MATCH (n)", "cypher") == "MATCH (n)"
    # Test with empty string
    assert remove_prefix("", "cypher") == ""
    # Test with empty prefix
    assert remove_prefix("test string", "") == "test string"
    # Test with prefix longer than string
    assert remove_prefix("MATCH", "cypherMATCH") == "MATCH"


def test_no_backticks() -> None:
    """Test if there are no backticks, so the original text should be returned."""
    query = "MATCH (n) RETURN n"
    output = extract_cypher(query)
    assert output == query


def test_backticks() -> None:
    """Test if there are backticks. Query from within backticks should be returned."""
    query: str = "You can use the following query: ```MATCH (n) RETURN n```"
    output: str = extract_cypher(query)
    assert output == "MATCH (n) RETURN n"


def test_allow_dangerous_requests_err() -> None:
    with pytest.raises(ValueError) as exc_info:
        KuzuQAChain.from_llm(
            llm=FakeLLM(),
            graph=FakeGraphStore(),
        )
    assert (
        "In order to use this chain, you must acknowledge that it can make "
        "dangerous requests by setting `allow_dangerous_requests` to `True`."
    ) in str(exc_info.value)


def test_kuzu_generation_prompt_structure() -> None:
    """Test KUZU_GENERATION_PROMPT has correct structure and variables."""
    assert isinstance(KUZU_GENERATION_PROMPT, PromptTemplate)
    required_variables = {"schema", "question"}
    assert set(KUZU_GENERATION_PROMPT.input_variables) == required_variables

    # Test prompt contains key instruction elements
    template = KUZU_GENERATION_PROMPT.template.lower()
    assert "cypher" in template
    assert "schema" in template
    assert "question" in template


def test_cypher_qa_prompt_structure() -> None:
    """Test CYPHER_QA_PROMPT has correct structure and variables."""
    assert isinstance(CYPHER_QA_PROMPT, PromptTemplate)
    required_variables = {"context", "question"}
    assert set(CYPHER_QA_PROMPT.input_variables) == required_variables

    # Test prompt contains key instruction elements
    template = CYPHER_QA_PROMPT.template.lower()
    assert "context" in template
    assert "question" in template


def test_llm_arg_combinations() -> None:
    # No llm
    with pytest.raises(ValueError) as exc_info:
        KuzuQAChain.from_llm(graph=FakeGraphStore(), allow_dangerous_requests=True)
    assert "Either `llm` or `cypher_llm` parameters must be provided" == str(exc_info.value)
    # llm only
    KuzuQAChain.from_llm(llm=FakeLLM(), graph=FakeGraphStore(), allow_dangerous_requests=True)
    # qa_llm only
    with pytest.raises(ValueError) as exc_info:
        KuzuQAChain.from_llm(
            qa_llm=FakeLLM(), graph=FakeGraphStore(), allow_dangerous_requests=True
        )
    assert "Either `llm` or `cypher_llm` parameters must be provided" == str(exc_info.value)
    # cypher_llm only
    with pytest.raises(ValueError) as exc_info:
        KuzuQAChain.from_llm(
            cypher_llm=FakeLLM(), graph=FakeGraphStore(), allow_dangerous_requests=True
        )
    assert "Either `llm` or `qa_llm` parameters must be provided along with `cypher_llm`" == str(
        exc_info.value
    )
    # llm + qa_llm
    KuzuQAChain.from_llm(
        llm=FakeLLM(),
        qa_llm=FakeLLM(),
        graph=FakeGraphStore(),
        allow_dangerous_requests=True,
    )
    # llm + cypher_llm
    KuzuQAChain.from_llm(
        llm=FakeLLM(),
        cypher_llm=FakeLLM(),
        graph=FakeGraphStore(),
        allow_dangerous_requests=True,
    )
    # qa_llm + cypher_llm
    KuzuQAChain.from_llm(
        qa_llm=FakeLLM(),
        cypher_llm=FakeLLM(),
        graph=FakeGraphStore(),
        allow_dangerous_requests=True,
    )
    # llm + qa_llm + cypher_llm
    with pytest.raises(ValueError) as exc_info:
        KuzuQAChain.from_llm(
            llm=FakeLLM(),
            qa_llm=FakeLLM(),
            cypher_llm=FakeLLM(),
            graph=FakeGraphStore(),
            allow_dangerous_requests=True,
        )
    assert (
        "You can specify up to two of 'cypher_llm', 'qa_llm'"
        ", and 'llm', but not all three simultaneously."
    ) == str(exc_info.value)
