# flake8: noqa
from langchain_core.prompts.prompt import PromptTemplate

CYPHER_GENERATION_TEMPLATE = """You are an expert in translating natural language questions into Cypher statements.
You will be provided with a question and a graph schema.
Use only the provided relationship types and properties in the schema to generate a Cypher
statement.
The Cypher statement could retrieve nodes, relationships, or both.
Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a
Cypher statement.

Task: Generate a Cypher statement to query a graph database.

Schema:
{schema}

The question is:
{question}"""

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
)

CYPHER_QA_TEMPLATE = """You are an AI assistant using Retrieval-Augmented Generation (RAG).
RAG enhances your responses by retrieving relevant information from a knowledge base.
You will be provided with a question and relevant context. Use only this context to answer
the question IN FULL SENTENCES.
Do not make up an answer. If you don't know the answer, say so clearly.
Always strive to provide concise, helpful, and context-aware answers.
Relevant context:
{context}

Question: {question}
Helpful Answer:"""

CYPHER_QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"], template=CYPHER_QA_TEMPLATE
)

KUZU_EXTRA_INSTRUCTIONS = """
Instructions:
1. Use only the provided node and relationship types and properties in the schema.
2. When returning results, return property values rather than the entire node or relationship.
3. When matching on a property, use the `LOWER()` function to match the property value.
4. Do not include triple backticks ``` in your response. Return only Cypher.
\n"""

KUZU_GENERATION_TEMPLATE = CYPHER_GENERATION_TEMPLATE.replace(
    "Generate Cypher", "Generate Kuzu Cypher"
).replace("Instructions:", KUZU_EXTRA_INSTRUCTIONS)

KUZU_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template=KUZU_GENERATION_TEMPLATE
)
