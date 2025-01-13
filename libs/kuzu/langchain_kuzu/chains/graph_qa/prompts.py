# flake8: noqa
from langchain_core.prompts.prompt import PromptTemplate

CYPHER_GENERATION_TEMPLATE = """Task:Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Schema:
{schema}
Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.

The question is:
{question}"""

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
)

CYPHER_QA_TEMPLATE = """You are an assistant that helps to form nice and human understandable answers.
The information part contains the provided information that you must use to construct an answer.
The provided information is authoritative, you must never doubt it or try to use your internal knowledge to correct it.
Make the answer sound as a response to the question. Do not mention that you based the result on the given information.
Here is an example:

Question: Which managers own Neo4j stocks?
Context:[manager:CTL LLC, manager:JANE STREET GROUP LLC]
Helpful Answer: CTL LLC, JANE STREET GROUP LLC owns Neo4j stocks.

Follow this example when generating answers.
If the provided information is empty, say that you don't know the answer.
Information:
{context}

Question: {question}
Helpful Answer:"""

CYPHER_QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"], template=CYPHER_QA_TEMPLATE
)

KUZU_EXTRA_INSTRUCTIONS = """
Instructions:
Generate the Kùzu dialect of Cypher with the following rules in mind:
1. When matching on a property, use the `LOWER()` function to match the property value.
2. Do not include triple backticks ``` in your response. Return only Cypher.
3. Do not return any notes or comments in your response.
\n"""

KUZU_GENERATION_TEMPLATE = CYPHER_GENERATION_TEMPLATE.replace(
    "Generate Cypher", "Generate Kùzu Cypher"
).replace("Instructions:", KUZU_EXTRA_INSTRUCTIONS)

KUZU_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template=KUZU_GENERATION_TEMPLATE
)
