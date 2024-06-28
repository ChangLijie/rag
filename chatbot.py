import os

from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.generators import HuggingFaceLocalGenerator
from haystack_integrations.components.retrievers.pgvector import (
    PgvectorEmbeddingRetriever,
)
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore

from tools.setting import HUGGING_FACE_TOKEN

# tutorial_running(27)
os.environ["PG_CONN_STR"] = "postgresql://admin:admin@postgres:5432/postgres"
# ---------------------pre work----------------------------------
# Initializing the DocumentStore
document_store = PgvectorDocumentStore(
    embedding_dimension=384,
    vector_function="cosine_similarity",
    recreate_table=False,
    search_strategy="hnsw",
)

# Define a Template Prompt
template = """
Answer the questions based on the given context.

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{ question }}
Answer:
"""

# Initialize a Generator

if "HF_API_TOKEN" not in os.environ:
    os.environ["HF_API_TOKEN"] = HUGGING_FACE_TOKEN

pipe = Pipeline()
embedding_model_path = "sentence-transformers/all-MiniLM-L6-v2"
llm_model_path = "HuggingFaceH4/zephyr-7b-beta"
# embedding_model_path = "~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/8b3219a92973c328a8e22fadcfa821b5dc75636a/"
# llm_model_path = "~/.cache/huggingface/hub/models--HuggingFaceH4--zephyr-7b-beta/snapshots/b70e0c9a2d9e14bd1e812d3c398e5f313e93b473/"
# pipe.from_template(PredefinedPipeline.INDEXING)
pipe.add_component(
    "embedder",
    SentenceTransformersTextEmbedder(
        model="./model/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/8b3219a92973c328a8e22fadcfa821b5dc75636a"
    ),
)
pipe.add_component(
    "retriever", PgvectorEmbeddingRetriever(document_store=document_store)
)
pipe.add_component("prompt_builder", PromptBuilder(template=template))
pipe.add_component(
    "llm",
    HuggingFaceLocalGenerator(
        model="./model/models--HuggingFaceH4--zephyr-7b-beta/snapshots/b70e0c9a2d9e14bd1e812d3c398e5f313e93b473",
        task="text2text-generation",
        generation_kwargs={"max_new_tokens": 100, "temperature": 0.9},
    ),
    # HuggingFaceAPIGenerator(api_type="serverless_inference_api", api_params={"model": "HuggingFaceH4/zephyr-7b-beta"}),
)
pipe.connect("embedder.embedding", "retriever.query_embedding")
pipe.connect("retriever", "prompt_builder.documents")
pipe.connect("prompt_builder", "llm")

while True:
    question = input("please enter your question : ")
    # question = (
    #     "What is EV2U-RMR2?"
    # )
    if question.lower() == "exit":
        break
    anwser = pipe.run(
        {
            "embedder": {"text": question},
            "prompt_builder": {"question": question},
            "llm": {"generation_kwargs": {"max_new_tokens": 350}},
        }
    )
    print(anwser["llm"]["replies"])
