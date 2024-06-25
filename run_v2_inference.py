from haystack.telemetry import tutorial_running
from haystack.document_stores.in_memory import InMemoryDocumentStore
from datasets import load_dataset
from haystack import Document
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
import pickle 
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders import PromptBuilder
import os
from getpass import getpass
from haystack.components.generators import OpenAIGenerator
from haystack import Pipeline, PredefinedPipeline


from haystack.components.writers import DocumentWriter
from haystack.components.converters import MarkdownToDocument, PyPDFToDocument, TextFileToDocument
from haystack.components.preprocessors import DocumentSplitter, DocumentCleaner
from haystack.components.routers import FileTypeRouter
from haystack.components.joiners import DocumentJoiner
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.generators import HuggingFaceAPIGenerator

from pathlib import Path
import os
from getpass import getpass
from config import HUGGING_FACE_TOKEN
# from tools.pdf_checker import pdf_handler
from tools.save_document import Database

from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
from haystack_integrations.components.retrievers.pgvector import PgvectorEmbeddingRetriever
from haystack import Document
# tutorial_running(27)
os.environ["PG_CONN_STR"] = "postgresql://admin:admin@postgres:5432/postgres"
#---------------------pre work----------------------------------
#Initializing the DocumentStore
document_store = PgvectorDocumentStore(
    embedding_dimension=384,
    vector_function="cosine_similarity",
    recreate_table=True,
    search_strategy="hnsw",
)

# #Fetch the Data
# doc_dir = "data/innodisk"
# # pdf_handler(dir=doc_dir)
# file_type_router = FileTypeRouter(mime_types=["text/plain", "application/pdf"])
# text_file_converter = TextFileToDocument()
# pdf_converter = PyPDFToDocument()
# document_joiner = DocumentJoiner()

# document_cleaner = DocumentCleaner()
# document_splitter = DocumentSplitter(split_by="word", split_length=150, split_overlap=50)

# #Initalize a Document Embedder
# document_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
# document_embedder.warm_up()


# #Write Documents to the DocumentStore
# document_writer = DocumentWriter(document_store)


# #create pipeline
# preprocessing_pipeline = Pipeline()
# preprocessing_pipeline.add_component(instance=file_type_router, name="file_type_router")
# preprocessing_pipeline.add_component(instance=text_file_converter, name="text_file_converter")
# preprocessing_pipeline.add_component(instance=pdf_converter, name="pypdf_converter")
# preprocessing_pipeline.add_component(instance=document_joiner, name="document_joiner")
# preprocessing_pipeline.add_component(instance=document_cleaner, name="document_cleaner")
# preprocessing_pipeline.add_component(instance=document_splitter, name="document_splitter")
# preprocessing_pipeline.add_component(instance=document_embedder, name="document_embedder")
# preprocessing_pipeline.add_component(instance=document_writer, name="document_writer")

# #connect pipeline
# preprocessing_pipeline.connect("file_type_router.text/plain", "text_file_converter.sources")
# preprocessing_pipeline.connect("file_type_router.application/pdf", "pypdf_converter.sources")
# preprocessing_pipeline.connect("text_file_converter", "document_joiner")
# preprocessing_pipeline.connect("pypdf_converter", "document_joiner")
# preprocessing_pipeline.connect("document_joiner", "document_cleaner")
# preprocessing_pipeline.connect("document_cleaner", "document_splitter")
# preprocessing_pipeline.connect("document_splitter", "document_embedder")
# preprocessing_pipeline.connect("document_embedder", "document_writer")


# preprocessing_pipeline.run({"file_type_router": {"sources": list(Path(doc_dir).glob("**/*"))}})
# Database(document_store).to_dict(path="./ivit.json") 

# with open('saved_dictionary.pkl', 'wb') as f:
#     pickle.dump(dictionary, f)

#---------------------RAG---------------------------------

#Define a Template Prompt
template = """
Answer the questions based on the given context.

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{ question }}
Answer:
"""

#Initialize a Generator

if "HF_API_TOKEN" not in os.environ:
    os.environ["HF_API_TOKEN"] = HUGGING_FACE_TOKEN

pipe = Pipeline()
# pipe.from_template(PredefinedPipeline.INDEXING)
pipe.add_component("embedder", SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"))
pipe.add_component("retriever", PgvectorEmbeddingRetriever(document_store=document_store))
pipe.add_component("prompt_builder", PromptBuilder(template=template))
pipe.add_component(
    "llm",
    HuggingFaceAPIGenerator(api_type="serverless_inference_api", api_params={"model": "HuggingFaceH4/zephyr-7b-beta"}),
)
pipe.connect("embedder.embedding", "retriever.query_embedding")
pipe.connect("retriever", "prompt_builder.documents")
pipe.connect("prompt_builder", "llm")

while(True):
    question = input("please enter your question : ")
    # question = (
    #     "What is EV2U-RMR2?"
    # )
    if question.lower()=="exit":
        break
    anwser = pipe.run(
        {
            "embedder": {"text": question},
            "prompt_builder": {"question": question},
            "llm": {"generation_kwargs": {"max_new_tokens": 350}},
        }
    )
    print(anwser['llm']['replies'])

    