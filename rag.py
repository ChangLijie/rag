import os
from pathlib import Path

from haystack import Pipeline
from haystack.components.converters import PyPDFToDocument, TextFileToDocument
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.joiners import DocumentJoiner
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.routers import FileTypeRouter
from haystack.components.writers import DocumentWriter
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore

# tutorial_running(27)
os.environ["PG_CONN_STR"] = "postgresql://admin:admin@postgres:5432/postgres"
# ---------------------pre work----------------------------------
# Initializing the DocumentStore
document_store = PgvectorDocumentStore(
    embedding_dimension=384,
    vector_function="cosine_similarity",
    recreate_table=True,
    search_strategy="hnsw",
)

# Fetch the Data
doc_dir = "data/new_data/"
# pdf_handler(dir=doc_dir)
file_type_router = FileTypeRouter(mime_types=["text/plain", "application/pdf"])
text_file_converter = TextFileToDocument()
pdf_converter = PyPDFToDocument()
document_joiner = DocumentJoiner()

document_cleaner = DocumentCleaner()
document_splitter = DocumentSplitter(
    split_by="word", split_length=150, split_overlap=50
)

e_model_path = ""
# Initalize a Document Embedder
document_embedder = SentenceTransformersDocumentEmbedder(
    model="sentence-transformers/all-MiniLM-L6-v2"
)
document_embedder.warm_up()


# Write Documents to the DocumentStore
document_writer = DocumentWriter(document_store)


# create pipeline
preprocessing_pipeline = Pipeline()
preprocessing_pipeline.add_component(instance=file_type_router, name="file_type_router")
preprocessing_pipeline.add_component(
    instance=text_file_converter, name="text_file_converter"
)
preprocessing_pipeline.add_component(instance=pdf_converter, name="pypdf_converter")
preprocessing_pipeline.add_component(instance=document_joiner, name="document_joiner")
preprocessing_pipeline.add_component(instance=document_cleaner, name="document_cleaner")
preprocessing_pipeline.add_component(
    instance=document_splitter, name="document_splitter"
)
preprocessing_pipeline.add_component(
    instance=document_embedder, name="document_embedder"
)
preprocessing_pipeline.add_component(instance=document_writer, name="document_writer")

# connect pipeline
preprocessing_pipeline.connect(
    "file_type_router.text/plain", "text_file_converter.sources"
)
preprocessing_pipeline.connect(
    "file_type_router.application/pdf", "pypdf_converter.sources"
)
preprocessing_pipeline.connect("text_file_converter", "document_joiner")
preprocessing_pipeline.connect("pypdf_converter", "document_joiner")
preprocessing_pipeline.connect("document_joiner", "document_cleaner")
preprocessing_pipeline.connect("document_cleaner", "document_splitter")
preprocessing_pipeline.connect("document_splitter", "document_embedder")
preprocessing_pipeline.connect("document_embedder", "document_writer")
preprocessing_pipeline.run(
    {"file_type_router": {"sources": list(Path(doc_dir).glob("**/*"))}}
)
