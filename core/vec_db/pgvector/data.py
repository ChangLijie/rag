import os
from os import walk
from pathlib import Path

from haystack import Pipeline
from haystack.components.converters import PyPDFToDocument, TextFileToDocument
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.joiners import DocumentJoiner
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.routers import FileTypeRouter


class Process:
    """
    Do data preprocess for RAG.
    """

    def __init__(self) -> None:
        self._init_tools()

    def _init_tools(self):
        """
        Init all tool for data process.
        """
        # TODO: More flexible
        self.pdf_converter = PyPDFToDocument()
        self.document_cleaner = DocumentCleaner(
            remove_empty_lines=True,
            remove_extra_whitespaces=True,
            remove_repeated_substrings=False,
        )
        self.document_splitter = DocumentSplitter(
            split_by="word", split_length=150, split_overlap=50
        )

    def run(self, data_folder: str) -> list:
        """
        Preprocess data for RAG.

        Args:
            data_folder (str): The path to data (Now only support PDF.)

        Returns:
            list: All document after clean and split.
        """
        docs = self.pdf_converter.run(
            sources=list(Path(data_folder).glob("**/*")),
            meta={"privacy": 0},
        )
        clear_docs = self.document_cleaner.run(documents=docs["documents"])
        split_docs = self.document_splitter.run(documents=clear_docs["documents"])
        return split_docs["documents"]
