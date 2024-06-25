import json
from pathlib import Path
from typing import Any, Dict

from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore

class Database(InMemoryDocumentStore):
    def to_disk(self, path: str):
        """Write the database and its' data to disk as a JSON file."""
        data: Dict[str, Any] = self.to_dict()
        data["documents"] = [doc.to_dict(flatten=False) for doc in self.storage.values()]
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def from_disk(cls, path: str) -> "Database":
        """Load the database and its' data from disk as a JSON file."""
        if Path(path).exists():
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                cls_object = cls.from_dict(data)
                cls_object.write_documents([Document(**doc) for doc in data["documents"]])
                return cls_object
            except Exception as e:
                return cls()
        else:
            return cls()