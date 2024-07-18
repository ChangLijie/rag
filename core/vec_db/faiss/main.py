from typing import Union

import numpy as np
from datasets import Dataset, load_dataset, load_from_disk

from core.img_text2text.main import Service


class Operator:
    """Faiss Operator"""

    def __init__(self, dataset_path: str, column: str = "embeddings") -> None:
        self.dataset = load_from_disk(dataset_path)
        self.column = column

    def save(self, save_path: str, em_type: str = "img") -> None:
        """
        Faiss store.

        Args:
            save_path (str): Path to save Faiss file.
        """
        model = Service()
        if em_type == "img":
            self.dataset = model.img_embedding(self.dataset)
        else:
            self.dataset = model.text_embedding(self.dataset)
        self.dataset.add_faiss_index(column=self.column)
        self.dataset.save_faiss_index(self.column, save_path)

    def load(self, faiss_path: str) -> None:
        """
        Load Faiss.

        Args:
            faiss_path (str): Path to Faiss file.
        """
        self.dataset.load_faiss_index(index_name=self.column, file=faiss_path)

    def search(self, q_vector: np.ndarray, top_k: int = 1) -> Union[int, str]:
        """
        Search relative result from Faiss.

        Args:
            q_vector (np.ndarray): Vector.
            top_k (int, optional): How many result to search. Defaults to 1.

        Returns:
            Union[int, str]: return Score and answer
        """
        scores, answer = self.dataset.get_nearest_examples(
            self.column, q_vector, k=top_k
        )
        return (scores, answer)


if __name__ == "__main__":
    from datasets import Dataset, load_from_disk

    from core.img_text2text.main import Service

    image_path = "./data/3ME3.png"
    clip = Service()
    img_vec = clip.get_image_vector(image_path=image_path)

    db = Operator(dataset="./embeddings/data")
    db.load(faiss_path="./embeddings/img_embedding.faiss")

    score, answer = db.search(img_vec)
    print(score, answer)
