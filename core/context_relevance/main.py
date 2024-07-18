from typing import Union

import numpy as np

from core.rag.main import Service as SentenceModel


class Service:
    def __init__(
        self,
        model: SentenceModel,
        threshold: int = 0.3,
    ) -> None:
        self.model = model
        self.threshold = threshold

    def _cosine_similarity(
        self, vec1: Union[list, np.ndarray], vec2: Union[list, np.ndarray]
    ) -> np.float64:
        """Whether two vector similar or not.

        Args:
            vec1 (Union[list, np.ndarray]): vector
            vec2 (Union[list, np.ndarray]): vector

        Returns:
            np.float64: The score of similarity.
        """
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        return dot_product / (norm_vec1 * norm_vec2)

    def compare(self, context1: str, context2: str) -> bool:
        """Compare the relative of two context.

        Args:
            context1 (str): context1.
            context2 (str): context2.

        Returns:
            bool: Is relevance or not .

        """
        vec_context1 = self.model.run(data=context1)
        vec_context2 = self.model.run(data=context2)
        return bool(
            self._cosine_similarity(vec1=vec_context1, vec2=vec_context2)
            > self.threshold
        )


if __name__ == "__main__":
    relecance_checher = Service()
    context1 = [0.1, 0.2]
    context2 = [0.3, 0.4]
    result = relecance_checher.compare(context1=context1, context2=context2)
    print(result)
