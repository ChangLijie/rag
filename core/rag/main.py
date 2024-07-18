from typing import Literal, Union

from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)


class Service:
    """
    Do RAG.
    """

    def __init__(
        self,
        model_path: str,
        type: str = Literal["text", "document"],
    ) -> None:
        self.model_path = model_path
        self.type = type
        self.model = self._load_model(model_path=self.model_path, type=self.type)

    def _load_model(self, model_path: str, type: str):
        """
        Load model from hugging face.

        Args:
            model_path (str): The model path.
                More information :https://docs.haystack.deepset.ai/reference/embedders-api#sentencetransformersdocumentembedder
            type (str) : To chose load model to do text embedding or document embedding.
        Returns:
            SentenceTransformersDocumentEmbedder: embedding model
        """
        if type == "text":
            embedder = SentenceTransformersTextEmbedder(model=model_path)
            embedder.warm_up()
        elif type == "document":
            embedder = SentenceTransformersDocumentEmbedder(model=model_path)

            embedder.warm_up()
        else:
            raise ModuleNotFoundError(
                f"Doesn't support type : '{type}',Plz choose 'text' or 'document'!"
            )

        return embedder

    def run(self, data: Union[list, str]) -> list:
        """Embedding.

        Args:
            data (Union[list, str]): Document or Prompt.

        Returns:
            list: List with serialized data.
        """
        if self.type == "document":
            emb_docs = self.model.run(documents=data)
            return emb_docs["documents"]
        elif self.type == "text":
            emb_prompt = self.model.run(text=data)
            return emb_prompt["embedding"]
        else:
            raise ModuleNotFoundError(
                f"Doesn't support type : '{type}',Plz choose 'text' or 'document'!"
            )


if __name__ == "__main__":
    import numpy as np

    def cosine_similarity(vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        print(type(vec1[0]))
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        return dot_product / (norm_vec1 * norm_vec2)

    data = [
        {
            "user": "What are the latest advancements in artificial intelligence?",
            "bot": " Recent advancements in artificial intelligence include improvements in natural language processing, such as the development of transformer models like GPT-3. These models have significantly enhanced the ability of machines to understand and generate human-like text.",
        },
        {
            "user": "Can you explain what transformer models are and how they work?",
            "bot": " Transformer models are a type of deep learning model introduced in the paper 'Attention is All You Need' by Vaswani et al. They use self-attention mechanisms to process input data in parallel, rather than sequentially, allowing for better handling of long-range dependencies and faster training times compared to traditional recurrent neural networks.",
        },
        {
            "user": "How have transformer models improved natural language processing?",
            "bot": "Transformer models have greatly improved natural language processing by enabling more accurate language understanding and generation. They have been used to create state-of-the-art language models like BERT and GPT-3, which excel at tasks such as text completion, translation, summarization, and question-answering.",
        },
        {
            "user": "What are some practical applications of these advancements?",
            "bot": "Practical applications of these advancements include chatbots and virtual assistants that can understand and respond to user queries more effectively, automated content generation, machine translation, sentiment analysis, and improved search engines that can understand the context and intent behind user queries.",
        },
    ]
    data_releative = [
        {
            "user": "What are the latest advancements in artificial intelligence?",
            "bot": "Recent advancements include improvements in natural language processing, such as the development of transformer models like GPT-3.",
        },
        {
            "user": "How does the immune system protect the body from diseases?",
            "bot": "The immune system protects the body by recognizing and attacking pathogens like bacteria and viruses.",
        },
        {
            "user": "Can you explain the process of photosynthesis in plants?",
            "bot": "Photosynthesis in plants involves converting light energy into chemical energy, using chlorophyll to produce glucose and oxygen from carbon dioxide and water.",
        },
        {
            "user": "What are the benefits of regular physical exercise?",
            "bot": "Regular physical exercise improves cardiovascular health, strengthens muscles, enhances flexibility, and boosts mental well-being.",
        },
    ]

    data_no_relative = [
        {
            "user": "What are the latest advancements in artificial intelligence?",
            "bot": "I recently watched a documentary about marine life.",
        },
        {
            "user": "Can you explain what transformer models are and how they work?",
            "bot": "My favorite hobby is painting landscapes on the weekends.",
        },
        {
            "user": "How have transformer models improved natural language processing?",
            "bot": "Did you know that Mount Everest is the highest mountain in the world?",
        },
        {
            "user": "What are some practical applications of these advancements?",
            "bot": "The best pizza I've ever had was in Italy.",
        },
    ]
    data_no_relative2 = [
        {
            "user": "What are the latest advancements in artificial intelligence?",
            "bot": "The capital of France is Paris, known for its historic landmarks and cultural heritage.",
        },
        {
            "user": "How does the immune system protect the body from diseases?",
            "bot": " The Grand Canyon is a significant natural landmark located in the state of Arizona in the United States.",
        },
        {
            "user": "Can you explain the process of photosynthesis in plants?",
            "bot": "Beethoven was a famous composer known for his symphonies and other classical music compositions.",
        },
        {
            "user": "What are the benefits of regular physical exercise?",
            "bot": "Quantum computing is an area of research focused on developing computers that use quantum bits instead of traditional binary bits.",
        },
    ]
    text_embedder = Service(
        model_path="./model/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/8b3219a92973c328a8e22fadcfa821b5dc75636a/",
        type="text",
    )
    for conversation in data_no_relative2:
        c1 = text_embedder.run(conversation["user"])
        c2 = text_embedder.run(conversation["bot"])

        print(type(cosine_similarity(c1, c2)))
