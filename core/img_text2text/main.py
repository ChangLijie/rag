from typing import Union

import numpy as np
import PIL
import torch
from datasets import Dataset
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer


class Service:
    """
    Image or Text embedding.
    Use HuggingFace

    """

    def __init__(
        self,
        model_path: str,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path

        self.model, self.processor, self.tokenizer = self._load_model(
            model_path=self.model_path
        )

    def _load_model(
        self, model_path: str
    ) -> Union[AutoModel, AutoImageProcessor, AutoTokenizer]:
        """
        Load model from HuggingFace.

        Args:
            model_path (str): The model path.

        Returns:
            Union[AutoModel, AutoImageProcessor, AutoTokenizer]:
                More information :https://huggingface.co/docs/transformers/v4.42.0/en/autoclass_tutorial#autoimageprocessor
        """
        model = AutoModel.from_pretrained(model_path).to(self.device)
        processor = AutoImageProcessor.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return model, processor, tokenizer

    def text_embedding(self, dataset: Dataset, column: str = "embeddings"):
        """
        Do text embedding

        Args:
            dataset (Dataset): Dataset . Type must is Dataset.
            column (str, optional): the key name for embedding vector. Defaults to embeddings.

        Returns:
            Dataset:Dataset after do embedding.
        """

        def text_process_function(example) -> Dataset:
            inputs = self.tokenizer(
                example["describe"], truncation=True, return_tensors="pt"
            ).to("cuda")
            embeddings = (
                self.model.get_text_features(**inputs)[0].detach().cpu().numpy()
            )

            return {column: embeddings}

        em_dataset = dataset.map(text_process_function)
        return em_dataset

    def img_embedding(self, dataset: Dataset, column: str = "embeddings") -> Dataset:
        """
        Do text embedding

        Args:
            dataset (Dataset): Dataset . Type must is Dataset.
            column (str, optional): the key name for embedding vector. Defaults to embeddings.

        Return:
            Dataset:Dataset after do embedding.
        """

        def img_process_function(example):
            image = Image.open(example["images"])
            inputs = self.processor(image, return_tensors="pt").to("cuda")
            embeddings = (
                self.model.get_image_features(**inputs)[0].detach().cpu().numpy()
            )

            return {"embeddings": embeddings}

        em_dataset = dataset.map(img_process_function)
        return em_dataset

    def get_image_vector(self, image: PIL) -> np.ndarray:
        """
        Get image vector.

        Args:
            image (PIL): image data.

        Returns:
            np.ndarray: image vector.
        """
        # image = Image.open(image_path)
        img_embedding = (
            self.model.get_image_features(
                **self.processor([image], return_tensors="pt", truncation=True).to(
                    "cuda"
                )
            )[0]
            .detach()
            .cpu()
            .numpy()
        )
        return img_embedding

    def get_text_vector(self, text: str) -> np.ndarray:
        """
        Get text vector.

        Args:
            image_path (str): text path

        Returns:
            np.ndarray: text vector.
        """

        text_embedding = (
            self.model.get_text_features(
                **self.tokenizer(text, truncation=True, return_tensors="pt").to("cuda")
            )[0]
            .detach()
            .cpu()
            .numpy()
        )
        return text_embedding


if __name__ == "__main__":
    from datasets import Dataset, load_from_disk

    model_dimension = 512
    clip = Service(
        model_path="./model/models--openai--clip-vit-base-patch16/snapshots/57c216476eefef5ab752ec549e440a49ae4ae5f3/"
    )
    # data embedding
    # dataset = load_from_disk("./embeddings/data")
    # em_text_dataset = clip.text_embedding(dataset=dataset)
    # em_img_dataset = clip.img_embedding(dataset=dataset)
    # if isinstance(em_text_dataset, Dataset):
    #     print("text_embedding is ok")
    # if isinstance(em_img_dataset, Dataset):
    #     print("em_img_dataset is ok")
    import numpy as np

    data_releative = [
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
            "bot": "  Recent advancements include improvements in natural language processing, such as the development of transformer models like GPT-3.",
        },
        {
            "user": "How does the immune system protect the body from diseases?",
            "bot": " The immune system protects the body by recognizing and attacking pathogens like bacteria and viruses.",
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

    def cosine_similarity(vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        return dot_product / (norm_vec1 * norm_vec2)

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
    print(cosine_similarity(np.array([0.14, 0.5, 0.6]), np.array([0.84, 0.1, 0.2])))
    # test get vector
    # img_vec = clip.get_image_vector(image_path="./data/wdssd.png")
    # for conversation in data_no_relative2:
    #     text_vec1 = clip.get_text_vector(conversation["user"])
    #     text_vec2 = clip.get_text_vector(conversation["bot"])
    #     print(cosine_similarity(text_vec1, text_vec2))
    # if i  sinstance(img_vec, np.ndarray) and (img_vec.shape[0] == 512):
    #     print("img_vec is ok!")
    # if isinstance(text_vec, np.ndarray) and (text_vec.shape[0] == 512):
    #     print("text_vec is ok!")
