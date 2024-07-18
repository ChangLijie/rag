import os

from haystack.components.generators.chat import HuggingFaceLocalChatGenerator
from haystack.dataclasses import ChatMessage

# from tools.setting import HUGGING_FACE_TOKEN

if "HF_API_TOKEN" not in os.environ:
    # os.environ["HF_API_TOKEN"] = HUGGING_FACE_TOKEN
    os.environ["HF_API_TOKEN"] = "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"


class Service:
    """
    Gen text.
    Use Haystack v2.2

    """

    def __init__(self, model_path) -> None:
        self.model_path = model_path

        self.model = self._load_model(model_path=self.model_path)
        self.model.warm_up()

    def _load_model(
        self,
        model_path: str,
        task: str = "text2text-generation",
        generation_kwargs: dict = {"max_new_tokens": 350, "temperature": 0.9},
    ) -> HuggingFaceLocalChatGenerator:
        """
        Load model from HuggingFace.

        Args:
            model_path (str): The repository of model on Huggingface.
            api_type (str): Using the HuggingFace online service.
                More information : https://huggingface.co/docs/api-inference/index

        Returns:
            HuggingFaceAPIChatGenerator: Haystack v2.2.
                More information : https://docs.haystack.deepset.ai/docs/huggingfaceapichatgenerator#in-a-pipeline
        """
        """Load model from HuggingFace.

        Args:
            model_path (str): The model path.
            task (str, optional):  Defaults to "text2text-generation".
                    The task for the Hugging Face pipeline.
                    Possible values are "text-generation" and "text2text-generation".
                    Generally, decoder-only models like GPT support "text-generation",
                    while encoder-decoder models like T5 support "text2text-generation".
                    If the task is also specified in the `huggingface_pipeline_kwargs`, this parameter will be ignored.
                    If not specified, the component will attempt to infer the task from the model name,
                    calling the Hugging Face Hub API.
            generation_kwargs (dict, optional): Defaults to {"max_new_tokens": 350, "temperature": 0.9}.
                    A dictionary containing keyword arguments to customize text generation.
                    Some examples: `max_length`, `max_new_tokens`, `temperature`, `top_k`, `top_p`, etc.
                    See Hugging Face's documentation for more information:
                    - - [customize-text-generation](https://huggingface.co/docs/transformers/main/en/generation_strategies#customize-text-generation)
                    - - [GenerationConfig](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationConfig)
                    - The only generation_kwargs we set by default is max_new_tokens, which is set to 512 tokens.

        Returns:
            HuggingFaceLocalChatGenerator: Haystack v2.2.+
                More information : https://docs.haystack.deepset.ai/reference/generators-api#huggingfacelocalchatgenerator
        """
        model = HuggingFaceLocalChatGenerator(
            model=model_path,
            task=task,
            generation_kwargs=generation_kwargs,
        )

        return model

    def run(self, prompt: list, max_tokens: int = 350) -> str:
        """
        Generate text.

        Args:
            prompt (list): List contain ChatMessage.
                More information about ChatMessage:https://docs.haystack.deepset.ai/docs/data-classes#chatmessage
            max_tokens (int): Maximum number of tokens that the model's output.

        Returns:
            str: Generate text.
        """
        text = self.model.run(
            messages=prompt,
            generation_kwargs={"max_tokens": max_tokens},
        )
        return text["replies"][0].content


if __name__ == "__main__":
    from haystack.dataclasses import ChatMessage

    prompt = "How r u ?"
    quesstion = [ChatMessage.from_user(prompt)]
    llm = Service(
        model_path="./model/models--HuggingFaceH4--zephyr-7b-beta/snapshots/b70e0c9a2d9e14bd1e812d3c398e5f313e93b473/",
    )
    text = llm.run(prompt=quesstion)

    print(text)
