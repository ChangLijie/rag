from typing import List, Union

from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage


class Service:
    """Prompt engineering"""

    def __init__(self) -> None:
        self.builder = ChatPromptBuilder()

        self.template = {
            "summary": """
                        Please provide a concise summary of the following conversation history. Focus on the key points and important details mentioned.

                        Conversation History:
                        {% for conversation in history %}
                            User: {{ conversation.user }}
                            Bot: {{ conversation.bot }}
                        {% endfor %}

                        Summary:
                        """,
        }

    def _summary_history(self, chat_history: list) -> list:
        """The template of prompt for summary conversation history.

        Returns:
            list:  The prompt. ex:[ChatMessage]
        """
        template = [ChatMessage.from_user(self.template["summary"])]

        prompt = self.builder.run(history=chat_history, template=template)

        return prompt["prompt"]

    def _build_instruction(self, instruction: list) -> list:
        """Init instruction.

        Returns:
            list:  The prompt. ex:[ChatMessage]
        """
        prompt = []
        if instruction is not None:
            for command in instruction:
                prompt.append(ChatMessage.from_system(command))
        return prompt

    def generate(
        self,
        history: Union[str, bool],
        retrieval: Union[str, bool],
        prompt: str,
        instruction: Union[list, None] = None,
    ) -> list:
        """Gen prompt.

        Args:
            instruction (Union[list, None], optional): System instruction. Defaults to None.
            history (str): Conversation history.
            retrieval (str): More information search from database.
            prompt (str): User question.

        Returns:
            list: The prompt. ex:[ChatMessage]
        """
        instruction_prompt = self._build_instruction(instruction=instruction)
        history_prompt = []
        retrieval_prompt = []
        user_prompt = [ChatMessage.from_user(content=prompt)]
        if isinstance(history, str):
            history_prompt.append(ChatMessage.from_assistant(content=history))
        if isinstance(retrieval, str):
            retrieval_prompt.append(ChatMessage.from_assistant(content=retrieval))

        print(instruction_prompt + history_prompt + retrieval_prompt + user_prompt)
        return instruction_prompt + history_prompt + retrieval_prompt + user_prompt


class GPT_Service(Service):
    pass


if __name__ == "__main__":
    prompt_engineering = Service()
    # summary_prompt = prompt_engineering._summary_history()
    # print(summary_prompt)
    prompt = prompt_engineering.generate(
        history="user introduces himself as jay and asks if the assistant can help him. The assistant greets jay and asks how it can assist him. Jay reveals that his job is to sell 3TE7, and the assistant supports this decision. Jay then inquires about who he can contact to purchase 3TE7, and the assistant suggests reaching out to jay.",
        retrieval="yes 123",
        prompt="ggg",
    )

    print(prompt)
