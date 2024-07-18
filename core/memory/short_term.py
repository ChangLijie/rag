class ChatHistory:
    """Short term memory"""

    def __init__(self, limit_len: int = 20) -> None:
        self.limit_len = limit_len
        self.chat_history = []

    def forget(self, idx: int) -> None:
        """Delete the idx memory.

        Args:
            idx (int): the number of chat_history.
        """
        self.chat_history.pop(idx)

    def _check(self) -> None:
        """Make sure the len of chat_history don't over the limit_len"""
        if len(self.chat_history) > self.limit_len:
            self.forget(idx=0)

    def remember(self, user_prompt: str, bot_answer: str) -> None:
        """Add new conversation.

        Args:
            user_prompt (str): question.
            bot_answer (str): llm generate.
        """
        self.chat_history.append({"user": user_prompt, "bot": bot_answer})
        self._check()

    def get(self) -> list:
        """Return memory.

        Returns:
            list: memory.
        """
        return self.chat_history


if __name__ == "__main__":
    s_memory = ChatHistory(limit_len=20)
    s_memory.remember(user_prompt="how r u ?", bot_answer="I'm fine!")

    print(s_memory.get())
