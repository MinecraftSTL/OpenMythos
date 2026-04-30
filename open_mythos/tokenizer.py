from transformers import AutoTokenizer

DEFAULT_MODEL_ID = "openai/gpt-oss-20b"


class MythosTokenizer:
    """
    OpenMythos 的 HuggingFace 分词器封装。

    参数:
        model_id (str): 用于 AutoTokenizer 的 HuggingFace 模型 ID 或路径。
            默认为 "openai/gpt-oss-20b"。

    属性:
        tokenizer: HuggingFace AutoTokenizer 的实例。

    示例:
        >>> tok = MythosTokenizer()
        >>> ids = tok.encode("Hello world")
        >>> s = tok.decode(ids)
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID):
        """
        初始化 MythosTokenizer。

        参数:
            model_id (str): HuggingFace 模型标识符或分词器文件路径。
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    @property
    def vocab_size(self) -> int:
        """
        返回分词器词汇表的大小。

        返回:
            int: 分词器词汇表中唯一 token 的数量。
        """
        return self.tokenizer.vocab_size

    def encode(self, text: str) -> list[int]:
        """
        将输入文本编码为 token ID 列表。

        参数:
            text (str): 要分词的输入文本字符串。

        返回:
            list[int]: 表示输入文本的整数 token ID 列表。
        """
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_ids: list[int]) -> str:
        """
        将 token ID 列表解码回文本字符串。

        参数:
            token_ids (list[int]): 要解码的整数 token ID 列表。

        返回:
            str: token ID 的解码字符串表示。
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
