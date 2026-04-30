import pytest
from open_mythos.tokenizer import MythosTokenizer


@pytest.fixture(scope="module")
def tokenizer():
    tok = MythosTokenizer()
    print(f"\n已加载分词器: {tok.tokenizer.name_or_path}")
    return tok


def test_loads(tokenizer):
    assert tokenizer is not None
    print(f"分词器: {tokenizer}")


def test_vocab_size(tokenizer):
    size = tokenizer.vocab_size
    print(f"词汇表大小: {size:,}")
    assert size > 0


def test_encode_returns_list_of_ints(tokenizer):
    ids = tokenizer.encode("Hello, world!")
    print(f"encode('Hello, world!') → {ids}")
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    assert len(ids) > 0


def test_encode_empty_string(tokenizer):
    ids = tokenizer.encode("")
    print(f"encode('') → {ids}")
    assert isinstance(ids, list)


def test_decode_returns_string(tokenizer):
    ids = tokenizer.encode("Hello, world!")
    text = tokenizer.decode(ids)
    print(f"decode({ids}) → '{text}'")
    assert isinstance(text, str)


def test_roundtrip(tokenizer):
    original = "The quick brown fox jumps over the lazy dog."
    ids = tokenizer.encode(original)
    recovered = tokenizer.decode(ids)
    print(f"原始文本:  '{original}'")
    print(f"token ID: {ids}")
    print(f"恢复文本: '{recovered}'")
    assert original in recovered or recovered in original


def test_encode_long_text(tokenizer):
    text = "OpenMythos is a recurrent depth transformer. " * 100
    ids = tokenizer.encode(text)
    print(f"长文本 ({len(text)} 字符) → {len(ids)} 个 token")
    assert len(ids) > 100


def test_custom_model_id():
    tok = MythosTokenizer(model_id="openai/gpt-oss-20b")
    print(f"自定义 model_id 词汇表大小: {tok.vocab_size:,}")
    assert tok.vocab_size > 0


def test_vocab_size_consistent(tokenizer):
    outer = tokenizer.vocab_size
    inner = tokenizer.tokenizer.vocab_size
    print(f"vocab_size 属性: {outer:,}  |  内部 tokenizer.vocab_size: {inner:,}")
    assert outer == inner


if __name__ == "__main__":
    pytest.main([__file__, "--verbose", "-s"])
