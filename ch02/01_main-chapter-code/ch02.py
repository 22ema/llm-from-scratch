import re
from importlib.metadata import version
import tiktoken

from tokenizer import SimpleTokenizerV1

def main_1(raw_text):
    preprocessed = re.split(r'([,.?!_"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]

    all_token = sorted(list(set(preprocessed)))
    all_token.extend(["<|endoftext|>", "<|unk|>"])
    vocab_size = len(all_token)

    vocab = {token: integer for integer, token in enumerate(all_token)}
    for i, x in vocab.items():
        print(i, x)

    tokenizer = SimpleTokenizerV1(vocab)
    text_1 = "Hello, do you like tea?"
    text_2 = "In the sunlit terraces of the palace."
    text = "<|endoftext|> ".join((text_1, text_2))
    ids = tokenizer.encode(text)
    text = tokenizer.decode(ids)
    print(ids)
    print(text)

if __name__ == "__main__":
    with open("./the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    print(f"전체 문자의 개수: {len(raw_text)}")
    main_1(raw_text)

