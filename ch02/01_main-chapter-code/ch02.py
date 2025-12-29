import re
import torch
import numpy as np
from importlib.metadata import version
from torch.utils.data import DataLoader
import tiktoken

from tokenizer import SimpleTokenizerV1, GPTDatasetV1

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


def main_2(text):
    tokenizer = tiktoken.get_encoding("gpt2")
    integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    print(integers)
    strings = tokenizer.decode(integers)
    print(strings)


def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader


if __name__ == "__main__":
    with open("./the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    print(f"전체 문자의 개수: {len(raw_text)}")
    dataloader = create_dataloader_v1(raw_text, 1, 4, 1, num_workers=8)
    data_iter = iter(dataloader)
    inputs, target = next(data_iter)

    context_length = 4
    output_dim = 256
    vocab_size = len(raw_text)

    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    token_embeddings =token_embedding_layer(inputs)

    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))

    input_embeddings = token_embeddings + pos_embeddings
    print(token_embeddings)
    print(pos_embeddings)
    print(input_embeddings)

