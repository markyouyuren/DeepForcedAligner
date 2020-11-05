from typing import List


class Tokenizer:

    def __init__(self, symbols: List[str], pad_token='_') -> None:
        self.symbols = symbols
        self.pad_token = pad_token
        self.idx_to_token = {i: s for i, s in enumerate(symbols, start=1)}
        self.idx_to_token[0] = pad_token
        self.token_to_idx = {s: i for i, s in self.idx_to_token.items()}
        self.vocab_size = len(self.symbols) + 1

    def __call__(self, sentence):
        seq = []
        ngrams = [''.join(j) for j in zip(*[sentence[i:] for i in range(2)])]
        for ng in ngrams:
            index = self.token_to_idx[ng]
            seq.append(index)
        return seq

    def decode(self, sequence):
        return ''.join([self.idx_to_token[int(t)] for t in sequence if int(t) in self.idx_to_token])
