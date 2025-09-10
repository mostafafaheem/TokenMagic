import regex as re
import collections
pat_str = "|".join(
    [
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""\p{N}{1,3}""",
        r""" ?[^\s\p{L}\p{N}]+[\r\n/]*""",
        r"""\s*[\r\n]+""",
        r"""\s+(?!\S)""",
        r"""\s+""",
    ]
)

class BPETokenizer:
    def __init__(self, pattern = None):
        self.pattern = re.compile(pat_str) if pattern is None else pattern

    def get_stats(tokens: list[str]):
        pass
    def train(self, vocabulary_size, text, verbose=False):
        
        if vocabulary_size < 2**8:
            raise ValueError("Vocab size must be at least 256 in order to encode all possible characters.")
        
        ids = {i:bytes([i]) for i in range(2**8)}

        num_merges = vocabulary_size - 2**8

        word_tokens: list[list[bytes]] = [
            [bytes([byte]) for byte in word.encode("utf-8")] for word in self.pattern.findall(text)
        ]

        while num_merges > 0:
            
            
        return 

    def encode(self, text: str) -> list[int]:
        pass
    def decode(self, tokens_ids: list[int]) -> str:
        pass
    
        # vocab = {bytes([i]):i for i in range(256)}

        # return vocab
# for i, x in bpe_train().items():
#     print(f"{i}, {x}")
# print(bpe_train())

