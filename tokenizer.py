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

    def train(self, vocabulary_size: int, text: str, verbose: bool=False):
        
        if vocabulary_size < 2**8:
            raise ValueError("Vocab size must be at least 256 in order to encode all possible characters.")
        
        ids = {i:bytes([i]) for i in range(2**8)}

        num_merges = vocabulary_size - 2**8

        word_tokens: list[list[bytes]] = [
            [bytes([byte]) for byte in word.encode("utf-8")] for word in self.pattern.findall(text)
        ]

        # print(word_tokens)
        merges = {}
        while num_merges > 0:
            pair_counts = collections.Counter()
            for word in word_tokens:
                for byte_pair in zip(word[:-1], word[1:]):
                    pair_counts[byte_pair] += 1
            if not pair_counts:
                break
            most_frequent_pair = max(pair_counts, key=pair_counts.get)
            merged_bytes = most_frequent_pair[0] + most_frequent_pair[1]
            merges[most_frequent_pair] = vocabulary_size - num_merges
            ids[vocabulary_size - num_merges] = merged_bytes

            merged_word_tokens = []
            for word in word_tokens:
                merged_word = []
                i = 0
                while i < len(word) - 1:
                    if (word[i], word[i + 1]) == most_frequent_pair:
                        merged_word.append(merged_bytes)
                        i += 2
                    else:
                        merged_word.append(word[i])
                        i += 1
                if i == len(word) - 1:
                    merged_word.append(word[i])
                merged_word_tokens.append(merged_word)
            
            word_tokens = merged_word_tokens

            num_merges -= 1
        print(merges)
        return ids

    def encode(self, text: str) -> list[int]:
        pass
    def decode(self, tokens_ids: list[int]) -> str:
        pass
