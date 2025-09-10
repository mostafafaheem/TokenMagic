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
        self.vocab = {bytes([i]):i for i in range(2**8)}
        self.merges = {}

    def train(self, vocabulary_size: int, text: str):
        
        if vocabulary_size < 2**8:
            raise ValueError("Vocab size must be at least 256 in order to encode all possible characters.")
        
        num_merges = vocabulary_size - 2**8

        word_tokens: list[list[bytes]] = [
            [bytes([byte]) for byte in word.encode("utf-8")] for word in self.pattern.findall(text)
        ]

        while num_merges > 0:
            pair_counts = collections.Counter()
            for word in word_tokens:
                for byte_pair in zip(word[:-1], word[1:]):
                    pair_counts[byte_pair] += 1
            if not pair_counts:
                break
            most_frequent_pair = max(pair_counts, key=pair_counts.get)
            merged_bytes = most_frequent_pair[0] + most_frequent_pair[1]
            self.merges[most_frequent_pair] = vocabulary_size - num_merges
            self.vocab[merged_bytes] = vocabulary_size - num_merges

            merged_word_tokens = []
            for word in word_tokens:
                merged_word = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and (word[i], word[i + 1]) == most_frequent_pair:
                        merged_word.append(merged_bytes)
                        i += 2
                    else:
                        merged_word.append(word[i])
                        i += 1
                merged_word_tokens.append(merged_word)
            
            word_tokens = merged_word_tokens
            num_merges -= 1

        return self.merges

    def encode(self, text: str, merges: dict[tuple[bytes, bytes], int]) -> list[int]:
        if not merges:
            raise ValueError("Tokenizer not trained. Call train() first.")

        tokens = []

        word_tokens = [
            [bytes([byte]) for byte in word.encode("utf-8")]
            for word in self.pattern.findall(text)
        ]

        for word in word_tokens:
            while len(word) >= 2:
                pairs = [(word[i], word[i + 1]) for i in range(len(word) - 1)]
                ranked = [(merges.get(pair, float("inf")), pair) for pair in pairs]
                min_rank, selected = min(ranked, key=lambda x: x[0], default=(float("inf"), None))

                if selected is None or min_rank == float("inf"):
                    break

                merged_word = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and (word[i], word[i + 1]) == selected:
                        merged_word.append(word[i] + word[i + 1])
                        i += 2
                    else:
                        merged_word.append(word[i])
                        i += 1
                word = merged_word

            for byte in word:
                
                token_id = self.vocab.get(byte)
                if token_id is None:
                    raise ValueError(f"Unknown token: {byte}")
                tokens.append(token_id)

        return tokens

    def decode(self, tokens_ids: list[int]) -> str:
        return 0