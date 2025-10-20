import regex as re
import collections
import unicodedata
import base64
import json

PATTERN_STRING = "|".join(
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

SPECIAL_TOKENS = {
    "<PAD>": 0,
    "<UNK>": 1,
    "<BOS>": 2,
    "<EOS>": 3,
}

class BPETokenizer:
    def __init__(self, pattern = None):
        self.pattern = PATTERN_STRING if pattern is None else pattern
        self.vocab: dict[bytes, int] = {bytes([i]):i for i in range(2**8)}
        self.lookup: dict[int, bytes] = {i:bytes([i]) for i in range(2**8)}
        self.merges: dict[(bytes, bytes), int] = {}
        self.special_tokens: dict[str, int] = {}

    def train(self, vocabulary_size: int, text: str):
        
        if vocabulary_size < 2**8:
            raise ValueError("Vocab size must be at least 256 in order to encode all possible characters.")
        
        num_merges = vocabulary_size - 2**8
        regex_obj = re.compile(self.pattern)

        word_tokens: list[list[bytes]] = [
            [seg.encode("utf-8")] if seg in SPECIAL_TOKENS
            else [bytes([b]) for b in word.encode("utf-8")]
            for seg in [p for p in re.split("(" + "|".join(map(re.escape, SPECIAL_TOKENS)) + ")", text) if p]
            for word in ([seg] if seg in SPECIAL_TOKENS else regex_obj.findall(seg))
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
            self.lookup[vocabulary_size - num_merges] = merged_bytes

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
        special_token_ids = {}
        final_vocab_size = len(self.vocab)
        for offset, (tok, _) in enumerate(SPECIAL_TOKENS.items()):
            idx = final_vocab_size + offset
            encoded = tok.encode("utf-8")
            self.vocab[encoded] = idx
            self.lookup[idx] = encoded
            special_token_ids[tok] = idx
        self.special_tokens = special_token_ids
        self.save("toktikv1")
        return self.merges

    def encode(self, text: str, merges: dict[tuple[bytes, bytes], int] = None) -> list[int]:
        if not merges:
            if not self.merges:
                raise ValueError("Tokenizer not trained. Call train() first.")
            merges = self.merges

        tokens = []
        
        regex_obj = re.compile(self.pattern)

        word_tokens: list[list[bytes]] = [
            [seg.encode("utf-8")] if seg in SPECIAL_TOKENS
            else [bytes([b]) for b in word.encode("utf-8")]
            for seg in [p for p in re.split("(" + "|".join(map(re.escape, SPECIAL_TOKENS)) + ")", text) if p]
            for word in ([seg] if seg in SPECIAL_TOKENS else regex_obj.findall(seg))
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

    def decode(self, tokens_ids: list[int], visusalize_control_caracters = False) -> str:
        out = []
        if visusalize_control_caracters:
            for token in tokens_ids:
                decoded = self.lookup[token].decode("utf-8", errors="replace")
                for ch in decoded:
                    if ch == "\n":
                        out.append("\\n")
                    elif ch == "\t":
                        out.append("\\t")
                    elif ch == "\r":
                        out.append("\\r")
                    elif unicodedata.category(ch)[0] == "C":  # other control chars
                        out.append(f"\\u{ord(ch):04x}")
                    else:
                        out.append(ch)
        else:
            for token in tokens_ids:
                out.append(self.lookup[token].decode("utf-8", errors="replace"))

        return "".join(out)


    def save(self, filename: str):

        model_file = filename + ".model.json"
        vocab_file = filename + ".vocab.json"

        model_data = {
            "pattern": self.pattern,
            "special_tokens": self.special_tokens, 
            "merges": [

                [base64.b64encode(a).decode("ascii"),
                 base64.b64encode(b).decode("ascii"),
                 rank]
                for (a, b), rank in self.merges.items()
            ],
        }
        with open(model_file, "w", encoding="utf-8") as f:
            json.dump(model_data, f, ensure_ascii=False, indent=2)

        vocab_data = {
            str(idx): base64.b64encode(token).decode("ascii")
            for token, idx in self.vocab.items()
        }

        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)

    def load(self, filename: str):

        model_file = filename + ".model.json"
        vocab_file = filename + ".vocab.json"

        with open(model_file, "r", encoding="utf-8") as f:
            model_data = json.load(f)

        self.pattern = model_data["pattern"]

        self.special_tokens = model_data["special_tokens"]

        self.merges = {
            (base64.b64decode(a), base64.b64decode(b)): rank
            for a, b, rank in model_data["merges"]
        }

        with open(vocab_file, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)

        self.vocab = {
            base64.b64decode(token): int(idx)
            for idx, token in vocab_data.items()
        }
        self.lookup = {idx: token for token, idx in self.vocab.items()}