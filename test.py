from tokenizer import BPETokenizer, SPECIAL_TOKENS, PATTERN_STRING
import tiktoken
import regex as re

tok = BPETokenizer()

regex_obj = re.compile(PATTERN_STRING)


with open("the-verdict.txt","r", encoding="utf-8") as f:
    verdict = f.read()
# text = "Hello, world!"
# hangul = "한국어 키보드"
text2 = "today\n  zpi -\n"
text3 = "hello\x00world\n<EOS>  <PAD><UNK>"
# # from tiktoken._educational import visualise_tokens
# print([
#             [seg.encode("utf-8")] if seg in SPECIAL_TOKENS
#             else [bytes([b]) for b in word.encode("utf-8")]
#             for seg in [p for p in re.split("(" + "|".join(map(re.escape, SPECIAL_TOKENS)) + ")", text3) if p]
#             for word in ([seg] if seg in SPECIAL_TOKENS else regex_obj.findall(seg))
#         ])
# print(tok.decode(tok.encode(text3, tok.train(10000, verdict)), visusalize_control_caracters=True))
tok.load('toktikv1')
print(tok.decode(tok.encode(text3), visusalize_control_caracters=True))
# print(tok.vocab)
# tok = tiktoken.get_encoding("o200k_base")

# print(tok.decode(tok.encode(text3)))