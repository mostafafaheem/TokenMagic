from tokenizer import BPETokenizer
import tiktoken
tok = BPETokenizer()

with open("the-verdict.txt","r", encoding="utf-8") as f:
    verdict = f.read()
# text = "Hello, world!"
# hangul = "한국어 키보드"
text2 = "today\n  zpi -\n"
text3 = "hello\x00world\n"
# # from tiktoken._educational import visualise_tokens

print(tok.decode(tok.encode(text3, tok.train(10000, verdict)), visusalize_control_caracters=True))
# tok.load('toktikv1')
# print(tok.decode(tok.encode(text2), visusalize_control_caracters=True))
# tok = tiktoken.get_encoding("o200k_base")

# print(tok.decode(tok.encode(text3)))