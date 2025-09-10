from tokenizer import BPETokenizer

tok = BPETokenizer()

text = "I'm 23 years-old!!!\nHello, world!"

print(tok.train(256, text))