import re

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items() }

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [ item.strip() for item in preprocessed if item.strip() ]
        preprocessed = [
            item if item in self.str_to_int
            else "<|unk|>"
            for item in preprocessed
        ]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

    @staticmethod
    def get_vocab(text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)

        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        all_words = sorted(set(preprocessed)) 
        all_words.extend(["<|endoftext|>", "<|unk|>"])
        return {token:integer for integer,token in enumerate(all_words)}



with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

vocab = SimpleTokenizerV2.get_vocab(raw_text)

tokenizer = SimpleTokenizerV2(vocab)

text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of the palace."

ids  = tokenizer.encode(text)
print("IDs: \n")
print(ids)


print("\n Decoded: \n")
print(tokenizer.decode(ids))

