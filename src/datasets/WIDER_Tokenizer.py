import json
import nltk

class WIDER_Tokenizer:
    def tokenize(self, sent):
        tokens = nltk.word_tokenize(sent.lower())
        # import pdb; pdb.set_trace()
        return [self.vocab[w] if w in self.vocab else 0 for w in tokens] + [1]
    def __init__(self, vocab_fn):
        with open(vocab_fn, "r") as f:
            vocab_stuff = json.load(f)
            self.vocab = vocab_stuff['vocab']
        
        
        