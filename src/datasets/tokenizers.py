import json
import nltk

class WIDER_Tokenizer:
    def tokenize(self, sent, max_length):
        tokens = nltk.word_tokenize(sent.lower())
        indexed_tokens = [self.vocab[w] if w in self.vocab else 0 for w in tokens] + [1]
        lengthed = [1] * max_length
        end_token = min(max_length,len(indexed_tokens))
        lengthed[:end_token] = indexed_tokens[:end_token]
        return lengthed
    
    def __init__(self, vocab_fn):
        with open(vocab_fn, "r") as f:
            vocab_stuff = json.load(f)
            self.vocab = vocab_stuff['vocab']
        
        
        