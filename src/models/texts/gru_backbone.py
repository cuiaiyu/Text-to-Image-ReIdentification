import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BiGRUBackbone(nn.Module):
    def __init__(self, embed_size, vocab_size=1287, caption_opt="bigru"):
        super(BiGRUBackbone, self).__init__()
        self.embed_size = embed_size
        self.embedding = nn.Embedding(vocab_size, 300)
        
        self.fc = nn.Linear(embed_size*2, embed_size)
        if caption_opt == "bigru":
            self.rnn = nn.GRU(input_size=300, 
                              hidden_size=embed_size, 
                              num_layers=1, 
                              bidirectional=True,
                              batch_first=True)
        self.init_weights()
        
    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)

    
    def forward(self, caps):
        # import pdb; pdb.set_trace()
        bs = caps.size(0)
        emb = self.embedding(caps)
        if emb.requires_grad:
            self.rnn.flatten_parameters()
        emb = pack_padded_sequence(emb, [caps.size(1)]*bs, batch_first=True, enforce_sorted=False)
        output, hidden = self.rnn(emb)
        sent_emb = hidden.transpose(0, 1).contiguous()
        sent_emb = sent_emb.view(-1, self.embed_size*2)
        sent_emb = self.fc(sent_emb)
        '''
        output = pad_packed_sequence(output, batch_first=True)[0]
        # --> batch  x seq_len x hidden_size*num_directions
        N, C, K = output.size()
        words_emb = output.sum(1)
        words_emb = self.fc(words_emb)
        
        return words_emb
        '''
        return sent_emb