from torch.utils import data
import torch, json
from datasets.tokenizers import WIDER_Tokenizer
from datasets.np_chunks import NPExtractor

def build_text_test_loader(cfg):
    TestDataset = WIDERNPTextDataset if cfg.np else WIDERTextDataset
    ds_text = TestDataset(anno_path=cfg.val_anno_path, 
                          vocab_fn=cfg.vocab_path, 
                          sent_token_length=cfg.sent_token_length, 
                          np_token_length=cfg.np_token_length, 
                          num_np_per_sent=cfg.num_np_per_sent,
                          debug=cfg.debug)
    return data.DataLoader(ds_text,
                               batch_size=cfg.batch_size,
                               shuffle=False, 
                               num_workers=cfg.num_workers,
                               pin_memory=True)


def text_test_np_collate_fn(batch):
    all_caps, all_nps, all_image_fn = [], [], []
    np2cap = []
    for i, (cap, nps, image_fn) in enumerate(batch):
        all_caps.append(cap)
        all_image_fn.append(image_fn)
        nps = nps[:6] if nps.size(0) > 6 else torch.cat((nps, torch.ones((6 - nps.size(0), 6)).long()))
                                                     
        all_nps.append(nps[None])
                                                     
        np2cap += [len(nps)] #[i]*len(nps)
        
    all_caps = torch.cat(all_caps)
    all_nps = torch.cat(all_nps)
    np2cap = torch.LongTensor(np2cap)
    return all_caps, all_nps, np2cap, all_image_fn
    
    

class WIDERTextDataset(data.Dataset):
    """
    Basic.
    Return 
        - caption (indexed) 
        - image_fn (private_key)
    """
    def __init__(self,anno_path, vocab_fn, split="val",
                 sent_token_length=40, np_token_length=6, num_np_per_sent=6,
                 debug=False):
        super(WIDERTextDataset, self).__init__()
        
        # load annotations
        with open(anno_path,'r') as f:
            anns = json.load(f)
        self.anns = [ann for ann in anns if ann['split']==split]
        self.anns = self.anns[:500] if debug else self.anns
        
        # init tokenizer
        self.sent_token_length = sent_token_length
        self.np_token_length = np_token_length
        self.num_np_per_sent = num_np_per_sent
        
        self.tokenizer = WIDER_Tokenizer(vocab_fn)
        
        # init captions data
        self.captions, self.images = [], []
        for ann in self.anns:
            for cap in ann['captions']:
                self.captions.append(cap)
                self.images.append(ann['file_path'])
                
        self.len = len(self.captions)
        
    def get_all_keys(self):
        return self.images
    
    def __len__(self):
        return self.len
    
    def _load_cap(self,index):
        cap = self.captions[index]
        cap_token = self.tokenizer.tokenize(cap, self.sent_token_length)
        cap_token = torch.LongTensor([cap_token])[0]
        return cap_token
    
    def __getitem__(self,index):
        cap = self._load_cap(index)
        image_fn = self.images[index]
        return cap, image_fn
    
    
    
class WIDERNPTextDataset(WIDERTextDataset):
    """
    Return:
        - caption (indexed)
        - np chunks (in-time processed)
        - image_fn (private_key)
    """
    def __init__(self,anno_path, vocab_fn, split="val",
                 sent_token_length=40, np_token_length=6, num_np_per_sent=6,
                 debug=False):
        super(WIDERNPTextDataset, self).__init__(anno_path=anno_path, 
                                                 vocab_fn=vocab_fn, 
                                                 split=split,
                                                 sent_token_length=sent_token_length, 
                                                 np_token_length=np_token_length, 
                                                 num_np_per_sent=num_np_per_sent,
                                                 debug=debug)
        self.np_extractor = NPExtractor()
     
    def _load_cap(self,index):
        len_sent = self.sent_token_length
        len_np = self.np_token_length
        N = self.num_np_per_sent
        cap = self.captions[index]
        cap_token = self.tokenizer.tokenize(cap, len_sent)
        cap_token = torch.LongTensor([cap_token])[0]
        
        nps = self.np_extractor.sent_parse(cap)
        nps = [torch.LongTensor([self.tokenizer.tokenize(np, len_np)]) for np in nps]
        num_nps = min(len(nps), N)
        nps = torch.cat(nps)
        nps = nps[:N] if nps.size(0) > N else torch.cat((nps, torch.ones((N - nps.size(0), len_np)).long()))
        
        return cap_token, nps, num_nps
    
    
    def __getitem__(self,index):
        cap, nps, num_nps = self._load_cap(index)
        image_fn = self.images[index]
        return cap, nps, num_nps, image_fn

