from torch.utils import data
import torch, json
from datasets.tokenizers import WIDER_Tokenizer
from datasets.np_chunks import NPExtractor

def build_text_test_loader(cfg):
    TestDataset = WIDERNPTextDataset if cfg.np else WIDERTextDataset
    ds_text = TestDataset(anno_path=cfg.val_anno_path, 
                          vocab_fn=cfg.vocab_path, 
                          debug=cfg.debug)
    if cfg.np:
        return data.DataLoader(ds_text,
                               batch_size=cfg.batch_size,
                               shuffle=False, 
                               num_workers=cfg.num_workers,
                               collate_fn=text_test_np_collate_fn,
                               pin_memory=True)
    else:
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
        nps = nps[:6] if nps.size(0) > 6 else torch.cat((nps, torch.ones((6 - nps.size(0), 8)).long()))
                                                     
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
    def __init__(self,anno_path, vocab_fn, split="val",token_length=40, debug=False):
        super(WIDERTextDataset, self).__init__()
        # load annotations
        with open(anno_path,'r') as f:
            anns = json.load(f)
        self.anns = [ann for ann in anns if ann['split']==split]
        self.anns = self.anns[:500] if debug else self.anns
        
        # init tokenizer
        self.token_length = token_length
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
        cap_token = self.tokenizer.tokenize(cap, 40)
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
    def __init__(self,anno_path, vocab_fn, split="val",token_length=40, debug=False):
        super(WIDERNPTextDataset, self).__init__(anno_path, vocab_fn, split, token_length, debug)
        self.np_extractor = NPExtractor()
     
    def _load_cap(self,index):
        cap = self.captions[index]
        cap_token = self.tokenizer.tokenize(cap, 40)
        nps = self.np_extractor.sent_parse(cap)
        nps = [torch.LongTensor([self.tokenizer.tokenize(np, 8)]) for np in nps]
        nps = torch.cat(nps)
        
        cap_token = torch.LongTensor([cap_token])
        return cap_token, nps
    
    
    def __getitem__(self,index):
        cap, nps = self._load_cap(index)
        image_fn = self.images[index]
        return cap, nps, image_fn

