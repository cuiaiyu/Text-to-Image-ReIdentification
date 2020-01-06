from datasets.WIDERTriplet import *
from datasets.tokenizers import WIDER_Tokenizer
from datasets.np_chunks import NPExtractor

def train_np_collate_fn(batch):
    ims, pos_ims, neg_ims = [], [], []
    caps, pos_caps, neg_caps = [], [], []
    nps, pos_nps, neg_nps = [], [], []
    n2c, pos_n2c, neg_n2c = [], [], []
    pids, pos_pids, neg_pids = [], [], []
    for i, (curr_img, cap, np, pid) in enumerate(batch):
        ims.append(curr_img[None])
        caps.append(cap);
        pids.append(pid);
        nps += np; 
        n2c += [len(np)] #[i] * len(np)
        
        #np = np[:6] if np.size(0) > 6 else torch.cat((np, torch.ones((6 - np.size(0), 6)).long()))
        #pos_np = np[:6] if pos_np.size(0) > 6 else torch.cat((pos_np, torch.ones((6 - pos_np.size(0), 6)).long()))
        #neg_np = np[:6] if neg_np.size(0) > 6 else torch.cat((neg_np, torch.ones((6 - neg_np.size(0), 6)).long()))
        #nps.append(np[None]); pos_nps.append(pos_np[None]); neg_nps.append(neg_np[None])
    
    ims = torch.cat(ims)
    
    caps = torch.cat(caps)
    
    pids = torch.LongTensor(pids)
    
    nps = torch.cat(nps)
    
    n2c = torch.LongTensor(n2c)
    
    return (ims, caps, nps, n2c, pids)
    
    
    
class WIDERTriplet_NP(WIDERTriplet):
    def __init__(self,anno_path,img_dir,vocab_fn, token_length=40,
                 split='train',transform=None,debug=False):
        super(WIDERTriplet_NP, self).__init__(anno_path, img_dir, split, transform, debug)
        self.token_length = token_length
        self.tokenizer = WIDER_Tokenizer(vocab_fn)
        self.np_extractor = NPExtractor()
        print("size of dataset: %d" % len(self.anns))
    
    def _load_cap(self,index,i=None):
        cap = self.anns[index]['captions']
        cap_token = self.tokenizer.tokenize(cap, 40)
        cap_token = torch.LongTensor([cap_token])
        nps = self.np_extractor.sent_parse(cap)
        nps = [torch.LongTensor([self.tokenizer.tokenize(np, 6)]) for np in nps]
        # nps = torch.cat(nps)
        return cap_token, nps
    
    def __getitem__(self,index):
        # sample
        # load image
        curr_img = self._load_img(index)
        # load caption
        cap, nps = self._load_cap(index)
        # load pid
        pid = self.ann2person[index]
        pid = self.person2label[pid]
        return (
            curr_img, cap, nps, pid
               )
    

        
        
class WIDERTriplet_Basic(WIDERTriplet):
    def __init__(self,anno_path,img_dir,vocab_fn, token_length=40,
                 split='train',transform=None,debug=False):
        super(WIDERTriplet_Basic, self).__init__(anno_path, img_dir, split, transform, debug)
        self.token_length = token_length
        self.tokenizer = WIDER_Tokenizer(vocab_fn)
        print("size of dataset: %d" % len(self.anns))
    
    def _load_cap(self,index,i=None):
        cap = self.anns[index]['captions']
        cap_token = self.tokenizer.tokenize(cap, 40)
        cap_token = torch.LongTensor([cap_token])[0]
        return cap_token
    
    def __getitem__(self,index):
        if self.train:
            # sample
            # load image
            curr_img = self._load_img(index)
            # load caption
            cap = self._load_cap(index)
            # load pid
            pid = self.ann2person[index]
            pid = self.person2label[pid]
            return (
                curr_img, cap, pid
                   )

def build_wider_dataloader(cfg):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose([
                                    transforms.Resize(cfg.dim),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    normalize,
                            ])
    TrainDataset = WIDERTriplet_NP if cfg.np else WIDERTriplet_Basic
        
    ds = TrainDataset(anno_path=cfg.anno_path,
                                img_dir=cfg.img_dir,
                                vocab_fn=cfg.vocab_path,
                                split='train',
                                transform=transform_train,
                                debug=cfg.debug)
    if cfg.np:
        dl = data.DataLoader(ds,
                             batch_size=cfg.batch_size,
                             shuffle=True,
                             num_workers=cfg.num_workers, 
                             collate_fn=train_np_collate_fn,
                             pin_memory=True)
    else:
        dl = data.DataLoader(ds,
                             batch_size=cfg.batch_size,
                             shuffle=True,
                             num_workers=cfg.num_workers, 
                             pin_memory=True)
    return dl