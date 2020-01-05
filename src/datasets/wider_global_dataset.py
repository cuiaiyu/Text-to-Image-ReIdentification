from datasets.WIDERTriplet import *
from datasets.tokenizers import WIDER_Tokenizer
from datasets.np_chunks import NPExtractor

def train_np_collate_fn(batch):
    ims, pos_ims, neg_ims = [], [], []
    caps, pos_caps, neg_caps = [], [], []
    nps, pos_nps, neg_nps = [], [], []
    n2c, pos_n2c, neg_n2c = [], [], []
    pids, pos_pids, neg_pids = [], [], []
    for i, (
            curr_img,pos_img,neg_img,
            cap, pos_cap, neg_cap,
            np, pos_np, neg_np,
            pid, pos_pid, neg_pid
               ) in enumerate(batch):
        ims.append(curr_img[None]); pos_ims.append(pos_img[None]); neg_ims.append(neg_img[None])
        caps.append(cap); pos_caps.append(pos_cap); neg_caps.append(neg_cap)
        pids.append(pid); pos_pids.append(pos_pid); neg_pids.append(neg_pid)
        nps += np; 
        n2c += [len(np)] #[i] * len(np)
        pos_nps += pos_np; 
        pos_n2c += [len(pos_np)] #[i] * len(pos_np)
        neg_nps += neg_np; 
        neg_n2c += [len(neg_np)] #[i] * len(neg_np)
        
        #np = np[:6] if np.size(0) > 6 else torch.cat((np, torch.ones((6 - np.size(0), 6)).long()))
        #pos_np = np[:6] if pos_np.size(0) > 6 else torch.cat((pos_np, torch.ones((6 - pos_np.size(0), 6)).long()))
        #neg_np = np[:6] if neg_np.size(0) > 6 else torch.cat((neg_np, torch.ones((6 - neg_np.size(0), 6)).long()))
        #nps.append(np[None]); pos_nps.append(pos_np[None]); neg_nps.append(neg_np[None])
    
    ims = torch.cat(ims)
    pos_ims = torch.cat(pos_ims)
    neg_ims = torch.cat(neg_ims)
    
    caps = torch.cat(caps)
    pos_caps = torch.cat(pos_caps)
    neg_caps = torch.cat(neg_caps)
    
    pids = torch.LongTensor(pids)
    pos_pids = torch.LongTensor(pos_pids)
    neg_pids = torch.LongTensor(neg_pids)
    
    nps = torch.cat(nps)
    pos_nps = torch.cat(pos_nps)
    neg_nps = torch.cat(neg_nps)
    
    n2c = torch.LongTensor(n2c)
    pos_n2c = torch.LongTensor(pos_n2c)
    neg_n2c = torch.LongTensor(neg_n2c)
    
    return (ims, pos_ims, neg_ims,
            caps, pos_caps, neg_caps,
            nps, pos_nps, neg_nps,
            n2c, pos_n2c, neg_n2c,
            pids, pos_pids, neg_pids,
           )
    
    
    
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
        pos_index,neg_index = self._triplet_sample(index)
        # load image
        curr_img = self._load_img(index)
        pos_img = self._load_img(pos_index)
        neg_img = self._load_img(neg_index)
        # load caption
        cap, nps = self._load_cap(index)
        pos_cap, pos_nps = self._load_cap(pos_index)
        neg_cap, neg_nps = self._load_cap(neg_index)
        # load pid
        pid = self.ann2person[index]
        pid = self.person2label[pid]
        pos_pid = self.ann2person[pos_index]
        pos_pid = self.person2label[pos_pid]
        neg_pid = self.ann2person[neg_index]
        neg_pid = self.person2label[neg_pid]
        return (
            curr_img,pos_img,neg_img,
            cap, pos_cap, neg_cap,
            nps, pos_nps, neg_nps,
            pid, pos_pid, neg_pid
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
            pos_index,neg_index = self._triplet_sample(index)
            # load image
            curr_img = self._load_img(index)
            pos_img = self._load_img(pos_index)
            neg_img = self._load_img(neg_index)
            # load caption
            cap = self._load_cap(index)
            pos_cap = self._load_cap(pos_index)
            neg_cap = self._load_cap(neg_index)
            # load pid
            pid = self.ann2person[index]
            pid = self.person2label[pid]
            pos_pid = self.ann2person[pos_index]
            pos_pid = self.person2label[pos_pid]
            neg_pid = self.ann2person[neg_index]
            neg_pid = self.person2label[neg_pid]
            return (
                curr_img,pos_img,neg_img,
                cap, pos_cap, neg_cap,
                pid, pos_pid, neg_pid
                   )
        else:
            image = self._load_img(index)
            cap1 = self._load_cap(index,0)
            cap2 = self._load_cap(index,1)
            return image,cap1,cap2

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