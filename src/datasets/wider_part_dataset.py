from datasets.WIDERTriplet import *
from datasets.tokenizers import WIDER_Tokenizer
from datasets.np_chunks import NPExtractor

class WIDERTriplet_Part(WIDERTriplet):
    def __init__(self, anno_path, img_dir, mask_dir, vocab_fn,  
                 sent_token_length=40, np_token_length=6, num_np_per_sent=6,
                 split='train', transform=None, debug=False):
        super(WIDERTriplet_Part, self).__init__(anno_path, img_dir, split, transform, debug)
        
        self.mask_dir = mask_dir
        self.toTensor = transforms.ToTensor()
        
        self.sent_token_length = sent_token_length
        self.np_token_length = np_token_length
        self.num_np_per_sent = num_np_per_sent
        
        self.tokenizer = WIDER_Tokenizer(vocab_fn)
        self.np_extractor = NPExtractor()
        print("size of dataset: %d" % len(self.anns))
    
    def _load_cap(self,index):
        len_sent = self.sent_token_length
        len_np = self.np_token_length
        N = self.num_np_per_sent
        cap = self.anns[index]['captions']
        cap_token = self.tokenizer.tokenize(cap, len_sent)
        cap_token = torch.LongTensor([cap_token])[0]
        
        nps = self.np_extractor.sent_parse(cap)
        nps = [torch.LongTensor([self.tokenizer.tokenize(np, len_np)]) for np in nps]
        num_nps = min(len(nps), N)
        nps = torch.cat(nps)
        nps = nps[:N] if nps.size(0) > N else torch.cat((nps, torch.ones((N - nps.size(0), len_np)).long()))
        
        return cap_token, nps, num_nps
    
    def _load_img(self,index):
        img_fn = self.anns[index]['file_path']
        mask_fn = img_fn.replace('/', '_')[:-4] + '.npy'
        img_path = os.path.join(self.img_dir,img_fn)
        mask_path = os.path.join(self.mask_dir,mask_fn)
        
        image = Image.open(img_path).convert('RGB')
        mask = torch.from_numpy(np.load(mask_path)).float()
        
        if self.transform:
            image = self.transform(image)
            
        if random.random() > 0.5:
            image = torch.flip(image, [2])
            mask = torch.flip(mask, [2])
        return image, mask
    
    def __getitem__(self,index):
        # sample
        # load image
        img, mask = self._load_img(index)
        # load caption
        cap, nps, num_nps = self._load_cap(index)
        # load pid
        pid = self.ann2person[index]
        pid = self.person2label[pid]
        return (img, mask, cap, nps, num_nps, pid)
    
class WIDERTriplet_NP(WIDERTriplet):
    def __init__(self,anno_path, img_dir, mask_dir, vocab_fn, 
                 sent_token_length=40, np_token_length=6, num_np_per_sent=6, 
                 split='train', transform=None, debug=False):
        super(WIDERTriplet_NP, self).__init__(anno_path, img_dir, split, transform, debug)
        
        self.sent_token_length = sent_token_length
        self.np_token_length = np_token_length
        self.num_np_per_sent = num_np_per_sent
        
        self.tokenizer = WIDER_Tokenizer(vocab_fn)
        self.np_extractor = NPExtractor()
        print("size of dataset: %d" % len(self.anns))
    
    def _load_cap(self,index):
        len_sent = self.sent_token_length
        len_np = self.np_token_length
        N = self.num_np_per_sent
        
        cap = self.anns[index]['captions']
        cap_token = self.tokenizer.tokenize(cap, len_sent)
        cap_token = torch.LongTensor([cap_token])[0]
        
        nps = self.np_extractor.sent_parse(cap)
        nps = [torch.LongTensor([self.tokenizer.tokenize(np, len_np)]) for np in nps]
        num_nps = min(len(nps), N)
        nps = torch.cat(nps)
        nps = nps[:N] if nps.size(0) > N else torch.cat((nps, torch.ones((N - nps.size(0), len_np)).long()))
        
        return cap_token, nps, num_nps
    
    def __getitem__(self,index):
        # sample
        # load image
        img = self._load_img(index)
        # load caption
        cap, nps, num_nps = self._load_cap(index)
        # load pid
        pid = self.ann2person[index]
        pid = self.person2label[pid]
        return (img, cap, nps, num_nps, pid)
    

        
        
class WIDERTriplet_Basic(WIDERTriplet):
    def __init__(self, anno_path, img_dir, mask_dir, vocab_fn,  
                 sent_token_length=40, np_token_length=6, num_np_per_sent=6,
                 split='train', transform=None, debug=False):
        super(WIDERTriplet_Basic, self).__init__(anno_path, img_dir, split, transform, debug)
        self.sent_token_length = sent_token_length
        self.tokenizer = WIDER_Tokenizer(vocab_fn)
        print("size of dataset: %d" % len(self.anns))
    
    def _load_cap(self,index,i=None):
        cap = self.anns[index]['captions']
        cap_token = self.tokenizer.tokenize(cap, self.sent_token_length)
        cap_token = torch.LongTensor([cap_token])[0]
        return cap_token
    
    def __getitem__(self,index):
        # load image
        curr_img = self._load_img(index)
        # load caption
        cap = self._load_cap(index)
        # load pid
        pid = self.ann2person[index]
        pid = self.person2label[pid]
        return (curr_img, cap, pid)

def build_wider_dataloader(cfg):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if cfg.mask:
        transform_train = transforms.Compose([transforms.Resize(cfg.dim), 
                                              transforms.ToTensor(), 
                                              normalize])
    else:
        transform_train = transforms.Compose([
                                              transforms.Resize(cfg.dim),
                                              # transforms.RandomCrop((260, 112)),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(), 
                                              normalize])
    if cfg.mask:
        TrainDataset = WIDERTriplet_Part
    elif cfg.np:
        TrainDataset = WIDERTriplet_NP
    else:
        TrainDataset = WIDERTriplet_Basic
        
    ds = TrainDataset(anno_path=cfg.anno_path,
                      img_dir=cfg.img_dir,
                      mask_dir=cfg.mask_dir, 
                      vocab_fn=cfg.vocab_path,
                      sent_token_length=cfg.sent_token_length, 
                      np_token_length=cfg.np_token_length, 
                      num_np_per_sent=cfg.num_np_per_sent,
                      split='train',
                      transform=transform_train,
                      debug=cfg.debug)
    dl = data.DataLoader(ds,
                         batch_size=cfg.batch_size,
                         shuffle=True,
                         num_workers=cfg.num_workers, 
                         pin_memory=True)
    return dl