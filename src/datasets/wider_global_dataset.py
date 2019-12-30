from datasets.WIDERTriplet import *
from datasets.WIDER_Tokenizer import *

class WIDERTriplet_Basic(WIDERTriplet):
    def __init__(self,anno_path,img_dir,vocab_fn, token_length=40,
                 split='train',transform=None,debug=False):
        super(WIDERTriplet_Basic, self).__init__(anno_path, img_dir, split, transform, debug)
        self.token_length = token_length
        self.tokenizer = WIDER_Tokenizer(vocab_fn)
        print("size of dataset: %d" % len(self.anns))
    
    def _load_cap(self,index,i=None):
        if not isinstance(i, int):
            i = 0 if random.random() > 0.5 else 1
        cap = self.anns[index]['captions'][i]
        indexed_tokens = self.tokenizer.tokenize(cap)
        lengthed = [1]*40
        lengthed[:min(40,len(indexed_tokens))] = indexed_tokens[:min(40,len(indexed_tokens))]
        tokens_tensor = torch.tensor([lengthed])
        return tokens_tensor[0] 
    
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

def build_wider_dataloader(anno_path,img_dir,vocab_fn, dim=(384,128),
                                    token_length=40,
                                    train=True,batch_size=64,num_workers=8,debug=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose([
                                    transforms.Resize(dim),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    normalize,
                            ])
    ds = WIDERTriplet_Basic(anno_path=anno_path,
                                img_dir=img_dir,
                                vocab_fn=vocab_fn,
                                token_length=token_length,
                                split='train' if train else 'val',
                                transform=transform_train,
                               debug=debug)
    dl = data.DataLoader(ds,
                         batch_size=batch_size,
                         shuffle=train,
                         num_workers=num_workers, 
                         pin_memory=True)
    return dl