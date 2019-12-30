from datasets.WIDERTriplet import *
from datasets.WIDER_Tokenizer import *

class WIDERTextDataset(data.Dataset):
    def __init__(self,anno_path, vocab_fn, split="val",token_length=40, debug=False):
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
        indexed_tokens = self.tokenizer.tokenize(cap)
        lengthed = [1]*40
        lengthed[:min(40,len(indexed_tokens))] = indexed_tokens[:min(40,len(indexed_tokens))]
        tokens_tensor = torch.tensor([lengthed])
        return tokens_tensor[0] 
    
    def __getitem__(self,index):
        cap = self._load_cap(index)
        image_fn = self.images[index]
        return cap, image_fn
    
class WIDERImageDataset(data.Dataset):
    def __init__(self,img_dir,transform=None,split='val',debug=False):
        super(WIDERImageDataset,self).__init__()
        self.img_dir = img_dir
        if split == 'val' or split == 'train':
            subfolders = os.listdir(img_dir)
            self.img_fns = []
            for subfolder in subfolders:
                for fn in os.listdir(os.path.join(img_dir,subfolder)):
                    self.img_fns.append(os.path.join(subfolder,fn))
        else:
            self.img_fns = os.listdir(img_dir)
        if debug:
            self.img_fns = self.img_fns[:500]
        #self.img_fns = [img for img in self.img_fns if img.endswith('.jpg')]
        self.transform=transform
    
    def get_all_keys(self):
        return self.img_fns
    
    def __len__(self):
        return len(self.img_fns)
    
    def _load_img(self,index):
        img_fn = self.img_fns[index]
        fn = os.path.join(self.img_dir,img_fn)
        image = Image.open(fn).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image,img_fn
    
    def __getitem__(self,index):
        image,image_fn = self._load_img(index)
        return image,image_fn
    
def build_wider_test_dataloader(anno_path, img_dir, vocab_fn, dim=(384, 128), num_workers=8, batch_size=64, debug=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    transform_test = transform=transforms.Compose([
                                    transforms.Resize(dim),
                                    transforms.ToTensor(),
                                    normalize,
                            ])
    ds_text = WIDERTextDataset(anno_path, vocab_fn, debug=debug)
    dl_text = data.DataLoader(ds_text,
                              batch_size=batch_size,
                              shuffle=False, 
                              num_workers=num_workers, 
                              pin_memory=True)
    
    ds_image = WIDERImageDataset(img_dir=img_dir, transform=transform_test, debug=debug)
    dl_image = data.DataLoader(ds_image,
                               batch_size=batch_size,
                               shuffle=False,
                               num_workers=num_workers, 
                               pin_memory=True)
    return dl_text, dl_image