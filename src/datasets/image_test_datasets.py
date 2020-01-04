from datasets.WIDERTriplet import *

def build_image_test_loader(cfg):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    transform_test = transforms.Compose([
                                    transforms.Resize(cfg.dim),
                                    transforms.ToTensor(),
                                    normalize,
                            ])
    ds_image = WIDERImageDataset(img_dir=cfg.val_img_dir, transform=transform_test, debug=False)
    return data.DataLoader(ds_image,
                               batch_size=cfg.batch_size,
                               shuffle=False,
                               num_workers=cfg.num_workers, 
                               pin_memory=True)
    
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
            self.img_fns = [fn for fn in self.img_fns if not fn.endswith(".DS_Store")]
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
    
