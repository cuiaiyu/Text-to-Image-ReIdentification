from torch.utils import data
import torchvision.transforms as transforms
import numpy as np
import pickle
from PIL import Image
import random,os,json
import collections
import torch, copy
import fastrand

class WIDERTriplet(data.Dataset):
    """
    root class of WIDER Triplet.
    """
    def __init__(self, anno_path, img_dir, split='train',transform=None,debug=False):
        """
        anno_path: path to annotations (json or pkl)
        img_dir: path to img dir
        """
        self.anno_path = anno_path
        self.img_dir = img_dir
        self.transform = transform
        self.train= (split=='train')
        
        # load annotations    
        self._load_anns(anno_path, split)
        
        
        # debug Mode?
        if debug:
            self.anns = self.anns[:1000]
        
        # build person maps
        self._build_person_map()
        
    def _build_person_map(self):
        """
        build map from ix2person and person2ix. (As a re-identification problem)
        """
        self.person2ann = collections.defaultdict(list)
        self.ann2person = {}
        for i,ann in enumerate(self.anns):
            self.person2ann[ann['id']].append(i)
            self.ann2person[i] = ann['id']
        self.len = len(self.ann2person)
        self.id2key = {i:key for i,key in enumerate(self.person2ann)}
        self.person2label = {key:i for i,key in enumerate(self.person2ann.keys())}
    
    def _load_anns(self, anno_path, split):
        """
        a helper constructor function. load all the annotations based on split.
        """
        if anno_path.endswith('pkl'):
            with open(anno_path,'rb') as f:
                anns = pickle.load(f)
        elif anno_path.endswith('json'):
            with open(anno_path,'r') as f:
                anns = json.load(f)
        print("[ds] load annotations from %s" % anno_path)
        if split == 'val2':
            self.anns = [ann for ann in anns if ann['split'] == 'val2']
        else:
            self.anns = [ann for ann in anns if ann['split'].startswith(split)] # or ann['split']=='val2']
            
        new_anns = []
        for ann in self.anns:
            caps = ann['captions']
            ann['captions'] = caps[0]
            new_anns.append(copy.deepcopy(ann))
            ann['captions'] = caps[1]
            new_anns.append(copy.deepcopy(ann))
        self.anns = new_anns
            
 
            
    def __len__(self):
        return len(self.ann2person)
    
    def _triplet_sample(self,index):
        """
        triplet sampling function
        """
        def _myrandint(up):
            #return fastrand.pcg32bounded(up)
            #return int(random.random()*up)
            return random.randint(0,up-1)
        
        curr_person = self.ann2person[index]
        pos_index=index
        while pos_index==index and len(self.person2ann[curr_person]) > 1:
            N = len(self.person2ann[curr_person])
            pid = _myrandint(N)
            pos_index=self.person2ann[curr_person][pid]
        neg_person = curr_person
        while neg_person == curr_person:
            nid = _myrandint(len(self.person2ann))
            neg_person = self.id2key[nid]
        M = len(self.person2ann[neg_person])
        nid = _myrandint(M)
        neg_index = self.person2ann[neg_person][nid]
        return pos_index,neg_index
    
    def _load_cap(self,index,i=None):
        """
        load captions
        """
        if not i:
            i = 0 if random.random() > 0.5 else 1
        tokens = self.anns[index]['captions'][i]
        return tokens
    
    def _load_img(self,index):
        """
        load image
        """
        fn = os.path.join(self.img_dir,self.anns[index]['file_path'])
        image = Image.open(fn).convert('RGB')
        #print(image.size)
        if self.transform:
            image = self.transform(image)
        return image
    
    def __getitem__(self,index):
        if self.train:
            # sample
            pos_index,neg_index = self._triplet_sample(index)
            # load image
            curr_img = self._load_img(index)
            pos_img = self._load_img(pos_index)
            neg_img = self._load_img(neg_index)
            # load caption
            curr_cap = self._load_cap(index)
            pos_cap = self._load_cap(pos_index)
            neg_cap = self._load_cap(neg_index)
            
            return curr_img,pos_img,neg_img,curr_cap,pos_cap,neg_cap,
        else:
            image = self._load_img(index)
            cap1 = self._load_cap(index,0)
            cap2 = self._load_cap(index,1)
            return image,cap1,cap2
    