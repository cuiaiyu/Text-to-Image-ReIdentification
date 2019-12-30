import torch
import numpy as np

from tqdm import tqdm
import itertools, collections
from .utils.wider_tools import write_result_file_image, eval

class Evaluator:
    def __init__(self, img_loader, cap_loader, gt_file_path,  embed_size):
        self.embed_size = embed_size
        self.img_loader = img_loader
        self.cap_loader = cap_loader
        self.gt_file_path = gt_file_path
        
        # keys
        self.idx2img = {i:name for i, name in enumerate(self.img_loader.dataset.get_all_keys())}
        self.idx2cap = {i:name for i, name in enumerate(self.cap_loader.dataset.get_all_keys())}
        
        
    def populate_img_db(self, encoder):
        pass
    
    def populate_cap_db(self, encoder):
        pass
    
    def retrieval(self):
        pass
    
    def format_result(self, scoremat, K=10):
        M, N = scoremat.shape
        result = {}
        result_all = {}
        for m in range(M):
            cap = self.idx2cap[m]
            img_idx_local = np.argsort(scoremat[m, :])[:K]
            img_idx = img_idx_local #[self.cheap_candidate_map[m, i] for i in img_idx_local]
            imgs = [self.idx2img[i] for i in img_idx]
            result[cap] = imgs
        self.results = result
        return result
    
    def compute_acc(self, scoremat, output_path):
        ret_dict = self.format_result(scoremat)
        write_result_file_image(ret_dict, file_name=output_path)
        acc = eval(output_path, self.gt_file_path)
        out = {'top-1': acc[0], 'top-5': acc[1], 'top-10': acc[2]}
        return out
    
    def evaluate(self, encoder, output_path="tmp.txt"):
        # compute global features
        self.populate_img_db(encoder)
        self.populate_cap_db(encoder)
    
        # global eval
        scoremat = self.retrieval()
        acc = self.compute_acc(scoremat, output_path)
        print("[global] R@1: %.4f | R@5: %.4f | R@10: %.4f" % (acc['top-1'], acc['top-5'], acc['top-10']))
        return acc

