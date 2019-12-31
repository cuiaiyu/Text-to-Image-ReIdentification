from .evaluator import Evaluator
from tqdm import tqdm
import torch

class GlobalEvaluator(Evaluator):
   
    def __init__(self, img_loader, cap_loader, gt_file_path,  embed_size, logger, dist_fn):
        super(GlobalEvaluator, self).__init__(img_loader, cap_loader, gt_file_path,  embed_size, logger)
        # dist fn
        self.dist = dist_fn
        
    def populate_img_db(self, encoder):
        K = self.embed_size
        global_imgs = torch.zeros((0,K)).cuda()
        encoder.eval()
        with torch.no_grad():
            for i, data in tqdm(enumerate(self.img_loader),desc='build db global imgs'):
                img, file_names = data
                img_em = encoder(img.cuda())
                global_imgs = torch.cat((global_imgs,img_em),0)
        self.global_imgs = global_imgs
        return global_imgs
    
    def populate_cap_db(self, encoder):
        K = self.embed_size
        encoder.eval()
        global_caps = torch.zeros((0,self.embed_size)).cuda()
        with torch.no_grad():
            for i, batch in tqdm(enumerate(self.cap_loader),desc='build db global caps'):
                caps, file_names = batch
                global_cap = encoder(caps.cuda())
                global_caps =  torch.cat((global_caps, global_cap),0)
        self.global_caps = global_caps
        return global_caps
    
    def retrieval(self):
        querys = self.global_caps.cpu().detach().numpy()
        candidates = self.global_imgs.cpu().detach().numpy()
        scoremat = self.dist(querys, candidates)
        return scoremat