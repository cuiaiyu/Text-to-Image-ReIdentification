from evaluators.evaluator import *
import torch
import numpy as np
from attentions.rga_attention import RGA_attend_one_to_many_batch
from sklearn.neighbors import DistanceMetric
from loss.loss import cos_distance
import torch.nn.functional as F


class NPEvaluator(Evaluator):
    def __init__(self, img_loader, cap_loader, gt_file_path,  embed_size, logger, dist_fn_opt):
        super(NPEvaluator, self).__init__(img_loader, cap_loader, gt_file_path,  embed_size, logger)
        # dist fn
        self.dist_fn_opt = dist_fn_opt
        if dist_fn_opt == 'euclidean':
            self.dist = DistanceMetric.get_metric('euclidean').pairwise
        else:
            self.dist = cos_distance
        
    def populate_img_db(self, encoder, img_mlp):
        K = self.embed_size
        self.global_imgs = []
        self.img_parts = []
        encoder.eval(); img_mlp.eval()
        with torch.no_grad():
            for i, data in tqdm(enumerate(self.img_loader),desc='build db global imgs'):
                img, file_names = data
                img_em, img_part = encoder(img.cuda())
                self.img_parts.append(img_mlp(img_part))
                self.global_imgs.append(img_em)
        self.global_imgs = torch.cat(self.global_imgs)
        self.img_parts = torch.cat(self.img_parts)
        return self.global_imgs
    
    def populate_cap_db(self, encoder, text_mlp):
        K = self.embed_size
        encoder.eval(); text_mlp.eval()
        self.global_caps = []
        self.cap_parts = []
        self.n2cs = []
        with torch.no_grad():
            for i, batch in tqdm(enumerate(self.cap_loader),desc='build db global caps'):
                caps, nps, n2c, file_names = batch
                N, M, T = nps.size()
                global_cap = encoder(caps.cuda())
                nps = encoder(nps.reshape(N*M, T).cuda())
                self.global_caps.append(global_cap)
                self.n2cs.append(n2c)
                self.cap_parts.append(text_mlp(nps).reshape(N, M, -1))
        self.global_caps = torch.cat(self.global_caps)
        self.cap_parts = torch.cat(self.cap_parts)
        self.n2cs = torch.cat(self.n2cs)
        return self.global_caps
    
    def regional_alignment_image(self, caps, img_parts, dist_fn_opt):
        scoremats = []
        N, M, K = img_parts.size()
        for cap in tqdm(caps, "scoremat_rga_img"):
            with torch.no_grad():
                parts = RGA_attend_one_to_many_batch(cap[None], img_parts, dist_fn_opt)
                if dist_fn_opt == "cosine":
                    scores = 1 - F.cosine_similarity(cap[None], parts)
                else:
                    scores = F.pairwise_distance(cap[None], parts)
                scoremats.append(scores.detach().cpu().numpy())
        return np.array(scoremats)
    
    def regional_alignment_text(self, imgs, cap_parts, n2cs, dist_fn_opt):
        scoremats = []
        N, M, K = cap_parts.size()
        for cap_part, n2c in tqdm(zip(cap_parts, n2cs), "scoremat_rga_cap(nps)"):
            with torch.no_grad():
                parts = RGA_attend_one_to_many_batch(imgs, cap_part[None,:n2c,:].expand(imgs.size(0), n2c, imgs.size(1)), dist_fn_opt)
                if dist_fn_opt == "cosine":
                    scores = 1 - F.cosine_similarity(imgs, parts)
                else:
                    scores = F.pairwise_distance(imgs, parts)
                scoremats.append(scores.detach().cpu().numpy())
        return np.array(scoremats)
    
    def evaluate(self, encoder, mlp_img, mlp_text, output_path="tmp.txt"):
        # compute global features
        self.populate_img_db(encoder, mlp_img)
        self.populate_cap_db(encoder, mlp_text)
    
        # global eval
        # scoremat = self.retrieval()
        scoremat_global, scoremat_img_rga, scoremat_cap_rga = self.retrieval()
        acc = self.compute_acc(scoremat_global, output_path)
        self.logger.log("[global] R@1: %.4f | R@5: %.4f | R@10: %.4f" % (acc['top-1'], acc['top-5'], acc['top-10']))
        acc = self.compute_acc(scoremat_img_rga, output_path)
        self.logger.log("[img_rga] R@1: %.4f | R@5: %.4f | R@10: %.4f" % (acc['top-1'], acc['top-5'], acc['top-10']))
        acc = self.compute_acc(scoremat_cap_rga, output_path)
        self.logger.log("[cap_rga] R@1: %.4f | R@5: %.4f | R@10: %.4f" % (acc['top-1'], acc['top-5'], acc['top-10']))
        
        acc = self.compute_acc(scoremat_global + 0.5*scoremat_img_rga + 0.5*scoremat_cap_rga, output_path)
        self.logger.log("[fusion] R@1: %.4f | R@5: %.4f | R@10: %.4f" % (acc['top-1'], acc['top-5'], acc['top-10']))
        return acc                         
            
    
    def retrieval(self):
        querys = self.global_caps.cpu().detach().numpy()
        candidates = self.global_imgs.cpu().detach().numpy()
        scoremat = self.dist(querys, candidates)
        scoremat2 = self.regional_alignment_image(self.global_caps, self.img_parts, self.dist_fn_opt)
        scoremat3 = self.regional_alignment_text(self.global_imgs, self.cap_parts, self.n2cs, self.dist_fn_opt)
        return scoremat, scoremat2, scoremat3