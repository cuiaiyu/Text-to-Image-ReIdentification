from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim 
import os

import torch.nn.functional as F
from models.encoder import Model, MLP
from loss.loss import triplet_cos_loss, crossmodal_triplet_loss

from attentions.rga_attention import RGA_attend_one_to_many_batch, RGA_attend_one_to_many



def build_graph_optimizer(models):
    if not isinstance(models, list):
        models = [models]
    params_to_optimize = []
    for model in models:
        if model and hasattr(model, '_parameters'):
            for param in model.parameters():
                if param.requires_grad == True:
                    params_to_optimize.append(param)
    return params_to_optimize

def regional_alignment_text(fulls, parts, p2fs, dist_fn_opt):
    start_index = 0
    aligned = []
    for i, jump in enumerate(p2fs):
        curr_parts = parts[start_index:start_index + jump]
        start_index += jump
        curr_full = fulls[i:i+1]
        aligned.append(RGA_attend_one_to_many(curr_full, curr_parts, dist_fn_opt))
    return torch.cat(aligned)

def regional_alignment_image(fulls, parts, dist_fn_opt):
    return RGA_attend_one_to_many_batch(fulls, parts, dist_fn_opt)
  
    
class Manager:
    def __init__(self, args, logger):
        self.log = logger.log
        self.cfg = args
        self._init_models()
        self._init_criterion()
    
    def _init_criterion(self):
        if self.cfg.dist_fn_opt == "cosine":
            self.triplet_loss = triplet_cos_loss
        elif self.cfg.dist_fn_opt == "euclidean":
            self.triplet_loss = nn.TripletMarginLoss()
        self.cls_loss = nn.CrossEntropyLoss()
        self.log("[Trainer][init] criterion initialized.")

    def _init_models(self):
        # encoder
        self.model = Model(embed_size=self.cfg.embed_size, 
                          image_opt=self.cfg.img_backbone_opt, 
                          caption_opt=self.cfg.cap_backbone_opt,
                          cap_embed_type=self.cfg.cap_embed_type,
                          img_num_cut=self.cfg.img_num_cut,
                          regional_embed_size=self.cfg.regional_embed_size).cuda()
        # id classifer
        self.id_cls = nn.Linear(self.cfg.embed_size, self.cfg.num_ids).cuda()
        # RGA image mlp
        self.rga_img_mlp = MLP(self.cfg.regional_embed_size, self.cfg.embed_size).cuda()
        # RGA text mlp
        self.rga_cap_mlp = MLP(self.cfg.embed_size, self.cfg.embed_size).cuda()
        # gpu
        self.all_models = {
            "model": self.model,
            "id_cls": self.id_cls, 
            "rga_img_mlp": self.rga_img_mlp,
            "rga_cap_mlp": self.rga_cap_mlp,
        }
        
        # load ckpt
        self.reset_ckpt()
        
        
        self.log("[Trainer][init] model initialized.")

    def reset_ckpt(self):
        self.start_epoch = 0
        self.acc_history = []
        self.best_acc = ({'top-1':0, 'top-1': 0, 'top-1': 0}, self.start_epoch)
        if self.cfg.load_ckpt_fn == "0":
            self.log("[Trainer][init] initialize fresh model.")
            return
        ckpt = torch.load(self.cfg.load_ckpt_fn)
        self.start_epoch = ckpt["epoch"] + 1
        self.acc_history = ckpt["acc_history"]
        for name, network in self.all_models.items():
            if name in ckpt:
                network.load_state_dict(ckpt[name], strict=False)
                self.log("[Trainer][init] load pre-trained %s from %s." % (name, self.cfg.load_ckpt_fn))

              
    def save_ckpt(self, epoch, acc, fn):
        # update acc history 
        self.acc_history.append((acc, epoch))
        if acc['top-1'] > self.best_acc[0]['top-1']:
            self.best_acc = (acc, epoch)
        # ckpt 
        ckpt = {
            "epoch": epoch,
            "acc_history": self.acc_history,
            "best_acc": self.best_acc,
            }
        for name, network in self.all_models.items():
            ckpt[name] = network.module.state_dict() if isinstance(network, nn.DataParallel) else network.state_dict()

        path = os.path.join(self.cfg.model_path, fn)
        torch.save(ckpt, path)
            
    def todevice(self, batch):
        ret = []
        for arg in batch:
            if isinstance(arg, torch.Tensor):
                arg = arg.cuda()
            ret.append(arg)
        return tuple(ret)
    
    def melt_img_layer(self, num_layer_to_melt=1):
        if isinstance(self.model, nn.DataParallel):
            self.model.module.img_backbone.melt_layer(8 - num_layer_to_melt)
        else:
            self.model.img_backbone.melt_layer(8 - num_layer_to_melt)
     
    def train_epoch_global(self, train_data, optimizer, epoch, note="train"):
        self.model.train()
        cum_tri_loss, cum_id_loss = 0.0, 0.0
        for i, data in tqdm(enumerate(train_data), "%s, epoch%d" % (note, epoch)):
            # load data
            data = self.todevice(data)
            (img,pos_img,neg_img, cap, pos_cap, neg_cap, pid, pos_pid, neg_pid) = data
            
            # encode
            img, pos_img, neg_img = self.model(img), self.model(pos_img), self.model(neg_img)
            cap, pos_cap, neg_cap = self.model(cap), self.model(pos_cap), self.model(neg_cap)

            # loss
            tri_loss =  crossmodal_triplet_loss(img,pos_img,neg_img, 
                                                  cap, pos_cap, neg_cap, 
                                                  self.triplet_loss, self.cfg.dist_fn_opt)  
            id_loss = self.cls_loss(self.id_cls(img), pid) + self.cls_loss(self.id_cls(cap), pid)
            loss = tri_loss + id_loss

            # backpropagation
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            # log
            cum_tri_loss += tri_loss.item()
            cum_id_loss += id_loss.item()
            if (i+1) % self.cfg.print_freq == 0:
                out_string = "[ep-%d, bs-%d] " % (epoch, i)
                out_string += "[tri-loss] %.6f, " % (cum_tri_loss / self.cfg.print_freq)
                out_string += "[id-loss] %.6f, " % (cum_id_loss / self.cfg.print_freq)
                self.log(out_string)
                cum_tri_loss, cum_id_loss = 0.0, 0.0
                
    def train_epoch_regional(self, train_data, optimizer, epoch, note="train"):
        self.model.train(); self.rga_img_mlp.train(); self.rga_cap_mlp.train()

        cum_tri_loss, cum_tri_image_regional_loss, cum_tri_text_regional_loss, cum_id_loss = 0.0, 0.0, 0.0, 0.0
        for i, data in tqdm(enumerate(train_data), "%s, epoch%d" % (note,epoch)):
            # load data
            data = self.todevice(data)
            (img, pos_img, neg_img, 
             cap, pos_cap, neg_cap,
             nps, pos_nps, neg_nps,
             n2c, pos_n2c, neg_n2c,
             pid, pos_pid, neg_pid) = data


            img, img_part = self.model(img)
            pos_img, pos_img_part = self.model(pos_img)
            neg_img, neg_img_part = self.model(neg_img)
            cap, pos_cap, neg_cap = self.model(cap), self.model(pos_cap), self.model(neg_cap)
            
            # N, M, T = nps.size()
            nps = self.rga_cap_mlp(self.model(nps))
            pos_nps = self.rga_cap_mlp(self.model(pos_nps))
            neg_nps = self.rga_cap_mlp(self.model(neg_nps))
            #nps = self.rga_cap_mlp(self.model(nps.reshape(-1, T))).reshape(N, M, -1)
            #pos_nps = self.rga_cap_mlp(self.model(pos_nps.reshape(-1, T))).reshape(N, M, -1)
            #neg_nps = self.rga_cap_mlp(self.model(neg_nps.reshape(-1, T))).reshape(N, M, -1)
            
            # part
            img_part = self.rga_img_mlp(img_part)
            pos_img_part = self.rga_img_mlp(pos_img_part)
            neg_img_part = self.rga_img_mlp(neg_img_part)

            img_part = RGA_attend_one_to_many_batch(cap, img_part, self.cfg.dist_fn_opt)
            pos_img_part = RGA_attend_one_to_many_batch(pos_cap, pos_img_part, self.cfg.dist_fn_opt)
            neg_img_part = RGA_attend_one_to_many_batch(neg_cap, neg_img_part, self.cfg.dist_fn_opt)
            cap_part = regional_alignment_text(img, nps, n2c, self.cfg.dist_fn_opt)
            pos_cap_part = regional_alignment_text(pos_img, pos_nps, pos_n2c, self.cfg.dist_fn_opt)
            neg_cap_part = regional_alignment_text(neg_img, neg_nps, neg_n2c, self.cfg.dist_fn_opt)
            #cap_part = RGA_attend_one_to_many_batch(img, nps, self.cfg.dist_fn_opt)
            #pos_cap_part = RGA_attend_one_to_many_batch(pos_img, pos_nps, self.cfg.dist_fn_opt)
            #neg_cap_part = RGA_attend_one_to_many_batch(neg_img, neg_nps, self.cfg.dist_fn_opt)

            # loss
            tri_loss =  crossmodal_triplet_loss(img,pos_img,neg_img, 
                                                  cap, pos_cap, neg_cap, 
                                                  self.triplet_loss, self.cfg.dist_fn_opt) 
            tri_image_regional_loss =  crossmodal_triplet_loss(img_part,pos_img_part,neg_img_part, 
                                                  cap, pos_cap, neg_cap, 
                                                  self.triplet_loss, self.cfg.dist_fn_opt) 
            tri_text_regional_loss =  crossmodal_triplet_loss(img,pos_img,neg_img, 
                                                  cap_part, pos_cap_part, neg_cap_part, 
                                                  self.triplet_loss, self.cfg.dist_fn_opt) 
            id_loss = self.cls_loss(self.id_cls(img), pid) +  self.cls_loss(self.id_cls(cap), pid)


            loss = tri_loss + tri_image_regional_loss  + tri_text_regional_loss + id_loss

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            # log
            cum_tri_loss += tri_loss.item()
            cum_tri_image_regional_loss += tri_image_regional_loss.item()
            cum_tri_text_regional_loss += tri_text_regional_loss.item()
            cum_id_loss += id_loss.item()
            
            if (i+1) % self.cfg.print_freq == 0:
                out_string = "[ep-%d, bs-%d] " % (epoch, i)
                out_string += "[id-loss] %.6f, " % (cum_id_loss / self.cfg.print_freq)
                out_string += "[tri-loss] %.6f, " % (cum_tri_loss / self.cfg.print_freq)
                out_string += "[img_rga] %.6f, " %  (cum_tri_image_regional_loss / self.cfg.print_freq)
                out_string += "[cap_rga] %.6f " % (cum_tri_text_regional_loss / self.cfg.print_freq)
                self.log(out_string)
                cum_tri_loss, cum_tri_image_regional_loss, cum_tri_text_regional_loss, cum_id_loss = 0.0, 0.0, 0.0, 0.0
               
            
            
    def train_epoch_id(self, train_data, optimizer, epoch, note="train"):
        self.model.train()
        self.id_cls.train()
        cum_loss = 0.0
        for i, data in tqdm(enumerate(train_data), "%s, epoch%d" % (note,epoch)):
            # load data
            data = self.todevice(data)
            (img,pos_img,neg_img, cap, pos_cap, neg_cap, pid, pos_pid, neg_pid) = data
            img = self.model(img)
            cap = self.model(cap)

            # loss
            loss = 0.0
            loss = loss + self.cls_loss(self.id_cls(img), pid) +  self.cls_loss(self.id_cls(cap), pid)
            cum_loss += loss.item()
            
            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

            # log
            if (i+1) % self.cfg.print_freq == 0:
                print("ep-%d, bs-%d, [id-loss] %.6f" % (epoch, i, cum_loss / self.cfg.print_freq))
                cum_loss = 0.0