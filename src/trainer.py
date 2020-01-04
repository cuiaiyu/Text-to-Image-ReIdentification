from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim 
import torchnet as tnt
import os

import torch.nn.functional as F
from models.model import build_dual_encoder
from loss.loss import triplet_cos_loss, crossmodal_triplet_loss

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
        self.cfg = args
        self._init_models()
        self._init_criterion()
        self.log = logger.log
    
    def _init_criterion(self):
        if self.cfg.dist_fn_opt == "cosine":
            self.triplet_loss = triplet_cos_loss
        elif self.cfg.dist_fn_opt == "euclidean":
            self.triplet_loss = nn.TripletMarginLoss()
        # self.cls_loss = nn.CrossEntropyLoss()
        self.log("[Trainer][init] criterion initialized.")

    def _init_models(self):
        self.model = Model(embed_size=self.cfg.embed_size, 
                          image_opt=self.cfg.img_backbone_opt, 
                          caption_opt=self.cfg.cap_backbone_opt,
                          cap_embed_type=self.cfg.cap_embed_type,
                          img_num_cut=self.cfg.img_num_cut,
                          regional_embed_size=self.cfg.regional_embed_size).cuda()
        self.rga_img_mlp = MLP(self.cfg.regional_embed_size, self.cfg.embed_size).cuda()
        self.rga_cap_mlp = MLP(self.cfg.embed_size, self.cfg.embed_size).cuda()
        
        # load ckpt
        self.reset_ckpt()
        
        # gpu
        self.all_models = {
            "model": self.model,
            "rga_img_mlp": self.rga_img_mlp,
            "rga_cap_mlp": self.rga_cap_mlp,
        }
        self.log("[Trainer][init] model initialized.")

    def reset_ckpt(self):
        self.start_epoch = 0
        self.acc_history = []
        self.best_acc = (0, self.start_epoch)
        if cfg.load_ckpt_fn == "0":
            self.log("[Trainer][init] initialize fresh model.")
            return
        ckpt = torch.load(cfg.load_ckpt_fn)
        self.start_epoch = ckpt["epoch"] + 1
        self.acc_history = ckpt["acc_history"]
        for network, name in self.all_models.items():
            if name in ckpt:
                network.load_state_dict(ckpt[name], False)
                self.log("[Trainer][init] load pre-trained %s from %s." % (network, cfg.load_ckpt_fn))

              
    def save_ckpt(self, epoch, acc, fn):
        # update acc history 
        self.acc_history.append((acc, epoch))
        if acc > self.best_acc[0]:
            self.best_acc = (acc, epoch)
        # ckpt 
        ckpt = {
            "epoch": epoch,
            "acc_history": self.acc_history,
            "best_acc": self.best_acc,
            }
        for network, name in self.all_models.items():
            ckpt[name] = network.module.state_dict() if isinstance(network, nn.DataParallel) else network.state_dict(),

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
        if isinstance(model, nn.DataParallel):
            self.model.module.img_backbone.melt_layer(8 - num_layer_to_melt)
        else:
            self.model.img_backbone.melt_layer(8 - num_layer_to_melt)
     
    def train_epoch_global(train_data, optimizer, epoch, note="train"):
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
            loss = tri_loss

            # backpropagation
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            # log
            cum_tri_loss += tri_loss.item()
            if (i+1) % self.cfg.print_freq == 0:
                out_string = "[ep-%d, bs-%d] " % (epoch, i)
                out_string += "[tri-loss] %.6f, " % cum_tri_loss / self.cfg.print_freq
                self.log(out_string)
                cum_tri_loss = 0.0
                
    def train_epoch_regional(train_data, optimizer, epoch, note="train"):
        self.model.train(); self.rga_img_mlp.train(); self.rga_cap_mlp.train()

        cum_tri_loss, cum_tri_image_regional_loss, cum_tri_text_regional_loss = 0.0, 0.0, 0.0
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
            
            N, M, T = nps.size()
            nps = self.rga_cap_mlp(self.model(nps.reshape(-1, T))).reshape(N, M, -1)
            pos_nps = self.rga_cap_mlp(self.model(pos_nps.reshape(-1, T))).reshape(N, M, -1)
            neg_nps = self.rga_cap_mlp(self.model(neg_nps.reshape(-1, T))).reshape(N, M, -1)
            
            # part
            img_part = self.rga_img_mlp(img_part)
            pos_img_part = self.rga_img_mlp(pos_img_part)
            neg_img_part = self.rga_img_mlp(neg_img_part)

            img_part = RGA_attend_one_to_many_batch(cap, img_part, self.cfg.dist_fn_opt)
            pos_img_part = RGA_attend_one_to_many_batch(pos_cap, pos_img_part, self.cfg.dist_fn_opt)
            neg_img_part = RGA_attend_one_to_many_batch(neg_cap, neg_img_part, self.cfg.dist_fn_opt)
            #cap_part = regional_alignment_text(img, nps, n2c, cfg.dist_fn_opt)
            #pos_cap_part = regional_alignment_text(pos_img, pos_nps, pos_n2c, cfg.dist_fn_opt)
            #neg_cap_part = regional_alignment_text(neg_img, neg_nps, neg_n2c, cfg.dist_fn_opt)
            cap_part = RGA_attend_one_to_many_batch(img, nps, self.cfg.dist_fn_opt)
            pos_cap_part = RGA_attend_one_to_many_batch(pos_img, pos_nps, self.cfg.dist_fn_opt)
            neg_cap_part = RGA_attend_one_to_many_batch(neg_img, neg_nps, self.cfg.dist_fn_opt)

            # loss
            tri_loss =  crossmodal_triplet_loss(img,pos_img,neg_img, 
                                                  cap, pos_cap, neg_cap, 
                                                  triplet_loss, self.cfg.dist_fn_opt) 
            tri_image_regional_loss =  crossmodal_triplet_loss(img_part,pos_img_part,neg_img_part, 
                                                  cap, pos_cap, neg_cap, 
                                                  triplet_loss, self.cfg.dist_fn_opt) 
            tri_text_regional_loss =  crossmodal_triplet_loss(img,pos_img,neg_img, 
                                                  cap_part, pos_cap_part, neg_cap_part, 
                                                  triplet_loss, self.cfg.dist_fn_opt) 


            loss = tri_loss + tri_image_regional_loss  + tri_text_regional_loss

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            # log
            cum_tri_loss += tri_loss.item()
            cum_tri_image_regional_loss += tri_image_regional_loss.item()
            cum_tri_text_regional_loss += tri_text_regional_loss.item()
            
            if (i+1) % self.cfg.print_freq == 0:
                out_string = "[ep-%d, bs-%d] " % (epoch, i)
                out_string += "[tri-loss] %.6f, " % cum_tri_loss / self.cfg.print_freq
                out_string += "[img_rga] %.6f, " %  cum_tri_image_regional_loss / self.cfg.print_freq
                out_string += "[cap_rga] %.6f " % cum_tri_text_regional_loss / self.cfg.print_freq
                self.log(out_string)
                cum_tri_loss, cum_tri_image_regional_loss, cum_tri_text_regional_loss = 0.0, 0.0, 0.0
               
            
            
def train_epoch_id(train_data, model, classifier, optimizer, cls_loss, note="train"):
    model.train()
    cum_loss = 0.0
    for i, data in tqdm(enumerate(train_data), "%s, epoch%d" % (note,epoch)):
        # load data
        (img,pos_img,neg_img, cap, pos_cap, neg_cap, pid, pos_pid, neg_pid) = data
        img, pos_img, neg_img = model(img.cuda()), model(pos_img.cuda()), model(neg_img.cuda())
        cap, pos_cap, neg_cap = model(cap.cuda()), model(pos_cap.cuda()), model(neg_cap.cuda())

        # loss
        loss = 0.0
        loss = loss + cls_loss(classifier(img), pid.cuda()) +  cls_loss(classifier(cap), pid.cuda())

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cum_loss += loss.item()

        # log
        if (i+1) % 64 == 0:
            print("batch %d, loss %.6f" % (i, cum_loss/64))
            cum_loss = 0.0
    return model
        


    
            





        
