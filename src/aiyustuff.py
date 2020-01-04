#!/usr/bin/env python
# coding: utf-8

# # Active FULL Training Code (Basic - Global)

# In[1]:

from datasets.wider_global_dataset import build_wider_dataloader
from datasets.text_test_datasets import build_text_test_loader
from datasets.image_test_datasets import build_image_test_loader
from models.encoder import Model, MLP
from evaluators.global_evaluator import GlobalEvaluator
from evaluators.np_evaluator import NPEvaluator
from loss.loss import crossmodal_triplet_loss, cos_distance
from loggers.logger import Logger
from tqdm import tqdm as tqdm
from sklearn.neighbors import DistanceMetric
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch

from configs.args import load_arg_parser
from attentions.rga_attention import RGA_attend_one_to_many_batch, RGA_attend_one_to_many

# ## config

# In[14]:

## Config
parser = load_arg_parser()
cfg = parser.parse_args()
root = cfg.data_root

# data path
cfg.anno_path = os.path.join(root, cfg.anno_path)
cfg.img_dir = os.path.join(root, cfg.img_dir)
cfg.val_anno_path = os.path.join(root, cfg.val_anno_path)
cfg.val_img_dir = os.path.join(root, cfg.val_img_dir)
cfg.gt_file_fn = os.path.join(root, cfg.gt_file_fn)

# meta data path
cfg.cheap_candidate_fn = os.path.join(root, cfg.cheap_candidate_fn)
cfg.vocab_path = os.path.join(root, cfg.vocab_path)

# sys path

cfg.dim = (384,128)
cfg.exp_name = "dist_fn_{}_imgbb_{}_capbb_{}_embed_size_{}_batch_{}_lr_{}_captype_{}_img_meltlayer_{}_cos_margin_{}_np_{}".format(
    cfg.dist_fn_opt,
    cfg.img_backbone_opt,
    cfg.cap_backbone_opt,
    cfg.embed_size,
    cfg.batch_size,
    cfg.lr,
    cfg.cap_embed_type,
    cfg.image_melt_layer,
    cfg.cos_margin,
    cfg.np)
cfg.model_path = os.path.join("/shared/rsaas/aiyucui2/wider_person", cfg.model_path, cfg.exp_name)
cfg.output_path = os.path.join("/shared/rsaas/aiyucui2/wider_person", cfg.output_path, cfg.exp_name)

if not os.path.exists(cfg.model_path):
    os.mkdir(cfg.model_path)
if not os.path.exists(cfg.output_path):
    os.mkdir(cfg.output_path)



logger = Logger(os.path.join(cfg.output_path, "log.txt"))
logger.log(str(cfg))

# ## Loading data

# train loader
train_loader = build_wider_dataloader(cfg)

# test loader (loading image and text separately)
test_text_loader = build_text_test_loader(cfg) 
test_image_loader = build_image_test_loader(cfg) 



# ## Define Model

# models initializations
model = Model(embed_size=cfg.embed_size, 
              image_opt=cfg.img_backbone_opt, 
              caption_opt=cfg.cap_backbone_opt,
              cap_embed_type=cfg.cap_embed_type,
              img_num_cut=cfg.img_num_cut,
              regional_embed_size=cfg.regional_embed_size).cuda()

img_mlp = MLP(cfg.regional_embed_size, cfg.embed_size).cuda()
cap_mlp = MLP(cfg.embed_size, cfg.embed_size).cuda()

if cfg.load_ckpt_fn != "0":
    logger.log("[Model] load pre-trained model from %s." % cfg.load_ckpt_fn)
    ckpt = torch.load(cfg.load_ckpt_fn)
    model.load_state_dict(ckpt["model"], False)
else:
    logger.log("[Model] Init fresh model.")
    
if cfg.num_gpus > 1:
    model = nn.DataParallel(model)
    
def triplet_cos_loss(x, pos, neg, margin=0.5):
    def cos_dist(x,y):
        # import pdb; pdb.set_trace()
        return 1 - torch.sum(x*y, 1) / (torch.norm(x, dim=1)*torch.norm(y, dim=1))
    pos_dist = cos_dist(x, pos)
    neg_dist = cos_dist(x, neg)
    scores = torch.clamp(pos_dist - neg_dist + margin, min=0)
    return scores.mean()

## dist functions
if cfg.dist_fn_opt == "euclidean":
    triplet_loss = nn.TripletMarginLoss()
elif cfg.dist_fn_opt == "cosine":
    triplet_loss = triplet_cos_loss


# ### Train Misc Setup
Evaluator = NPEvaluator if cfg.np else GlobalEvaluator
evaluator = Evaluator(img_loader=test_image_loader, 
                          cap_loader=test_text_loader, 
                          gt_file_path=cfg.gt_file_fn,
                          embed_size=cfg.embed_size,
                          logger=logger,
                          dist_fn_opt="euclidean")
cos_evaluator = Evaluator(img_loader=test_image_loader, 
                          cap_loader=test_text_loader, 
                          gt_file_path=cfg.gt_file_fn,
                          embed_size=cfg.embed_size,
                          logger=logger,
                          dist_fn_opt="cosine")


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

# In[13]:

def train_epoch_global(train_data, model, img_mlp_rga, cap_mlp_rga, optimizer, triplet_loss, logger, note="train"):
    model.train()
    cum_tri_loss, cum_id_loss = 0.0, 0.0
    for i, data in tqdm(enumerate(train_data), "%s, epoch%d" % (note,epoch)):
        # load data
        (img,pos_img,neg_img, cap, pos_cap, neg_cap, pid, pos_pid, neg_pid) = data
        img, pos_img, neg_img = model(img.cuda()), model(pos_img.cuda()), model(neg_img.cuda())
        cap, pos_cap, neg_cap = model(cap.cuda()), model(pos_cap.cuda()), model(neg_cap.cuda())
        
        # loss
        tri_loss =  crossmodal_triplet_loss(img,pos_img,neg_img, 
                                              cap, pos_cap, neg_cap, 
                                              triplet_loss, cfg.dist_fn_opt)  
        loss = tri_loss
        
        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
       
        
        # log
        cum_tri_loss += tri_loss.item()
        if (i+1) % 64 == 0:
            logger.log("batch %d, [tri-loss] %.6f" % (i, cum_tri_loss/64))
            cum_tri_loss = 0.0
    return model

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
  
    
def train_epoch_regional(train_data, model, img_mlp_rga, cap_mlp_rga, optimizer, triplet_loss, logger, note="train"):
    model.train(); img_mlp_rga.train(); cap_mlp_rga.train()
    
    cum_tri_loss, cum_tri_image_regional_loss, cum_tri_text_regional_loss = 0.0, 0.0, 0.0
    for i, data in tqdm(enumerate(train_data), "%s, epoch%d" % (note,epoch)):
        # load data
        (img, pos_img, neg_img, 
         cap, pos_cap, neg_cap,
         nps, pos_nps, neg_nps,
         n2c, pos_n2c, neg_n2c,
         pid, pos_pid, neg_pid) = data
        
        
        img, img_part = model(img.cuda())
        pos_img, pos_img_part = model(pos_img.cuda())
        neg_img, neg_img_part = model(neg_img.cuda())
        cap, pos_cap, neg_cap = model(cap.cuda()), model(pos_cap.cuda()), model(neg_cap.cuda())
        N, M, T = nps.size()
        nps, pos_nps, neg_nps = model(nps.cuda().reshape(-1, T)), model(pos_nps.cuda().reshape(-1, T)), model(neg_nps.cuda().reshape(-1, T))
        
        # part
        img_part, pos_img_part, neg_img_part = img_mlp_rga(img_part), img_mlp_rga(pos_img_part), img_mlp_rga(neg_img_part)
        
        nps, pos_nps, neg_nps = cap_mlp_rga(nps.reshape(N, M, -1)), cap_mlp_rga(pos_nps.reshape(N, M, -1)), cap_mlp_rga(neg_nps.reshape(N, M, -1))
        
        img_part = RGA_attend_one_to_many_batch(cap, img_part, cfg.dist_fn_opt)
        pos_img_part = RGA_attend_one_to_many_batch(pos_cap, pos_img_part, cfg.dist_fn_opt)
        neg_img_part = RGA_attend_one_to_many_batch(neg_cap, neg_img_part, cfg.dist_fn_opt)
        #cap_part = regional_alignment_text(img, nps, n2c, cfg.dist_fn_opt)
        #pos_cap_part = regional_alignment_text(pos_img, pos_nps, pos_n2c, cfg.dist_fn_opt)
        #neg_cap_part = regional_alignment_text(neg_img, neg_nps, neg_n2c, cfg.dist_fn_opt)
        cap_part = RGA_attend_one_to_many_batch(img, nps, cfg.dist_fn_opt)
        pos_cap_part = RGA_attend_one_to_many_batch(pos_img, pos_nps, cfg.dist_fn_opt)
        neg_cap_part = RGA_attend_one_to_many_batch(neg_img, neg_nps, cfg.dist_fn_opt)
        
        # loss
        tri_loss =  crossmodal_triplet_loss(img,pos_img,neg_img, 
                                              cap, pos_cap, neg_cap, 
                                              triplet_loss, cfg.dist_fn_opt) 
        tri_image_regional_loss =  crossmodal_triplet_loss(img_part,pos_img_part,neg_img_part, 
                                              cap, pos_cap, neg_cap, 
                                              triplet_loss, cfg.dist_fn_opt) 
        tri_text_regional_loss =  crossmodal_triplet_loss(img,pos_img,neg_img, 
                                              cap_part, pos_cap_part, neg_cap_part, 
                                              triplet_loss, cfg.dist_fn_opt) 
        
        
        loss = tri_loss + tri_image_regional_loss  + tri_text_regional_loss
        
        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
       
        
        # log
        cum_tri_loss += tri_loss.item()
        cum_tri_image_regional_loss += tri_image_regional_loss.item()
        cum_tri_text_regional_loss += tri_text_regional_loss.item()
        
        if (i+1) % 64 == 0:
            logger.log("batch %d, [tri-loss] %.6f, [img_rga] %.6f, [cap_rga] %.6f" % (i, 
                                                                                      cum_tri_loss/64, 
                                                                                     cum_tri_image_regional_loss / 64, 
                                                                                     cum_tri_text_regional_loss / 64))
            cum_tri_loss, cum_tri_image_regional_loss, cum_tri_text_regional_loss = 0.0, 0.0, 0.0
    return model



train_epoch = train_epoch_regional if cfg.np else train_epoch_global
# stage 1 - image channel forzen
if isinstance(model, nn.DataParallel):
    model.module.img_backbone.melt_layer(8)
else:
    model.img_backbone.melt_layer(8)
# 
param_to_optimize = build_graph_optimizer([model, img_mlp, cap_mlp])
optimizer = optim.Adam(param_to_optimize, lr=1e-3, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10)
for epoch in range(cfg.num_epochs_stage1):
    model = train_epoch(train_loader, model, img_mlp, cap_mlp, optimizer, triplet_loss, logger, "train-stage-1")
    if cfg.np:
        acc = evaluator.evaluate(model, img_mlp, cap_mlp)
    else:
        acc = evaluator.evaluate(model)
    logger.log('[euclidean][global] R@1: %.4f | R@5: %.4f | R@10: %.4f' % (acc['top-1'], acc['top-5'], acc['top-10']))
    if cfg.np:
        acc = cos_evaluator.evaluate(model, img_mlp, cap_mlp)
    else:
        acc = cos_evaluator.evaluate(model)
    logger.log('[cosine   ][global] R@1: %.4f | R@5: %.4f | R@10: %.4f' % (acc['top-1'], acc['top-5'], acc['top-10']))
    scheduler.step()
torch.save({
    "model": model.state_dict() if not isinstance(model, nn.DataParallel) else model.module.state_dict(),
    "cfg": cfg,
}, os.path.join(cfg.model_path, "stage1.pt"))
    
    
# stage 2 - train all
if isinstance(model, nn.DataParallel):
    model.module.img_backbone.melt_layer(8 - cfg.image_melt_layer)
else:
    model.img_backbone.melt_layer(8 - cfg.image_melt_layer)
param_to_optimize = build_graph_optimizer([model, img_mlp, cap_mlp])
optimizer = optim.Adam(param_to_optimize, lr=1e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10)
for epoch in range(cfg.num_epochs_stage2):
    model = train_epoch(train_loader, model, img_mlp, cap_mlp, optimizer, triplet_loss, logger, "train-stage-2")
    if cfg.np:
        acc = evaluator.evaluate(model,  img_mlp, cap_mlp)
    else:
        acc = evaluator.evaluate(model)
    logger.log('[euclidean][global] R@1: %.4f | R@5: %.4f | R@10: %.4f' % (acc['top-1'], acc['top-5'], acc['top-10']))
    if cfg.np:
        acc = cos_evaluator.evaluate(model,  img_mlp, cap_mlp)
    else:
        acc = cos_evaluator.evaluate(model)
    logger.log('[cosine   ][global] R@1: %.4f | R@5: %.4f | R@10: %.4f' % (acc['top-1'], acc['top-5'], acc['top-10']))
    scheduler.step()
torch.save({
    "model": model.state_dict() if not isinstance(model, nn.DataParallel) else model.module.state_dict(),
    "cfg": cfg,
}, os.path.join(cfg.model_path,  "stage2.pt"))


# In[ ]:




