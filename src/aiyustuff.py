#!/usr/bin/env python
# coding: utf-8

# # Active FULL Training Code (Basic - Global)

# In[1]:


from datasets.wider_global_dataset import build_wider_dataloader
from datasets.wider_global_test_dataset import build_wider_test_dataloader
from models.encoder import Model
from evaluators.global_evaluator import GlobalEvaluator
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
cfg.exp_name = "dist_fn_{}_imgbb_{}_capbb_{}_embed_size_{}_batch_{}_lr_{}_captype_{}_img_meltlayer_{}_cos_margin_{}".format(
    cfg.dist_fn_opt,
    cfg.img_backbone_opt,
    cfg.cap_backbone_opt,
    cfg.embed_size,
    cfg.batch_size,
    cfg.lr,
    cfg.cap_embed_type,
    cfg.image_melt_layer,
    cfg.cos_margin)
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
train_loader = build_wider_dataloader(anno_path=cfg.anno_path,
                                    img_dir=cfg.img_dir,
                                    vocab_fn=cfg.vocab_path, 
                                    dim=cfg.dim,
                                    token_length=40,
                                    train=True,
                                    batch_size=cfg.batch_size,
                                    num_workers=8,
                                    debug=cfg.debug)

# test loader (loading image and text separately)
test_text_loader, test_image_loader = build_wider_test_dataloader(anno_path=cfg.val_anno_path,
                                                              img_dir=cfg.val_img_dir,
                                                              vocab_fn=cfg.vocab_path, 
                                                              dim=cfg.dim,
                                                              batch_size=cfg.batch_size,
                                                              num_workers=8,
                                                              debug=cfg.debug)



# ## Define Model

# In[6]:


 


# In[7]:

# models initializations
model = Model(embed_size=cfg.embed_size, 
              image_opt=cfg.img_backbone_opt, 
              caption_opt=cfg.cap_backbone_opt,
              cap_embed_type=cfg.cap_embed_type).cuda()

if cfg.load_ckpt_fn != "0":
    logger.log("[Model] load pre-trained model from %s." % cfg.load_ckpt_fn)
    ckpt = torch.load(cfg.load_ckpt_fn)
    model.load_state_dict(ckpt["model"])
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
    dist_fn = DistanceMetric.get_metric('euclidean').pairwise
    triplet_loss = nn.TripletMarginLoss()
elif cfg.dist_fn_opt == "cosine":
    dist_fn = cos_distance
    triplet_loss = triplet_cos_loss


# ### Train Misc Setup
evaluator = GlobalEvaluator(img_loader=test_image_loader, 
                          cap_loader=test_text_loader, 
                          gt_file_path=cfg.gt_file_fn,
                          embed_size=cfg.embed_size,
                          logger=logger,
                          dist_fn=DistanceMetric.get_metric('euclidean').pairwise)
cos_evaluator = GlobalEvaluator(img_loader=test_image_loader, 
                          cap_loader=test_text_loader, 
                          gt_file_path=cfg.gt_file_fn,
                          embed_size=cfg.embed_size,
                               logger=logger, 
                          dist_fn=cos_distance)


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

def train_epoch(train_data, model, optimizer, triplet_loss, note="train"):
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
        cum_tri_loss += tri_loss.item()
        
        # log
        if (i+1) % 64 == 0:
            logger.log("batch %d, [tri-loss] %.6f" % (i, cum_tri_loss/64))
            cum_tri_loss = 0.0
    return model

# stage 1 - image channel forzen
if isinstance(model, nn.DataParallel):
    model.module.img_backbone.melt_layer(8)
else:
    model.img_backbone.melt_layer(7)
param_to_optimize = build_graph_optimizer([model])
optimizer = optim.Adam(param_to_optimize, lr=1e-3, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20)

for epoch in range(cfg.num_epochs_stage1):
    logger.log("-----stage1, epoch %d-----" % epoch)
    model = train_epoch(train_loader, model, optimizer, triplet_loss, "train-stage-1")
    acc = evaluator.evaluate(model)
    logger.log('[euclidean][global] R@1: %.4f | R@5: %.4f | R@10: %.4f' % (acc['top-1'], acc['top-5'], acc['top-10']))
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
param_to_optimize = build_graph_optimizer([model])
optimizer = optim.Adam(param_to_optimize, lr=2e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20)

for epoch in range(cfg.num_epochs_stage2):
    logger.log("-----stage2, epoch %d-----" % epoch)
    model = train_epoch(train_loader, model, optimizer, triplet_loss, "train-stage-2")
    acc = evaluator.evaluate(model)
    logger.log('[euclidean][global] R@1: %.4f | R@5: %.4f | R@10: %.4f' % (acc['top-1'], acc['top-5'], acc['top-10']))
    acc = cos_evaluator.evaluate(model)
    logger.log('[cosine   ][global] R@1: %.4f | R@5: %.4f | R@10: %.4f' % (acc['top-1'], acc['top-5'], acc['top-10']))
    scheduler.step()
torch.save({
    "model": model.state_dict() if not isinstance(model, nn.DataParallel) else model.module.state_dict(),
    "cfg": cfg,
}, os.path.join(cfg.model_path,  "stage2.pt"))


# In[ ]:




