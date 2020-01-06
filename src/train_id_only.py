from datasets.wider_global_dataset_new import build_wider_dataloader
from datasets.text_test_datasets import build_text_test_loader
from datasets.image_test_datasets import build_image_test_loader
from manager import Manager, build_graph_optimizer
from evaluators.global_evaluator import GlobalEvaluator
from evaluators.np_evaluator import NPEvaluator

from loggers.logger import Logger
from tqdm import tqdm as tqdm
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
from configs.args import load_arg_parser

#------------------
## args
#------------------
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
cfg.img_num_cut = 1 if not cfg.np else cfg.img_num_cut
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
cfg.exp_name += "_id_only"
cfg.model_path = os.path.join("/shared/rsaas/aiyucui2/wider_person", cfg.model_path, cfg.exp_name)
cfg.output_path = os.path.join("/shared/rsaas/aiyucui2/wider_person", cfg.output_path, cfg.exp_name)

if not os.path.exists(cfg.model_path):
    os.mkdir(cfg.model_path)
if not os.path.exists(cfg.output_path):
    os.mkdir(cfg.output_path)

logger = Logger(os.path.join(cfg.output_path, "log.txt"))
logger.log(str(cfg))


#------------------
## agents
#------------------
# Data loaders
train_loader = build_wider_dataloader(cfg)
cfg.num_ids = len(train_loader.dataset.person2label.values())
test_text_loader = build_text_test_loader(cfg) 
test_image_loader = build_image_test_loader(cfg) 

# Evaluators 
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
# Trainer
manager = Manager(cfg, logger)


#------------------
## Train
#------------------
# Stage 1 (ID)
logger.log("======== [Stage 1] ============")
manager.melt_img_layer(num_layer_to_melt=0)
param_to_optimize = build_graph_optimizer([manager.model, manager.id_cls])
optimizer = optim.Adam(param_to_optimize, lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10)

for epoch in range(cfg.num_epochs_stage1):
    manager.train_epoch_id(train_loader, optimizer, epoch, "train-stage-1")
    acc = evaluator.evaluate(manager.model)
    logger.log('[stage-1, ep-%d][euclidean][global] R@1: %.4f | R@5: %.4f | R@10: %.4f' % (epoch, acc['top-1'], acc['top-5'], acc['top-10']))
    acc = cos_evaluator.evaluate(manager.model)
    logger.log('[stage-1, ep-%d][cosine   ][global] R@1: %.4f | R@5: %.4f | R@10: %.4f' % (epoch, acc['top-1'], acc['top-5'], acc['top-10']))
    scheduler.step()
    manager.save_ckpt(epoch, acc, 'stage_1_id_last.pt')
if cfg.num_epochs_stage1:
    manager.save_ckpt(epoch, acc, 'id_initialized.pt')

# Stage 2 (ID)
logger.log("======== [Stage 2] ============")
manager.melt_img_layer(num_layer_to_melt=8)
param_to_optimize = build_graph_optimizer([manager.model, manager.id_cls])
optimizer = optim.Adam(param_to_optimize, lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5)

for epoch in range(cfg.num_epochs_stage2):
    manager.train_epoch_id(train_loader, optimizer, epoch, "train-stage-2")
    acc = evaluator.evaluate(manager.model)
    logger.log('[stage-2, ep-%d][euclidean][global] R@1: %.4f | R@5: %.4f | R@10: %.4f' % (epoch, acc['top-1'], acc['top-5'], acc['top-10']))
    acc = cos_evaluator.evaluate(manager.model)
    logger.log('[stage-2, ep-%d][cosine   ][global] R@1: %.4f | R@5: %.4f | R@10: %.4f' % (epoch, acc['top-1'], acc['top-5'], acc['top-10']))
    scheduler.step()
    manager.save_ckpt(epoch, acc, 'stage_2_id_last.pt')