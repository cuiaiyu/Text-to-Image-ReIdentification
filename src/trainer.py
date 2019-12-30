from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim 
import torchnet as tnt
import os

import torch.nn.functional as F
from models.model import build_dual_encoder

class Manager:
    def __init__(self, args):
        self.args = args
        self._init_models()
        self._init_optimizer()
        self._init_criterion()
        self.log_file = os.path.join(args.log_dir, args.experiment_name + '.txt')
    
    def _init_criterion(self):
        self.triplet_loss = nn.TripletMarginLoss()
        self.cls_loss = nn.BCELoss()
        if self.args.dist_fn == "cos":
            self.dist = F.cosine_similarity
        elif self.args.dist_fn == "euclidean":
            self.dist = F.pairwise_distance
        # self.cls_loss = nn.CrossEntropyLoss()
        self.log("[Trainer][init] criterion initialized.")

    def _init_models(self):
        self.encoder = build_dual_encoder(self.args)
        self.reset_ckpt(self.args.load_model_path)
        # freeze models
        self.encoder.img_encoder.melt(self.args.n_img_melt_layer)
        self.encoder.cap_encoder.melt(self.args.n_cap_melt_layer)
        # gpu
        self.encoder.cuda()
        self.log("[Trainer][init] model initialized.")

    def _init_optimizer(self):
        # params to optimize
        parameters_to_optimize = []
        for param in self.model.parameters():
            if param.requires_grad == True:
                parameters_to_optimize.append(param)
                
        if self.args.optimizer_opt == "adam":
            self.optimizer = optim.Adam(
                parameters_to_optimize,
                lr=self.args.lr,
                weight_decay=self.args.weight_decay,
            )
        else:
            self.log("not implemented!")
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.step_size)
        self.log("[Trainer][init] optimizer initialized.")

    def log(self, out_string):
        print(out_string)
        f=open(self.log_file, "a+")
        f.write(out_string)
        f.write("\n")
        f.close()

    def reset_ckpt(self, load_model_path):
        self.start_epoch = 0
        self.acc_history = []
        self.best_acc = (0, self.start_epoch)
        if load_model_path != "0":
            ckpt = torch.load(load_model_path)
            self.start_epoch = ckpt["epoch"] + 1
            self.acc_history = ckpt["acc_history"]
            self.encoder.load_state_dict(ckpt["encoder"])
            self.log("[Trainer][init] load pre-trained model from %s." % load_model_path)
        else:
            self.log("[Trainer][init] initialize fresh model.")
              
    def update_ckpt(self, epoch, acc, fn=None):
        # update acc history 
        self.acc_history.append((acc, epoch))
        if acc > self.best_acc[0]:
            self.best_acc = (acc, epoch)
        # ckpt 
        if fn:
            ckpt = {
                "epoch": epoch,
                "acc_history": self.acc_history,
                "best_acc": self.best_acc,
                "encoder": self.encoder.module.state_dict() if isinstance(self.encoder, nn.DataParallel) else self.encoder.state_dict(),
            }
            path = os.path.join(self.args.ckpt_dir, self.args.experiment_name, fn)
            torch.save(ckpt, path)
            
    def todevice(self, batch):
        ret = []
        for arg in batch:
            if isinstance(arg, torch.Tensor):
                arg = arg.cuda()
            ret.append(arg)
        return tuple(ret)

    def encode(self, x, mask, m2i):
        global_feature_maps = self.encoder.encode(x)
        local_features = self.cropper(global_feature_maps, mask)
        global_features = self.cropper(global_feature_maps)
        return global_features, local_features
    

    def train_epoch(self, train_loader, epoch):
        global_loss_meter = tnt.meter.AverageValueMeter()
        self.encoder.train()
        for i, data in tqdm(enumerate(train_loader), "epoch%d" % epoch):
            data = self.todevice(data)
            (img, pos_img, neg_img, cap, pos_cap, neg_cap, # input
             mimg, pos_mimg, neg_mimg, mcap, pos_mcap, neg_mcap, # masks
             m2i, pos_m2i, neg_m2i, m2c, pos_m2c, neg_m2c, # index map
             ilabel, pos_ilabel, neg_ilabel, clabel, pos_clabel, neg_clabel, # label
             ) = data

            # encode
            glo_img, loc_img = self.encode(img, mimg, m2i)
            glo_pimg, loc_pimg = self.encode(pos_img, pos_mimg, pos_m2i)
            glo_nimg, loc_nimg = self.encode(neg_img, neg_mimg, neg_m2i)

            glo_cap, loc_cap = self.encode(cap, mcap, m2c)
            glo_pcap, loc_pcap = self.encode(pos_cap, pos_mcap, pos_m2c)
            glo_ncap, loc_ncap = self.encode(neg_cap, neg_mcap, neg_m2c)

            # loss
            if self.args.flag_global_triplet_loss:
                global_loss = self.global_triplet_loss(glo_img, glo_pimg, glo_nimg, glo_cap, glo_pcap, glo_ncap)
                loss = loss + global_loss
            if self.args.flag_local_triplet_loss:
                pass
             
             # backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            global_loss_meter.add(global_loss.data.cpu())

            # log 
            if i % self.args.print_freq == 0:
                out_string = "[ep-%d, %d] " % (epoch, i)
                out_string += "[glo_tri_loss] %.4f +- %.4f,  " % global_loss_meter.value()
                self.log(out_string)
                global_loss_meter.reset()
            





        
