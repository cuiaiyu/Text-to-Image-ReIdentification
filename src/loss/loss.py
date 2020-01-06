import torch
import numpy as np
import torch.nn.functional as F

def cos_distance(x,y):
    eps = 1e-12
    dot_prod = np.dot(x,y.transpose(1,0))
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    y_norm = np.linalg.norm(y, axis=1, keepdims=True)
    tmp = dot_prod / (np.dot(x_norm, y_norm.transpose(1,0)) + eps)
    return 1 - tmp


def triplet_cos_loss_inner(x, pos, neg, margin=0.2):
    def cos_dist(x,y):
        # import pdb; pdb.set_trace()
        return F.cosine_similarity(x, y) 
        # return 1 - torch.sum(x*y, 1) / (torch.norm(x, dim=1)*torch.norm(y, dim=1))
    pos_dist = cos_dist(x, pos)
    neg_dist = cos_dist(x, neg)
    scores = torch.clamp(margin - pos_dist + neg_dist, min=0)
    return scores.sum()

def triplet_cos_loss(imgs, caps, pids):
    loss = 0.0
    N = imgs.size(0)
    for i, pid in enumerate(pids):
        neg_imgs = imgs[pids != pid]
        neg_caps = imgs[pids != pid]
        pos_imgs = imgs[i:i+1].expand_as(neg_imgs)
        pos_caps = caps[i:i+1].expand_as(neg_caps)
        cap_tri_loss = triplet_cos_loss_inner(pos_imgs, pos_caps, neg_caps)
        img_tri_loss = triplet_cos_loss_inner(pos_caps, pos_imgs, neg_imgs)
        loss = loss + cap_tri_loss + img_tri_loss
    return loss / N


def crossmodal_triplet_loss(img, pos_img, neg_img, cap, pos_cap, neg_cap, triplet_loss, dist_fn_opt):
    loss = 0.0
    #loss = loss + triplet_loss(img, pos_img, neg_img)
    #loss = loss + triplet_loss(cap, pos_cap, neg_cap)
    loss = loss + triplet_loss(img, cap, neg_cap)
    loss = loss + triplet_loss(cap, img, neg_img)
        
    return loss

def crossmodal_triplet_old(img, pos_img, neg_img, cap, pos_cap, neg_cap, triplet_loss, dist_fn_opt):
    loss = 0.0
    if True: #dist_fn_opt == "euclidean":
        loss = loss + triplet_loss(img, pos_img, neg_img)
        loss = loss + triplet_loss(cap, pos_cap, neg_cap)

        loss = loss + triplet_loss(img, pos_cap, neg_cap)
        loss = loss + triplet_loss(img, pos_cap, neg_img)
        loss = loss + triplet_loss(img, pos_img, neg_cap)
        loss = loss + triplet_loss(cap, pos_img, neg_img)
        loss = loss + triplet_loss(cap, pos_img, neg_cap)
        loss = loss + triplet_loss(cap, pos_cap, neg_img)

        loss = loss + triplet_loss(cap, img, neg_cap)
        loss = loss + triplet_loss(cap, img, neg_img)
        
    elif dist_fn_opt == "cosine":
        same = torch.Tensor(img.size(0)).fill_(1).cuda()
        diff = torch.Tensor(img.size(0)).fill_(-1).cuda()
        
        loss = loss + triplet_loss(img, cap, same)
        loss = loss + triplet_loss(img, img, same)
        loss = loss + triplet_loss(pos_img, pos_cap, same)

        loss = loss + triplet_loss(img, neg_img, diff)
        loss = loss + triplet_loss(cap, neg_img, diff)
        loss = loss + triplet_loss(img, neg_cap, diff)
        loss = loss + triplet_loss(cap, neg_cap, diff)

        loss = loss + triplet_loss(pos_img, neg_img, diff)
        loss = loss + triplet_loss(pos_cap, neg_img, diff)
        loss = loss + triplet_loss(pos_img, neg_cap, diff)
        loss = loss + triplet_loss(pos_cap, neg_cap, diff)
    return loss