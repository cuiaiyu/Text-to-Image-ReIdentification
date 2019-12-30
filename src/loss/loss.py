import torch

def crossmodal_triplet_loss(img, pos_img, neg_img, cap, pos_cap, neg_cap, triplet_loss, dist_fn_opt):
    loss = 0.0
    if dist_fn_opt == "euclidean":
        loss = loss + triplet_loss(img, pos_img, neg_img)
        loss = loss + triplet_loss(cap, pos_cap, neg_cap)

        loss = loss + triplet_loss(img, pos_cap, neg_cap)
        loss = loss + triplet_loss(img, pos_cap, neg_img)
        loss = loss + triplet_loss(img, pos_img, neg_cap)
        loss = loss + triplet_loss(cap, pos_img, neg_img)
        loss = loss + triplet_loss(cap, pos_img, neg_cap)
        loss = loss + triplet_loss(cap, pos_cap, neg_img)

        loss = loss + triplet_loss(pos_cap, pos_img, neg_cap)
        loss = loss + triplet_loss(pos_cap, pos_img, neg_img)
        loss = loss + triplet_loss(cap, img, neg_cap)
        loss = loss + triplet_loss(cap, img, neg_img)
    elif dist_fn_opt == "cosine":
        same = torch.Tensor(img.size(0)).fill_(1).cuda()
        diff = torch.Tensor(img.size(0)).fill_(-1).cuda()
        loss = loss + triplet_loss(img, pos_img, same)
        loss = loss + triplet_loss(cap, pos_cap, same)
        loss = loss + triplet_loss(img, cap, same)
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


