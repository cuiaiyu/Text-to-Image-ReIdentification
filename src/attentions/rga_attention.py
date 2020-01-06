from .utils import cos_dist, euclid_dist
import torch.nn.functional as F
import torch

def RGA_attend_one_to_many(full, parts, dist_fn_opt):
    full = full.expand_as(parts)
    if dist_fn_opt == "euclidean":
        dists = - F.pairwise_distance(full, parts)
    else:
        dists = F.cosine_similarity(full, parts)
    weights = F.softmax(dists, 0).view(-1, 1).expand_as(parts)
    return torch.sum(weights * parts, 0, keepdim=True)

def RGA_attend_one_to_many_batch(fulls, parts, dist_fn_opt):
    """
    fulls: NxT
    parts: N x M x T
    """
    if fulls.size() != parts.size():
        fulls = fulls[:,None,:].expand_as(parts)
    dist_fn = euclid_dist if dist_fn_opt == "euclidean" else cos_dist
    dists = - dist_fn(fulls, parts)
    N, M, T = parts.size()
    weights = F.softmax(dists, 1).view(N, M, 1).expand_as(parts)
    return torch.sum(weights * parts, 1)    