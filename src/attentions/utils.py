import torch
import torch.nn.functional as F

def euclid_dist(x, y):
    """
    x: N x M x T
    y: N x M x T
    """
    return torch.sqrt((x - y) ** 2).sum(2)


def cos_dist(x,y):
    N, M, T = x.size()
    # import pdb; pdb.set_trace()
    fenzi = torch.sum(x.reshape(N*M, T) * y.reshape(N*M, T), 1)
    x_norm = torch.norm(x.reshape(N*M, T), dim=1)
    y_norm = torch.norm(y.reshape(N*M, T), dim=1)
    s = 1 -  fenzi / (x_norm * y_norm)
    #s = F.cosine_similarity(x.reshape(N*M, T), y.reshape(N*M, T))
    return s.view(N, M)