### adapted from anm_hsic.py to allow differentiation in pytorch

from __future__ import division
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def rbf_dot(pattern1, pattern2, deg):
    size1 = pattern1.size()
    size2 = pattern2.size()

    G = torch.sum(pattern1 * pattern1, 1).reshape(size1[0], 1).to(device)
    H = torch.sum(pattern2 * pattern2, 1).reshape(size2[0], 1).to(device)

    Q = G.repeat(1, size2[0]).to(device)
    R = H.T.repeat(size1[0], 1).to(device)

    H = Q + R - 2 * torch.matmul(pattern1, pattern2.T)

    H = torch.exp(-H / 2 / (deg ** 2))

    return H


def hsic_gam(X, Y, alph=0.5):
    """
	X, Y are numpy vectors with row - sample, col - dim
	alph is the significance level
	auto choose median to be the kernel width
	"""
    n = X.size()[0]

    # ----- width of X -----
    Xmed = X.to(device)

    G = torch.sum(Xmed * Xmed, 1).reshape(n, 1)
    Q = G.repeat(1, n)
    R = G.T.repeat(n, 1)

    dists = Q + R - 2 * torch.matmul(Xmed, Xmed.T).to(device)
    dists = dists - torch.tril(dists)
    dists = dists.reshape(n ** 2, 1)

    width_x = torch.sqrt(0.5 * torch.median(dists[dists > 0]))

    # ----- width of Y -----
    Ymed = Y.to(device)

    G = torch.sum(Ymed * Ymed, 1).reshape(n, 1)
    Q = G.repeat(1, n)
    R = G.T.repeat(n, 1)

    dists = (Q + R - 2 * torch.matmul(Ymed, Ymed.T))
    dists = dists - torch.tril(dists)
    dists = dists.reshape(n ** 2, 1)

    width_y = torch.sqrt(0.5 * torch.median(dists[dists > 0]))

    # bone = torch.ones((n, 1), dtype = float)
    H = (torch.eye(n).to(device) - torch.ones((n, n), dtype=float).to(device) / n)

    K = rbf_dot(X, X, width_x).to(device).double()
    L = rbf_dot(Y, Y, width_y).to(device).double()

    Kc = torch.matmul(torch.matmul(H, K), H)
    Lc = torch.matmul(torch.matmul(H, L), H)

    testStat = torch.sum(Kc.T * Lc) / n

    return testStat
