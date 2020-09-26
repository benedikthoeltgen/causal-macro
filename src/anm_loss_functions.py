# setup
import torch
import torch.nn.functional as F
from src.anm_hsic_torch import hsic_gam as hsic_torch


# different parts of the loss function needed for the double VAE

def xy_mse_loss(xy_out, XY_batch):
    _, _, xy_pred, _, _ = xy_out
    MSE = F.mse_loss(xy_pred, XY_batch, reduction='mean')
    return MSE


def xy_KLD_loss(xy_out):
    x_trans, y_trans, _, x_logvar, y_logvar = xy_out
    KLDx = -0.5 * torch.mean(1 + x_logvar - x_trans.pow(2) - x_logvar.exp())
    KLDy = -0.5 * torch.mean(1 + y_logvar - y_trans.pow(2) - y_logvar.exp())
    return KLDx + KLDy


def conn_mse_loss(xy_out, y_trans_pred):
    _, y_trans, _, _, _ = xy_out
    return F.mse_loss(y_trans, y_trans_pred, reduction='mean') / torch.var(y_trans)


def conn_hsic_loss(xy_out, y_trans_pred):
    x_trans, y_trans, _, _, _ = xy_out
    y_res = y_trans - y_trans_pred
    return hsic_torch(x_trans, y_res)
