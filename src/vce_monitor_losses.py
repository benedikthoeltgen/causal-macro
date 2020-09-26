# setup
import numpy as np
import torch
import torch.nn.functional as F


# calculate losses, append to loss vector and print progress

def monitor_losses(epoch, beta, gamma, best_loss, X_tr, X_ts, Y_tr, Y_ts,
                   xout_ts, yout_ts, xout_tr, yout_tr, lossesX, lossesY, TlossesX, TlossesY):
    # validation set losses
    Y_pred_ts, X_mu, X_logvar, X_bottle, Y_bottle_pred = xout_ts
    X_pred_ts, Y_mu, Y_logvar, Y_bottle, X_bottle_pred = yout_ts
    val_lossX_MSE = F.mse_loss(Y_pred_ts, Y_ts, reduction='mean').item()
    val_lossX_KLD = -0.5 * torch.mean(1 + X_logvar - X_mu.pow(2) - X_logvar.exp()).item()
    val_lossX_bottle = F.mse_loss(Y_bottle_pred, Y_mu).item()
    val_lossY_MSE = F.mse_loss(X_pred_ts, X_ts, reduction='mean').item()
    val_lossY_KLD = -0.5 * torch.mean(1 + Y_logvar - Y_mu.pow(2) - Y_logvar.exp()).item()
    val_lossY_bottle = F.mse_loss(X_bottle_pred, X_mu).item()
    val_lossX_all = val_lossX_MSE + beta * val_lossX_KLD + gamma * val_lossX_bottle
    val_lossY_all = val_lossY_MSE + beta * val_lossY_KLD + gamma * val_lossX_bottle
    val_loss_all = val_lossX_all + val_lossY_all

    # training set losses
    Y_pred, X_mu, X_logvar, X_bottle, Y_bottle_pred = xout_tr
    X_pred, Y_mu, Y_logvar, Y_bottle, X_bottle_pred = yout_tr
    tr_lossX_MSE = F.mse_loss(Y_pred, Y_tr, reduction='mean').item()
    tr_lossX_KLD = -0.5 * torch.mean(1 + X_logvar - X_mu.pow(2) - X_logvar.exp()).item()
    tr_lossX_bottle = F.mse_loss(Y_bottle_pred, Y_mu).item()
    tr_lossY_MSE = F.mse_loss(X_pred, X_tr, reduction='mean').item()
    tr_lossY_KLD = -0.5 * torch.mean(1 + Y_logvar - Y_mu.pow(2) - Y_logvar.exp()).item()
    tr_lossY_bottle = F.mse_loss(X_bottle_pred, X_mu).item()
    tr_lossX_all = val_lossX_MSE + beta * val_lossX_KLD + gamma * val_lossX_bottle
    tr_lossY_all = val_lossY_MSE + beta * val_lossY_KLD + gamma * val_lossX_bottle

    # print progress: X val component losses, X and Y overall training and val losses
    if epoch % 50 == 49 or epoch == 9:
        stdX = np.exp(0.5 * X_logvar.detach().numpy())
        stdY = np.exp(0.5 * Y_logvar.detach().numpy())
        print(f"ep {epoch}: {round(val_lossX_MSE, 2)} {round(val_lossX_KLD, 2)} {round(val_lossX_bottle, 2)} "
              f"{round(tr_lossX_all, 2)} {round(val_lossX_all, 2)} {round(tr_lossY_all, 2)} {round(val_lossY_all)} "
              f"|X|= {str(sum(np.mean(stdX, axis=0) < .95))} "
              f"|Y|= {str(sum(np.mean(stdY, axis=0) < .95))}")

    # save val losses as [MSE, KLD, bottle, all]
    # X
    lossesX[0].append(val_lossX_MSE)
    lossesX[1].append(val_lossX_KLD)
    lossesX[2].append(val_lossX_bottle)
    lossesX[3].append(val_lossX_all)
    # Y
    lossesY[0].append(val_lossY_MSE)
    lossesY[1].append(val_lossY_KLD)
    lossesY[2].append(val_lossY_bottle)
    lossesY[3].append(val_lossY_all)

    # save train losses as [MSE, KLD, bottle, all]
    # X
    TlossesX[0].append(tr_lossX_MSE)
    TlossesX[1].append(tr_lossX_KLD)
    TlossesX[2].append(tr_lossX_bottle)
    TlossesX[3].append(tr_lossX_all)
    # Y
    TlossesY[0].append(tr_lossY_MSE)
    TlossesY[1].append(tr_lossY_KLD)
    TlossesY[2].append(tr_lossY_bottle)
    TlossesY[3].append(tr_lossY_all)

    return val_loss_all
