# setup
from src.anm_loss_functions import xy_mse_loss, xy_KLD_loss, conn_mse_loss


# calculate losses, append to loss vector and print progress

def monitor_losses(epoch, xy_out_tr, y_trans_pred_tr, xy_out_ts, y_trans_pred_ts, XY_tr, XY_ts, hsic_list,
                   conn_weight, hsic_weight, losses, Tlosses):
    # test set
    val_loss_xy_MSE = xy_mse_loss(xy_out_ts, XY_ts).item()
    val_loss_xy_KLD = xy_KLD_loss(xy_out_ts).item()
    val_loss_conn = conn_mse_loss(xy_out_ts, y_trans_pred_ts).item()
    avg_hsic = sum(hsic_list) / len(hsic_list)
    val_loss = val_loss_xy_MSE + val_loss_xy_KLD * .01 + val_loss_conn * conn_weight + avg_hsic * hsic_weight  # val loss
    # training set
    tr_loss_xy_MSE = xy_mse_loss(xy_out_tr, XY_tr).item()
    tr_loss_xy_KLD = xy_KLD_loss(xy_out_tr).item()
    tr_loss_conn = conn_mse_loss(xy_out_tr, y_trans_pred_tr).item()

    # print progress: val losses, training losses
    if epoch % 50 == 49 or epoch == 9:
        print(f"ep {epoch}: xy: {round(val_loss_xy_MSE, 3)} {round(val_loss_xy_KLD, 2)} "
              f"conn: {round(val_loss_conn, 3)} {round(avg_hsic, 2)}")

    # list validation set losses
    losses[0].append(val_loss_xy_MSE)
    losses[1].append(val_loss_xy_KLD)
    losses[2].append(val_loss_conn)
    losses[3].append(avg_hsic)

    # list training set losses
    Tlosses[0].append(tr_loss_xy_MSE)
    Tlosses[1].append(tr_loss_xy_KLD)
    Tlosses[2].append(tr_loss_conn)

    return val_loss
