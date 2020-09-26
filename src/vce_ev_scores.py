# setup
from sklearn.metrics import explained_variance_score


# calculate and print EV scores for both microstate and bottleneck prediction as well as bottleneck size

def print_ev_scores(b, g, Xnp, Ynp, xout, yout, stdX, stdY, testset, x_scaler, y_scaler):
    Y_pred, X_bottle, _, _, Y_bottle_pred = xout
    X_pred, Y_bottle, _, _, X_bottle_pred = yout

    X_pred_raw = x_scaler.inverse_transform(X_pred.detach().numpy())
    Y_pred_raw = y_scaler.inverse_transform(Y_pred.detach().numpy())

    X_ts_raw = Xnp[testset]
    Y_ts_raw = Ynp[testset]

    # determine variables in the bottlenecks (the neurons that carry information)
    X_var = X_bottle.detach()[:, stdX < .95]
    Y_var = Y_bottle.detach()[:, stdY < .95]
    X_var_pred = X_bottle_pred.detach().numpy()[:, stdX < .95]  # Mean stds above .95 indicate the noise is pure noise.
    Y_var_pred = Y_bottle_pred.detach().numpy()[:, stdY < .95]  # Value is arbitrary. Another option would be .99.

    ev_y_raw = round(explained_variance_score(Y_pred_raw, Y_ts_raw, multioutput='variance_weighted'), 2)
    ev_x_raw = round(explained_variance_score(X_pred_raw, X_ts_raw, multioutput='variance_weighted'), 2)
    out_string_prepend = f"g={str(g)}, b= {str(b)}"

    if sum(stdX < .95) != 0 and sum(stdY < .95) != 0:  # only if bottlenecks nonempty
        print(f"{out_string_prepend} X: {ev_y_raw}, Bottleneck: "
              f"{round(explained_variance_score(Y_var_pred, Y_var, multioutput='variance_weighted'), 2)}")

        print(f"{out_string_prepend} X: {ev_x_raw}, Bottleneck: "
              f"{round(explained_variance_score(X_var_pred, X_var, multioutput='variance_weighted'), 2)}"
              f", |X|={str(sum(stdX < .95))}, |Y|={str(sum(stdY < .95))}")
    else:
        print(f"{out_string_prepend} X: {ev_y_raw}, Bottleneck: Empty")
        print(f"{out_string_prepend} Y: {ev_x_raw}, Bottleneck: Empty")
