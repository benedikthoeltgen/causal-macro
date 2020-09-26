# setup
from matplotlib import pyplot as plt
from src.anm_hsic import hsic_gam
from sklearn import linear_model


# make plots, compute and print HSIC scores for ANM analysis

def anm_analysis(regress, data):
    # will test whether first variable (named x here) causes the second (y)
    x = data[0].detach().numpy().reshape(-1, 1)
    y = data[1].detach().numpy().reshape(-1, 1)

    # linear regression
    lm = linear_model.LinearRegression()
    lm.fit(x, y)
    y_pred = lm.predict(x)
    y_res = y - y_pred

    # plots
    plt.scatter(x, y, marker='.', s=.1)  # x vs y plus y-pred
    plt.scatter(x, y_pred, marker='.', s=.1)
    plt.show()
    plt.scatter(x, y_res, marker='.', s=.1)  # x_res vs y_res
    plt.show()

    # hsic
    testScore, alpha = hsic_gam(x, y_res, .05)

    regressed_on = 'y' if regress == 'x' else 'x'
    print(f'{regress} ~ {regressed_on}_res:', round(testScore, 3), round(alpha, 3))
