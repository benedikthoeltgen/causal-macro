# setup
import numpy as np
import joblib

np.random.seed(1423)


# get list with 32 entries that sum to specified shift size
def distributed_list(shift, size):
    shifts = []
    for i in range(size - 1):
        shifts.append(shift / 20 * np.random.randn() + shift)
    shifts.append(shift * size - sum(shifts))
    return shifts


def distribute_topbottom(array, shift_sum, half):
    if half not in ['top','bottom']:
        raise Exception('half param must be either top or bottom')
    hop = 0 if half == 'top' else 4
    shifts = distributed_list(shift_sum, 32)
    for i in range(8):
        for j in range(4):
            array[i, j + hop] += shifts[i + 8 * j]
    return array


def distribute_rightleft(array, shift_sum, half):
    if half not in ['left','right']:
        raise Exception('half param must be either right or left')
    hop = 0 if half == 'left' else 4
    shifts = distributed_list(shift_sum, 32)
    for i in range(4):
        for j in range(8):
            array[i + hop, j] += shifts[i + 4 * j]
    return array


def generate_simulated_data(size=10000):
    _X, _Y, _var = [], [], []

    # generate 10000 samples
    for n in range(size):
        # C; X1 pre-noise
        C = np.random.uniform(-1, 1)
        X1_clean = C + np.random.uniform(-.2, .2)

        # X2 pre-noise
        X2_clean = np.random.uniform(-1, 1)

        # build X
        X = np.random.uniform(-.2, .2, (8, 8))
        X = distribute_rightleft(X, X1_clean, 'left')
        X = distribute_rightleft(X, X2_clean, 'right')

        # calculate X1, X2
        X1 = X[:4, :4].mean()
        X2 = X[4:, :].mean()

        # build Y
        Y1_clean = C ** 3 + np.random.uniform(-.2, .2)
        Y2_clean = np.tanh(X2) + np.random.uniform(-.2, .2)
        Y = np.random.uniform(-.2, .2, (8, 8))
        Y = distribute_topbottom(Y, Y1_clean, 'bottom')
        Y = distribute_topbottom(Y, Y2_clean, 'top')

        # calculate Y1, Y2
        Y1 = Y[:4, 4:].mean()
        Y2 = Y[:, :4].mean()

        # convert into 1-dim vectors
        X = np.reshape(X, (64,))
        Y = np.reshape(Y, (64,))

        # append to lists
        _X.append(X)
        _Y.append(Y)
        _var.append([X1, X2, Y1, Y2, C])

    return np.array(_X), np.array(_Y), np.array(_var)


def save_data(X, Y, var):
    # save X, Y, and variables
    joblib.dump(np.array(X), 'X_data.pkl')
    joblib.dump(np.array(Y), 'Y_data.pkl')
    joblib.dump(np.array(var), 'true_variables.pkl')
