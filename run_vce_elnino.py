### run VCE on climate data

# setup
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from src.vce import VCE, loss_function
from src.vce_monitor_losses import monitor_losses
from src.vce_ev_scores import print_ev_scores
import joblib
import pickle



np.random.seed(1423)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load data
Xnp, Ynp, coords = joblib.load('data/ElNino/elnino_data.pkl')

# normalize
x_scaler = StandardScaler().fit(Xnp)
y_scaler = StandardScaler().fit(Ynp)
X = torch.from_numpy(x_scaler.transform(Xnp)).float().to(device)
Y = torch.from_numpy(y_scaler.transform(Ynp)).float().to(device)

# random partition
shuffled_ids = np.random.permutation(X.shape[0])
trainset = shuffled_ids[:int(X.shape[0]*.9)]
testset = shuffled_ids[int(X.shape[0]*.9):]
joblib.dump((testset,trainset), 'data/ElNino/test+trainset.pkl')

X_tr = X[trainset]
Y_tr = Y[trainset]
X_ts = X[testset]
Y_ts = Y[testset]






################################################################################
#####     Define Training
################################################################################



def train(epochs, g, b):
    gamma = pow(10,(-g))
    beta = pow(10,(-b))
    
    #best loss init
    best_loss = np.inf
    best_loss_epoch = 0
                    
    for epoch in range(epochs):
        permutation = torch.randperm(X_tr.size()[0])
        
        for i in range(0, X_tr.size()[0], 128):                  # minibatch loop
            indices = permutation[i:i+128]
            X_batch = X_tr[indices]
            Y_batch = Y_tr[indices]
            
            optimizer16.zero_grad()
            xout, yout = vce16((X_batch, Y_batch))
            Y_pred, xmu, xlogvar, _, Y_bottle_pred = xout
            X_pred, ymu, ylogvar, _, X_bottle_pred = yout
            
            lossX_vae = loss_function(Y_pred, Y_batch, xmu, xlogvar, beta)
            lossX_bottle = F.mse_loss(Y_bottle_pred, ymu)
            lossX = lossX_vae + gamma * lossX_bottle            # loss specification
            
            lossY_vae = loss_function(X_pred, X_batch, ymu, ylogvar, beta)
            lossY_bottle = F.mse_loss(X_bottle_pred, xmu)
            lossY = lossY_vae + gamma * lossY_bottle            # loss specification
            
            lossXY = lossX + lossY
            lossXY.backward()
            optimizer16.step()
            
        
        # monitor losses
        if epoch % 10 == 9:
            #calculate training and test set outputs for losses
            xout_ts, yout_ts = vce16((X_ts, Y_ts))        
            xout_tr, yout_tr = vce16((X_tr, Y_tr))
            #append losses and print progress
            val_loss_all = monitor_losses(epoch, beta, gamma, best_loss, X_tr, X_ts, Y_tr, Y_ts,
                                         xout_ts, yout_ts, xout_tr, yout_tr, lossesX, lossesY, TlossesX, TlossesY)
            #save net for early stopping
            if val_loss_all < best_loss:
                torch.save(vce16.state_dict(), f'results/ElNino/vce_nn_params/XYg{g}b{b}.pth')
                best_loss = val_loss_all
                best_loss_epoch = epoch
                
    #load best loss net and save activations and noise std
    vce16.load_state_dict(torch.load(f'results/ElNino/vce_nn_params/XYg{g}b{b}.pth'))
    xout, yout = vce16((X,Y))
    _, X_bottle, logvarX, _, _ = xout
    _, Y_bottle, logvarY, _, _ = yout
    #save variable activations
    joblib.dump(X_bottle.detach().cpu().numpy(), f'results/ElNino/variables/XtoYg{g}b{b}.pkl')
    joblib.dump(Y_bottle.detach().cpu().numpy(), f'results/ElNino/variables/YtoXg{g}b{b}.pkl')
    #save noise stds
    stdX = np.exp(0.5*logvarX.detach().cpu().numpy())
    stdY = np.exp(0.5*logvarY.detach().cpu().numpy())
    joblib.dump(stdX, f'results/ElNino/vce_stds/XtoYg{g}b{b}.pkl')
    joblib.dump(stdY, f'results/ElNino/vce_stds/YtoXg{g}b{b}.pkl')
    
    print('best loss: ', round(best_loss,2), ' in epoch ', best_loss_epoch, \
          "   |X|="+str(sum(np.mean(stdX, axis=0) < .95)), 
          "|Y|="+str(sum(np.mean(stdY, axis=0) < .95)))







################################################################################
#####     Execute Training
################################################################################



# run for different gamma (=10^-g) and beta (=10^-b)

if __name__ == "__main__":
    for g in (0, 1, 2):
        for b in (0, 1, 2):
            ### initialize...
            #...nets
            vce16 = VCE(X.size()[1], Y.size()[1], 265, 16, 512, noisy=True).to(device)
            #...optimizers
            optimizer16 = torch.optim.Adam(vce16.parameters(), lr= 1e-4, weight_decay = 1e-4)
            #...loss lists
            lossesX = [[],[],[],[]]
            lossesY = [[],[],[],[]]
            TlossesX = [[],[],[],[]]
            TlossesY = [[],[],[],[]]
            
            #### train nets
            print('g', g, ' b', b)
            train(1500, g, b)
            
            ### save losses
            with open(f'results/ElNino/vce_losses/XtoYg{g}b{b}.py', 'wb') as fp:
                pickle.dump(lossesX, fp)
            with open(f'results/ElNino/vce_losses/YtoXg{g}b{b}.py', 'wb') as fp:
                pickle.dump(lossesY, fp)
            with open(f'results/ElNino/vce_losses/trXtoYg{g}b{b}.py', 'wb') as fp:
                pickle.dump(TlossesX, fp)
            with open(f'results/ElNino/vce_losses/trYtoXg{g}b{b}.py', 'wb') as fp:
                pickle.dump(TlossesY, fp)
            


################################################################################
#####     Analyse trained VCEs in terms of EV scores and bottleneck size
################################################################################


            ### explained variance scores without noise
            vce16 = VCE(X.size()[1], Y.size()[1], 265, 16, 512, noisy=False).to(device)
            vce16.load_state_dict(torch.load(f'results/ElNino/vce_nn_params/XYg{g}b{b}.pth', map_location=device))
            stdX = np.mean(joblib.load(f'results/Elnino/vce_stds/XtoYg{g}b{b}.pkl'), axis=0)
            stdY = np.mean(joblib.load(f'results/Elnino/vce_stds/YtoXg{g}b{b}.pkl'), axis=0)
            xout, yout = vce16((X_ts,Y_ts))
            print_ev_scores(b, g, Xnp, Ynp, xout, yout, stdX, stdY, testset, x_scaler, y_scaler)

print('Done') 



### EV scores for microstate and bottleneck prediction, bottleneck size

# g=0 b=0  X:  0.70    bottle:  0.82
# g=0 b=0  Y: -0.01    bottle:  0.43    |X|=3 |Y|=2
# g=0 b=1  X:  0.83    bottle:  0.82
# g=0 b=1  Y:  0.52    bottle:  0.85    |X|=4 |Y|=5 ###
# g=0 b=2  X:  0.86    bottle:  0.88
# g=0 b=2  Y:  0.64    bottle:  0.88    |X|=7 |Y|=7 ##
# g=1 b=0  X:  0.71    bottle: -0.92
# g=1 b=0  Y:  0.15    bottle: -0.96    |X|=3 |Y|=3
# g=1 b=1  X:  0.85    bottle: -0.03
# g=1 b=1  Y:  0.67    bottle:  0.41    |X|=6 |Y|=9
# g=1 b=2  X:  0.90    bottle:  0.46
# g=1 b=2  Y:  0.76    bottle:  0.46    |X|=13 |Y|=13 #
# g=2 b=0  X:  0.71    bottle: -5.51
# g=2 b=0  Y:  0.21    bottle: -4.99    |X|=3 |Y|=4
# g=2 b=1  X:  0.86    bottle: -5.24
# g=2 b=1  Y:  0.69    bottle: -3.45    |X|=7 |Y|=10
# g=2 b=2  X:  0.90    bottle: -0.96
# g=2 b=2  Y:  0.78    bottle: -0.94    |X|=16 |Y|=16