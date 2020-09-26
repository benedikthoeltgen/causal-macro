
# setup
import numpy as np
import torch
import joblib
from src.anm_double_vae import DoubleVAE
from src.anm_loss_functions import xy_mse_loss, xy_KLD_loss, conn_mse_loss, conn_hsic_loss
from src.anm_monitor_losses import monitor_losses
from src.anm_analysis import anm_analysis
import pickle


# select VCE by hyperparameters (beta = 10^-b, gamma = 10^-g)
g = 'g0'
b = 'b1'


np.random.seed(1423)

# load activations
X_activ = joblib.load('results/ElNino/variables/XtoY'+g+b+'.pkl')
Y_activ = joblib.load('results/ElNino/variables/YtoX'+g+b+'.pkl')






##############################################################
#####     Define Training
##############################################################



def train(epochs, conn_weight, hsic_weight):
    
    #best loss init
    best_loss = np.inf
    best_loss_epoch = 0
    
    hsic_fac = 0
    conn_fac = conn_weight
    
    for epoch in range(epochs):
        permutation = torch.randperm(XY_tr.size()[0])
        hsic_list = []
        
        if epoch == 50:
            hsic_fac = hsic_weight
        
        for i in range(0, XY_tr.size()[0], 256):            # minibatch loop
            indices = permutation[i:i+256]
            XY_batch = XY_tr[indices]
            
            optimizer.zero_grad()
            xy_out, y_trans_pred = vae(XY_batch)
            
            loss_xy_MSE = xy_mse_loss(xy_out, XY_batch)
            loss_xy_KLD = xy_KLD_loss(xy_out)
            loss_conn = conn_mse_loss(xy_out, y_trans_pred)
            loss_hsic = conn_hsic_loss(xy_out, y_trans_pred)
            loss = loss_xy_MSE + loss_xy_KLD*.01 + loss_conn*conn_fac + loss_hsic*hsic_fac   # loss
            
            loss.backward()
            optimizer.step()
            if epoch % 10 == 9:
                hsic_list.append(loss_hsic.item())
      
        ### LOSSES
        if epoch % 10 == 9:
            xy_out_tr, y_trans_pred_tr = vae(XY_tr) 
            xy_out_ts, y_trans_pred_ts = vae(XY_ts) 
            val_loss = monitor_losses(epoch, xy_out_tr, y_trans_pred_tr, xy_out_ts, y_trans_pred_ts, 
                                      XY_tr, XY_ts, hsic_list, conn_weight, hsic_weight, losses, Tlosses)
            
            #early stopping
            if val_loss < best_loss:
                torch.save(vae.state_dict(), 'results/Elnino/anm_vae_training/nn_params_'+xory+g+b+n+'.pth')
                best_loss = val_loss
                best_loss_epoch = epoch
    print('best loss: ', round(best_loss,3), ' in epoch ', best_loss_epoch)
    
    #load best loss net and compute outputs
    vae.load_state_dict(torch.load('results/Elnino/anm_vae_training/nn_params_'+xory+g+b+n+'.pth'))
    xy_out, _ = vae(XY)
    x_trans, y_trans, _, _, _ = xy_out
    #save activations etc.
    joblib.dump((x_trans.detach().numpy(),
                 y_trans.detach().numpy()), 'results/Elnino/transformed_variables/trans_xy_'+xory+g+b+n+'.pkl')
    
    
    
    
    
    


##############################################################
#####     Execute Training
##############################################################


    

for neuron in (8,11,15):
    for xory in ('x', 'y'):

        n = 'n'+str(neuron)

        # activation vectors for chosen neuron
        Xnp = np.transpose(X_activ)[neuron].reshape(-1,1)
        Ynp = np.transpose(Y_activ)[neuron].reshape(-1,1)
        # concatenate and convert
        if xory == 'x':
            XYnp = np.concatenate((Xnp, Ynp), axis=1)
        # concatenate and convert
        if xory == 'y':
            XYnp = np.concatenate((Ynp, Xnp), axis=1)
        XY = torch.Tensor(XYnp)
        print('')

        # initialize net
        vae = DoubleVAE()
        vae.layer1.weight = torch.nn.Parameter(torch.tensor([[1]]).float())
        vae.layer1.bias = torch.nn.Parameter(torch.tensor([[0]]).float())
        vae.layer1.requires_grad_(False)
            
        # randomized training and validation set
        shuffled_ids = np.random.permutation(XY.shape[0])
        trainset = shuffled_ids[:int(XY.shape[0]*.9)]
        testset = shuffled_ids[int(XY.shape[0]*.9):]
        joblib.dump((testset,trainset), 'results/Elnino/anm_vae_training/test+trainset_'+xory+g+b+n+'.pkl')
        XY_tr = XY[trainset]
        XY_ts = XY[testset]
    
        ### initialize
        optimizer = torch.optim.Adam(vae.parameters(), lr= 1e-3)
        losses = [[],[],[],[]]
        Tlosses = [[],[],[]]
        
        ### train
        print(n, xory)
        train(1, conn_weight=1, hsic_weight=1)                                               ### train
        
        ### save losses
        with open('results/ElNino/anm_vae_training/losses_'+xory+g+b+n+'.py', 'wb') as fp:
            pickle.dump(losses, fp)
        with open('results/ElNino/anm_vae_training/losses_'+xory+g+b+n+'_tr.py', 'wb') as fp:
            pickle.dump(Tlosses, fp)
                


##############################################################
#####     Compute HSIC Scores and Plot
##############################################################
        

        vae.load_state_dict(torch.load('results/ElNino/anm_vae_training/nn_params_'+xory+g+b+n+'.pth'))
        xy_out, _ = vae(XY)
        anm_analysis(xory, xy_out)



### n8 x
# x ~ y_res: 2.189 0.608

### n8 y
# y ~ x_res: 4.64 0.61

### n11 x
# x ~ y_res: 1.337 0.597

### n11 y
# y ~ x_res: 0.99 0.59

### n15 x
# x ~ y_res: 0.987 0.563

### n15 y
# y ~ x_res: 1.41 0.56