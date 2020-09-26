# setup
import torch
import torch.nn as nn


# define double VAE network used for finding transformed variables

class DoubleVAE(nn.Module):
    def __init__(self):
        super(DoubleVAE, self).__init__()
        self.xlayer1 = nn.Linear(1, 32)
        self.xlayer31 = nn.Linear(32, 1)
        self.xlayer32 = nn.Linear(32, 1)
        self.xlayer4 = nn.Linear(1, 32)
        self.xlayer6 = nn.Linear(32, 1)
        self.ylayer1 = nn.Linear(1, 32)
        self.ylayer31 = nn.Linear(32, 1)
        self.ylayer32 = nn.Linear(32, 1)
        self.ylayer4 = nn.Linear(1, 32)
        self.ylayer6 = nn.Linear(32, 1)
        nn.init.xavier_normal_(self.xlayer1.weight)
        nn.init.xavier_normal_(self.xlayer4.weight)
        nn.init.xavier_normal_(self.ylayer1.weight)
        nn.init.xavier_normal_(self.ylayer4.weight)
        self.layer1 = nn.Linear(1, 1)

    def xencode(self, x):
        h2 = torch.tanh(self.xlayer1(x))
        x_trans = self.xlayer31(h2)
        x_logvar = self.xlayer32(h2)
        return x_trans, x_logvar

    def yencode(self, y):
        h2 = torch.tanh(self.ylayer1(y))
        y_trans = self.ylayer31(h2)
        y_logvar = self.ylayer32(h2)
        return y_trans, y_logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def xdecode(self, x_trans):
        x_trans = self.xlayer6(torch.tanh(self.xlayer4(x_trans)))
        return x_trans

    def ydecode(self, y_trans):
        y_pred = self.ylayer6(torch.tanh(self.ylayer4(y_trans)))
        return y_pred

    def xyforward(self, xy):
        x = xy[:, 0].reshape(-1, 1)
        y = xy[:, 1].reshape(-1, 1)
        x_trans, x_logvar = self.xencode(x)
        y_trans, y_logvar = self.yencode(y)
        x_z = self.reparameterize(x_trans, x_logvar)
        y_z = self.reparameterize(y_trans, y_logvar)
        x_pred = self.xdecode(x_z)
        y_pred = self.ydecode(y_z)
        return x_trans, y_trans, torch.cat((x_pred, y_pred), axis=1), x_logvar, y_logvar

    def forward(self, xy):
        xy_out = self.xyforward(xy)
        y_trans_pred = self.layer1(xy_out[0])
        return xy_out, y_trans_pred
