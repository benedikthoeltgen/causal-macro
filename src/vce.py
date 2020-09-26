import torch
import torch.nn as nn
import torch.nn.functional as F


class VCE(nn.Module):
    def __init__(self, X_size, Y_size, h1_size, h2_size, h3_size, noisy):
        self.noisy = noisy
        h3_part_size = int(h3_size / h2_size)
        super(VCE, self).__init__()
        self.xfc1 = nn.Linear(X_size, h1_size)
        nn.init.xavier_normal_(self.xfc1.weight)
        self.xfc21 = nn.Linear(h1_size, h2_size)
        self.xfc22 = nn.Linear(h1_size, h2_size)
        self.xfc3 = nn.ModuleList()
        self.xadd = nn.ModuleList()
        for i in range(h2_size):
            self.xfc3.append(nn.Linear(1, h3_part_size))
            nn.init.xavier_normal_(self.xfc3[i].weight)
            self.xadd.append(nn.Linear(1, 1))
        self.xfc4 = nn.Linear(h3_size, Y_size)

        self.yfc1 = nn.Linear(Y_size, h1_size)
        nn.init.xavier_normal_(self.yfc1.weight)
        self.yfc21 = nn.Linear(h1_size, h2_size)
        self.yfc22 = nn.Linear(h1_size, h2_size)
        self.yfc3 = nn.ModuleList()
        self.yadd = nn.ModuleList()
        for i in range(h2_size):
            self.yfc3.append(nn.Linear(1, h3_part_size))
            nn.init.xavier_normal_(self.yfc3[i].weight)
            self.yadd.append(nn.Linear(1, 1))
        self.yfc4 = nn.Linear(h3_size, X_size)

    def xencode(self, x):
        h1 = torch.tanh(self.xfc1(x))
        return self.xfc21(h1), self.xfc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def xdecode(self, z):
        z_list = [torch.reshape(z[:, 0], (-1, 1))]
        h3 = torch.tanh(self.xfc3[0](z_list[0]))
        z_pred = self.xadd[0](z_list[0])
        for i in range(1, z.size()[1]):
            z_list.append(torch.reshape(z[:, i], (-1, 1)))
            h3 = torch.cat((h3, torch.tanh(self.xfc3[i](z_list[i]))), axis=1)
            z_pred = torch.cat((z_pred, self.xadd[i](z_list[i])), axis=1)
        return self.xfc4(h3), z_pred

    def yencode(self, x):
        h1 = torch.tanh(self.yfc1(x))
        return self.yfc21(h1), self.yfc22(h1)

    def ydecode(self, z):
        z_list = [torch.reshape(z[:, 0], (-1, 1))]
        h3 = torch.tanh(self.yfc3[0](z_list[0]))
        z_pred = self.yadd[0](z_list[0])
        for i in range(1, z.size()[1]):
            z_list.append(torch.reshape(z[:, i], (-1, 1)))
            h3 = torch.cat((h3, torch.tanh(self.yfc3[i](z_list[i]))), axis=1)
            z_pred = torch.cat((z_pred, self.yadd[i](z_list[i])), axis=1)
        return self.yfc4(h3), z_pred

    def forward(self, xy):
        x, y = xy
        xmu, xlogvar = self.xencode(x)
        ymu, ylogvar = self.yencode(y)
        if self.noisy:
            xz = self.reparameterize(xmu, xlogvar)
            yz = self.reparameterize(ymu, ylogvar)
            xy_pred, xz_pred = self.xdecode(xz)
            yy_pred, yz_pred = self.ydecode(yz)
        else:
            xy_pred, xz_pred = self.xdecode(xmu)
            yy_pred, yz_pred = self.ydecode(ymu)
        xout = (xy_pred, xmu, xlogvar, xmu, xz_pred)
        yout = (yy_pred, ymu, ylogvar, ymu, yz_pred)
        return xout, yout


# Reconstruction + KL divergence losses averaged over all elements and batch
def loss_function(recon_x, x, mu, logvar, beta):
    MSE = F.mse_loss(recon_x, x, reduction='mean')
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + beta * KLD
