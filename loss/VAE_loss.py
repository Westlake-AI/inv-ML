import torch
import torch.nn as nn


class VAEloss(object):
    """ VAE
    Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    paper: https://arxiv.org/abs/1312.6114
    """
    def __init__(self,):
        pass

    def SetEpoch(self, e):
        pass

    def CalLosses(self, inputs):
        x, mu, logvar, recon_x, last_3d = inputs
        criterion = nn.MSELoss().cuda()
        BCE = criterion(x, recon_x)

        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return [BCE, KLD]
