# import math
import torch
import torch.nn as nn
# import torch.nn.functional as F


class VAE_MLP(nn.Module):

    def __init__(self, args, full_mc=True, sync_bn=True):
        super().__init__()

        self.full_mc = full_mc
        self.args = args
        self.NetworkStructure = args['NetworkStructure']
        # NetworkStructure = self.NetworkStructure
        self.index_latent = (len(args['NetworkStructure'])-1)*2 - 1

        self.name_list = ['input']
        self.plot_index_list = [0]

        # Encoder
        self.Encoder = nn.Sequential()
        for i in range(len(self.NetworkStructure)-2):
            self.Encoder.add_module(
                'layer{}'.format(i),
                nn.Linear(self.NetworkStructure[i], self.NetworkStructure[i+1])
            )

            if i == len(self.NetworkStructure)-2:
                pass
            else:
                self.Encoder.add_module(
                    'relu{}'.format(i),
                    nn.LeakyReLU()
                )
        # Decoder
        self.Decoder = nn.Sequential()
        for i in range(len(self.NetworkStructure)-1):
            j = len(self.NetworkStructure)-1-i
            self.Decoder.add_module(
                'layerhat{}'.format(j),
                nn.Linear(self.NetworkStructure[j], self.NetworkStructure[j-1])
            )
            if i == len(self.NetworkStructure)-2:
                pass
            else:
                self.Decoder.add_module(
                    'reluhat{}'.format(j),
                    nn.LeakyReLU()
                )
        # latent variable, using Guassian noise
        self.mean_head = nn.Linear(
            self.NetworkStructure[-2], self.NetworkStructure[-1])
        self.var_head = nn.Linear(
            self.NetworkStructure[-2], self.NetworkStructure[-1])

    def reparameterize(self, mu, logvar):
        """random Gaussian noise for latent space"""
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, input_data):

        input_data = input_data.view(input_data.shape[0], -1)
        feature = self.Encoder(input_data)
        mean_latent = self.mean_head(feature)
        var_latent = self.var_head(feature)

        latent = self.reparameterize(mean_latent, var_latent)
        output = self.Decoder(latent)

        output_info = [input_data, mean_latent, var_latent, output, feature]
        self.input_data = input_data
        self.feature = feature
        self.mean_latent = mean_latent
        self.output = output

        return output_info