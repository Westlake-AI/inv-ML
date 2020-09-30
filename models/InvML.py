import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class InvLeakyReLU(nn.Module):
    """ Invertible Bi-LeakyReLU with alpha """
    def __init__(self, alpha=2, inplace=False, invertible=False):
        super(InvLeakyReLU, self).__init__()
        self.alpha = alpha
        self.inplace = inplace
        self.invertible = invertible
        self.weight = None

    def set_invertible(self, invertible=False):
        # print('set invertible in LeakyReLU', invertible)
        self.invertible = invertible

    def forward(self, x):
        if self.invertible == False:
            return torch.max(1 / self.alpha * x, self.alpha * x)
        else:
            return torch.min(1 / self.alpha * x, self.alpha * x)
    
    def extra_repr(self):
        inplace_str = ', inplace=True' if self.inplace else ''
        return 'alpha_inplace={}{}'.format(self.negative_slope, inplace_str)


class InvML_MLP(nn.Module):
    """ inv-ML MLP baseline """

    def __init__(self, args, mode='encoder', device="cuda"):

        super().__init__()
        self.args = args
        self.Structure = args['NetworkStructure']  # MLP
        self.epoch = 0
        self.device = device
        # 1. network
        self.name_list = ['input']  # input, encoder layers, decoder layers
        self.plot_index_list = [0]  # index of layers for plot
        self.network = nn.ModuleList()
        self.layer_num = len(args['NetworkStructure']['layer']) - 1
        self.extraHead = nn.ModuleList()
        self.extraLayer = self.args['ExtraHead']['layer']
        # 2. add inv relue
        if args['ratio']['orth'] > 0:  # add inv to leaky
            self.inv_leaky = True
        else:
            self.inv_leaky = False
        # 3. Enc, Dec weight index
        self.enc_weight_index = []
        self.dec_weight_index = []
        # Extra-Head index
        self.extra_index = []
        # 4. mode: ['encoder', 'decoder']
        self.mode = mode
        # 5. jump link for Decoder
        self.add_jump = self.args["add_jump"]
        # 6. zero padding mode
        if args["InverseMode"]["mode"] == "ZeroPadding":
            self.zero_padding = args["InverseMode"]["padding"]
            tmp_padding = []
        else:
            self.zero_padding = []

        # Encoder
        for i in range(len(self.Structure["layer"])-1):
            
            self.network.append(
                nn.Linear(self.Structure["layer"][i], self.Structure["layer"][i+1], bias=False)
            )
            self.name_list.append('EN{}_{}->{}'.format(i+1,
                self.Structure["layer"][i], self.Structure["layer"][i+1]))
            self.enc_weight_index.append(len(self.name_list) - 2) # add weight index
            # 1) set required_grad
            if self.Structure["Enc_require_gard"][i] == 0:
                self.network[-1].weight.requires_grad = False
            
            # 2) add Extra-Head (project) for DR
            if len(self.extraLayer) > 0:
                if self.extraLayer[i] > 0:
                    self.extraHead.append(
                        nn.Linear(self.Structure["layer"][i], self.extraLayer[i], bias=False)
                    )
                    self.extra_index.append( {"index":len(self.network)-1, "extra":len(self.extra_index)} )
            # 3) add weight padding
            if len(self.zero_padding) > 0:
                tmp_padding.append(self.zero_padding[i])
            
            # 4) add relu
            if self.Structure["relu"][i] > 0:
                if self.args['ReluType']["type"] == "Leaky":
                    self.network.append(nn.LeakyReLU(self.args['ReluType']["Enc_alpha"], inplace=False))
                elif self.args['ReluType']["type"] == "InvLeaky":
                    self.network.append(
                        InvLeakyReLU(self.args['ReluType']["Enc_alpha"] , inplace=False, invertible=False)
                    )
                else:
                    raise NotImplementedError
                self.name_list.append('ENRELU{}_{}->{}'.format(
                    i+1, self.Structure["layer"][i], self.Structure["layer"][i+1]))
                # add relu padding
                if len(self.zero_padding) > 0:
                    tmp_padding.append(self.zero_padding[i])
            self.plot_index_list.append(len(self.name_list) - 1)
        
        # Decoder (not sharing params with Encoder)
        for i in range(len(self.Structure["layer"])-1, 0, -1):  # 10 -> 1
            if self.Structure["inv_Dec"] == 0:
                
                self.network.append(
                    nn.Linear(self.Structure["layer"][i], self.Structure["layer"][i-1], bias=False)
                )
                self.name_list.append('DE{}*_{}->{}'.format(
                    i-1, self.Structure["layer"][i], self.Structure["layer"][i-1]))
                self.dec_weight_index.append(len(self.name_list) - 2) # add weight index
                # set required_grad
                if self.Structure["Dec_require_gard"][i-1] == 0:
                    self.network[-1].weight.requires_grad = False
                
                # add relu
                if self.Structure["relu"][i-1] > 0:
                    if self.args['ReluType']["type"] == "Leaky":
                        self.network.append(nn.LeakyReLU(self.args['ReluType']["Dec_alpha"]))
                    elif self.args['ReluType']["type"] == "InvLeaky":
                        self.network.append(
                            InvLeakyReLU(self.args['ReluType']["Enc_alpha"], inplace=False, invertible=self.inv_leaky) # add inv
                        )
                    else:
                        raise NotImplementedError
                    self.name_list.append('DERELU{}*_{}->{}'.format(
                        i-1, self.Structure["layer"][i], self.Structure["layer"][i-1]))
            else:
                # add relu
                if self.Structure["relu"][i-1] > 0:
                    if self.args['ReluType']["type"] == "Leaky":
                        self.network.append(nn.LeakyReLU(self.args['ReluType']["Dec_alpha"]))
                    elif self.args['ReluType']["type"] == "InvLeaky":
                        self.network.append(
                            InvLeakyReLU(self.args['ReluType']["Enc_alpha"], inplace=False, invertible=self.inv_leaky) # add inv
                        )
                    else:
                        raise NotImplementedError
                    self.name_list.append('DERELU{}*_{}->{}'.format(
                        i-1, self.Structure["layer"][i], self.Structure["layer"][i-1]))
                # set Linear
                self.network.append(
                    nn.Linear(self.Structure["layer"][i], self.Structure["layer"][i-1], bias=False)
                )
                self.name_list.append('DE{}*_{}->{}'.format(
                    i-1, self.Structure["layer"][i], self.Structure["layer"][i-1]))
                self.dec_weight_index.append(len(self.name_list) - 2) # add weight index
                # set required_grad
                if self.Structure["Dec_require_gard"][i-1] == 0:
                    self.network[-1].weight.requires_grad = False
            # add plot
            self.plot_index_list.append(len(self.name_list)-1)
        # print(self.network)

        # padding for Enc and Dec
        if len(self.zero_padding) > 0:
            latent_pad = self.zero_padding[-1]
            self.zero_padding = []
            for t in tmp_padding:
                self.zero_padding.append(t)
            #self.zero_padding.append(latent_pad)
            for i in range(len(tmp_padding)-1, -1, -1):
                self.zero_padding.append(tmp_padding[i])
    
    def SetEpoch(self, epoch):
        self.epoch = epoch
    
    def SetMode(self, mode):
        self.mode = mode
    
    def params_transfer(self):
        """ transfer Enc's params to Dec, when Dec is forzen """

        for i in range(self.layer_num):
            enc_i = self.enc_weight_index[i]
            dec_i = self.dec_weight_index[-i-1] if i > 0 else self.dec_weight_index[-1]
            
            row, col = self.network[dec_i].weight.shape
            if row == col:
                #self.network[dec_i].weight = Parameter(self.network[enc_i].weight.t() )  # transpose
                self.network[dec_i].weight = Parameter(torch.inverse(self.network[enc_i].weight) )  # inverse
            else:
                self.network[dec_i].weight = Parameter(torch.pinverse(self.network[enc_i].weight) )  # fake inverse
            # print('weight enc={}, dec={}'.format(self.network[enc_i].weight.shape, self.network[dec_i].weight.shape))
            if self.Structure["Dec_require_gard"][i] == 0:
                self.network[dec_i].weight.requires_grad = False

    def GetGradual(self, step=[0, 0, 0]):
        """ gradual change loss weight [ascendant, descendant] """
        if step[2] > 0:  # 0 -> 1
            gradual = 0
            if self.epoch >= step[0]:
                if self.epoch < step[1]:
                    gap = step[1] - step[0]
                    gradual = (self.epoch - step[0]) / gap
                else:
                    gradual = 1
        else:  # 1 -> 0
            gradual = 1
            if self.epoch >= step[0]:
                if self.epoch < step[1]:
                    gap = step[1] - step[0]
                    gradual -= (self.epoch - step[0]) / gap
                else:
                    gradual = 0
        return gradual

    def decode_forward(self, input_data):
        """ only run decoder """
        input_c = input_data
        tmp_res = None
        for i, layer in enumerate(self.network):
            input_c = torch.tensor(input_c, dtype=torch.float32).to(self.device)
            # 1. normal
            output_c = layer(input_c) # for add jump
            
            # 2. jump (Dec)
            try:
                w_shape = layer.weight.shape
                if w_shape[0] < w_shape[1]:
                    # add interpolation
                    tmp_res = torch.zeros(15, input_c.shape[1]).to(self.device)
                    tmp_res[0] = input_c[0]
                    tmp_res[1] = input_c[0] * 0.94 + input_c[1] * 0.06
                    tmp_res[2] = input_c[0] * 0.88 + input_c[1] * 0.12
                    tmp_res[3] = input_c[0] * 0.82 + input_c[1] * 0.18
                    tmp_res[4] = input_c[0] * 0.76 + input_c[1] * 0.24
                    tmp_res[5] = input_c[0] * 0.70 + input_c[1] * 0.30
                    tmp_res[6] = input_c[0] * 0.65 + input_c[1] * 0.35
                    # mid
                    tmp_res[7] = input_c[0] * 0.5 + input_c[1] * 0.5
                    # next id
                    tmp_res[8] = input_c[0] * 0.35 + input_c[1] * 0.65
                    tmp_res[9] = input_c[0] * 0.30 + input_c[1] * 0.70
                    tmp_res[10] = input_c[0] * 0.24 + input_c[1] * 0.76
                    tmp_res[11] = input_c[0] * 0.18 + input_c[1] * 0.82
                    tmp_res[12] = input_c[0] * 0.12 + input_c[1] * 0.88
                    tmp_res[13] = input_c[0] * 0.06 + input_c[1] * 0.94
                    tmp_res[14] = input_c[1]
                    tmp_res = torch.tensor(tmp_res, dtype=torch.float32).to(self.device)
                elif w_shape[0] > w_shape[1]:
                    output_c = tmp_res  # only for 'decoder'
            except:
                pass

            input_c = output_c
        return output_c

    def forward(self, input_data):
        """ baseline forward """
        # interplotation task
        if self.args["Task"] == 'interplotation':
            return self.decode_forward(input_data)

        input_data = input_data.view(input_data.shape[0], -1)
        input_c = input_data
        
        # info
        output_info = [input_data, ]  # store input and each layer output
        grad_info = [None]  # store requires_grad
        extra_info = []
        pad_info = []
        # extra
        tmp_res = None
        extra_i = 0 # extra head
        use_jump = self.mode == 'decoder' or self.args["InverseMode"]["mode"] == "ZeroPadding"

        for i, layer in enumerate(self.network):
            # 1. normal
            output_c = layer(input_c) # for add jump
            # save output before padding
            pad_info.append(output_c)

            # 2. zero padding, use for CS
            if len(self.zero_padding) > 0:
                if i < len(self.zero_padding):
                    pad_bool = False
                    if self.zero_padding[i] > 0 and self.zero_padding[i] < output_c.shape[1]:
                        if self.mode == 'encoder' and i < self.layer_num * 2 - 1:  # padding latent, not decoder
                            pad_bool = True
                        elif self.mode == 'decoder' and i < self.layer_num * 2 - 2:  # not for decoder
                            pad_bool = True
                    
                    if pad_bool:
                        # if gradual_pad:
                        #     pad_num = int(self.zero_padding[i] * gradual_padding)
                        #     if pad_num <= 0:
                        #         pad_num = 1
                        # else:
                        #     pad_num = self.zero_padding[i]
                        pad_num = self.zero_padding[i]
                        pad_shape = (output_c.shape[0], output_c.shape[1] - pad_num)
                        # print("padding shape:", pad_shape)
                        zeros = torch.zeros(pad_shape).to(self.device)
                        # pad output
                        output_c[:, self.zero_padding[i]:] = zeros  # padding with zeros

            # 3. jump (Dec)
            if self.add_jump:  # Dec without DR
                try:
                    grad_info.append(layer.weight.requires_grad)
                    w_shape = layer.weight.shape
                    # if layer.weight.shape == (2, 50) or layer.weight.shape == (3, 784): # prev
                    if w_shape[0] < w_shape[1] // 2:
                        tmp_res = input_c
                        # print('hit 1', tmp_res.shape)
                    #elif layer.weight.shape == (50, 2) or layer.weight.shape == (784, 3): # prev
                    elif w_shape[0] // 2 > w_shape[1]:
                        output_c = tmp_res  # only for 'decoder'
                except:
                    #print("index {}: relu.".format(i))
                    grad_info.append(None)

            # 4. extra Head (Enc)
            if len(self.extra_index) > 0:
                if extra_i < len(self.extra_index):
                    if self.extra_index[extra_i]["index"] == i:
                        output_e = self.extraHead[ self.extra_index[extra_i]["extra"] ](input_c)
                        extra_info.append(output_e)
                        extra_i += 1

            # normal
            output_info.append(output_c)
            input_c = output_c
        
        # list of each layer output
        return {'output': output_info, 'weight': self.network.parameters(), "grad": grad_info, \
                "extra": extra_info, "padding": pad_info}

