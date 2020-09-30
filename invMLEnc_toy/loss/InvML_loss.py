import torch
import torch.nn as nn
from torch.autograd import Variable


class InvMLLosses(object):
    """ Inv-ML baseline Loss for MLP """

    def __init__(self, args, k=5, cuda=True):

        self.nk = args['MAEK']
        self.NetworkStructure = args['NetworkStructure']
        self.layer_num = len(args['NetworkStructure']['layer']) - 1
        # latent index
        self.index = {}
        self.index["latent"] = self.layer_num
        self.index["layer"] = args["index"]
        for i,r in enumerate(args['NetworkStructure']["relu"]):
            self.index["latent"] += r
        self.args = args
        self.device = cuda
        self.epoch = 0

        # loss weight
        self.AE_w = args["AEWeight"]
        self.LIS_w = args["LISWeght"]
        self.Orth_w = args["OrthWeight"]
        self.gradual = {"AE":0, "LIS":0, "Orth":0, "pad":0,
                        "push": {"cross_w":0, "enc_w":0, "dec_w":0, "each_w":0, "extra_w":0},
                    }  # update 0716
        # weight sum
        self.w_sum = {'AE':0, 'each':0,
                    'cross':0, 'enc_forward':0, 'dec_forward':0, 'enc_backward':0, 'dec_backward':0,}
        for a in self.AE_w["each"]:
            self.w_sum['AE'] += a
        self.w_sum['AE'] /= len(self.AE_w["each"])
        for k in list(self.LIS_w.keys()):  # use all keys except "_w"
            if k.find("_gradual") == -1 and k.find("_w") == -1:
                for w in self.LIS_w[k]:
                    self.w_sum[k] += w
                self.w_sum[k] /= len(self.LIS_w[k])
        # loss dict
        self.loss = {}
        for k in self.w_sum.keys():
            if self.w_sum[k] > 0 and k != 'AE':
                self.loss["dist"] = 0
                self.loss["push"] = 0
                self.loss["angle"] = 0
                break
        if self.args["ratio"]["orth"] > 0:  # add
            self.loss["orth"] = 0
        # CS padding
        self.CS = self.args["InverseMode"]["mode"] == "CSinverse" or self.args["InverseMode"]["mode"] == "ZeroPadding" \
                    and args["ratio"]["pad"] > 0 and len(args["InverseMode"]["pad_w"]) == len(args["InverseMode"]["padding"])
        if self.CS == True:
            self.InverseMode = self.args["InverseMode"]
            self.loss["pad"] = 0
        if self.w_sum['AE'] > 0:
            self.loss["AE"] = 0

    def SetEpoch(self, epoch):
        self.epoch = epoch

    def ReconstructionLoss(self, pred, target):
        """ MSELoss for reconstruction """
        criterion = nn.MSELoss().cuda()
        loss = criterion(pred, target)
        return loss

    def MorphicLossItem(self, data1, data2):
        """ Isomap Loss (LIS constrain) cross two layers """
        d_data1, kNN_mask_data1 = self.epsilonball(data1)
        d_data2, kNN_mask_data2 = self.epsilonball(data2)
        # LIS loss and push-away loss
        loss_mae_distance, loss_mae_mutex = self.DistanceLossYuanLinHUChi(
            data1, data2, d_data1, d_data2, kNN_mask_data1, kNN_mask_data2)
        # angle loss
        if self.args['ratio']['angle'] < 0.01:
            loss_mae_ang = loss_mae_distance / 10000
        else:
            loss_mae_ang = self.AngleLoss(  # fast version
                data1, data2, kNN_mask_data1, kNN_mask_data2)
        
        return loss_mae_distance, loss_mae_ang, loss_mae_mutex

    def kNNGraph(self, data):
        """ K nearest graph """
        k = self.nk
        batch_size = data.shape[0]  # [batch_size, n]

        x = data.to(self.device)
        y = data.to(self.device)
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(1, -2, x, y.t())  # dist = x^2 + y^2 - 2xy = ||x - y||^2
        d = dist.clamp(min=1e-8).sqrt()  # for numerical stabili

        kNN_mask = torch.zeros((batch_size, batch_size,), device=self.device)
        s_, indices = torch.sort(d, dim=1)

        indices = indices[:, :k+1]
        for i in range(kNN_mask.size(0)):
            kNN_mask[i, :][indices[i]] = 1
        kNN_mask[torch.eye(kNN_mask.shape[0], dtype=bool)] = 0
        
        return d, kNN_mask.bool()

    def epsilonball(self, data):
        """ r ball graph """
        epcilon = self.args['epcilon']

        x = data.to(self.device)
        y = data.to(self.device)
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(1, -2, x, y.t())  # dist = x^2 + y^2 - 2xy = ||x - y||^2
        d = dist.clamp(min=1e-8).sqrt()  # for numerical stabili

        kNN_mask = (d < epcilon).bool()  # indictor of neighbor graph

        return d, kNN_mask

    def DistanceLoss(self, data1, data2, d_data1, d_data2,
                     kNN_mask_data1, kNN_mask_data2):

        norml_data1 = torch.sqrt(torch.tensor(float(data1.shape[1])))
        norml_data2 = torch.sqrt(torch.tensor(float(data2.shape[1])))

        D1_1 = (d_data1 / norml_data1)[kNN_mask_data1]
        D1_2 = (d_data2 / norml_data2)[kNN_mask_data1]
        Error1 = (D1_1 - D1_2) / 1
        loss2_1 = torch.norm(Error1) / torch.sum(kNN_mask_data1)

        D2_1 = (d_data1 / norml_data1)[kNN_mask_data2]
        D2_2 = (d_data2 / norml_data2)[kNN_mask_data2]
        Error2 = (D2_1 - D2_2) / 1
        loss2_2 = torch.norm(Error2) / torch.sum(kNN_mask_data2)

        loss_mae_d = 1*loss2_1 + 2*loss2_2

        return loss_mae_d

    def DistanceLossYuanLinHUChi(self, data1, data2, d_data1, d_data2,
                                 kNN_mask_data1, kNN_mask_data2):
        """ LIS Loss (dist loss + push-away loss)
            
            Dist Loss is the difference of two L2_norm / sqrt(dim).
            Push-Away Loss (mae_mutex), gradually decrease from 0.8 to 0.
        """
        norml_data1 = torch.sqrt(torch.tensor(float(data1.shape[1])))  # sqrt(dim)
        norml_data2 = torch.sqrt(torch.tensor(float(data2.shape[1])))

        mask_u = kNN_mask_data1
        D1_1 = (d_data1 / norml_data1)[mask_u]
        D1_2 = (d_data2 / norml_data2)[mask_u]
        Error1 = (D1_1 - D1_2) / 1  # scaling
        loss2_1 = torch.norm(Error1) / torch.sum(mask_u)

        # D2_1 = (d_data1/norml_data1)[False == kNN_mask_data1]
        D2_2 = (d_data2/norml_data2)[kNN_mask_data1 == False]
        Error2 = (0 - D2_2) / 1  # scaling
        loss2_2 = torch.norm(Error2[Error2 > -1 * self.args['regularB']]) / \
            torch.sum(kNN_mask_data1 == False)

        loss_mae_distance = 1 * loss2_1
        loss_mae_mutex = -1 * loss2_2  # -1 * self.mutex * loss2_2 # ori

        return loss_mae_distance, loss_mae_mutex

    def Cossimi(self, data):
        eps = 1e-8
        a_n, b_n = data.norm(dim=2)[:, :, None], data.norm(dim=2)[:, :, None]
        a_norm = data / torch.max(a_n, eps * torch.ones_like(a_n))
        b_norm = data / torch.max(b_n, eps * torch.ones_like(b_n))
        out = torch.matmul(a_norm, b_norm.permute(0, 2, 1))

        return torch.acos(out.clamp(min=-(1-1e-6), max=(1-1e-6)))

    def AngleLoss(self, data, datap1, kNN_mask1, kNN_mask2):

        batch_size = data.size(0)

        data_y = data.expand(batch_size, batch_size, -1).permute(1, 0, 2)
        data_x = data.expand(batch_size, batch_size, -1)
        datap1_y = datap1.expand(batch_size, batch_size, -1).permute(1, 0, 2)
        datap1_x = datap1.expand(batch_size, batch_size, -1)

        relative_data1_1 = torch.sub(data_y, data_x)[
            kNN_mask1].view(batch_size, self.nk, -1)
        relative_datap1_2 = torch.sub(datap1_y, datap1_x)[
            kNN_mask1].view(batch_size, self.nk, -1)

        angle_data1_1 = self.Cossimi(relative_data1_1)
        angle_datap1_2 = self.Cossimi(relative_datap1_2)

        angle_loss1 = torch.norm(angle_data1_1 - angle_datap1_2,
                                 dim=(1, 2)) / batch_size

        angle_loss = 1 * angle_loss1  # + 3 * angle_loss2
        return torch.mean(angle_loss)

    def ShowkNNGraph(self, data, kNN_mask, label):
        import matplotlib.pyplot as plt

        label_numpy = label.detach().numpy()
        data_numpy = data.detach().numpy()

        batchsize = kNN_mask.shape[0]

        for i in range(10):
            point = data_numpy[label_numpy == i]
            plt.scatter(point[:, 0], point[:, 1], s=10)

        for i in range(batchsize):
            for j in range(batchsize):
                if kNN_mask[i, j] == 1:
                    plt.plot([data[i, 0], data[j, 0]], [
                        data[i, 1], data[j, 1]], 'grey', linewidth=0.5)
    
    def ortho_norm_l2(self, weights, orth_w=list(), version='v3', mode='full'):
        """ Orthogonal Norm Regularization """
        orth_loss = None

        for i,W in enumerate(weights):
            if W.ndimension() < 2 and W.requires_grad == False:
                continue
            elif W.requires_grad == False: # except requires_grad = False
                continue
            else:

                if W.shape[0] != W.shape[1] and mode == 'full':  # except W.col == W.row
                    continue
                #assert(W.shape[0] == W.shape[1])

                cols = W[0].numel()
                rows = W.shape[0]
                w1 = W.view(-1, cols)
                wt = torch.transpose(w1, 0, 1)
                if (rows > cols):
                    m  = torch.matmul(wt, w1)
                    ident = Variable(torch.eye(cols,cols), requires_grad=True)
                else:
                    m = torch.matmul(w1, wt)
                    ident = Variable(torch.eye(rows,rows), requires_grad=True)
                
                ident = ident.cuda()
                w_tmp = (m - ident)

                v1 = w_tmp
                norm1 = torch.norm(v1, 2)
                v2 = torch.div(v1, norm1)
                v3 = torch.matmul(w_tmp, v2)

                # add orth_weight
                cur_weight = 1
                if i < len(orth_w):  # add orth_w if given
                    cur_weight = orth_w[i]

                if version == 'v3':
                    if orth_loss is None:
                        orth_loss = ((torch.norm(v3, 2))**2) * cur_weight
                    else:
                        orth_loss += ((torch.norm(v3, 2))**2) * cur_weight
                else:  # 'v1'
                    if orth_loss is None:
                        orth_loss = ((torch.norm(v1, 2))**2) * cur_weight
                    else:
                        orth_loss += ((torch.norm(v1, 2))**2) * cur_weight
                
        return orth_loss

    def SetGradual(self):
        # AE
        if self.epoch >= self.AE_w["AE_gradual"][0]:  # 0 -> 1.0
            if self.epoch < self.AE_w["AE_gradual"][1]:
                gap = self.AE_w["AE_gradual"][1] - self.AE_w["AE_gradual"][0]
                self.gradual["AE"] = (self.epoch - self.AE_w["AE_gradual"][0]) / gap
            else:
                self.gradual["AE"] = 1
        # LIS
        if self.epoch >= self.LIS_w["LIS_gradual"][0]:  # 0 -> 1.0
            if self.epoch < self.LIS_w["LIS_gradual"][1]:
                gap = self.LIS_w["LIS_gradual"][1] - self.LIS_w["LIS_gradual"][0]
                self.gradual["LIS"] = (self.epoch - self.LIS_w["LIS_gradual"][0]) / gap
            else:
                self.gradual["LIS"] = 1
        # Orth
        if self.epoch >= self.Orth_w["Orth_gradual"][0]:  # 0 -> 1.0
            if self.epoch < self.Orth_w["Orth_gradual"][1]:
                gap = self.Orth_w["Orth_gradual"][1] - self.Orth_w["Orth_gradual"][0]
                self.gradual["Orth"] = (self.epoch - self.Orth_w["Orth_gradual"][0]) / gap
            else:
                self.gradual["Orth"] = 1
        #print(self.gradual)
    
    def GetGradual(self, step=[0,0,0]):
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
    
    def PaddingLoss(self, train_info):
        """ force elements in [padding[i]:] to zeros """
        pad_loss = 0
        padding, pad_w = self.InverseMode["padding"], self.InverseMode["pad_w"]
        mode = self.InverseMode["loss_type"]
        assert(len(train_info) // 2 + 1 >= len(padding))

        # pad each layer with given zeros
        for i in range(len(padding)):
            pad_shape = (train_info[i].shape[0], train_info[i].shape[1] - padding[i])
            zeros = torch.zeros(pad_shape).to(self.device)
            if mode == "L2":
                criterion = nn.MSELoss().cuda()
                pad_loss += criterion(train_info[i][:, padding[i]:], zeros) * pad_w[i]
            elif mode == "L1":
                criterion = nn.L1Loss().cuda()
                pad_loss += criterion(train_info[i][:, padding[i]:], zeros) * pad_w[i]
            elif mode == 'p-norm':
                p = self.GetGradual(self.InverseMode["p_gradual"]) + 1
                if p > 1:
                    pad_loss += torch.sum( torch.norm(train_info[i][:, padding[i]:] - zeros, p) ) * pad_w[i]
                else:
                    criterion = nn.L1Loss().cuda()
                    pad_loss += criterion(train_info[i][:, padding[i]:], zeros) * pad_w[i]
            else:
                pass
        return pad_loss
    
    def CalLosses(self, train_info):
        """ cal Loss and output to train_info """
        if type(train_info) == type(dict()):
            weight_info = train_info['weight']
            extra_info  = train_info["extra"]
            grad_info   = train_info['grad']
            pad_info    = train_info["padding"]
            train_info  = train_info['output']
        train_info[0] = train_info[0].view(train_info[0].shape[0], -1)

        # @ zero loss, set gradual changing
        for k in self.loss.keys():
            self.loss[k] = 0
        # gradually change:
        self.gradual["AE"]   = self.GetGradual(self.AE_w["AE_gradual"])
        self.gradual["LIS"]  = self.GetGradual(self.LIS_w["LIS_gradual"])
        self.gradual["Orth"] = self.GetGradual(self.Orth_w["Orth_gradual"])
        if self.CS:
            self.gradual["pad"] = self.GetGradual(self.InverseMode["pad_gradual"])
        for k in self.gradual["push"].keys():  # push-away gradual
            self.gradual["push"][k] = self.GetGradual(self.LIS_w["push_gradual"][k])

        # @ LIS Loss cross-layer (input -- latent)
        if self.LIS_w['cross'][0] > 0:
            _dist, _angle, _push = self.MorphicLossItem(train_info[0], train_info[self.index["latent"]])
            self.loss["dist"]  = _dist  * self.LIS_w['cross'][0] * self.gradual["LIS"]  * self.LIS_w["cross_w"][0]
            self.loss["angle"] = _angle * self.LIS_w['cross'][0] * self.gradual["LIS"]  * self.LIS_w["cross_w"][1]
            self.loss["push"]  = _push  * self.LIS_w['cross'][0] * self.gradual["LIS"]  * self.LIS_w["cross_w"][2] \
                                    * self.gradual["push"]["cross_w"]
        
        # LIS and reconstruct Enc and Dec
        for i in range(self.layer_num):

            # index
            enc_i = self.index["layer"][i]
            dec_i = self.index["layer"][-i-1]

            # @ AE Rect loss: each-layer
            if self.AE_w['each'][i] > 0:
                self.loss["AE"] += (self.ReconstructionLoss(train_info[enc_i], train_info[dec_i]) / self.layer_num) \
                                    * self.AE_w['each'][i] / self.w_sum["AE"] * self.gradual["AE"]

            # @ LIS Loss: each-layer
            if self.LIS_w['each'][i] > 0:
                _dist, _angle, _push = self.MorphicLossItem(train_info[enc_i], train_info[dec_i])
                self.loss["dist"]  += _dist  * self.LIS_w['each'][i] / self.w_sum['each'] * self.gradual["LIS"]  * self.LIS_w['each_w'][0]
                self.loss["angle"] += _angle * self.LIS_w['each'][i] / self.w_sum['each'] * self.gradual["LIS"]  * self.LIS_w['each_w'][1]
                self.loss["push"]  += _push  * self.LIS_w['each'][i] / self.w_sum['each'] * self.gradual["LIS"]  * self.LIS_w['each_w'][2] \
                                             * self.gradual["push"]["each_w"]
        
            # @ LIS Loss forward
            if self.LIS_w['enc_forward'][i] > 0:
                _dist, _angle, _push = self.MorphicLossItem(train_info[0], train_info[enc_i])
                self.loss["dist"]  += _dist  * self.LIS_w['enc_forward'][i] / self.w_sum["enc_forward"] \
                                             * self.gradual["LIS"] * self.LIS_w['enc_forward_w'][0]
                self.loss["angle"] += _angle * self.LIS_w['enc_forward'][i] / self.w_sum["enc_forward"] \
                                            * self.gradual["LIS"] * self.LIS_w['enc_forward_w'][1]
                self.loss["push"]  += _push * self.LIS_w['enc_forward'][i] / self.w_sum["enc_forward"] \
                                            * self.gradual["LIS"] * self.LIS_w['enc_forward_w'][2] * self.gradual["push"]["enc_w"]
            if self.LIS_w['dec_forward'][i] > 0:
                _dist, _angle, _push = self.MorphicLossItem(train_info[0], train_info[dec_i])
                self.loss["dist"]  += _dist * self.LIS_w['dec_forward'][i] / self.w_sum["dec_forward"] \
                                            * self.gradual["LIS"] * self.LIS_w['dec_forward_w'][0]
                self.loss["angle"] += _angle * self.LIS_w['dec_forward'][i] / self.w_sum["dec_forward"] \
                                            * self.gradual["LIS"] * self.LIS_w['dec_forward_w'][1]
                self.loss["push"]  += _push * self.LIS_w['dec_forward'][i] / self.w_sum["dec_forward"] \
                                            * self.gradual["LIS"] * self.LIS_w['dec_forward_w'][2] * self.gradual["push"]["dec_w"]
            # @ LIS Loss backward
            if self.LIS_w['enc_backward'][i] > 0:
                _dist, _angle, _push = self.MorphicLossItem(train_info[self.index["latent"]], train_info[enc_i])
                self.loss["dist"] += _dist  * self.LIS_w['enc_backward'][i] / self.w_sum["enc_backward"] \
                                            * self.gradual["LIS"] * self.LIS_w['enc_backward_w'][0]
                self.loss["angle"] += _angle * self.LIS_w['enc_backward'][i] / self.w_sum["enc_backward"] \
                                             * self.gradual["LIS"] * self.LIS_w['enc_backward_w'][1]
                self.loss["push"] += _push  * self.LIS_w['enc_backward'][i] / self.w_sum["enc_backward"] \
                                            * self.gradual["LIS"] * self.LIS_w['enc_backward_w'][2] * self.gradual["push"]["enc_w"]
            if self.LIS_w['dec_backward'][i] > 0:
                _dist, _angle, _push = self.MorphicLossItem(train_info[self.index["latent"]], train_info[dec_i])
                self.loss["dist"] += _dist  * self.LIS_w['dec_backward'][i] / self.w_sum["dec_backward"] \
                                            * self.gradual["LIS"] * self.LIS_w['dec_backward_w'][0]
                self.loss["angle"] += _angle * self.LIS_w['dec_backward'][i] / self.w_sum["dec_backward"] \
                                             * self.gradual["LIS"] * self.LIS_w['dec_backward_w'][1]
                self.loss["push"] += _push  * self.LIS_w['dec_backward'][i] / self.w_sum["dec_backward"] \
                                            * self.gradual["LIS"] * self.LIS_w['dec_backward_w'][2] * self.gradual["push"]["dec_w"]
            
            # @ CS loss: add zeros
            if self.CS == True:  # add 0716
                self.loss["pad"] = self.PaddingLoss(pad_info) * self.gradual["pad"]
        
        # @ Orth Loss
        if self.args["ratio"]["orth"] > 0:
            _orth = self.ortho_norm_l2(weight_info, orth_w=self.args["OrthWeight"]["each"])
            if _orth is not None:
                self.loss["orth"] = _orth * self.gradual["Orth"]

        # @ Global loss weight
        for k in self.loss.keys():
            self.loss[k] *= self.args["ratio"][k]
        
        # @ Extra Head: LIS Loss (input -- extra-Head)
        if len(extra_info) > 0:
            for extra_i in range(len(extra_info)):
                _dist, _angle, _push = self.MorphicLossItem(train_info[0], extra_info[extra_i])
                if self.args['ExtraHead'].get('push_w', False):
                    _push *= self.args['ExtraHead']["push_w"][extra_i]
                self.loss["dist"]  += _dist * self.args['ExtraHead']['weight'][extra_i]
                self.loss["angle"] += _angle * self.args['ExtraHead']['weight'][extra_i]
                self.loss["push"]  += _push * self.args['ExtraHead']['weight'][extra_i] * self.gradual["push"]["extra_w"]

        return self.loss
