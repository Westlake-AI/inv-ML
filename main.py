import os
import numpy as np
import random as rd
import json
import time
import argparse

import torch
import torch.utils.data
from torch import optim

import gifploter
from dataset import dataset, dataloader
from trainer.invML_trainer import invML_trainer
from generator.samplegenerater import SampleIndexGenerater


def PlotLatenSpace(model, batch_size, datas, labels, loss_caler, gif_ploter, device,
                   path='./', name='none', full=True, save_plot=True):
    """use to test the model and plot the latent space

    Arguments:
        model {torch model} -- a model need to train
        batch_size {int} -- batch size
        datas {tensor} -- the train data
        labels {label} -- the train label, for unsuprised method, it is only used in plot fig

    Keyword Arguments:
        path {str} -- the path to save the fig (default: {'./'})
        name {str} -- the name of current fig (default: {'no name'})
    """
    model.eval()
    train_loss_sum = [0, 0, 0, 0, 0, 0]
    num_train_sample = datas.shape[0]

    if full == True:
        for batch_idx in torch.arange(0, (num_train_sample-1)//batch_size + 1):
            start_number = (batch_idx * batch_size).int()
            end_number = torch.min(torch.tensor(
                [batch_idx*batch_size+batch_size, num_train_sample])).int()

            data = datas[start_number:end_number].float()
            label = labels[start_number:end_number]

            data = data.to(device)
            label = label.to(device)
            # train info
            train_info = model(data)
            loss_dict = loss_caler.CalLosses(train_info)
            if type(train_info) == type(dict()):
                train_info = train_info['output']

            for i, k in enumerate(list(loss_dict.keys())):
                train_loss_sum[i] += loss_dict[k].item()

            if batch_idx == 0:
                latent_point = []
                for train_info_item in train_info:
                    latent_point.append(train_info_item.detach().cpu().numpy())

                label_point = label.cpu().detach().numpy()
            else:
                for i, train_info_item in enumerate(train_info):
                    latent_point_c = train_info_item.detach().cpu().numpy()
                    latent_point[i] = np.concatenate(
                        (latent_point[i], latent_point_c), axis=0)

                label_point = np.concatenate(
                    (label_point, label.cpu().detach().numpy()), axis=0)
    
        gif_ploter.AddNewFig(
            latent_point, label_point,
            title_=path+'/'+name +
                '__AE_' + str(4)[:4] + '__MAE_'+ str(4)[:4],
            loss=train_loss_sum,
            save=save_plot
        )

    else:
        data = datas.to(device)
        label = labels.to(device)
        
        eval_info = model(data)
        if type(eval_info) == type(dict()):
            eval_info = eval_info['output']

        latent_point = []
        for info_item in eval_info:
            latent_point.append(info_item.detach().cpu().numpy())

        label_point = label.cpu().detach().numpy()
        
        gif_ploter.AddNewFig(
            latent_point, label_point,
            title_=path+'/'+'result', loss=None, save=save_plot
        )


def SetSeed(seed):
    """function used to set a random seed """
    SEED = seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    rd.seed(SEED)
    np.random.seed(SEED)


def SetParam(mode='encoder'):
    # config param
    param = dict(
        # regular default params
        EPOCHS = 8000,
        LEARNINGRATE = 1e-3,
        BATCHSIZE = 800,
        N_dataset = 800,
        regularB = 3,
        epcilon = 0.23,
        MAEK = 20,
        LOGINTERVAL = 1.0,
        PlotForloop = 2000,
        sampleMethod = 'normal', # 'normal', 'near'
        # choose
        Noise = 0.0,
        add_jump = False,
        NetworkStructure = dict(
            layer = [3, 100, 100, 100, 100, 2],
            relu = [1, 1, 1, 1, 0],
            Enc_require_gard = [1, 1, 1, 1, 1],
            Dec_require_gard = [0, 0, 0, 0, 0],
            inv_Enc = 0, inv_Dec = 1,
        ),
        # Extra Head (DR project)
        ExtraHead = dict(
            layer = [], # None
            weight = [], #[0, 0, 0, 0, 0, 0],
        ),
        # ReLU
        ReluType = dict(
            type = "Leaky", # "InvLeaky"
            Enc_alpha = 0.1,
            Dec_alpha = 10,
        ),
        # LIS
        LISWeght = dict(
            cross = [0,],
            enc_forward      = [0,   0,   0,   0,   0],
            dec_forward      = [0,   0,   0,   0,   0],
            enc_backward     = [0,   0,   0,   0,   0],
            dec_backward     = [0,   0,   0,   0,   0],
            each             = [0,   0,   0,   0,   0],
            # [dist, angle, push]
            cross_w        = [1, 1, 1],
            enc_forward_w  = [1, 1, 1],
            dec_forward_w  = [1, 1, 1],
            enc_backward_w = [1, 1, 1],
            dec_backward_w = [1, 1, 1],
            each_w         = [1, 1, 1],
            extra_w        = [1, 1, 1],
            # [start, end, mode]
            LIS_gradual = [0,    0,     1],
            push_gradual = dict( # add 0716
                cross_w = [500,  1000,  0],  # 1 -> 0
                enc_w   = [500,  1000,  0],
                dec_w   = [500,  1000,  0],
                each_w  = [500,  1000,  0],
                extra_w = [500,  1000,  0],
            ),
        ),
        # AE layer
        AEWeight = dict(
            each = [],
            AE_gradual = [0, 0, 1],  # [start, end, mode]
        ),
        # Orth
        OrthWeight = dict(
            each = [0, 0, 0, 0, 0],
            Orth_gradual = [0, 0, 1],  # [start, end, mode]
        ),
    )
    # cmd argsparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-D", "--DATASET",
                        default='SwissRoll', type=str,
                        choices=['usps','mnist','Kmnist','Fmnist','coil-20','cifar-10',
                                'SwissRoll','SCurve','sphere'])
    parser.add_argument("-C", "--numberClass",
                        default=10, type=int,)
    parser.add_argument("-M", "--Model",
                        default='MLP', choices=['MLP'])
    parser.add_argument("-SD", "--SEED", default=0, type=int)
    parser.add_argument("-V", "--Version",
                        #default="v1",   # v1.0, swiss roll, sphere
                        default="v1.1", # v1.1, MNIST, FMNIST, etc
                        type=str)
    parser.add_argument("-Task", "--Task",
                        default="train", choices=['train', "interplotation"])  
    # new params
    parser.add_argument("-Name", "--ExpName", default="Inv", type=str)
    parser.add_argument("-R", "--ratio",
                        default={"AE": 0.005,
                                "dist": 1,
                                "angle": 0,
                                "push": 1,
                                "orth": 0,
                                "pad": 0,
                            },
                        type=dict, help='the weight for each loss item for inv-ML or MLAE')
    parser.add_argument("-Mode", "--InverseMode",
                        default={
                                "mode": "pinverse",  # ["pinverse", "CSinverse", "ZeroPadding"],
                                "padding": [0, 0, 0, 0, 0],
                                "pad_w":   [0, 0, 0, 0, 0],
                                "pad_gradual": [0, 0, 1],  # [start, end, mode]
                                "p_gradual": [2000, 4000, 0], # p-norm: p=2 -> p=1
                        },
                        type=dict)
    # useless param
    parser.add_argument("-NAME", "--SaveName", default="None", type=str)
    parser.add_argument("-T", "--Test",
                        default=1, type=int, choices=[1, 2, 3, 4,  5, 6, 7, 8])
    args = parser.parse_args()

    # to param dict
    args = parser.parse_args()
    args = args.__dict__
    param.update(args)

    # use config file to update param: param['Test']
    from test_config import import_test_config

    new_param = import_test_config(param['Test'], mode=mode)
    param.update(new_param)
    
    # save name
    if param["SaveName"] == "None":
        path_file = "./{}".format(param['Model'])
        for k in param['ratio'].keys():
            path_file += '_'+k+str(param['ratio'][k])
    else:
        path_file = param["SaveName"]
    
    path = os.path.join(param['ExpName'], path_file)
    if not os.path.exists(path):
        os.makedirs(path)
    
    return param, path


def SetModel(param):
    from models.InvML import InvML_MLP
    from loss.InvML_loss import InvMLLosses
    
    if param['Model'] == 'MLP':
        Model = InvML_MLP(param).to(device)
        param['index'] = Model.plot_index_list  # network index
        loss = InvMLLosses(args=param, cuda=device)

    return Model, loss


def SetModelVAE(param):
    from models.VAE import VAE_MLP
    from loss.VAE_loss import VAEloss

    Model = VAE_MLP(param).to(device)
    loss = VAEloss()
    return Model, loss


def single_test(new_param=dict(), mode='encoder', device='cuda'):
    param, path = SetParam(mode)
    SetSeed(param['SEED'])
    # updata new params
    param.update(new_param)
    # params transfer
    transfer_bool = True
    for g in param["NetworkStructure"]["Dec_require_gard"]:
        if g > 0:
            transfer_bool = False

    # load the data
    if param['DATASET'] != 'SwissRoll' and param['DATASET'] != 'SCurve':
        train_data, test_data, train_label, test_label = dataloader.LoadRealData(param, device=device,)
    else:
        train_data, train_label, test_data, test_label = dataset.LoadData(
            dataname=param['DATASET'],
            train_number=param['N_dataset'], test_number=param['N_dataset'],
            noise=param['Noise'], randomstate=param['SEED'], remove='star'
        )
    param['BATCHSIZE'] = min(param['BATCHSIZE'], train_data.shape[0])

    # init the model, set mode
    Model, loss_caler = SetModel(param)
    Model.SetMode(mode)
    # optimizer = optim.Adam(Model.parameters(), lr=param['LEARNINGRATE']) # ori
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, Model.parameters()), lr=param['LEARNINGRATE'])
    
    sample_index_generater_train = SampleIndexGenerater(
        train_data, param['BATCHSIZE'], method=param['sampleMethod'], choos_center='normal'
    )
    gif_ploter = gifploter.GIFPloter(param, Model)

    # load .pth of pertrained encoder
    if mode == 'decoder' or param.get("resume_from", False):
        path_pth = param["ExpName"]+".pth"
        # path_pth = '{}/checkpoints/9000.pth'.format(param["ExpName"].split("_")[0])
        path_pth = param.get("resume_from", param["ExpName"]+".pth")

        checkpoint = torch.load(path_pth)
        state_dict = checkpoint['state_dict']
        
        model_dict = Model.state_dict()
        pretrained = {}
        if mode == "decoder":
            pretrained = {k: v for k, v in state_dict.items() if k in model_dict}
        else:
            # init only orth W
            pretrained = {k: v for k, v in state_dict.items() if k in model_dict and v.shape[0] == v.shape[1]}
        model_dict.update(pretrained)
        Model.load_state_dict(model_dict)
        print('load pretrained encoder model: {}.'.format(path_pth))

    # training
    for epoch in range(param['EPOCHS'] + 1):
        if param['EPOCHS'] <= 0:
            break
        # start a trainer
        loss_sum = invML_trainer(Model, loss_caler, epoch, train_data, train_label,
                                optimizer, device,
                                sample_index_generater_train,
                                batch_size=param['BATCHSIZE'],
                                verbose=epoch % param['PlotForloop']==0)
        # update plot loss
        loss_interval = 200
        if epoch == 0 or epoch % loss_interval == 0 and epoch % param['PlotForloop'] != 0:
            gif_ploter.update_loss(loss_sum)
        # plot GIF
        if epoch % param['PlotForloop'] == 0 and epoch > 0:
            # transfer
            if mode == 'encoder' and transfer_bool:
                Model.params_transfer()
            # plot
            name = 'epoch_' + str(epoch).zfill(5)
            PlotLatenSpace(
                Model, param['BATCHSIZE'],
                train_data, train_label, loss_caler, gif_ploter, device,
                path=path, name=name,
            )
            # checkpoints
            if mode == 'encoder':
                if not os.path.exists(os.path.join(param["ExpName"], "checkpoints")):
                    os.mkdir(os.path.join(param["ExpName"], "checkpoints"))
                state = {'state_dict': Model.state_dict()}
                torch.save(state, os.path.join(param["ExpName"], "checkpoints", str(epoch)+".pth"))
    # save final .pth
    if mode == 'encoder':
        state = {'state_dict': Model.state_dict()}
        torch.save(state, param["ExpName"]+"_Dec"+".pth")
        print('save encoder model as ', param["ExpName"]+"_Dec"+".pth")
    
    # test
    data = test_data.float()
    data = data.to(device)
    label = test_label.to(device)
    Model.eval()
    test_info = Model(data)

    # add visualization
    save_result = True
    if save_result and mode == 'decoder':
        save_name = param["ExpName"].split("_")[0]
        np.save(os.path.join(save_name, "input.npy"), test_data.cpu().numpy())
        np.save(os.path.join(save_name, "result.npy"), test_info['output'][-1].detach().cpu().numpy())
        print("save test result as .npy.")

    # plot test img (True/False)
    plot_test = False
    if mode == 'decoder':
        plot_test = True

    if plot_test:
        name = 'test_result'
        PlotLatenSpace(
            Model, param['BATCHSIZE'],
            data, label, loss_caler, gif_ploter, device,
            path=path, name=name,
            full=False,
        )
    
    return param["ExpName"], param["Test"]


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. test encoder
    new_param = {}
    
    expName, testName = single_test(new_param, "encoder", device)
    # expName, testName = "Orth", 1 # decoder for 1
    # expName, testName = "Orth2", 2 # decoder for 2
    # expName, testName = "Orth3", 3 # decoder for 3
    # expName, testName = "Orth4", 4 # decoder for 4
    # expName, testName = "Orth5", 5 # decoder for 5
    # expName, testName = "Orth6", 6 # decoder for 6
    # expName, testName = "Orth7", 7 # decoder for 7
    # expName, testName = "Orth8", 8 # decoder for 8

    # test decoder based on encoder param
    new_param = {"ExpName": expName+"_Dec", "Test": testName}

    # 2. test decoder
    _,_ = single_test(new_param, "decoder", device)