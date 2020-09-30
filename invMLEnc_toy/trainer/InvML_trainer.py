import datetime
import os
import torch


def InvML_trainer(model, loss_caler, epoch, train_data, train_label,
        optimizer, device, sample_index_generater, batch_size, verbose=False):
    """ one loop Trainer for MLAE model for MLP

    Arguments:
        model {torch model} -- a model need to train
        loss_caler {torch model} -- a model used to get the loss
        epoch {int} -- current epoch
        train_data {tensor} -- the train data
        train_label {label} -- the train label, for unsuprised method, it is only used in plot fig
        sample_index_generater {class} -- a train index generater
        batch_size {int} -- batch size
    """
    # train the model for one loop
    model.train()
    train_loss_sum = [0, 0, 0, 0, 0, 0]
    loss_caler.SetEpoch(epoch)

    num_train_sample = train_data.shape[0]
    sample_index_generater.Reset()

    num_batch = (num_train_sample - 0.5) // batch_size + 1
    for batch_idx in torch.arange(0, num_batch):

        sample_index = sample_index_generater.CalSampleIndex(batch_idx)
        data = train_data[sample_index].float()
        label = train_label[sample_index]

        # init optimizer, add data to device
        optimizer.zero_grad()
        data = data.to(device)
        label = label.to(device)
        # forward
        train_info = model(data)
        loss_dict = loss_caler.CalLosses(train_info)
        # backward
        for i, k in enumerate(list(loss_dict.keys())):
            loss_dict[k].backward(retain_graph=True)
            train_loss_sum[i] += loss_dict[k].item()
        # opt update
        optimizer.step()
        if verbose:
            print('Train Epoch: {}, [sample_num: {}]\tLoss: {}'.format(
                epoch, num_train_sample,
                [(k+":", round(loss_dict[k].item(), 6)) for k in loss_dict.keys()]
            ))

    # epoch log
    if verbose:
        # time_stamp = datetime.datetime.now()
        # print("time_stamp:" + time_stamp.strftime('%Y.%m.%d-%H:%M:%S'))
        print('====> Epoch: {},\tAverage loss: {}'.format(
            epoch,
            [round(train_loss_sum[i] / num_batch, 6) for i in range(len(train_loss_sum))]
        ))
    
    return train_loss_sum