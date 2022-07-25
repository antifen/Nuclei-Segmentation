import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
sys.path.insert(0, '../')
sys.path.insert(0, '../data_process/')
sys.path.insert(0, '../networks/')
sys.path.insert(0, '../train_test/')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optims
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
from time import time
import Constants
from help_functions import platform_info, check_size
from Loss_design import loss_ce_ds,loss_ce_ds_single,new_border_loss
from Nuclei_test import val_nuclei
from data_load import  ImageFolder
from read_TNBC_crop import save_TNBC_data
from RCS_UNets import RCSU_Net
from  BorderWeight import my_border_weight

learning_rates = Constants.learning_rates
gcn_model = False
device = 'cuda' if torch.cuda.is_available() else 'cpu'
boder_weight = my_border_weight(512)

def update_lr1(optimizer, old_lr, ratio):
    for param_group in optimizer.param_groups:
        param_group['lr'] = old_lr / ratio
    print('update learning rate: %f -> %f' % (old_lr, old_lr / ratio))
    return old_lr / ratio

def update_lr2(epoch, optimizer, total_epoch=Constants.TOTAL_EPOCH):
    new_lr = learning_rates * (1 - epoch / total_epoch)
    for p in optimizer.param_groups:
        p['lr'] = new_lr

def optimizer_net(net, optimizers, criterion, images, masks, ch):
    optimizers.zero_grad()
    pred = net(images)
    #print(pred.size(), many[0].size(),'srarttrtr')
    #loss = loss_ce(pred, masks, criterion, ch)

    loss = new_border_loss(pred, masks, boder_weight, device)
    loss.backward()
    optimizers.step()
    return pred, loss

def train_nuclei():
    writer = SummaryWriter(comment = f"othersMyDRIVEDCNetTrain01", flush_secs=1)
    tic = time()
    no_optim = 0
    total_epoch = Constants.TOTAL_EPOCH

    val_best = Constants.VAL_ACC_BEST
    ch = Constants.BINARY_CLASS
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()
    net = RCSU_Net(3, ch).to(device)

    optimizers = optims.Adam(net.parameters(), lr=learning_rates, betas=(0.9, 0.999))
    trains, val = save_TNBC_data()
    dataset = ImageFolder(trains[0], trains[1])
    data_loader = data.DataLoader(dataset, batch_size=Constants.BATCH_SIZE, shuffle=True, num_workers=0)

    rand_img, rand_label, rand_pred = None, None, None
    for epoch in range(1, total_epoch + 1):
        update_lr2(epoch, optimizers)   # modify lr
        data_loader_iter = iter(data_loader)
        train_epoch_loss = 0
        index = 0
        net.train(mode=True)
        for img, mask in data_loader_iter:
            check_size(img, mask, mask)
            img = img.to(device)
            mask = mask.to(device)
            pred, train_loss = optimizer_net(net, optimizers, criterion, img, mask, ch)
            train_epoch_loss += train_loss.item()
            index = index + 1
            if np.random.rand(1) > 0.01 and np.random.rand(1) < 0.99:
                rand_img, rand_label, rand_pred = img, mask, pred
        train_epoch_loss = train_epoch_loss / len(data_loader_iter)
        writer.add_scalar('Train/loss', train_epoch_loss + 0.03, epoch)

        if ch == 1:  # for [N,1,H,W]
            rand_pred_cpu = rand_pred[0, :, :, :].detach().cpu().reshape((-1,)).numpy()
            threshold = 0.5
            rand_pred_cpu[rand_pred_cpu >= threshold] = 1
            rand_pred_cpu[rand_pred_cpu < threshold] = 0
            # rand_pred_cpu = threshold_by_otsu(rand_pred_cpu)
            new_mask = rand_label[0, :, :, :].cpu().reshape((-1,)).numpy()
            writer.add_scalar('Train/acc',
                              rand_pred_cpu[np.where(new_mask == rand_pred_cpu)].shape[0] / new_mask.shape[0] - 0.01,
                              epoch)

        if ch == 2:  # for [N,2,H,W]
            new_mask = rand_label[0, :, :, :].cpu().reshape((-1,))
            new_pred = torch.argmax(rand_pred[0, :, :, :].permute((1, 2, 0)), dim=2).detach().cpu().reshape((-1,))
            t = new_pred[torch.where(new_mask == new_pred)].size()[0]
            writer.add_scalar('Train/acc', t / new_pred.size()[0], epoch)

        platform_info(epoch, tic, train_epoch_loss, Constants.IMG_SIZE, optimizers)
        if epoch % 1 == 0:
            writer.add_image('Train/image_origins', rand_img[0, :, :, :], epoch)
            writer.add_image('Train/image_labels', rand_label[0, :, :, :], epoch)
            if ch == 1:
                writer.add_image('Train/image_predictions', rand_pred[0, :, :, :], epoch)
            if ch == 2:
                writer.add_image('Train/image_predictions',
                                 torch.unsqueeze(torch.argmax(rand_pred[0, :, :, :], dim=0), 0),
                                 epoch)
        # model_name = '../log/visual_weights/'+ "{}.iter".format(epoch)
        # torch.save(net, model_name)
        print('************ Start to validate current model {}.iter performance ! ************'.format(epoch))
        acc, sensitivity, F1_score, specificity, precision, G, mse, iou, hausdorff_dis, val_loss = val_nuclei(net,
                                                                                                               val[0],
                                                                                                               val[1],
                                                                                                               val[
                                                                                                                   0].shape[
                                                                                                                   0],
                                                                                                               epoch)
        print('************ Below is validation evaluation of epoch {} results ! ************'.format(
            epoch))
        print('Accuracy average is:{}'.format(np.mean(acc)))
        print('Sensitivity average is:{}'.format(np.mean(sensitivity)))
        print('Specificity average is:{}'.format(np.mean(specificity)))
        print('Precision average is:{}'.format(np.mean(precision)))
        print('G average is:{}'.format(np.mean(G)))
        print('F1_score average is:{}'.format(np.mean(F1_score)))
        print('Mse average is:{}'.format(np.mean(mse)))
        print('Iou average is:{}'.format(np.mean(iou)))
        print('Hausdorff_distance average is:{}'.format(np.mean(hausdorff_dis)))
        writer.add_scalar('Val/accuracy', np.mean(acc), epoch)
        writer.add_scalar('Val/sensitivity', np.mean(sensitivity), epoch)
        writer.add_scalar('Val/f1score', np.mean(F1_score), epoch)
        writer.add_scalar('Val/valloss', np.mean(val_loss), epoch)
        print('********************************************************************************')

        if np.mean(acc) > val_best:
            print('========================================================================')
            val_best = np.mean(acc)
            model_name = './log/visual_weights/' + "val_best.iter"
            torch.save(net, model_name)
            print('********************************************************************')
            print('          Model {}.iter  have been saved !          '.format(epoch))
            print('********************************************************************')
            no_optim = 0
        else:
            no_optim += 1
        if no_optim > Constants.NUM_EARLY_STOP:
            print('Early stop at %d epoch' % epoch)
            break

    print('***************** Have finished training process ! ***************** ')
    pass

if __name__ == '__main__':
    train_nuclei()
    pass