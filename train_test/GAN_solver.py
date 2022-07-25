import sys
sys.path.append('../data_process/')
sys.path.append('../networks/')
sys.path.append('../')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optimizer
import torch.utils.data as Pdata
import torch.nn.functional as F
from data_load import ImageFolder, get_Object_info
# import gan_ultils as utils
from GAN_model import generator, discriminator
import Constants_gan
from Nuclei_test import val_nuclei
from torch.utils.tensorboard import SummaryWriter

device = 'cuda' if torch.cuda.is_available() else 'cpu'

'''
We use GAN (generative adversial networks) to increase RCSAU-Net robustness to segment nuclei objects 
Our method is based on image pairs strategy 
'''
class PatchDataset(Pdata.Dataset):
    def __init__(self, img_arr, img_label):
        self.img_arr = torch.tensor(img_arr, dtype=torch.float32)
        self.img_label = torch.tensor(img_label, dtype=torch.float32)

    def __len__(self):
        return self.img_arr.size()[0]

    def __getitem__(self, index):
        self.data_img = self.img_arr[index]
        self.data_label = self.img_label[index]
        return self.data_img, self.data_label

class gen(nn.Module):
    def __init__(self):
        super(gen, self).__init__()
        self.gene = generator()
    def forward(self, x):
        return self.gene(x)

class dis(nn.Module):
    def __init__(self):
        super(dis, self).__init__()
        self.dise = discriminator(Constants_gan.discriminator_type)
    def forward(self, x):
        return self.dise(x)

class Solver():
    def __init__(self):
        self.dataset = Constants_gan.dataset
        self.flags = Constants_gan.is_test
        self.train_sample_num = 20
        self.best_auc_sum = 0.
        self.alpha_recip = 1. * Constants_gan.ratio_gan2seg if  Constants_gan.ratio_gan2seg > 0 else 0
        self.mse_ratio = Constants_gan.mse_ratio
        # trains, val = get_drive_data()
        # self.images, self.vessel = trains[0], trains[1]
        # self.val_img,  self.val_vessel = val[0], val[1]
        # self.test_img, self.test_vessel = get_drive_data(is_train=False)
        # print('have loaded all train images, shape(img, vessel) is: ', self.images.shape, self.vessel.shape)
        # print('have loaded all val images, shape(img,vessel, mask)is:', self.val_img.shape, self.val_vessel.shape,
        #       self.test_img.shape)
        self.epoch = Constants_gan.iters
        self.train_interval = Constants_gan.train_interval

    def train_phase(self, gen, dis, X, Y):
        d_loss, g_loss = 0, 0
        g_samples = gen(X)
        real_pair = torch.cat([X, Y], dim=1)
        fake_pair = torch.cat([X, g_samples], dim=1)

        d_logit_real = dis(real_pair)
        d_logit_fake = dis(fake_pair)

        for d_i in range(0, Constants_gan.train_interval):
            d_loss_real = nn.BCELoss()(d_logit_real, torch.ones_like(d_logit_real)) +\
                          self.mse_ratio * F.smooth_l1_loss(d_logit_real, torch.ones_like( d_logit_real))
            d_loss_fake = nn.BCELoss()(d_logit_fake, torch.zeros_like(d_logit_fake)) +\
                          self.mse_ratio * F.smooth_l1_loss(d_logit_fake, torch.zeros_like(d_logit_fake))
            d_loss = d_loss_real + d_loss_fake

        for g_i in range(0, Constants_gan.train_interval):
            gan_loss = nn.BCELoss()(d_logit_fake.detach(), torch.ones_like(d_logit_fake))+\
                       self.mse_ratio * F.smooth_l1_loss(d_logit_fake.detach(), torch.ones_like(d_logit_fake))
            seg_loss = nn.BCELoss()(g_samples, Y) + self.mse_ratio * F.smooth_l1_loss(g_samples, Y)
            g_loss =  gan_loss + self.alpha_recip *seg_loss

        return d_loss, g_loss, g_samples

    def train_GAN(self):
        best_acc = -1
        gene_me= generator().to(device)
        dis_me  = discriminator().to(device)
        trains, val = get_Object_info()
        images, vessel = trains[0], trains[1]
        val_img,  val_vessels = val[0], val[1]
        datasets = PatchDataset(images, vessel)
        train_loader = Pdata.DataLoader(datasets, batch_size=Constants_gan.batch_size, shuffle=True, num_workers=0)
        dis_op = optimizer.Adam(gene_me.parameters(), lr=Constants_gan.learning_rate * 1, betas=(Constants_gan.beta1, 0.999))
        gen_op = optimizer.Adam(dis_me.parameters(), lr=Constants_gan.learning_rate, betas=(Constants_gan.beta1, 0.999))
        writer = SummaryWriter(comment = f"LR_train02")
        rand_img, rand_label, rand_pred = None, None, None
        for iter_time in range(0, self.epoch):
            print('********************** Start to train {} Epoch ! ***********************'.format(iter_time))
            d_t_loss, g_t_loss = 0, 0
            gene_me.train()
            dis_me.train()
            for index, data in enumerate(train_loader):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                gen_op.zero_grad()
                dis_op.zero_grad()
                d_loss, g_loss, g_samples = self.train_phase(gene_me, dis_me, inputs, labels)
                d_t_loss = d_t_loss + d_loss.item()
                g_t_loss = g_t_loss + g_loss.item()
                print('Have finished number {} Epoch  mini batch {} training !'.format(iter_time, index))
                d_loss.backward(retain_graph=True)
                g_loss.backward()
                gen_op.step()
                dis_op.step()
                if np.random.rand(1) > 0.2 and np.random.rand(1) < 0.8:
                    rand_img, rand_label, rand_pred = inputs, labels, g_samples

            if iter_time % 1 == 0:
                writer.add_image('Train/image_origins', rand_img[0, :, :, :], iter_time)
                writer.add_image('Train/image_labels', rand_label[0, :, :, :], iter_time)
                writer.add_image('Train/image_predictions', rand_pred[0, :, :, :], iter_time)

            print( 'After Epoch {}, d_loss is {}, g_loss is {}.'.format(iter_time,
                                                                           d_t_loss / len(train_loader),
                                                                           g_t_loss / len(train_loader)))
            writer.add_scalar('Loss/D_train', d_t_loss / len(train_loader), iter_time)
            writer.add_scalar('Loss/G_train', g_t_loss / len(train_loader), iter_time)

            new_masks = labels[0, :, :, :].cpu().clone()
            new_preds = g_samples[0, :, :, :].cpu().clone()
            new_masks = new_masks.reshape((-1, ))
            new_preds = new_preds.reshape((-1, ))
            new_preds[torch.where(new_preds >= 0.5)] = 1
            new_preds[torch.where(new_preds < 0.5)] = 0
            t_t = new_preds[torch.where(new_masks == new_preds)].size()[0]
            writer.add_scalar('ACC/train', t_t / new_preds.size()[0], iter_time)

            print('************ Start to validate current model {}.iter performance ! ************'.format(iter_time))
            acc, sen, f1score,specificity, precision, G, mse, iou, hausdorff_dis, val_loss = val_nuclei(gene_me, val_img, val_vessels, val_img.shape[0], iter_time)
            writer.add_scalar('Val/accuracy', np.mean(acc), iter_time)
            writer.add_scalar('Val/sensitivity', np.mean(sen), iter_time)
            writer.add_scalar('Val/f1score', np.mean(f1score), iter_time)
            writer.add_scalar('Val/valloss', np.mean(val_loss), iter_time)
            print('*********************************************************************')
            if iter_time%50 ==0:
                self.save_model(gene_me, iter_time)
            if np.mean(acc) > best_acc:
                best_acc = np.mean(acc)
                self.save_model(gene_me, 'val_best')
        self.save_model(gene_me, self.epoch)
        return

    def save_model(self, net, iter_time):
        if iter_time == 'val_best':
            model_name = "../log/visual_weights/Object/val_best.iter"
            print('=============================================================================')
            print('                 Best Model parameter have been saved !                   ')
            print('=============================================================================\n')
        else:
            model_name = "../log/visual_weights/Object/{}.iter".format(iter_time)
        torch.save(net, model_name)

    def load_model(self, turns =100):
        print('[*] Reading checkpoint...')
        path = './logs/'+str(turns)+'.iter'
        net = torch.load(path)
        return net

if __name__ == '__main__':
    solver = Solver()
    print('--------------------- Have loaded all dataset images ! ---------------------')
    solver.train_GAN()
    print('--------------------- Have finished training process ! ---------------------')