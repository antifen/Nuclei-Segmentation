# --------------------------------------------------------
# import time
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import collections
import numpy as np

# from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optimizer
import torch.utils.data as Pdata

from data import Dataset
import gan_ultils as utils
from gan_model import generator, discriminator
import Constant

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class PatchDataset(Pdata.Dataset):
    def __init__(self, imgs_arr, imgs_label):
        self.imgs_arr = torch.tensor(imgs_arr, dtype=torch.float32)
        self.imgs_label = torch.tensor(imgs_label, dtype=torch.float32)

    def __len__(self):
        return self.imgs_arr.size()[0]

    def __getitem__(self, index):
        self.data_img = self.imgs_arr[index]
        self.data_label = self.imgs_label[index]
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
        self.dise = discriminator()

    def forward(self, x):
        return self.dise(x)


class GAN(nn.Module):
    def __init__(self):
        super(GAN, self).__init__()
        self.gen = generator()
        self.dis = discriminator()
        self.alpha_recip = 1. / Constant.ratio_gan2seg if Constant.ratio_gan2seg > 0 else 0

    def forward(self, X, Y):
        d_loss, g_loss = 0, 0
        g_samples = self.gen(X)
        real_pair = torch.cat([X, Y], dim=1)           # real pairs
        fake_pair = torch.cat([X, g_samples], dim=1)   # fake pairs
        print('value range check:', torch.max(g_samples), torch.max(Y), torch.min(g_samples), torch.min(Y))
        d_logit_real = self.dis(real_pair)
        d_logit_fake = self.dis(fake_pair)

        for g_i in range(0, Constant.train_interval):
            # print('train generator')
            gan_loss = nn.MSELoss()(d_logit_fake.detach(), torch.ones_like(d_logit_fake))
            seg_loss = nn.MSELoss()(g_samples, Y)
            #             gan_loss = torch.square(d_logit_fake.detach() - torch.ones_like(d_logit_fake)).mean()
            #             seg_loss = torch.square(g_samples - Y).mean()
            g_loss = self.alpha_recip * gan_loss + seg_loss

        for d_i in range(0, Constant.train_interval):
            # print('train discriminator')
            #             print(torch.max(real_pair), torch.max(fake_pair), d_logit_real, d_logit_fake)
            d_loss_real = nn.MSELoss()(d_logit_real, torch.ones_like(d_logit_real))
            d_loss_fake = nn.MSELoss()(d_logit_fake, torch.zeros_like(d_logit_fake))
            #             d_loss_real = torch.square(d_logit_real - torch.ones_like(d_logit_real)).sum()
            #             d_loss_fake = torch.square(d_logit_fake - torch.zeros_like(d_logit_fake)).sum()
            d_loss = d_loss_real + d_loss_fake

        return d_loss, g_loss


class Solver():
    def __init__(self):

        self.dataset = 'DRIVE'
        self.flags = Constant.is_test
        self.train_sample_num = 20

        self.dataset = Dataset(self.dataset, self.flags)

        self.best_auc_sum = 0.

        self.alpha_recip = 1. / Constant.ratio_gan2seg if Constant.ratio_gan2seg > 0 else 0

        self.images, self.cells, self.mask = self.dataset.val_imgs, self.dataset.val_cells, self.dataset.val_masks
        # self.images,self.cells,self.mask = self.dataset.read_val_data()

        self.train_img, self.train_cells = self.obtain_train_set()

        # self.images = self.channel_first(self.images)
        # self.cells = self.channel_first(self.cells)
        # self.mask = self.channel_first(self.mask)

        self.train_img = self.channel_first(self.train_img)
        self.train_cells = self.channel_first(self.train_cells)

        print('have loaded all train images, shape(img, cells) is: ', self.train_img.shape, self.train_cells.shape)
        print('have loaded all val images, shape(img,cells, mask)is:', self.images.shape, self.cells.shape,
              self.mask.shape)
        # print(np.max(self.train_cells), np.max(self.train_img))

        self.epoch = Constant.iters
        self.train_interval = Constant.train_interval

    def channel_first(self, data):
        return np.transpose(data, (0, 3, 1, 2))

    def obtain_train_set(self):
        train_img, train_cells = self.dataset.train_next_batch(self.train_sample_num)
        #         print(train_img.shape, train_cells,'------------')

        return train_img, train_cells

    #     def set_requires_grad(self, net: nn.Module, mode=True):
    #         for p in net.parameters():
    #             p.requires_grad_(mode)

    def train_phase(self, gen, dis, X, Y):
        d_loss, g_loss = 0, 0
        g_samples = gen(X)
        real_pair = torch.cat([X, Y], dim=1)
        #         print(X.size(), g_samples.size())
        fake_pair = torch.cat([X, g_samples], dim=1)
        #         print('value range check:', torch.max(g_samples), torch.max(Y), torch.min(g_samples), torch.min(Y))
        d_logit_real = dis(real_pair)
        d_logit_fake = dis(fake_pair)

        for d_i in range(0, Constant.train_interval):
            # print('here 1')
            #             print(torch.max(real_pair), torch.max(fake_pair), d_logit_real, d_logit_fake)
            #             d_loss_real = nn.MSELoss()(d_logit_real, torch.ones_like(d_logit_real))
            #             d_loss_fake = nn.MSELoss()(d_logit_fake, torch.zeros_like(d_logit_fake))
            d_loss_real = nn.BCELoss()(d_logit_real, torch.ones_like(d_logit_real))
            d_loss_fake = nn.BCELoss()(d_logit_fake, torch.zeros_like(d_logit_fake))
            d_loss = d_loss_real + d_loss_fake
        #             self.set_requires_grad(self.dis, False)
        #             print(d_logit_real,d_logit_fake,'------------------')

        for g_i in range(0, Constant.train_interval):
            # print('here 2')
            #             gan_loss = nn.MSELoss()(d_logit_fake.detach(), torch.ones_like(d_logit_fake))
            #             seg_loss = nn.MSELoss()(g_samples, Y)
            gan_loss = nn.BCELoss()(d_logit_fake.detach(), torch.ones_like(d_logit_fake))
            seg_loss = nn.BCELoss()(g_samples, Y)
            g_loss = self.alpha_recip * gan_loss + seg_loss
        #             self.set_requires_grad(self.dis, True)
        #             self.set_requires_grad(self.gen, True)

        return d_loss, g_loss

    def train_GAN(self):
        gene_me = gen().to(device)
        dis_me = dis().to(device)

        datasets = PatchDataset(self.train_img, self.train_cells)
        train_loader = Pdata.DataLoader(datasets, batch_size=Constant.batch_size, shuffle=True,
                                        num_workers=0)

        dis_op = optimizer.Adam(gene_me.parameters(), lr=Constant.learning_rate * 1, betas=(Constant.beta1, 0.99))
        gen_op = optimizer.Adam(dis_me.parameters(), lr=Constant.learning_rate, betas=(Constant.beta1, 0.99))

        for iter_time in range(0, self.epoch):
            d_t_loss, g_t_loss = 0, 0

            for index, data in enumerate(train_loader):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                # train discriminator and train generator
                gen_op.zero_grad()
                dis_op.zero_grad()
                d_loss, g_loss = self.train_phase(gene_me, dis_me, inputs, labels)
                d_t_loss = d_t_loss + d_loss
                g_t_loss = g_t_loss + g_loss
                print('after epoch {}, d_loss is {}, g_loss is {}.'.format(iter_time + 1, d_t_loss.item(),
                                                                           g_t_loss.item()))
                d_loss.backward(retain_graph=True)
                g_loss.backward()
                gen_op.step()
                dis_op.step()

                d_t_loss, g_t_loss = 0, 0

            if iter_time % 100 == 0 and iter_time != 0:
                # use validation dataset to verify current model
                print('-------------evaluation---------------')
                auc_sum = self.eval_GAN(self.images.shape[0], self.images, self.cells, self.mask, phase='train')
                if self.best_auc_sum < auc_sum:
                    self.best_auc_sum = auc_sum
                    self.save_model(self.gen, iter_time)

        self.save_model(gene_me, self.epoch)

                # d_t_loss, g_t_loss = 0, 0

    def test_GAN(self):
        net = self.load_model()
        net.eval()

        datasets = PatchDataset(self.train_img, self.train_cells)
        train_loader = Pdata.DataLoader(datasets, batch_size=Constant.batch_size, shuffle=True,
                                        num_workers=0)
        for index, data in enumerate(train_loader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outs = net(inputs)
            print(outs.size(), '-------------haha------------')
            generated_cell = outs.reshape((1, 640, 640)).permute((1, 2, 0)).detach().cpu().numpy()
            print('labels.size()', labels.size())
            ground_truth_cell = labels.reshape((1, 640, 640)).permute((1, 2, 0)).detach().cpu().numpy()
            print('check generated cells', generated_cell.shape)
            print('check ground_truth_cells', ground_truth_cell.shape)
            utils.visualize(generated_cell, './logs/' + str(index) + 'cells')
            utils.visualize(ground_truth_cell, './logs/' + str(index) + 'ground_truth')

        self.eval_GAN(self.images.shape[0], self.images, self.cells, self.mask, phase='tests')

    def plot(self, x_imgs, samples, y_imgs, iter_time, idx=None, save_file=None, phase='train'):
        pass

    def print_info(self, iter_time, name, loss):
        if np.mod(iter_time, self.flags.print_freq) == 0:
            ord_output = collections.OrderedDict([(name, loss), ('dataset', self.flags.dataset),
                                                  ('discriminator', self.flags.discriminator),
                                                  ('train_interval', np.float32(self.flags.train_interval)),
                                                  ('gpu_index', self.flags.gpu_index)])
            utils.print_metrics(iter_time, ord_output)

    '''
    use val dataset to validation
    '''

    def eval_GAN(self, num_datas, images, cells, mask, phase='train'):

        ori_shape = (584, 565)
        num_data, imgs, cells, masks = num_datas, images, cells, mask

        generated = []
        for iter_ in range(num_data):
            x_img = imgs[iter_]
            x_img = np.expand_dims(x_img, axis=0)  # (H, W, C) to (1, H, W, C)
            # measure inference time
            x_img = np.transpose(x_img, (0, 3, 1, 2))
            x_img = torch.tensor(x_img, dtype=torch.float32).cuda()
            if phase == 'train':
                generated_cells = self.sample_imgs(x_img)  # generate segment map (1, H, W, 1)
            else:
                path = './logs/700.iter'
                net = torch.load(path)
                generated_cells = net(x_img)
            # generated_cells = np.expand_dims(np.expand_dims(cells[iter_, :, :, ], axis=2), axis=0)
            # assumption is the same with ground-truth
            generated_cells = generated_cells.permute((0, 2, 3, 1)).detach().cpu().numpy()
            print('check generated cells', generated_cells.shape)

            generated.append(np.squeeze(generated_cells, axis=(0, 3)))  # (1, H, W, 1) to (H, W)

        generated = np.asarray(generated)
        # calculate measurements
        auc_sum = self.measure(generated, cells, masks, num_data)

        if phase == 'test':
            # save test images
            segmented_cells = utils.remain_in_mask(generated, masks)
            # crop to original image shape
            imgs_ = utils.crop_to_original(imgs, ori_shape)
            cropped_cells = utils.crop_to_original(segmented_cells, ori_shape)
            cells_ = utils.crop_to_original(cells, ori_shape)

        return auc_sum

    def sample_imgs(self, x_data):
        '''
        :param x_data:
        :return:  generator results
        '''
        # return self.sess.run(self.g_samples, feed_dict={self.X: x_data})
        x_data = torch.tensor(x_data).cuda()
        predicts = self.gen(x_data)
        return predicts

    '''
    the method input parameters must be (h,w,c) format !
    '''

    def measure(self, generated, cells, masks, iter_time):
        import collections
        # masking
        cells_in_mask, generated_in_mask = utils.pixel_values_in_mask(
            cells, generated, masks)
        # evaluate Area Under the Curve of ROC and Precision-Recall
        auc_roc = utils.AUC_ROC(cells_in_mask, generated_in_mask)
        auc_pr = utils.AUC_PR(cells_in_mask, generated_in_mask)
        # binarize to calculate Dice Coeffient
        binarys_in_mask = utils.threshold_by_otsu(generated, masks)
        dice_coeff = utils.dice_coefficient_in_train(cells_in_mask, binarys_in_mask)
        acc, sensitivity, specificity = utils.misc_measures(cells_in_mask, binarys_in_mask)
        score = auc_pr + auc_roc + dice_coeff + acc + sensitivity + specificity
        # auc_sum for saving best model in training
        auc_sum = auc_roc + auc_pr
        # print information
        ord_output = collections.OrderedDict([('auc_pr', auc_pr), ('auc_roc', auc_roc),
                                              ('dice_coeff', dice_coeff), ('acc', acc),
                                              ('sensitivity', sensitivity), ('specificity', specificity),
                                              ('score', score), ('auc_sum', auc_sum),
                                              ])
        utils.print_metrics(iter_time, ord_output)
        return auc_sum

    def save_model(self, net, iter_time):

        model_name = "./logs/{}.iter".format(iter_time)
        torch.save(net, model_name)

        print('===================================================')
        print('                     Model saved!                  ')
        print(' Best auc_sum: {:.3}'.format(self.best_auc_sum))
        print('===================================================\n')

    def load_model(self):
        print(' [*] Reading checkpoint...')
        path = './logs/700.iter'
        net = torch.load(path)
        return net

if __name__ == '__main__':
    solver = Solver()
    print('---------------------have loaded all dataset ---------------------------')
    # solver.train_GAN()
    print('---------------------- finish trained process -----------------------')
    solver.test_GAN()
    print('---------------------- finish tested process -----------------------')