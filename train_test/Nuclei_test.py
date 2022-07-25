import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
sys.path.append('../')
sys.path.insert(0, '../data_process/')
sys.path.insert(0, '../networks/')

import Constants
from old_evaluation import misc_measures, roc_pr_curve, threshold_by_otsu
from data_load import get_Contour_info, get_Object_info, ImageFolder
from difference_nuclei import retina_color_different
from overlap_HE import overlap_contour, overlap_object, overlap_object_contour, overlap_gt_co
from Evaluation import Emetrics
from Instance import instance_seg_results

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optims
import torch.utils.data as data
from torch.autograd import Variable as V
import sklearn.metrics as metrics
import cv2
import os
import seaborn as sns
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model(path):
    print('[*] Reading checkpoint...')
    net = torch.load(path, map_location = device)
    return net

def visualize(data, filename):
    '''
    :param data:     input is 3d tensor of a image,whose size is (H*W*C)
    :param filename:
    :return:         saved into filename positions
    '''
    assert (len(data.shape) == 3)  # height*width*channels
    # print data
    if data.shape[2] == 1:  # in case it is black and white
        data = np.reshape(data, (data.shape[0], data.shape[1]))
    if np.max(data) > 1:
        img = Image.fromarray(data.astype(np.uint8))  # the image is already 0-255
    else:
        img = Image.fromarray((data * 255).astype(np.uint8))  # the image is between 0-1
    img.save(filename + '.png')
    return img


def val_nuclei(net, imgs, masks, length, epoch=0, ch=Constants.BINARY_CLASS):
    acc, sensitivity, specificity, precision, G, F1_score, mse, iou, hausdorff_dis, val_loss = [], [], [], [], [], [], [], [], [], []
    net.eval()
    with torch.no_grad():
        for iteration in range(0, length):
            net.eval()
            x_img = imgs[iteration]
            x_img = np.expand_dims(x_img, axis=0)  # from (H, W, C) to (1, H, W, C)
            x_img = torch.tensor(x_img, dtype=torch.float32).to(device)
            print('Number {} image of val set size is: {}'.format(iteration,x_img.size()))
            generated_vessel = crop_eval(net, x_img)
            # vl = loss_ce(generated_vessel.detach().cpu().reshape((-1,)),
            #                   torch.tensor(masks[iteration].reshape((-1,)), dtype=torch.float),
            #                   nn.CrossEntropyLoss())
            # val_loss.append(vl.numpy())
            generated_vessel = generated_vessel.permute((0, 2, 3, 1)).detach().cpu().numpy()
            if ch == 1:   # for [N, 1, H, W]
                visualize(np.asarray(generated_vessel[0, :, :, :, ]),
                          Constants.visual_results + 'val_prob_pic' + str(iteration))
                #threshold = 0.5
                #generated_vessel[generated_vessel >= threshold] = 1
                #generated_vessel[generated_vessel < threshold] = 0
                generated_vessel = threshold_by_otsu(generated_vessel)
            if ch == 2:   # for [N, H, W, 2]
                generated_vessel = np.expand_dims(np.argmax(generated_vessel, axis=3), axis=3)
            generated_vessel = np.squeeze(generated_vessel, axis=0)     # from (1, H, W, 1) to (H, W, 1)
            visualize(np.asarray(generated_vessel), Constants.visual_results + 'val_pic' + str(iteration))
            metrics_current = misc_measures(masks[iteration].reshape((-1,)), generated_vessel.reshape((-1,)), False)
            acc.append(metrics_current[0])
            sensitivity.append(metrics_current[1])
            specificity.append(metrics_current[2])
            precision.append(metrics_current[3])
            G.append(metrics_current[4])
            F1_score.append(metrics_current[5])
            mse.append(metrics_current[6])
            iou.append(metrics_current[7])
            hausdorff_dis.append(metrics_current[8])

    return acc, sensitivity, F1_score,specificity, precision, G, mse, iou, hausdorff_dis, val_loss,

def heatmap_retinal(data, path):
    ax = sns.heatmap(data, cmap='viridis')
    sns.despine(fig=None, ax=None, top=True, right=True, left=True, bottom=True, offset=None, trim=False)
    plt.tick_params(labelleft=False, left=False, labelbottom= False,bottom=False)
    plt.savefig(path)
    plt.show()
    plt.close()

def crop_eval(net, image, crop_size=Constants.resize_drive):
    '''
    :param net:       proposed net
    :param image:     image is tensor form of [N, C, H, W], shape is (584, 565)
    :param crop_size: 512 default
    :return:          (584, 565)
    '''
    d_h, d_w, h, w = image.size()[2] - crop_size, image.size()[3] - crop_size, image.size()[2], image.size()[3]
    crop_lu_im = image[:, :, 0:h - d_h, 0:w - d_w]
    crop_ld_im = image[:, :, 0:h - d_h, d_w:w]
    crop_ru_im = image[:, :, d_h:h, 0:w - d_w]
    crop_rd_im = image[:, :, d_h:h, d_w:w]

    # lu,_ = net(crop_lu_im)
    # ru,_ = net(crop_ld_im)
    # ld,_ = net(crop_ru_im)
    # rd,_ = net(crop_rd_im)

    lu = net(crop_lu_im)
    ru = net(crop_ld_im)
    ld = net(crop_ru_im)
    rd = net(crop_rd_im)

    if Constants.BINARY_CLASS ==1:
        new_image = torch.zeros_like(torch.unsqueeze(image[:,0,:,:,], dim=1))
    else:
        new_image = torch.zeros(size=(image.size()[0], 2, image.size()[2],image.size()[3]))

    for i in range(0, h):
        for j in range(0, w):
            if i >= d_h and j >= d_w and i < crop_size and j < crop_size:
                new_image[:, :, i, j] = (lu[:, :, i, j] + ru[:, :, i, j - d_w] + ld[:, :, i - d_h, j] + rd[:, :,
                                                                                                        i - d_h,
                                                                                                        j - d_w]) / 4
            if i >= 0 and j >= 0 and i < d_h and j < d_w:
                new_image[:, :, i, j] = lu[:, :, i, j]
            if i >= 0 and j >= d_w and i < d_h and j < crop_size:
                new_image[:, :, i, j] = (lu[:, :, i, j] + ru[:, :, i, j - d_w]) / 2
            if i >= 0 and j >= crop_size and i < d_h:
                new_image[:, :, i, j] = ru[:, :, i, j - d_w]
            if i >= d_h and j >= 0 and i < crop_size and j < d_w:
                new_image[:, :, i, j] = (lu[:, :, i, j] + ld[:, :, i - d_h, j]) / 2
            if i >= d_h and j >= crop_size and i < crop_size:
                new_image[:, :, i, j] = (ru[:, :, i, j - d_w] + rd[:, :, i - d_h, j - d_w]) / 2
            if i >= crop_size and j >= 0 and j < d_w:
                new_image[:, :, i, j] = ld[:, :, i - d_h, j]
            if i >= crop_size and j >= d_w and j < crop_size:
                new_image[:, :, i, j] = (ld[:, :, i - d_h, j] + rd[:, :, i - d_h, j - d_w]) / 2
            if i >= crop_size and j > crop_size:
                new_image[:, :, i, j] = rd[:, :, i - d_h, j - d_w]

    return new_image.to(device)

def test_nuclei_Contour(path, ch=Constants.BINARY_CLASS):
    images, masks, origin_test = get_Contour_info(is_train=False)
    acc, sensitivity, specificity, precision, recall, F1_score, Aji, hausdorff_dis, Dice = [], [], [], [], [], [], [], [], []
    pr_g, pr_l = [], []
    with torch.no_grad():
        net = load_model(path)
        net.eval()
        for iter_ in range(int(Constants.total_MONUCLEI_test)):
            x_img = images[iter_]
            x_img = np.expand_dims(x_img, axis=0)
            x_img = torch.tensor(x_img, dtype=torch.float32).to(device)
            generated_vessel = crop_eval(net, x_img)
            print('[*] calculate image {} , please waiting ...'.format(iter_))
            generated_vessel = generated_vessel.permute((0, 2, 3, 1)).detach().cpu().numpy()
            generated_vessel_ = generated_vessel
            if ch == 1:  # for [N, 1, H, W]
                pr_g.append(generated_vessel.reshape((-1,)).tolist())
                pr_l.append(masks[iter_].reshape((-1,)).tolist())
                heatmap_retinal(generated_vessel_[0, :, :, 0,], Constants.visual_results + str(iter_) + 'heatmap')
                visualize(np.asarray(generated_vessel_[0, :, :, :, ]), Constants.visual_results + str(iter_) + 'prob')
                threshold = 0.5
                generated_vessel_[generated_vessel_ >= threshold] = 1
                generated_vessel_[generated_vessel_ < threshold] = 0
                generated_vessel[generated_vessel>= threshold] = 1
                generated_vessel[generated_vessel < threshold] = 0
                # generated_vessel = threshold_by_otsu(generated_vessel)
                # generated_vessel_ = threshold_by_otsu(generated_vessel_)
            if ch == 2:  # for [N, 2, H, W]
                generated_vessel = np.expand_dims(np.argmax(generated_vessel_, axis=3), axis=3)
                pr_g.append(generated_vessel.reshape((-1,)).tolist())
                pr_l.append(masks[iter_].reshape((-1,)).tolist())
            generated_vessel_ = np.squeeze(generated_vessel_, axis=0)   # From (1, H, W, 1) to (H, W, 1)
            visualize(np.transpose(origin_test[iter_],(1,2,0)), Constants.visual_results + str(iter_) + 'origin')
            visualize(np.asarray(generated_vessel_), Constants.visual_results + str(iter_) + 'seg')
            visualize(origin_test[iter_].transpose((1,2,0)),Constants.visual_results + str(iter_) + 'image')
            visualize(masks[iter_].transpose((1,2,0)),Constants.visual_results + str(iter_) + 'mask')
            new_color_retina = retina_color_different(np.asarray(generated_vessel_), masks[iter_].transpose((1,2,0)),
                                   Constants.visual_results + str(iter_) + 'different')  # different map
            visualize(overlap_contour(np.asarray(origin_test[iter_]), new_color_retina),Constants.visual_results + str(iter_) + 'overlap')
            ACC, SEN, SPE, Precision, Recall, F1_Score, AJI, hds, dice1 = Emetrics(masks[iter_], generated_vessel)
            # metrics_current = misc_measures(masks[iter_].reshape((-1,)), generated_vessel.reshape((-1,)))
            # metrics_current = misc_measures(masks[iter_].reshape((-1,)), generated_vessel.reshape((-1,)))
            acc.append(ACC)
            sensitivity.append(SEN)
            specificity.append(SPE)
            precision.append(Precision)
            recall.append(Recall)
            F1_score.append(F1_Score)
            Aji.append(AJI)
            hausdorff_dis.append(hds)
            Dice.append(dice1)
            print(
                'test image: {} Metrics **** Accuracy is : {}, sensitivity is : {}, specificity is : {},precision is : {},recall is : {},F1_score is : {},AJI is : {},'
                'Hausdorff distance is: {},Dice1 is: {} ****'.format(iter_, ACC, SEN,SPE, Precision,Recall,
                                                                     F1_Score, AJI, hds, dice1))
        AUC_prec_rec, AUC_ROC = roc_pr_curve(np.array(pr_l).reshape((-1,)), np.array(pr_g).reshape((-1,)),
                                             Constants.visual_results)
        path_files_saved = Constants.visual_results + 'evaluation.txt'
        print('********************** Final test results has been saved in to {} **********************'.format(
            path_files_saved))
        str_a = 'Area of PR curve is: {}, Area of ROC curve is: {}'.format(AUC_prec_rec, AUC_ROC)
        str_b = 'Accuracy average is: {}, std is: {}'.format(np.mean(acc), np.std(acc))
        str_c = 'Sensitivity average is: {}, std is: {}'.format(np.mean(sensitivity), np.std(sensitivity))
        str_d = 'Specificity average is: {}, std is: {}'.format(np.mean(specificity), np.std(specificity))
        str_e = 'Precision average is: {}, std is: {}'.format(np.mean(precision), np.std(precision))
        str_f = 'Recall average is: {}, std is: {}'.format(np.mean(recall), np.std(recall))
        str_g = 'F1_Score average is:{}, std is: {}'.format(np.mean(F1_score), np.std(F1_score))
        str_h = 'AJI average is: {}, std is: {}'.format(np.mean(Aji), np.std(Aji))
        str_i = 'Hausdorff distance: {}, std is: {}'.format(np.mean(hausdorff_dis), np.std(hausdorff_dis))
        str_j = 'Dice1: {}, std is: {}'.format(np.mean(Dice), np.std(Dice))
        f = open(path_files_saved, 'w', encoding='utf-8')
        f.write(str_a + '\n')
        f.write(str_b + '\n')
        f.write(str_c + '\n')
        f.write(str_d + '\n')
        f.write(str_e + '\n')
        f.write(str_f + '\n')
        f.write(str_g + '\n')
        f.write(str_h + '\n')
        f.write(str_i + '\n')
        f.write(str_j + '\n')
        f.close()
        print('****************** Have finished predicted all images, all results saved ! ******************')

def test_nuclei_Object(path, ch = Constants.BINARY_CLASS):
    images, masks, origin_test = get_Object_info(is_train=False)
    acc, sensitivity, specificity, precision, recall, F1_score, Aji, hausdorff_dis, Dice = [], [], [], [], [], [], [], [], []
    pr_g, pr_l = [], []
    with torch.no_grad():
        net = load_model(path)
        net.eval()
        for iter_ in range(int(Constants.total_MONUCLEI_test)):
            x_img = images[iter_]
            x_img = np.expand_dims(x_img, axis=0)
            x_img = torch.tensor(x_img, dtype=torch.float32).to(device)
            generated_vessel = crop_eval(net, x_img)
            print('[*] calculate image {} , please waiting ...'.format(iter_))
            generated_vessel = generated_vessel.permute((0, 2, 3, 1)).detach().cpu().numpy()
            generated_vessel_ = generated_vessel
            if ch == 1:  # for [N, 1, H, W]
                pr_g.append(generated_vessel.reshape((-1,)).tolist())
                pr_l.append(masks[iter_].reshape((-1,)).tolist())
                heatmap_retinal(generated_vessel_[0, :, :, 0,], Constants.visual_results + str(iter_) + 'heatmap')
                visualize(np.asarray(generated_vessel_[0, :, :, :, ]), Constants.visual_results + str(iter_) + 'prob')
                threshold = 0.5
                generated_vessel_[generated_vessel_ >= threshold] = 1
                generated_vessel_[generated_vessel_ < threshold] = 0
                generated_vessel[generated_vessel>= threshold] = 1
                generated_vessel[generated_vessel < threshold] = 0
                # generated_vessel = threshold_by_otsu(generated_vessel)
                # generated_vessel_ = threshold_by_otsu(generated_vessel_)
            if ch == 2:  # for [N, 2, H, W]
                generated_vessel = np.expand_dims(np.argmax(generated_vessel_, axis=3), axis=3)
                pr_g.append(generated_vessel.reshape((-1,)).tolist())
                pr_l.append(masks[iter_].reshape((-1,)).tolist())
            generated_vessel_ = np.squeeze(generated_vessel_, axis=0)   # From (1, H, W, 1) to (H, W, 1)
            visualize(np.transpose(origin_test[iter_],(1,2,0)), Constants.visual_results + str(iter_) + 'origin')
            visualize(np.asarray(generated_vessel_), Constants.visual_results + str(iter_) + 'seg')
            visualize(origin_test[iter_].transpose((1,2,0)),Constants.visual_results + str(iter_) + 'image')
            visualize(masks[iter_].transpose((1,2,0)),Constants.visual_results + str(iter_) + 'mask')
            new_color_retina = retina_color_different(np.asarray(generated_vessel_), masks[iter_].transpose((1,2,0)),
                                   Constants.visual_results + str(iter_) + 'different')  # different map
            visualize(overlap_object(np.asarray(origin_test[iter_]), new_color_retina), Constants.visual_results + str(iter_) + 'overlap')
            visualize(instance_seg_results(Constants.visual_results + str(iter_) + 'seg.png'), Constants.visual_results + str(iter_) + 'instance_seg')
            ACC, SEN, SPE, Precision, Recall, F1_Score, AJI, hds, dice1 = Emetrics(masks[iter_], generated_vessel)
            # metrics_current = misc_measures(masks[iter_].reshape((-1,)), generated_vessel.reshape((-1,)))
            # metrics_current = misc_measures(masks[iter_].reshape((-1,)), generated_vessel.reshape((-1,)))
            acc.append(ACC)
            sensitivity.append(SEN)
            specificity.append(SPE)
            precision.append(Precision)
            recall.append(Recall)
            F1_score.append(F1_Score)
            Aji.append(AJI)
            hausdorff_dis.append(hds)
            Dice.append(dice1)
            print(
                'test image: {} Metrics **** Accuracy is : {}, sensitivity is : {}, specificity is : {},precision is : {},recall is : {},F1_score is : {},AJI is : {},'
                'Hausdorff distance is: {},Dice1 is: {} ****'.format(iter_, ACC, SEN,SPE, Precision,Recall,
                                                                     F1_Score, AJI, hds, dice1))
        AUC_prec_rec, AUC_ROC = roc_pr_curve(np.array(pr_l).reshape((-1,)), np.array(pr_g).reshape((-1,)),
                                             Constants.visual_results)
        path_files_saved = Constants.visual_results + 'evaluation.txt'
        print('********************** Final test results has been saved in to {} **********************'.format(
            path_files_saved))
        str_a = 'Area of PR curve is: {}, Area of ROC curve is: {}'.format(AUC_prec_rec, AUC_ROC)
        str_b = 'Accuracy average is: {}, std is: {}'.format(np.mean(acc), np.std(acc))
        str_c = 'Sensitivity average is: {}, std is: {}'.format(np.mean(sensitivity), np.std(sensitivity))
        str_d = 'Specificity average is: {}, std is: {}'.format(np.mean(specificity), np.std(specificity))
        str_e = 'Precision average is: {}, std is: {}'.format(np.mean(precision), np.std(precision))
        str_f = 'Recall average is: {}, std is: {}'.format(np.mean(recall), np.std(recall))
        str_g = 'F1_Score average is:{}, std is: {}'.format(np.mean(F1_score), np.std(F1_score))
        str_h = 'AJI average is: {}, std is: {}'.format(np.mean(Aji), np.std(Aji))
        str_i = 'Hausdorff distance: {}, std is: {}'.format(np.mean(hausdorff_dis), np.std(hausdorff_dis))
        str_j = 'Dice1: {}, std is: {}'.format(np.mean(Dice), np.std(Dice))
        f = open(path_files_saved, 'w', encoding='utf-8')
        f.write(str_a + '\n')
        f.write(str_b + '\n')
        f.write(str_c + '\n')
        f.write(str_d + '\n')
        f.write(str_e + '\n')
        f.write(str_f + '\n')
        f.write(str_g + '\n')
        f.write(str_h + '\n')
        f.write(str_i + '\n')
        f.write(str_j + '\n')
        f.close()
        print('****************** Have finished predicted all images, all results saved ! ******************')

def test_final_merge(path_contour, path_object):
    images, masks_contour, origin_test = get_Contour_info(is_train=False)
    _, masks_object, _ = get_Object_info(is_train=False)
    with torch.no_grad():
        net_contour = load_model(path_contour)
        net_object  = load_model(path_object)
        net_contour.eval()
        net_object.eval()

        for iter_ in range(int(Constants.total_MONUCLEI_test)):
            x_img = images[iter_]
            x_img = np.expand_dims(x_img, axis=0)
            x_img = torch.tensor(x_img, dtype=torch.float32).to(device)
            generated_contour = crop_eval(net_contour, x_img)
            generated_object = crop_eval(net_object, x_img)
            print('[*] calculate image {} , please waiting ...'.format(iter_))
            generated_contour = generated_contour.permute((0, 2, 3, 1)).detach().cpu().numpy()
            generated_object = generated_object.permute((0, 2, 3, 1)).detach().cpu().numpy()

            if True:
                threshold = 0.5
                generated_object[generated_object >= threshold] = 1
                generated_object[generated_object < threshold] = 0
                generated_contour[generated_contour >= threshold] = 1
                generated_contour[generated_contour < threshold] = 0
                # generated_vessel = threshold_by_otsu(generated_vessel)
                # generated_vessel_ = threshold_by_otsu(generated_vessel_)

            generated_contour = np.squeeze(generated_contour, axis=0)  # From (1, H, W, 1) to (H, W, 1)
            generated_object = np.squeeze(generated_object, axis=0)    # From (1, H, W, 1) to (H, W, 1)
            print('==================== Here ==================== ', origin_test[iter_].shape)
            visualize(np.transpose(origin_test[iter_], (1, 2, 0)), Constants.visual_results + str(iter_) + 'origin')
            visualize(overlap_object_contour(origin_test[iter_], generated_contour, generated_object), Constants.visual_results + str(iter_) + 'final_seg')
            visualize(overlap_gt_co(origin_test[iter_], masks_contour[iter_], masks_object[iter_]), Constants.visual_results + str(iter_) + 'mask')

    pass

if __name__ == '__main__':

    path_contour = '../log/visual_weights/Contour/val_best.iter'
    path_object = '../log/visual_weights/Object/val_best.iter'

    # test_nuclei_Contour(path_contour)
    test_nuclei_Object(path_object)
    # test_final_merge(path_contour, path_object)

    pass