from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from PIL import Image
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage import distance_transform_edt
import  cv2
from skimage.transform import rotate
from skimage.transform import rescale, resize
import os
import time
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import ImageEnhance



# size_h = 448
# size_w = 448

# patch_h = size_h
# patch_w = size_w

def randomRotate90(image, mask, u=0.5):
    if np.random.random() < u:
        image=np.rot90(image)
        mask=np.rot90(mask)
    return image, mask

def randomHorizontalFlip(image, mask):
    if np.random.random() < 0.5:
        image = np.flipud(image)
        mask = np.flipud(mask)
    return  image, mask

def randomVerticleFlip(image, mask):
    if np.random.random() < 0.5:
        image = np.fliplr(image)
        mask = np.fliplr(mask)
    return  image, mask


def deformation_set(image, mask,
                           shift_limit=(-0.1, 0.1),
                           scale_limit=(-0.2, 0.2),
                           rotate_limit=(-0.2, 0.2),
                           aspect_limit=(-0.1, 0.1),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    # print('--------------', image.shape, mask.shape)
    masks = mask
    img = image

    if np.random.random() < u:
#         height, width, channel = img.shape

#         angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
#         scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
#         aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
#         sx = scale * aspect / (aspect ** 0.5)
#         sy = scale / (aspect ** 0.5)
#         dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
#         dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)


#         cc = np.math.cos(angle / 180 * np.math.pi) * sx
#         ss = np.math.sin(angle / 180 * np.math.pi) * sy
#         rotate_matrix = np.array([[cc, -ss], [ss, cc]])

#         box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
#         box1 = box0 - np.array([width / 2, height / 2])
#         box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])


#         box0 = box0.astype(np.float32)
#         box1 = box1.astype(np.float32)
#         mat = cv2.getPerspectiveTransform(box0, box1)
#         img = cv2.warpPerspective(img, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
#                                     borderValue=(
#                                         0, 0,
#                                         0,))
#         masks = cv2.warpPerspective(masks, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
#                                    borderValue=(
#                                        0, 0,
#                                        0,))
#         # print('--------**-------',image.shape, mask.shape)
#         masks = np.expand_dims(masks, axis=2)
        # print('-------<>------',image.shape, masks.shape)

        img, masks = randomHorizontalFlip(img, masks)
        img, masks = randomVerticleFlip(img, masks)
        img, masks = randomRotate90(img, masks)

        # img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
        # masks = np.array(masks, np.float32).transpose(2, 0, 1) / 255.0
        # img = np.array(img, np.float32).transpose(1, 2, 0)
        # masks = np.array(masks, np.float32).transpose(1, 2, 0)
        # print(img.shape, masks.shape,'================')

    return img, masks

def read_all_images(path_images, all_images,size_h, size_w):

    '''
    :param path_images:  the path of data array
    :param all_images:   all_images is numpy array to contain all images channel 1 and 3 is different
    :return:             N H W C
    '''
    index = 0
    for _, _, content in os.walk(path_images):

        # content.sort(key=lambda x: int(x[0:2]))
        print('read content sequence is : \n', content)

        for p_img in list(content):
                img = Image.open(path_images + p_img)
                # print(p_img)
                # enh_con = ImageEnhance.Sharpness(img)
                # img     = enh_con.enhance(factor=2.1)   # ADD augment

                img = img.resize((size_h, size_w))
                img_arr = np.asarray(img)
                # img_arr = randomHueSaturationValue(img_arr)  # add
                if(len(img_arr.shape)==2):   # (1024, 1024)
                    new_img_arr = np.reshape(img_arr,(img_arr.shape[0],img_arr.shape[1], 1)) # (1024, 1024, 1)
                    all_images[index, :, :, :, ] = new_img_arr
                else:
                    all_images[index, :, :, :,] = img_arr           # (1024, 1024, 3)
                index =index+1
    print(' this directory of ({}) has total {} images and images tensor is {}'.format(path_images, index, all_images.shape))
    # visualize(group_images(all_images, 5),'./testimages2'+str(np.random.randint(1,100)))
    return all_images



def group_images(data, per_row):
    '''
    :param data:     both N H W C and N C H W is OK ！
    :param per_row:  every images number in a row
    :return:         images channel last
    '''
    assert data.shape[0]%per_row==0
    if (data.shape[1]==1 or data.shape[1]==3):
        data = np.transpose(data,(0,2,3,1))      # change data format into channel last !
    all_stripe = []
    for i in range(int(data.shape[0]/per_row)):
        stripe = data[i*per_row]
        for k in range(i*per_row+1, i*per_row+per_row):
            stripe = np.concatenate((stripe,data[k]),axis=1)
        all_stripe.append(stripe)
    totimg = all_stripe[0]
    for i in range(1,len(all_stripe)):
        totimg = np.concatenate((totimg,all_stripe[i]),axis=0)
    return totimg

# visualize image (as PIL image, NOT as matplotlib!)
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


def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    images = image
    if np.random.random() < u:
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1]+1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        images = cv2.merge((h, s, v))
        #image = cv2.merge((s, v))
        # image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    return images


def read_MO_images(size_h,size_w):

    total_imgs = 20
    path_images = '../MONUCLEI/training/image/'
    path_gt = '../MONUCLEI/training/gt/'

    all_images = np.empty(shape=(total_imgs, size_h, size_w, 3))
    all_masks  = np.empty(shape=(total_imgs, size_h, size_w, 1))
    all_images = read_all_images(path_images, all_images,size_h,size_w)
    all_masks  = read_all_images(path_gt, all_masks,size_h,size_w)

    print('============= have read all images ==============')
    # print(all_images.shape, all_masks.shape)
    return all_images, all_masks


def data_auguments(aug_num,size_h, size_w):

    all_images, all_masks = read_MO_images(size_h, size_w)         # original data
    # print('image and gt shape is:', all_images.shape, all_masks.shape)
    img_list = []
    gt_list = []
    for nums in range(0, aug_num):
        for i_d in range(0, all_images.shape[0]):
            aug_img, aug_gt = deformation_set(all_images[i_d, :, :, :, ], all_masks[i_d, :, :, :, ])
            img_list.append(np.expand_dims(aug_img, axis=0))
            gt_list.append(np.expand_dims(aug_gt, axis=0))
    img_au = np.concatenate(img_list, axis=0)
    gt_au = np.concatenate(gt_list, axis=0)
    # print(img_au.shape, gt_au.shape)
    # visualize(group_images(all_masks, 5), './image_test')
    return img_au,gt_au

def gt_reshape_3d(gt):
    gt2 = np.zeros(shape=gt.shape)
    gt3 = np.zeros(shape=gt.shape)
    gt = np.concatenate([gt, gt2, gt3], axis=3)     # here they are both (B,H,W,3) shape
    # print('======= now 3 ch gt size is ', gt.shape)
    return gt

def data_for_train(aug_num,size_h, size_w):
    all_images, all_masks = data_auguments(aug_num, size_h, size_w)
    print('image and gt shape is:', all_images.shape, all_masks.shape)
    img = np.array(all_images, np.float32).transpose(0,3,1,2) / 255.0
    mask = np.array(all_masks, np.float32).transpose(0,3,1,2) / 255.0
    # img  = all_images
    # mask = all_masks
    mask[mask >= 0.5] = 1
    mask[mask <= 0.5] = 0
    # print(img.shape, mask.shape)
    #  data water
    index = np.arange(img.shape[0])
    np.random.shuffle(index)
    img  = img[index, :, :, :]
    mask = mask[index, :, :]


    return img, mask


def rand_test1():
    gt_path1 = './dataset/DRIVE/training/1st_manual/21_manual1.gif'
    gt_path2 = './dataset/DRIVE/training/1st_manual/22_manual1.gif'
    gtre1 = np.asarray(Image.open(gt_path1))
    gtre2 = np.asarray(Image.open(gt_path2))

    gtre1 = np.expand_dims(gtre1, axis=2)
    gtre2 = np.expand_dims(gtre2, axis=2)
    gtre1 = np.expand_dims(gtre1, axis=0)
    gtre2 = np.expand_dims(gtre2, axis=0)
    print(gtre1.shape)
    list_gt = np.concatenate([gtre1, gtre2], axis=0)
    print(list_gt.shape)

    image_path1 = './dataset/DRIVE/training/images/21_training.tif'
    image_path2 = './dataset/DRIVE/training/images/22_training.tif'
    imgre1 = np.asarray(Image.open(image_path1))
    imgre2 = np.asarray(Image.open(image_path2))
    imgre1 = np.expand_dims(imgre1, axis=0)
    imgre2 = np.expand_dims(imgre2, axis=0)
    list_img = np.concatenate([imgre1, imgre2], axis=0)

    img = list_img
    gt = list_gt  # here gt (B,H,W,1) shape
    gt2 = np.zeros(shape=gt.shape)
    gt3 = np.zeros(shape=gt.shape)
    gt = np.concatenate([gt, gt2, gt3], axis=3)  # here they are both (B,H,W,3) shape

    print('image and gt shape is:',img.shape, gt.shape)
    img_list = []
    gt_list  = []
    for i_d in range(0, img.shape[0]):
         aug_img, aug_gt = deformation_set(img[i_d,:,:,:,], gt[i_d,:,:,:,])
         img_list.append(aug_img)
         gt_list.append(aug_gt)
        # visualize(img[0], './image_train2')

        # gt = np.reshape(gt, (-1, size_h, size_w, 3))

        # print('before visualization gt0 size is: ', gt[0].shape)
        # visualize(gt[0], './image_test2')
        # new_gt = gt[:, :, :, 0]
        # new_gt[np.where(gt[:, :, :, 0] >= 1)] = 1
        # new_gt = np.expand_dims(new_gt, axis=3)
        # visualize(new_gt[0], './image_test')

if __name__ == '__main__':
    images, mask = data_for_train(2,448,448)
    visualize(group_images(images, 5),'../sample0')
    visualize(group_images(mask, 5),'../sample1')
    print('---end---')


    pass