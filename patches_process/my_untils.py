import numpy as np
import random
from PIL import Image
import cv2
import os

'''
imgs are 4D array N 1 H W , grey image
'''

path_images = '../MONUCLEI/train/image/'
path_gt  = '../MONUCLEI/train/gt/'
path_masks  ='../MONUCLEI/train/mask/'


test_my_path = '../test2/group3'

img_height = 1000
img_width = 1000
n_imgs = 20


def size_change(dir_img="training/image/", dir_save="my_training/image/", sizes=1024):
    size = (sizes, sizes)  # size after resize
    list_img = os.listdir(dir_img)  # get all image names in directory
    for img_name in list_img:  # iterator
        pri_image = Image.open(dir_img + img_name)
        tmppath = dir_save + img_name.replace('.bmp', '') + '.bmp'  # images name to be saved
        pri_image.resize(size, Image.ANTIALIAS).save(tmppath)  # save resize images


def rgb2gray(rgb):
    '''
    :param rgb:   N C H W
    :return:      N C H W
    '''
    assert (len(rgb.shape) == 4)  # 4D arrays
    assert (rgb.shape[1] == 3)
    bn_imgs = rgb[:, 0, :, :] * 0.299 + rgb[:, 1, :, :] * 0.587 + rgb[:, 2, :, :] * 0.114
    bn_imgs = np.reshape(bn_imgs, (rgb.shape[0], 1, rgb.shape[2], rgb.shape[3]))
    return bn_imgs


def histo_equalized(imgs):
    assert (len(imgs.shape) == 4)  # 4D arrays
    assert (imgs.shape[1] == 1)  # check the channel is 1
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i, 0] = cv2.equalizeHist(np.array(imgs[i, 0], dtype=np.uint8))
    return imgs_equalized


# CLAHE (Contrast Limited Adaptive Histogram Equalization)
def clahe_equalized(imgs):
    assert (len(imgs.shape) == 4)  # 4D arrays
    assert (imgs.shape[1] == 1)  # check the channel is 1
    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i, 0] = clahe.apply(np.array(imgs[i, 0], dtype=np.uint8))
    return imgs_equalized


# ===== normalize over the dataset
def dataset_normalized(imgs):
    assert (len(imgs.shape) == 4)  # 4D arrays
    assert (imgs.shape[1] == 1)  # check the channel is 1
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs - imgs_mean) / imgs_std
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (
                    np.max(imgs_normalized[i]) - np.min(imgs_normalized[i]))) * 255
    return imgs_normalized


def adjust_gamma(imgs, gamma=1.0):
    assert (len(imgs.shape) == 4)  # 4D arrays
    assert (imgs.shape[1] == 1)  # check the channel is 1
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    new_imgs = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        new_imgs[i, 0] = cv2.LUT(np.array(imgs[i, 0], dtype=np.uint8), table)
    return new_imgs


def visualize(data, filename):
    '''
    :param data:      the visual data must be channel last ! H W C
    :param filename:
    :return:          save into filename path .png format !
    '''

    assert (len(data.shape) == 3)  # height*width*channels
    img = None
    if data.shape[0] == 1 and data.shape[1] > 1:
        data = np.transpose(data, (1, 2, 0))
    if data.shape[2] == 1:  # in case it is black and white
        data = np.reshape(data, (data.shape[0], data.shape[1]))
    if np.max(data) > 1:
        img = Image.fromarray(data.astype(np.uint8))  # the image is already 0-255
    else:
        img = Image.fromarray((data * 255).astype(np.uint8))  # the image is between 0-1
    img.save(filename + '.png')
    print('===========================>visualize function have saved into ', filename + '.png')
    return img


def group_images(data, per_row):
    '''
    :param data:     both N H W C and N C H W is OK ！
    :param per_row:  every images number in a row
    :return:         images channel last
    '''
    assert data.shape[0] % per_row == 0
    if (data.shape[1] == 1 or data.shape[1] == 3):
        data = np.transpose(data, (0, 2, 3, 1))  # change data format into channel last !
    all_stripe = []
    for i in range(int(data.shape[0] / per_row)):
        stripe = data[i * per_row]
        for k in range(i * per_row + 1, i * per_row + per_row):
            stripe = np.concatenate((stripe, data[k]), axis=1)
        all_stripe.append(stripe)
    totimg = all_stripe[0]
    for i in range(1, len(all_stripe)):
        totimg = np.concatenate((totimg, all_stripe[i]), axis=0)
    return totimg


def read_all_images(path_images, all_images):
    '''
    :param path_images:  the path of data array
    :param all_images:   all_images is numpy array to contain all images channel 1 and 3 is different
    :return:             N H W C and N C H W
    '''
    index = 0
    for _, _, content in os.walk(path_images):

        # content.sort(key=lambda x: int(x[0:-7]))
        print('read content sequence is : \n', content)
        for p_img in list(content):
            img = Image.open(path_images + p_img)
            img_arr = np.asarray(img)
            if (len(img_arr.shape) == 2):  # (1024, 1024)
                new_img_arr = np.reshape(img_arr, (img_arr.shape[0], img_arr.shape[1], 1))  # (1024, 1024, 1)
                all_images[index, :, :, :, ] = new_img_arr
            else:
                all_images[index, :, :, :, ] = img_arr  # (1024, 1024, 3)
            index = index + 1
    print(' this directory of ({}) has total {} images and images tensor is {}'.format(path_images, index,
                                                                                       all_images.shape))
    return all_images, np.transpose(all_images, (0, 3, 1, 2))


def test_big_imgs(all_images, all_gts, all_masks, path=test_my_path):
    '''
    :param all_images: all_images,all_gts,all_masks
    :param all_gts:
    :param all_masks:
    :return:           save a random group images into files!
    '''
    random_id = np.random.randint(0, 20)
    test_original_row = np.empty((3, img_height, img_width, 3))
    test_original_row[0] = all_images[random_id]
    test_original_row[1] = all_gts[random_id]
    test_original_row[2] = all_masks[random_id]
    visualize(group_images(test_original_row, 3), path)
    print('===========================> have saved sample 3 images! ')


if __name__ == '__main__':
    all_images = np.empty(shape=(n_imgs, img_height, img_width, 3))
    all_gts = np.empty(shape=(n_imgs, img_height, img_width, 1))
    all_masks = np.empty(shape=(n_imgs, img_height, img_width, 1))

    all_images, call_images = read_all_images(path_images, all_images)
    all_gts, call_gts = read_all_images(path_gt, all_gts)
    all_masks, call_masks = read_all_images(path_masks, all_masks)

    test_big_imgs(all_images, all_gts, all_masks)

    grey_all_images = rgb2gray(call_images)
    grey_all_images = clahe_equalized(grey_all_images)
    grey_all_images = dataset_normalized(grey_all_images)
    grey_all_images = adjust_gamma(grey_all_images, gamma=1.0)
    print('now gray image size is : ', grey_all_images.shape)



