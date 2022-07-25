import sys

sys.path.append('../train_test/')
sys.path.append('../../data_process/')
sys.path.append('../')
sys.path.append('../../')
print(sys.path)
import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np

from image_process import rgb2gray, clahe_equalized, dataset_normalized, adjust_gamma
from data_ultils import group_images, visualize, label2rgb
import Constants
import warnings

warnings.filterwarnings("ignore")

save_contour_1 = 'Sample_' + Constants.dataset_name+'_Contour_train'
save_contour_2 = 'Sample_' + Constants.dataset_name+'_Contour_test'
save_object_1 = 'Sample_' + Constants.dataset_name+'_Object_train'
save_object_2 = 'Sample_' + Constants.dataset_name+'_Object_test'

def visual_sample(images, mask, path, per_row=5):
    visualize(group_images(images, per_row), Constants.visual_samples + path + '_GreyImage')
    visualize(group_images(mask, per_row), Constants.visual_samples + path + '_Label')

def load_from_npy(npy_path):
    arrays = np.load(npy_path)
    print('have loaded all arrays from ', npy_path)
    return arrays

def get_Contour_info(val_ratio=0.1, is_train=True):
    images = load_from_npy(Constants.path_nuclei_train_image)
    mask = load_from_npy(Constants.path_nuclei_train_label)
    images_test = load_from_npy(Constants.path_nuclei_test_image)
    mask_test = load_from_npy(Constants.path_nuclei_test_label)

    images_val = load_from_npy(Constants.path_nuclei_val_image)
    mask_val = load_from_npy(Constants.path_nuclei_val_label)

    images = rgb2gray(images)
    images = dataset_normalized(images)
    images = clahe_equalized(images)
    images = adjust_gamma(images, 1.0)
    
    origin_test = images_test
    images_test = rgb2gray(images_test)
    images_test = dataset_normalized(images_test)
    images_test = clahe_equalized(images_test)
    images_test = adjust_gamma(images_test, 1.0)

    images_val = rgb2gray(images_val)
    images_val = dataset_normalized(images_val)
    images_val = clahe_equalized(images_val)
    images_val = adjust_gamma(images_val, 1.0)

    images = images / 255.                           # reduce to 0-1 range
    images_test = images_test / 255.
    images_val = images_val / 255.

    mask[np.where(mask==1)] =1                       # reduce to 0-1 range
    mask[np.where(mask != 1)] = 0
    mask_val[np.where(mask_val ==1)] = 1
    mask_val[np.where(mask_val != 1)] = 0
    mask_test[np.where(mask_test==1)] =1
    mask_test[np.where(mask_test != 1)] = 0

    if is_train is True:
        print('Check loaded image and mask size and max value:', images.shape, mask.shape, np.max(images), np.max(mask))
        print('Succeed in loading all Drive Train, Val and Test images!')
        visual_sample(images_test[0:10, :, :, :, ], mask_test[0:10, :, :, :, ], save_contour_2, per_row=5)
        visual_sample(images[0:20, :, :, :, ], mask[0:20, :, :, :, ], save_contour_1, per_row=5)

    train_list = [images[:, :, :, :, ], mask[:, :, :, :, ]]
    val_list = [images_val[:, :, :, :, ], mask_val[:, :, :, :, ]]
    if is_train is True:
        return train_list, val_list
    else:
        return images_test, mask_test, origin_test


def get_Object_info(val_ratio=0.1, is_train=True):
    images = load_from_npy(Constants.path_nuclei_train_image)
    mask = load_from_npy(Constants.path_nuclei_train_label)
    images_test = load_from_npy(Constants.path_nuclei_test_image)
    mask_test = load_from_npy(Constants.path_nuclei_test_label)

    images_val = load_from_npy(Constants.path_nuclei_val_image)
    mask_val = load_from_npy(Constants.path_nuclei_val_label)

    images = rgb2gray(images)
    images = dataset_normalized(images)
    images = clahe_equalized(images)
    images = adjust_gamma(images, 1.0)

    origin_test = images_test
    images_test = rgb2gray(images_test)
    images_test = dataset_normalized(images_test)
    images_test = clahe_equalized(images_test)
    images_test = adjust_gamma(images_test, 1.0)

    images_val = rgb2gray(images_val)
    images_val = dataset_normalized(images_val)
    images_val = clahe_equalized(images_val)
    images_val = adjust_gamma(images_val, 1.0)

    images = images / 255.      # reduce to 0-1 range
    images_test = images_test / 255.
    images_val = images_val / 255.

    mask[np.where(mask == 1)] = 0
    mask[np.where(mask==2)] =1  # reduce to 0-1 range
    mask[np.where(mask != 1)] = 0
    mask_val[np.where(mask_val == 1)] = 0
    mask_val[np.where(mask_val ==2)] = 1
    mask_val[np.where(mask_val != 1)] = 0
    mask_test[np.where(mask_test == 1)] = 0
    mask_test[np.where(mask_test==2)] = 1
    mask_test[np.where(mask_test != 1)] = 0

    if is_train is True:
        print('Check loaded image and mask size and max value:', images.shape, mask.shape, np.max(images), np.max(mask))
        print('Succeed in loading all Drive Train, Val and Test images!')
        visual_sample(images_test[0:10, :, :, :, ], mask_test[0:10, :, :, :, ],save_object_2, per_row=5)
        visual_sample(images[0:20, :, :, :, ], mask[0:20, :, :, :, ],save_object_1, per_row=5)

    train_list = [images[:, :, :, :, ], mask[:, :, :, :, ]]
    val_list = [images_val[:, :, :, :, ], mask_val[:, :, :, :, ]]
    if is_train is True:
        return train_list, val_list
    else:
        return images_test, mask_test, origin_test

class ImageFolder(data.Dataset):
    '''
    Image is RGB original image, mask is one hot GT and label is grey image to visual
    img and mask is necessary while label is alternative !
    '''
    def __init__(self, img, mask, label=None):
        self.img = img
        self.mask = mask
        self.label = label

    def __getitem__(self, index):
        imgs = torch.from_numpy(self.img[index]).float()
        masks = torch.from_numpy(self.mask[index]).float()
        if self.label is not None:
            label = torch.from_numpy(self.label[index]).float()
            return imgs, masks, label
        else:
            return imgs, masks

    def __len__(self):
        assert self.img.shape[0] == self.mask.shape[0],  'The number of images must be equal to labels'
        return self.img.shape[0]


if __name__ == '__main__':
    get_Contour_info()
    get_Object_info()

    pass
