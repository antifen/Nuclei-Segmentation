import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np


from data_ultils import data_for_train,group_images,visualize



def get_train_val_data(numbers =2, val_ratio = 0.2, size_h =128, size_w =128):

    images, mask = data_for_train(numbers, size_h, size_w)
    visualize(group_images(images, 5), '../sample0')
    visualize(group_images(mask, 5), '../sample1')

    print('========  data and mask have been saved into png format ==========',np.max(images))

    val_num = int(images.shape[0] * val_ratio)

    train_list = [images[val_num:, :, :, :, ], mask[val_num:, :, :, :, ]]
    val_list = [images[0:val_num, :, :, :, ], mask[0:val_num, :, :, :, ]]

    return train_list, val_list



class ImageFolder(data.Dataset):

    def __init__(self,img, mask):
        self.img = img
        self.mask = mask

    def __getitem__(self, index):

        imgs  = torch.from_numpy(self.img[index])
        masks = torch.from_numpy(self.mask[index])

        return imgs, masks

    def __len__(self):
        assert self.img.shape[0] == self.mask.shape[0], 'The number of images must be equal to labels'
        return self.img.shape[0]


if __name__ == '__main__':

    get_train_val_data()

