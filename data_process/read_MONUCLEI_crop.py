import sys
sys.path.append('../')

import  numpy as np
import os
import matplotlib.pyplot as plt
from  PIL import  Image
import cv2
import Constants
from data_ultils import  read_all_images, data_shuffle

path_images_MONUCLEI = '../dataset/MONUCLEI/training/image/'
path_gt_MONUCLEI = '../dataset/MONUCLEI/training/gt/'
path_images_test_MONUCLEI = '../dataset/MONUCLEI/test/image/'
path_gt_test_MONUCLEI = '../dataset/MONUCLEI/test/gt/'
path_images_val_MONUCLEI ='../dataset/MONUCLEI/val/image/'
path_gt_val_MONUCLEI = '../dataset/MONUCLEI/val/gt/'

def read_numpy_into_npy(arrays, path):
    np.save(path, arrays)
    print('have saved all arrays in to path ', path)

def load_from_npy(npy_path):
    arrays = np.load(npy_path)
    print('have loaded all arrays from ', npy_path)
    return arrays

def read_MONUCLEI_images(size_h,size_w, path_images, path_gt,total_imgs, mask_ch =1):
    all_images = np.empty(shape=(total_imgs, size_h, size_w, 3))
    all_masks  = np.empty(shape=(total_imgs, size_h, size_w, mask_ch))
    all_images = read_all_images(path_images, all_images,size_h, size_w,type ='non_resize')
    all_masks  = read_all_images(path_gt, all_masks, size_h, size_w,type ='non_resize')
    print('============= have read all images ! ==============')
    return all_images, all_masks

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

def crop_images(image, mask, crop_size = Constants.resize_MONUCLEI, kid = True):
    select_id = np.random.randint(0, 6)
    d_h, d_w, h, w =  image.shape[0] - crop_size, image.shape[1] - crop_size,image.shape[0],image.shape[1]
    crop_lu_im,  crop_lu_ma = image[d_h:h, d_w:w, :,], mask[d_h:h, d_w:w, :,]
    crop_ld_im,  crop_ld_ma = image[d_h:h, 0:w-d_w, :, ], mask[d_h:h, 0:w-d_w, :, ]
    crop_ru_im,  crop_ru_ma = image[0:h - d_h, d_w:w, :, ], mask[0:h - d_h, d_w:w, :, ]
    crop_rd_im,  crop_rd_ma = image[0:h - d_h, 0:w-d_w, :, ], mask[0:h - d_h, 0:w-d_w, :, ]

    crop_img, crop_mask = None, None
    crop_img_2, crop_mask_2 = None, None
    if select_id ==0:
        crop_img, crop_mask = np.expand_dims(crop_lu_im, axis=0), np.expand_dims(crop_lu_ma, axis=0)
        crop_img_2, crop_mask_2 = np.expand_dims(crop_ld_im, axis=0), np.expand_dims(crop_ld_ma, axis=0)
    if select_id ==1:
        crop_img, crop_mask = np.expand_dims(crop_lu_im, axis=0), np.expand_dims(crop_lu_ma, axis=0)
        crop_img_2, crop_mask_2 = np.expand_dims(crop_ru_im, axis=0), np.expand_dims(crop_ru_ma, axis=0)
    if select_id ==2:
        crop_img, crop_mask = np.expand_dims(crop_rd_im, axis=0), np.expand_dims(crop_rd_ma, axis=0)
        crop_img_2, crop_mask_2 = np.expand_dims(crop_lu_im, axis=0), np.expand_dims(crop_lu_ma, axis=0)
    if select_id ==3:
        crop_img, crop_mask = np.expand_dims(crop_ld_im, axis=0), np.expand_dims(crop_ld_ma, axis=0)
        crop_img_2, crop_mask_2 = np.expand_dims(crop_ru_im, axis=0), np.expand_dims(crop_ru_ma, axis=0)
    if select_id ==4:
        crop_img, crop_mask = np.expand_dims(crop_rd_im, axis=0), np.expand_dims(crop_rd_ma, axis=0)
        crop_img_2, crop_mask_2 = np.expand_dims(crop_ld_im, axis=0), np.expand_dims(crop_ld_ma, axis=0)
    if select_id ==5:
        crop_img, crop_mask = np.expand_dims(crop_rd_im, axis=0), np.expand_dims(crop_rd_ma, axis=0)
        crop_img_2, crop_mask_2 = np.expand_dims(crop_ru_im, axis=0), np.expand_dims(crop_ru_ma, axis=0)

    if kid is True:
        return np.concatenate([crop_img, crop_img_2], axis=0), np.concatenate([crop_mask, crop_mask_2], axis=0)
    else:
        return np.concatenate([np.expand_dims(crop_lu_im, axis=0), np.expand_dims(crop_ru_im, axis=0),np.expand_dims(crop_ld_im, axis=0),np.expand_dims(crop_rd_im, axis=0)], axis=0),\
               np.concatenate([np.expand_dims(crop_lu_ma, axis=0), np.expand_dims(crop_ru_ma, axis=0),np.expand_dims(crop_ld_ma, axis=0),np.expand_dims(crop_rd_ma, axis=0)], axis=0)

def deformation_set(image, mask,
                           shift_limit=(-0.1, 0.1),
                           scale_limit=(-0.2, 0.2),
                           rotate_limit=(-180.0, 180.0),
                           aspect_limit=(-0.1, 0.1),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    print('deformation_set size check: ', image.shape, mask.shape)

    start_angele, per_rotate = -180, 90
    rotate_num = - start_angele // per_rotate * 2
    image_set, mask_set = [], []

    crop_im, crop_ma = crop_images(image, mask, kid = False)
    image_set.append(crop_im)
    mask_set.append(crop_ma)

    for rotate_id in range(0, rotate_num):
        masks = mask
        img = image
        height, width, channel = img.shape
        sx, sy = 1., 1.
        angle = start_angele + rotate_id * per_rotate
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)
        cc = np.cos(angle / 180 * np.pi) * sx
        ss = np.sin(angle / 180 * np.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])
        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height],])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])
        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        img = cv2.warpPerspective(img, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(0, 0, 0,))
        masks = cv2.warpPerspective(masks, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(0, 0, 0,))
        img, masks = randomHorizontalFlip(img, masks)
        img, masks = randomVerticleFlip(img, masks)
        masks = np.expand_dims(masks, axis=2)               #
        # image_set.append(np.expand_dims(img, axis=0))
        # mask_set.append(np.expand_dims(masks, axis=0))
        crop_im, crop_ma = crop_images(img, masks)
        image_set.append(crop_im)
        mask_set.append(crop_ma)
        # print(img.shape, masks.shape,'====================')
    aug_img  = np.concatenate([image_set[i] for i in range(0, len(image_set))],axis=0)
    aug_mask = np.concatenate([mask_set[i] for i in range(0, len(mask_set))], axis=0)
    return aug_img, aug_mask

def data_auguments(aug_num,size_h, size_w,path_images, path_gt,total_imgs, mask_ch, augu=True):

    all_images, all_masks = read_MONUCLEI_images(size_h, size_w,path_images, path_gt,total_imgs, mask_ch)         # original data
    if augu is False:
        return all_images, all_masks
    # print('image and gt shape is:', all_images.shape, all_masks.shape)
    img_list = []
    gt_list = []
    for nums in range(0, aug_num):
        for i_d in range(0, all_images.shape[0]):
            aug_img, aug_gt = deformation_set(all_images[i_d, :, :, :, ], all_masks[i_d, :, :, :, ])
            img_list.append(aug_img)
            gt_list.append(aug_gt)
    img_au = np.concatenate(img_list, axis=0)
    gt_au = np.concatenate(gt_list, axis=0)
    # print(img_au.shape, gt_au.shape)
    # visualize(group_images(all_masks, 5), './image_test')
    print('After augment, data and label shape is : {} and {}'.format(img_au.shape, gt_au.shape))
    return img_au,gt_au

def data_for_train(aug_num,size_h, size_w,path_images, path_gt,total_imgs, mask_ch,augu):
    all_images, all_masks = data_auguments(aug_num, size_h, size_w, path_images, path_gt,total_imgs, mask_ch,augu)
    print('image and gt shape is:', all_images.shape, all_masks.shape)
    img = np.array(all_images, np.float32).transpose(0,3,1,2) / 255.0
    mask = np.array(all_masks, np.float32).transpose(0,3,1,2)

    if Constants.class_nums == 1:
        mask[mask >= 0.5] = 1
        mask[mask < 0.5] = 0

    print('label value max is {} and min is {}'.format(np.max(mask), np.min(mask)))
    #  data shuffle
    if augu is True:
        index = np.arange(img.shape[0])
        np.random.shuffle(index)
        img  = img[index, :, :, :]
        mask = mask[index, :, :]
    return img, mask

def save_MONUCLEI_data(mum_arg = 1):
    images, mask = data_for_train(mum_arg, Constants.size_h,Constants.size_w,
                                  path_images_MONUCLEI, path_gt_MONUCLEI, Constants.total_MONUCLEI, mask_ch=1, augu=True)
    images_test, mask_test = data_for_train(mum_arg, Constants.size_h,Constants.size_w,
                                  path_images_test_MONUCLEI, path_gt_test_MONUCLEI, Constants.total_MONUCLEI_test, mask_ch=1, augu=False)
    images_val, mask_val = data_for_train(mum_arg, Constants.size_h,Constants.size_w,
                                  path_images_val_MONUCLEI, path_gt_val_MONUCLEI, Constants.total_MONUCLEI_val, mask_ch=1, augu=False)

    try:
        read_numpy_into_npy(images,Constants.path_nuclei_train_image)
        read_numpy_into_npy(mask, Constants.path_nuclei_train_label)
        read_numpy_into_npy(images_test,Constants.path_nuclei_test_image)
        read_numpy_into_npy(mask_test, Constants.path_nuclei_test_label)
        read_numpy_into_npy(images_val,Constants.path_nuclei_val_image)
        read_numpy_into_npy(mask_val, Constants.path_nuclei_val_label)
        print('========  All DRIVE train and test data has been saved ! ==========')
    except:
        print('File save exception has happened, please check ! ')
    pass

def check_bst_data():
    a = load_from_npy(Constants.path_nuclei_train_image)
    b = load_from_npy(Constants.path_nuclei_train_label)
    c = load_from_npy(Constants.path_nuclei_test_image)
    d = load_from_npy(Constants.path_nuclei_test_label)
    e = load_from_npy(Constants.path_nuclei_val_image)
    f = load_from_npy(Constants.path_nuclei_val_label)
    print('Check npy files data shape and value !')
    print('============================================================')
    print(a.shape, b.shape, c.shape, d.shape,e.shape, f.shape)
    print(np.max(a),np.max(b),np.max(c), np.max(d),np.max(e), np.max(f))
    print('============================================================')

if __name__ == '__main__':
    save_MONUCLEI_data()
    check_bst_data()
    pass