import sys
sys.path.append('../')
import numpy as np
import os
import Constants
from data_ultils import  read_all_images, deformation_set, data_shuffle
from PIL import Image

def read_numpy_into_npy(arrays, path):
    np.save(path, arrays)
    print('have saved all arrays in to path ', path)

def load_from_npy(npy_path):
    arrays = np.load(npy_path)
    print('have loaded all arrays from ', npy_path)
    return arrays

def data_auguments(aug_num,  path_img, agument = True):
    all_images, all_masks = perpare_data(path_img)          # original data
    if agument is True:
        img_list = []
        gt_list = []
        for nums in range(0, aug_num):
            for i_d in range(0, all_images.shape[0]):
                aug_img, aug_gt = deformation_set(all_images[i_d, :, :, :, ], all_masks[i_d, :, :, :, ])
                img_list.append(np.expand_dims(aug_img, axis=0))
                gt_list.append(np.expand_dims(aug_gt, axis=0))
        img_au = np.concatenate(img_list, axis=0)
        gt_au = np.concatenate(gt_list, axis=0)
        return img_au,gt_au
    else:
        return all_images, all_masks


def data_for_train(aug_num,path_img, agument = True):
    all_images, all_masks = data_auguments(aug_num,  path_img, agument)
    print('image and gt shape is:', all_images.shape, all_masks.shape)
    img = np.array(all_images, np.float32).transpose(0,3,1,2) / 255.0
    mask = np.array(all_masks, np.float32).transpose(0,3,1,2) / 255
    # print(img.shape, mask.shape)

    #  data shuffle
    print(img.shape[0],'===========')
    index = np.arange(img.shape[0])
    np.random.shuffle(index)
    img  = img[index, :, :, :]
    mask = mask[index, :, :]
    print(np.max(img), np.max(mask),'==================')

    return img, mask

def save_data(agnum, path_images, agument = True):

    img, mask = data_for_train(agnum,path_images,  agument = agument)
    try:
        read_numpy_into_npy(img,Constants.path_image_TBNC)
        read_numpy_into_npy(mask, Constants.path_label_TBNC)
        print('========  all train and test data has been saved ! ==========')
        return True
    except:
        return False

def perpare_data(path_images, augu = True):
    content = os.listdir(path_images)
    content.sort(key=lambda x: int(x[x.find('_')+1:]))
    con_gt = [i for i in content if 'GT' in i]
    con_image = [i for i in content if 'Slide' in i]

    list_img = []
    list_gt = []

    for p_img in list(con_image):
        for  sub_content in os.listdir(path_images + p_img):
            list_img.append(np.asarray(Image.open(path_images +'/'+ p_img +'/'+sub_content).convert("RGB")))
            # print(np.asarray(Image.open(path_images + '/'+ p_img + '/' + sub_content).convert("RGB")).shape,'====')
    list_img = np.concatenate(np.expand_dims(list_img, axis=0), axis =0)


    for p_img in list(con_gt):
        for  sub_content in os.listdir(path_images + p_img):
            list_gt.append(np.asarray(Image.open(path_images +'/'+ p_img +'/'+sub_content)))
            # print(np.asarray(Image.open(path_images + '/' + p_img +'/' + sub_content)).shape,'====')
    list_gt = np.concatenate(np.expand_dims(list_gt, axis=0), axis =0)


    if augu is True:
        index = np.arange(list_img.shape[0])
        np.random.shuffle(index)
        list_img  = list_img[index, :, :, :]
        list_gt = list_gt[index, :, :]

    return list_img /255 , np.expand_dims(list_gt, axis = 3) /255

def save_TNBC_data(path = '../dataset/TNBC/', is_train = True):
    image, gt = perpare_data(path)
    print('output image and label size: ', image.shape, gt.shape)
    images, mask = image[0:28,:,:,:,], gt[0:28,:,:,:,]
    images_val, mask_val =  image[28:30,:,:,:,], gt[28:30,:,:,:,]
    images_test, mask_test =  image[30:50,:,:,:,], gt[30:50,:,:,:,]

    # if is_train is True:
    #     return [images, mask], [images_val, mask_val]
    # else:
    #     return [images_test, mask_test]

    if is_train is True:
        return [np.transpose(images, (0,3,1,2)), np.transpose(mask,(0,3,1,2))], \
               [np.transpose(images_val, (0,3,1,2)), np.transpose(mask_val, (0,3,1,2))]
    else:
        return [np.transpose(images_test, (0,3,1,2)), np.transpose(mask_test, (0,3,1,2)),np.transpose(images_test, (0,3,1,2))]

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
    save_TNBC_data('../dataset/TNBC/')
    check_bst_data()

    pass