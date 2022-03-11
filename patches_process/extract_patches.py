import numpy as np
import random
from PIL import Image
import cv2
import os

from my_untils  import  read_all_images,test_big_imgs,rgb2gray,clahe_equalized,dataset_normalized,\
    adjust_gamma,group_images,visualize,size_change,test_big_imgs
from base_utils import segment_line,shape_check,value_range_check,net_predict_reshape4d,\
    tensor4d_channels_last,tensor4d_channels_first


# path_images = '../DRIVE/training/images/'
# path_gt  = '../DRIVE/training/1st_manual/'
# path_masks  = '../DRIVE/training/mask/'
#
# test_my_path = '../DRIVE/myimg'

img_height = 1000
img_width  = 1000
n_imgs = 30


def size_changes(path_origin, path_save, size):
    # size_change(dir_img="training/image/", dir_save="my_training/image/", sizes=1024)
    size_change(dir_img=path_origin, dir_save=path_save, sizes = size)
    print('========== have finished image resize ! ===========')

def extract_orders(imgs, patch_h, patch_w):
    '''
    :param imgs:       N  C  H  W
    :param patch_h:
    :param patch_w:
    :param stride_h:
    :param stride_w:
     Nimgs is total images for training
    :return:          N  C  H  W
    '''
    assert (imgs.shape[1]==1 or imgs.shape[1]==3)
    Nimgs = int(imgs.shape[2]//(patch_h) * imgs.shape[3]//(patch_w))
    new_imgs = np.empty(shape=(Nimgs*imgs.shape[0], imgs.shape[1], patch_h, patch_w))
    i = 0
    for j in range(0, imgs.shape[0]):
        for h in range(0, imgs.shape[2]//patch_h):
            for w in range(0, imgs.shape[3]//patch_w):
                new_imgs[i, :, :, :,] = imgs[j, :, h*patch_h:(h+1)*patch_h, w*patch_w:(w+1)*patch_w]
                i = i + 1
    print('image patch numbers are:', i)
    return new_imgs

def extract_random(imgs, gts, total_nums, patch_h, patch_w):
    assert (imgs.shape[1] == 1 or imgs.shape[1] == 3)   # check imgs channel first
    assert (gts.shape[1] == 1 or gts.shape[1] == 3)     # check gts channel first
    new_imgs = np.empty(shape=(total_nums, imgs.shape[1], patch_h, patch_w))
    new_gts = np.empty(shape=(total_nums, gts.shape[1], patch_h, patch_w))
    i = 0
    while i < total_nums:
        random_h = np.random.randint(0, imgs.shape[2] - patch_h)
        random_w = np.random.randint(0, imgs.shape[3] - patch_w)
        random_id = np.random.randint(0, imgs.shape[0])
        new_imgs[i, :, :, :, ] = imgs[random_id, :, random_h:random_h+patch_h, random_w:random_w + patch_w]
        new_gts[i, :, :, :, ] = gts[random_id, :, random_h:random_h + patch_h, random_w:random_w + patch_w]
        i = i + 1
    return new_imgs, new_gts

def extract_random_center(imgs, gts, total_nums, patch_h, patch_w):
    '''
     totals_num must be divided by imgs numbers !
    :param imgs:
    :param gts:
    :param total_nums:
    :param patch_h:
    :param patch_w:
    :return:
    '''
    assert (imgs.shape[1] == 1 or imgs.shape[1] == 3)   # check imgs channel first
    assert (gts.shape[1] == 1 or gts.shape[1] == 3)     # check gts channel first
    new_imgs = np.empty(shape=(total_nums, imgs.shape[1], patch_h, patch_w))
    new_gts = np.empty(shape=(total_nums, gts.shape[1], patch_h, patch_w))
    i = 0
    k = 0
    out_points = 0
    while i < total_nums and k<imgs.shape[0]:
        for j in range(0, total_nums//imgs.shape[0]):
            center_h = random.randint(patch_h//2, imgs.shape[2] - patch_h//2)
            center_w = random.randint(patch_w//2, imgs.shape[3] - patch_w//2)
            while True:
                if is_patch_inside_FOV(center_h,center_w,imgs.shape[2]//2,imgs.shape[3]//2, patch_h) is False:
                    out_points +=1
                    center_h = random.randint(patch_h // 2, imgs.shape[2] - patch_h // 2)
                    center_w = random.randint(patch_w // 2, imgs.shape[3] - patch_w // 2)
                else:
                    break
            new_imgs[i, :, :, :, ] = imgs[k, :, center_h-patch_h//2:center_h+patch_h//2,
                                         center_w- patch_w//2:center_w + patch_w//2]
            new_gts[i, :, :, :, ] = gts[k, :, center_h-patch_h//2:center_h + patch_h//2,
                                        center_w-patch_w//2:center_w + patch_w//2]
            i = i + 1
        k = k + 1
    print('now the k is{} and the i is{}, all number of out_points is {}'.format(k,i,out_points))
    return new_imgs, new_gts

def is_patch_inside_FOV(x,y,img_w,img_h,patch_h):
    x_1 = x - int(img_h/2) # origin (0,0) shifted to image center
    y_1 = y - int(img_w/2)  # origin (0,0) shifted to image center
    R_inside = 270 - int(patch_h * np.sqrt(2.0) / 2.0) #radius is 270 (from DRIVE db docs), minus the patch diagonal (assumed it is a square #this is the limit to contain the full patch in the FOV
    radius = np.sqrt((x_1*x_1)+(y_1*y_1))
    if radius < R_inside:
        return True
    else:
        return False


def gt_to_color(imgs):
    '''
    :param imgs: image size is N C H W
    :return:
    '''
    value_range_check(imgs)
    color_imgs = np.zeros(shape=(imgs.shape[0], 3, imgs.shape[2], imgs.shape[3]))
    for iter in range(0, imgs.shape[0]):
        for height in range(0, imgs.shape[2]):
            for width in range(0, imgs.shape[3]):
                if imgs[iter,0,height,width] == 1:        # green is contour
                    color_imgs[iter, 1, height,width] = 255
                elif imgs[iter,0,height,width] == 2:      # blue is object
                    color_imgs[iter, 2, height, width] = 255
                else:                                     # red  is object
                    color_imgs[iter, 0, height, width] = 255
    return color_imgs, color_imgs.transpose((0,2,3,1))

# Divide all the full_imgs in pacthes
def extract_ordered_overlap(full_imgs, patch_h, patch_w, stride_h, stride_w, channel):
    assert (len(full_imgs.shape) == 4)  # 4D arrays
    assert (full_imgs.shape[1] == channel)  # check the channel is 1 or 3
    img_h = full_imgs.shape[2]  # height of the full image
    img_w = full_imgs.shape[3]  # width of the full image
    assert ((img_h - patch_h) % stride_h == 0 and (img_w - patch_w) % stride_w == 0)

    num_patches_one = ((img_h - patch_h) // stride_h + 1) \
                      * ((img_w - patch_w) // stride_w + 1)

    num_patches_total = num_patches_one * full_imgs.shape[0]
    print("Number of patches on h : ", (img_h - patch_h) // stride_h + 1)
    print("Number of patches on w : ", (img_w - patch_w) // stride_w + 1)
    print("number of patches per image: {}, totally for this testing dataset: {}"
          .format(num_patches_one, num_patches_total))
    patches = np.empty((num_patches_total, full_imgs.shape[1], patch_h, patch_w), dtype=np.float16)
    iter_total = 0  # iter over the total number of patches (N_patches)
    for i in range(full_imgs.shape[0]):  # loop over the full images
        for h in range((img_h - patch_h) // stride_h + 1):
            for w in range((img_w - patch_w) // stride_w + 1):
                patch = full_imgs[i, :, h * stride_h:(h * stride_h) + patch_h, w * stride_w:(w * stride_w) + patch_w]
                patches[iter_total] = patch
                iter_total += 1  # total
    assert (iter_total == num_patches_total)
    return patches  # array with all the full_imgs divided in patches


def paint_border_overlap(full_imgs, patch_h, patch_w, stride_h, stride_w, channel):
    assert (len(full_imgs.shape) == 4)  # 4D arrays
    assert (full_imgs.shape[1] == channel)  # check the channel is 1 or 3
    img_h = full_imgs.shape[2]  # height of the full image
    img_w = full_imgs.shape[3]  # width of the full image
    leftover_h = (img_h - patch_h) % stride_h  # leftover on the h dim
    leftover_w = (img_w - patch_w) % stride_w  # leftover on the w dim

    # extend dimension of img h by adding zeros
    if leftover_h != 0:
        print("the side H is not compatible with the selected stride of {}".format(stride_h))
        print("img_h: {}, patch_h: {}, stride_h: {}".format(img_h, patch_h, stride_h))
        print("(img_h - patch_h) MOD stride_h: ", leftover_h)
        print("So the H dim will be padded with additional {} pixels ".format(stride_h - leftover_h))
        tmp_full_imgs = np.zeros((full_imgs.shape[0], full_imgs.shape[1], img_h + (stride_h - leftover_h), img_w))
        tmp_full_imgs[0:full_imgs.shape[0], 0:full_imgs.shape[1], 0:img_h, 0:img_w] = full_imgs
        full_imgs = tmp_full_imgs

    # extend dimension of img w by adding zeros
    if leftover_w != 0:  # change dimension of img_w
        print("the side W is not compatible with the selected stride of {}".format(stride_w))
        print("img_w: {}, patch_w: {}, stride_w: {}".format(img_w, patch_w, stride_w))
        print("(img_w - patch_w) MOD stride_w: ", leftover_w)
        print("So the W dim will be padded with additional {} pixels ".format(stride_w - leftover_w))
        tmp_full_imgs = np.zeros(
            (full_imgs.shape[0], full_imgs.shape[1], full_imgs.shape[2], img_w + (stride_w - leftover_w)))
        tmp_full_imgs[0:full_imgs.shape[0], 0:full_imgs.shape[1], 0:full_imgs.shape[2], 0:img_w] = full_imgs
        full_imgs = tmp_full_imgs
    print("new full images shape:", full_imgs.shape)
    return full_imgs

def get_loss_weight(patch_height, patch_width, mode, border = 16):
    loss_weight = np.zeros((patch_height, patch_width))
    center_x = patch_height /2 - 1
    center_y = patch_width / 2 - 1
    if mode == 0:
        return None

    for k in range(patch_height//2):
        for i in range(k, patch_width - k):
            loss_weight[k, i] = k
            loss_weight[i, k] = k
            loss_weight[patch_height - k - 1, i] = k
            loss_weight[i, patch_width - k - 1] = k
    max_value = np.max(loss_weight)
    max_value = float(max_value)
    if mode == 4:
        # in this mode, loss weight outside is 0, inner is 1
        loss_weight[np.where(loss_weight < border)] = 0
        loss_weight[np.where(loss_weight >= border)] = 1
        loss_weight = np.reshape(loss_weight, (patch_width * patch_height))
    else:
        if mode == 1:
            loss_weight = loss_weight/max_value * loss_weight/max_value
        elif mode == 2:
            loss_weight = loss_weight/max_value
        elif mode == 3:
            loss_weight = np.sqrt(loss_weight/max_value)

        loss_weight = np.reshape(loss_weight, (patch_width * patch_height))
        weight_sum = patch_height * patch_width
        cur_sum = np.sum(loss_weight)
        loss_weight *= weight_sum/cur_sum
    #     loss_weight = np.reshape(loss_weight[:,:,0], (patch_width * patch_height))
    #     loss_weight += 0.01
    #     weight_sum = patch_height * patch_width
    #     cur_sum = np.sum(loss_weight)
    #     loss_weight *= weight_sum/cur_sum

        #loss_weight = np.reshape(loss_weight[:,0], (patch_width*patch_height,1))
    result = loss_weight
    print("shape of loss_weight:", result.shape)
    return result

def recompose_overlap(preds, img_h, img_w, stride_h, stride_w, loss_weight=None):
    assert (len(preds.shape) == 4)  # 4D arrays
    #assert (preds.shape[1] == channel)
    patch_h = preds.shape[2]
    patch_w = preds.shape[3]

    N_patches_h = (img_h - patch_h) // stride_h + 1
    N_patches_w = (img_w - patch_w) // stride_w + 1
    N_patches_img = N_patches_h * N_patches_w

    print("N_patches_h: ", N_patches_h)
    print("N_patches_w: ", N_patches_w)
    print("N_patches_img: ", N_patches_img)
    assert (preds.shape[0] % N_patches_img == 0)
    N_full_imgs = preds.shape[0] // N_patches_img

    print("According to the dimension inserted, there are {} full images (of {} x {} each)"
          .format(N_full_imgs, img_h, img_w))

    full_prob = np.zeros(
        (N_full_imgs, preds.shape[1], img_h, img_w))  # initialize to zero mega array with sum of Probabilities
    full_sum = np.zeros((N_full_imgs, preds.shape[1], img_h, img_w))

    k = 0  # iterator over all the patches

    # extract each patch
    center = [patch_h / 2, patch_w / 2]
    expand = patch_h / 2
    left = center[1] - expand
    right = center[1] + expand
    top = center[0] - expand
    bottom = center[0] + expand

    if loss_weight is not None:
        weight = np.reshape(loss_weight, (patch_h, patch_w))
        weight += 0.000000001
    else:
        weight = 1

    print(top,bottom,left,right,img_h,img_w)
    top = int(top)
    bottom = int(bottom)
    left = int(left)
    right = int(right)

    for i in range(N_full_imgs):
        for h in range((img_h - patch_h) // stride_h + 1):
            for w in range((img_w - patch_w) // stride_w + 1):
                full_prob[i, :, h * stride_h + top:(h * stride_h) + bottom,
                          w * stride_w + left:(w * stride_w) + right] +=preds[k, :, top:bottom, left:right]*weight
                full_sum[i, :, h * stride_h + top:(h * stride_h) + bottom,
                         w * stride_w + left:(w * stride_w) + right] += weight
                k += 1
    #assert (k == preds.shape[0])
    #assert (np.min(full_sum) >= 0.0)  # must larger than 0
    #print(np.min(full_sum))

    final_avg = full_prob / (full_sum + 0.0000000001)
    #print("the shape of prediction result", final_avg.shape)
    #print("max value of prediction result", np.max(final_avg))
    #assert (np.max(final_avg) <= 1.01)  # max value for a pixel is 1.0
    #assert (np.min(final_avg) >= 0.0)  # min value for a pixel is 0.0
    return final_avg


#Recompone the full images with the patches
def recompone(data,N_h,N_w):
    assert (data.shape[1]==1 or data.shape[1]==3)  #check the channel is 1 or 3
    assert(len(data.shape)==4)
    N_pacth_per_img = N_w*N_h
    assert(data.shape[0]%N_pacth_per_img == 0)
    N_full_imgs = data.shape[0]/N_pacth_per_img
    patch_h = data.shape[2]
    patch_w = data.shape[3]
    N_pacth_per_img = N_w*N_h
    #define and start full recompone
    full_recomp = np.empty((N_full_imgs,data.shape[1],N_h*patch_h,N_w*patch_w))
    k = 0  #iter full img
    s = 0  #iter single patch
    while (s<data.shape[0]):
        #recompone one:
        single_recon = np.empty((data.shape[1],N_h*patch_h,N_w*patch_w))
        for h in range(N_h):
            for w in range(N_w):
                single_recon[:,h*patch_h:(h*patch_h)+patch_h,w*patch_w:(w*patch_w)+patch_w]=data[s]
                s+=1
        full_recomp[k]=single_recon
        k+=1
    assert (k==N_full_imgs)
    return full_recomp



def test_order_extract_cell1():
    test_path_images = '../images/'
    test_path_gt = '../gt/'
    change_test_path_gt = '../gts/'
    n_test_images = 12
    size = 1024

    size_changes(test_path_gt, change_test_path_gt, size=size)
    all_images = np.empty(shape=(n_test_images, size, size, 3))
    all_gts = np.empty(shape=(n_test_images, size, size, 1))

    all_images, call_images = read_all_images(test_path_images, all_images)
    all_gts, call_gts = read_all_images(change_test_path_gt, all_gts)

    print(call_images.shape)
    results_patch = extract_orders(call_images, 32, 32)

    for i in range(0, int(results_patch.shape[0] / 32 / 32)):
        visualize(group_images(results_patch[32 * 32 * i:32 * 32 * (i + 1), :, :, :, ], 32),
                  '../gts_color/' + str(i) + 'origin')

    results_gt = extract_orders(call_gts, 32, 32)
    results_color_gt, color_gt = gt_to_color(results_gt)

    for i in range(0, int(results_patch.shape[0] / 32 / 32)):
        visualize(group_images(color_gt[32 * 32 * i:32 * 32 * (i + 1), :, :, :, ], 32),
                  '../gts_color/' + str(i) + 'c_gt')

def test_order_extract_cell2():
    test_path_images = '../images/'
    change_test_path_gt = '../gts/'
    n_test_images = 12
    size = 1024
    all_images = np.empty(shape=(n_test_images, size, size, 3))
    all_gts = np.empty(shape=(n_test_images, size, size, 1))
    all_images, call_images = read_all_images(test_path_images, all_images)
    all_gts, call_gts = read_all_images(change_test_path_gt, all_gts)
    results_patch = extract_orders(call_images, 48, 32, 6, 6)
    print(results_patch.shape)
    results_gt = extract_orders(call_gts, 48, 32, 6, 6)
    print(results_gt.shape)
    results_color_gt, color_gt = gt_to_color(results_gt)
    # test_big_imgs(results_patch[0], results_color_gt[0], results_gt[0], path='../test/retinal_group')
    visualize(results_patch[100].transpose((1, 2, 0)), '../test/retinal_pt')
    visualize(color_gt[100], '../test/retinal_gtc')
    visualize(results_gt[100].transpose((1, 2, 0)), '../test/retinal_gt')


if __name__ == '__main__':
    '''
    the test example use cell data as example 
    '''

    # test_path_images = '../images/'
    # change_test_path_gt = '../gts/'
    # n_test_images = 12
    # size = 1024
    # all_images = np.empty(shape=(n_test_images, size, size, 3))
    # all_gts = np.empty(shape=(n_test_images, size, size, 1))
    #
    #
    # all_images, call_images = read_all_images(test_path_images, all_images)
    # all_gts, call_gts = read_all_images(change_test_path_gt, all_gts)
    # raimg, ragt = extract_random_center(call_images, call_gts, 1200, 64, 48)
    # results_color_gt, color_gt = gt_to_color(ragt)
    # visualize(raimg[33].transpose((1, 2, 0)), '../test/cell_pt')
    # visualize(color_gt[33], '../test/cell_gtc')
    # visualize(ragt[33].transpose((1, 2, 0)), '../test/cell_gt')

    pass