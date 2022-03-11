
# test_conv2d.py   用tensorflow 调整图片大小
# dataformat NHWC

import numpy as np





# gradient_descent(0.01)
#
# print(np.concatenate([np.random.rand(10,12,12,3),np.random.rand(10,12,12,3)],axis=1).shape)
# print(np.argmax(np.random.rand(10, 12, 12, 3), axis=3).shape)
# print(int(5))

# import os
# from PIL import Image
# from patch_seg import seg_patches_images
# for filename in os.listdir(r"./" + "my_testing/image/"):
#     img_arr =  np.asarray(Image.open("my_testing/image/" + '/' + filename))
#     img_arr = img_arr.transpose((2,0,1))
#     print(img_arr.shape)
#     img_arr = img_arr.reshape((1, 3, img_arr.shape[1], img_arr.shape[2]))
#     img_arr = img_arr.transpose((0, 2, 3, 1))
#     patches, _  = seg_patches_images(img_arr, 128, 128)
#     print(patches.shape)
from skimage.transform import rescale, resize, rotate
from PIL import  Image
import matplotlib.pyplot as plt

def yeis():
    while True:
        print('0000000000000')
        yield 12

next (yeis())


import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage import distance_transform_edt
from scipy.ndimage.filters import gaussian_filter
from skimage.transform import rescale

def elastic_transform(img, gt, alpha, sigma, alpha_affine, random_state=None):
    # the img and gt must be in uint8 data type
    if random_state is None:
        random_state = np.random.RandomState(None)
    img_shape = img.shape
    gt_shape = gt.shape
    shape_size = img_shape[:2]
    # print(shape)
    # print(img)
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)

    img = cv2.warpAffine(img, M, shape_size[::-1], borderMode=cv2.BORDER_CONSTANT)
    img = np.reshape(img, img_shape)
    gt = cv2.warpAffine(gt, M, shape_size[::-1], borderMode=cv2.BORDER_CONSTANT)
    ax = random_state.rand(*shape_size) * 2 - 1
    ay = random_state.rand(*shape_size) * 2 - 1
    max_channel = max(img_shape[2], gt_shape[2])
    ax_all = np.zeros((img_shape[0], img_shape[1], max_channel))
    ay_all = np.zeros((img_shape[0], img_shape[1], max_channel))
    # print(img_shape, gt_shape)
    for i in range(max_channel):
        ax_all[:, :, i] = ax
        ay_all[:, :, i] = ay
    ax_img = ax_all[:, :, :img_shape[2]]
    ay_img = ay_all[:, :, :img_shape[2]]
    dx = gaussian_filter(ax_img, sigma) * alpha
    dy = gaussian_filter(ay_img, sigma) * alpha
    x, y, z = np.meshgrid(np.arange(img_shape[1]), np.arange(img_shape[0]), np.arange(img_shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
    img = map_coordinates(img, indices, order=1, mode='constant').reshape(img_shape)

    # gt = cv2.warpAffine(gt, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
    ax_gt = ax_all[:, :, :gt_shape[2]]
    ay_gt = ay_all[:, :, :gt_shape[2]]

    dx = gaussian_filter(ax_gt, sigma) * alpha
    dy = gaussian_filter(ay_gt, sigma) * alpha
    x, y, z = np.meshgrid(np.arange(gt_shape[1]), np.arange(gt_shape[0]), np.arange(gt_shape[2]))
    indices_gt = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
    gt = map_coordinates(gt, indices_gt, order=1, mode='constant').reshape(gt_shape)

    return img, gt

def rescale_transform(img, gt):
    scale = np.random.uniform(0.5,1.5,1) # range in 0.5-1.5
    #print(scale)
    #scale = random.random() * 1.75 + 0.25  # range in 0.25-2
    size = img.shape[0]
    print('channel =',img.shape[2])
    scale =0.6
    if scale >= 1:
        img_scaled = rescale(img, scale)
        gt_scaled = rescale(gt, scale)
        img_scaled = img_scaled[:size,:size, :]
        gt_scaled = gt_scaled[:size, :size, :]
    else:
        img_scaled_1 = resize(img, (int (img.shape[0]* scale) ,int (img.shape[1]* scale )))
        gt_scaled_1 = resize(gt, (int (gt.shape[0]* scale) ,int (gt.shape[1]* scale )))
        #print(img_scaled_1.dtype)
        img_scaled = np.zeros(img.shape)
        gt_scaled = np.zeros(gt.shape)
        offset = (img_scaled.shape[0] - img_scaled_1.shape[0])//2
        print(img_scaled_1.shape, gt_scaled_1.shape)    # this channel is 2 ??? why
        print(img_scaled.shape, gt_scaled.shape)
        img_scaled[offset:offset+img_scaled_1.shape[0], offset:offset+img_scaled_1.shape[1], :]=img_scaled_1
        gt_scaled[offset:offset+img_scaled_1.shape[0], offset:offset+img_scaled_1.shape[1], :]=gt_scaled_1
    # img_scaled *= 255
    # gt_scaled *= 255
    # img_scaled = img_scaled.astype(np.uint8)
    # gt_scaled = gt_scaled.astype(np.uint8)
    print('img_scaled shape',img_scaled.shape)
    print('gt_scaled shape', gt_scaled.shape)

    return img_scaled, gt_scaled



# --------------test of above elastic transform ----------------
img_reads = Image.open('my_testing/test_results/original11.png')
img_reads_list = np.asarray(img_reads)
# img_patches = np.random.rand(10,img_reads_list.shape[0],img_reads_list.shape[1],img_reads_list.shape[2])
# img_reads_list = np.transpose(img_reads_list, (0, 3, 1, 2))
img_rotate = rotate(img_reads_list, 60)*255
print(img_rotate.shape)
print(img_rotate)
# img_rotate = np.transpose(img_rotate, (0, 2, 3, 1))
img = Image.fromarray(img_rotate.astype(np.uint8))
# plt.plot(img)
# plt.show()

img.save('my_testing/test_results/original111.png')
img_rotate[:,:,:] = img_rotate[::-1,:,:,]
Image.fromarray(img_rotate.astype(np.uint8)).save('my_testing/test_results/original222.png')

# ------------------------------- image rotate changes!-------------------------

my_rotate1    = Image.open('my_testing/test_results/original11.png')
my_rotate_arr = np.asarray(my_rotate1)
new_arr =np.zeros((my_rotate_arr.shape[0],my_rotate_arr.shape[1],my_rotate_arr.shape[2]))
for cs in range(0,my_rotate_arr.shape[2]):
    new_arr [:,:,cs]= my_rotate_arr[:,:,cs].T
print('flip original images:',new_arr.shape)
Image.fromarray(new_arr.astype(np.uint8)).save('my_testing/test_results/original333.png')
# ----------------------------------------- elastic transform ------------------------------------


from patch_seg import  label2rgb

my_test_gt    = Image.open('my_testing/gt/1_gt.bmp')
single_channel_gt = np.asarray(my_test_gt)
single_channel_gt = np.reshape(single_channel_gt, (1, 1, single_channel_gt.shape[0],single_channel_gt.shape[1]))
results = label2rgb(single_channel_gt, 3)      # labeled with 3 kinds of color ! ( label2rgb input is (n,1,h,w))
results = np.transpose(results,(0,2,3,1))
print('visualization is:',results.shape)
Image.fromarray(results[0].astype(np.uint8)).save('my_testing/test_results/gt111.bmp')

# --- now result[0]_gt size is (h,w,3)
my_test_origin    = Image.open('my_testing/image/1_sample.bmp')
origin = np.asarray(my_test_origin)
assert (origin.shape == results[0].shape)
img , gt = elastic_transform(origin, results[0],  550, 12, 10)    # image parameter is (h ,w, 3)
print('having completed elastic transform:',img.shape, gt.shape)
Image.fromarray(img.astype(np.uint8)).save('my_testing/test_results/origins9999.bmp')
Image.fromarray(gt.astype(np.uint8)).save('my_testing/test_results/gts9999.bmp')
index2 = np.arange(13)
np.random.shuffle(index2)
print(index2)
# ----------------------------------------------- test of rescale -----------------------------

resimg1 = Image.open('my_testing/test_results/original11.png')
resimg2    = Image.open('my_testing/test_results/original12.png')
print('resimg1  and resimg2 before transform of rescale:',np.asarray(resimg1).shape,np.asarray(resimg2).shape)
resimg1, resimg2 =rescale_transform(np.asarray(resimg1),np.asarray(resimg2))
print('resimg1  and resimg2 after transform of rescale:',resimg1.shape,resimg2.shape)
print('-------having finished rescaled of images !')

