from keras import layers
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, core, Dropout
from keras.layers import Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.layers.core import Activation
from keras.callbacks import ModelCheckpoint, TensorBoard, Callback
from keras import backend as K
from keras.utils import plot_model
from keras.optimizers import SGD
from keras.activations import selu, relu
from keras.layers.core import Lambda
import tensorflow as tf
import numpy as np
from keras.models import model_from_json
from sklearn.metrics import accuracy_score
from keras.backend import clear_session
from PIL import Image
import os

from patch_seg import get_color, label2rgb,visualize,seg_patches_images,seg_patches_gt,group_images, \
     load_images, test_images, resuts_compare_gts, change_gtpatch_3ch

from models import make_parallel, unet, unet_backbone

from help_functions import size_change, extract_ordered_overlap, recompose_overlap, recompose, paint_border,\
    get_loss_weight,paint_border_overlap,pred_to_imgs


# test_images(20,20)
# resuts_compare_gts (20,20)

patch_h = 128
patch_w = 128
batch_size =16
'''
1 consider to take the train set all images as train set, first prepare and load data for training 

2 patch for all images (WSI images and labels)

3 send all of these images into segmentation network to finish semantic segment

4 get the predict results, and reshape to 4d tensors which can be visualized by manuscript method 

'''
def prepare_trains():
     all_train_imgs = load_images('my_testing/image', 3)
     print('original WSI images size are:', all_train_imgs.shape)

     all_patches, _ = seg_patches_images(all_train_imgs, patch_h , patch_w )
     print('all WSI images patch size are:',all_patches.shape)

     all_gts = load_images('my_testing/gt',-1)

     print('original gt bmp size are:',all_gts.shape)



     all_patches_gt, _ = seg_patches_gt(all_gts, patch_h , patch_w)

     print('one channel gt patches size are:',all_patches_gt.shape)

     all_patches_gt = change_gtpatch_3ch(all_patches_gt)

     print('all gt patches size (3 channels) are:', all_patches_gt.shape)

     print('-----------------------------having loaded all patches of both original images and corresponding gt into tensors !------------------')

     return all_patches, all_patches_gt

def size_changes():
     size_change(dir_img = "test/image/", dir_save = "my_testing/image/", sizes=1024)
     size_change(dir_img="test/gt/", dir_save="my_testing/gt/", sizes=1024)

size_changes()
all_patches, all_patches_gt = prepare_trains()

n_ch = all_patches.shape[3]
patch_height = all_patches.shape[1]
patch_width  = all_patches.shape[2]


index_p = 0
for filename in os.listdir(r"./" + "my_testing/image/"):
    model = model_from_json(open('experiments' + '/unet' + '_architecture.json').read(), custom_objects={'tf': tf})
    model.load_weights('experiments/unet_last_weight.h5')
    img_arr =  np.asarray(Image.open("my_testing/image/" + '/' + filename))
    print(img_arr.shape)
    visualize(img_arr, 'my_testing/test_results/original' + str(index_p))

    img_arr = np.transpose(img_arr, (2,0,1))

    img_arr = np.reshape(img_arr , (1,3, img_arr.shape[1], img_arr.shape[2]))

    test_imgs = paint_border_overlap(img_arr, patch_h, patch_w, 16, 16, 3)
    # expanding the images !
    print('expanding size is : ', test_imgs.shape)
    #  overlapping extract !
    img_arr_patches =  extract_ordered_overlap(test_imgs, patch_h, patch_w, 16, 16, 3)
    print('all one image patches size is : ',img_arr_patches.shape)

    # img_arr_patches /= 255.0

    # patches expand
    shape = img_arr_patches.shape
    num_patches = shape[0] / batch_size + 1
    expand_patches_imgs_test = np.zeros((int(num_patches * batch_size - shape[0]), shape[1], shape[2], shape[3]),
                                        )
    patches_imgs_test = np.concatenate((img_arr_patches, expand_patches_imgs_test), axis=0)

    # Calculate the predictions , model predict the patch category
    prediction = model.predict(patches_imgs_test, batch_size=batch_size, verbose=1)
    # acc = accuracy_score(np.argmax(all_patches_gt, axis=2).reshape(-1), np.argmax(prediction, axis=2).reshape(-1))
    # print('the prediction accuracy of current image',index_p,' is:',acc)
    prediction = prediction [:shape[0], :, :]
    print("predicted patches size :", prediction.shape)
    print("max value of the predicted patches {}".format(np.max(prediction[:, :, 1])))
    # ===== Convert the prediction arrays in corresponding images
    pred_to_imgss = pred_to_imgs(prediction, real_value=1)
    print('pred imgss shape is:', pred_to_imgss.shape)

    # merge overlapping patches
    pred_imgs = recompose_overlap(pred_to_imgss, 1024,
                                     1024, 16, 16,
                                     3, [get_loss_weight(patch_h, patch_w, 2)])  # predictions
    print("the final pred imgs shape is: ", pred_imgs.shape)

    index_p = index_p + 1

    visualize(group_images(np.transpose(pred_imgs, (0, 2, 3, 1)), 1), 'my_testing/test_results/result' + str(index_p))
