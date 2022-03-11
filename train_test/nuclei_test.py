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


from patch_seg import get_color, label2rgb,visualize,seg_patches_images,seg_patches_gt,group_images, \
     load_images, test_images, resuts_compare_gts, change_gtpatch_3ch

from models import make_parallel, unet, unet_backbone

from help_functions import size_change


# test_images(20,20)
# resuts_compare_gts (20,20)

patch_h = 128
patch_w = 128
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
     size_change(dir_img = "training/image/", dir_save = "my_testing/image/", sizes=1024)
     size_change(dir_img="training/gt/", dir_save="my_testing/gt/", sizes=1024)

size_changes()
all_patches, all_patches_gt = prepare_trains()

n_ch = all_patches.shape[3]
patch_height = all_patches.shape[1]
patch_width  = all_patches.shape[2]

# print(unet_backbone(Input(shape=(3, 64, 64)), 'relu').shape)
model = unet(n_ch, patch_height, patch_width, 3, act='selu',
          loss_weight=None, sample_weight_mode=None,
          GPU_num=1, net_name='unet', fine_tune=0, pretrain_model='')

print(' check the model output size is:', model.output_shape)

all_patches    = np.transpose(all_patches,(0,3,1,2))

print(' before gt_current size is:',all_patches_gt.shape)

all_patches_gt = np.transpose(all_patches_gt, (0,3,1,2))
all_patches_gt = np.reshape(all_patches_gt,(all_patches_gt.shape[0], 3, patch_h*patch_w,))
all_patches_gt = np.transpose(all_patches_gt, (0,2,1))

print(' now gt_current size is:',all_patches_gt.shape)


# Load the saved model
model = model_from_json(open('experiments' + '/unet' + '_architecture.json').read(), custom_objects={'tf': tf})
model.load_weights('experiments/unet_last_weight.h5')


predictions = model.predict(all_patches, batch_size=16, verbose=1)
acc = accuracy_score(np.argmax(all_patches_gt, axis=2).reshape(-1), np.argmax(predictions, axis=2).reshape(-1))
print(' prediction accuracy after the network is:', acc)

predictions = np.transpose(predictions, (0,2,1))
predictions = np.reshape(predictions,(predictions.shape[0], 3, patch_h, patch_w))
predictions = np.transpose(predictions, (0,2,3,1))
print(predictions.shape)
new_predictions = np.zeros((predictions.shape[0], predictions.shape[1], predictions.shape[2]))
for index_pre1 in range(0, predictions.shape[0]):
     for index_pre2 in range(0, predictions.shape[1]):
          for index_pre3 in range(0, predictions.shape[2]):
               if predictions[index_pre1, index_pre2, index_pre3, 1] > 0.3:
                    new_predictions[index_pre1, index_pre2, index_pre3] = 1
               if predictions[index_pre1, index_pre2, index_pre3, 2] > 0.3:
                    new_predictions[index_pre1, index_pre2, index_pre3] = 2
new_predictions = new_predictions.reshape((predictions.shape[0], predictions.shape[1], predictions.shape[2],1))


print('now the prediction shape is:',new_predictions.shape)
new_predictions2 = np.transpose(new_predictions, (0,3,1,2))     # (896, 128, 128, 1) ------> (896, 1, 128, 128)
color_patches = label2rgb(new_predictions2,3)
color_patches2 = np.transpose(color_patches, (0,2,3,1))
print('now the color_patches shape is:',color_patches2.shape)


nums_per = color_patches2.shape[0] //(1024//patch_h)**2
per_size = (1024//patch_h)**2
print('nums_per is:',nums_per)
for index_whole in range(0,nums_per):
     visualize(group_images(color_patches2[index_whole*per_size:(index_whole+1)*per_size], 1024//patch_h),'my_testing/test_results/result'+str(index_whole))


# change 3 classes into RGB color
# contour and region segmentation details
