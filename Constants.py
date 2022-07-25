# MONUCLEI  and TNBC  Dataset are adopted  respectively !

# basic hypo-parameters setting
IMG_SIZE = (512, 512)
resize_drive = 512
BATCH_SIZE =4
BINARY_CLASS = 1
VAL_ACC_BEST = -1
TOTAL_EPOCH = 200
learning_rates = 1e-4
NUM_EARLY_STOP = 600

class_nums = 2
dataset_name = 'MONUCLEI'
resize_MONUCLEI = 512
size_h, size_w = 1000, 1000

# experiment configure and saved path
visual_samples = '../log/visual_samples/Object/'
saved_path =  '../log/visual_weights/Object/'
visual_results = '../log/visual_results/Object/'

# path of images saved into directory !
path_nuclei_train_image = '../dataset/npy/tempt/' + dataset_name + '/train_image.npy'
path_nuclei_train_contour = '../dataset/npy/tempt/'+ dataset_name + '/train_contour.npy'
path_nuclei_train_object = '../dataset/npy/tempt/'+ dataset_name +'/train_object.npy'
path_nuclei_train_label = '../dataset/npy/tempt/'+ dataset_name +'/train_label.npy'

path_nuclei_val_image = '../dataset/npy/tempt/' + dataset_name+'/val_image.npy'
path_nuclei_val_contour = '../dataset/npy/tempt/'+ dataset_name+'/val_contour.npy'
path_nuclei_val_object = '../dataset/npy/tempt/'+ dataset_name+'/val_object.npy'
path_nuclei_val_label = '../dataset/npy/tempt/'+ dataset_name+'/val_label.npy'

path_nuclei_test_image = '../dataset/npy/tempt/' +dataset_name+'/test_image.npy'
path_nuclei_test_contour = '../dataset/npy/tempt/'+dataset_name+'/test_contour.npy'
path_nuclei_test_object = '../dataset/npy/tempt/'+dataset_name+'/test_object.npy'
path_nuclei_test_label = '../dataset/npy/tempt/'+dataset_name+'/test_label.npy'

# image number info
total_MONUCLEI_val = 1
total_MONUCLEI_train = 16
total_MONUCLEI = 16
total_MONUCLEI_test = 14