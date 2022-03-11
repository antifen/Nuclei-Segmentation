import numpy as np
import random
from PIL import Image
import cv2
import os

def segment_line():
    print('------------------------------------分割线------------------------------------')

def shape_check(tensor, name =''):
    print('check {} shape is: {}'.format(name, tensor.shape))

def value_range_check(tensor, name = ''):
    print('check {} value range is from {} to {}'.format(name, np.min(tensor), np.max(tensor)))

def net_predict_reshape4d(tensor, classification, Height ,Width):
    '''
    :param tensor:  N  C  H*W =====> N C H W      N H*W C =====> N C H W
    :return:        N C H W
    '''
    if tensor.shape[1]==classification:
        tensor = np.reshape(tensor, (tensor.shape[0], tensor.shape[1], Height, Height))
        return tensor
    elif tensor.shape[2]==classification:
        tensor = np.transpose(tensor, (0,2,1))
        tensor = np.reshape(tensor,(tensor.shape[0],tensor.shape[1],Height,Height))
        return tensor
    else:
        print('wrong happen !')
        raise Exception

def tensor4d_channels_last(tensor):
    '''
    :param tensor: N H W C and N H W C
    :return: N H W C
    '''
    if tensor.shape[3]==1 or tensor.shape[3]==3:
        if tensor.shape[1] > 3 and tensor.shape[2] > 3:
            return tensor
        else:
            print('I am not sure tensor_channels_last!')
            raise Exception
    if tensor.shape[1] == 1 or tensor.shape[1] == 3:
        return np.transpose(tensor, (0,2,3,1))

def tensor4d_channels_first(tensor):
    '''
    :param tensor: N H W C and N H W C
    :return:   N C H W
    '''
    if tensor.shape[3] == 1 or tensor.shape[3] == 3:
        return np.transpose(tensor, (0,3,1,2))
    if tensor.shape[1] == 1 or tensor.shape[1] == 3:
        if tensor.shape[2] > 3 and tensor.shape[3] > 3:
            return tensor
        else:
            print('I am not sure tensor_channels_first !')
            raise Exception




if __name__ == '__main__':
    tensor = np.random.rand(2,3,12,12)
    shape_check(tensor,'my tensor')
    value_range_check(tensor,'my tensor')
    pre_tensor = np.random.rand(20,12*12,3)
    new_predicts = net_predict_reshape4d(pre_tensor, 3,12,12)
    shape_check(new_predicts)
    value_range_check(tensor, 'my tensor2')
    segment_line()
    shape_check(tensor4d_channels_last(tensor))
    pass
