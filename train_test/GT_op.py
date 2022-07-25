import numpy as np
import sys
sys.path.insert(0, '../src/')
sys.path.insert(0, '../lib/')

def gt_to_class(gt):
    '''
    in this method
    :param gt:   N C H W
    :return:     N Class H W
    '''
    gt_arr0 = np.zeros(shape=(gt.shape[0], 1, gt.shape[2], gt.shape[3]))
    gt_arr1 = np.zeros(shape=(gt.shape[0], 1, gt.shape[2], gt.shape[3]))
    print(np.max(gt), np.min(gt))
    gt_arr0[np.where(gt == 0) ] = 1
    print('get 2-C ground-truth')
    gt_arr1[np.where(gt >0)]   = 1
    gt_arr = np.concatenate([gt_arr0, gt_arr1], axis=1)
    return gt_arr

