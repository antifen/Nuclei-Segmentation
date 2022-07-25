import os
import numpy as np
from time import time

def platform_info(epoch, tic, train_epoch_loss, img_size, optimizers):
    print('********')
    print('Epoch:', epoch, ' time:', int(time() - tic))
    print('train_loss:', train_epoch_loss)
    print('image size:', img_size)
    print('learn ratio is:', optimizers.state_dict()['param_groups'][0]['lr'])
    print('********')

def check_size(img, mask, label):
    print('=======****======= image size is  :{} and label size is :{} =======****======='.format(img.size(), label.size()))

def other_info():
    pass

def consistency_check():
    pass



