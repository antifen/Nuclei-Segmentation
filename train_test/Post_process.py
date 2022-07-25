import sys
sys.path.append('../')
sys.path.append('./')
import torch
import Constants
device = 'cuda' if torch.cuda.is_available() else 'cpu'

'''
some methods remained to be updated !
'''

def common_merge_manchenism(net, image, crop_size=Constants.resize_drive):
    '''
    :param net:       proposed net
    :param image:     image is tensor form of [N, C, H, W], shape is (584, 565)
    :param crop_size: 512 default
    :return:          (584, 565)
    '''
    d_h, d_w, h, w = image.size()[2] - crop_size, image.size()[3] - crop_size, image.size()[2], image.size()[3]
    crop_lu_im = image[:, :, 0:h - d_h, 0:w - d_w]
    crop_ld_im = image[:, :, 0:h - d_h, d_w:w]
    crop_ru_im = image[:, :, d_h:h, 0:w - d_w]
    crop_rd_im = image[:, :, d_h:h, d_w:w]

    lu = net(crop_lu_im)
    ru = net(crop_ld_im)
    ld = net(crop_ru_im)
    rd = net(crop_rd_im)

    if Constants.BINARY_CLASS ==1:
        new_image = torch.zeros_like(torch.unsqueeze(image[:,0,:,:,], dim=1))
    else:
        new_image = torch.zeros(size=(image.size()[0], 2, image.size()[2],image.size()[3]))

    for i in range(0, h):
        for j in range(0, w):
            if i >= d_h and j >= d_w and i < crop_size and j < crop_size:
                new_image[:, :, i, j] = (lu[:, :, i, j] + ru[:, :, i, j - d_w] + ld[:, :, i - d_h, j] + rd[:, :,
                                                                                                        i - d_h,
                                                                                                        j - d_w]) / 4
            if i >= 0 and j >= 0 and i < d_h and j < d_w:
                new_image[:, :, i, j] = lu[:, :, i, j]
            if i >= 0 and j >= d_w and i < d_h and j < crop_size:
                new_image[:, :, i, j] = (lu[:, :, i, j] + ru[:, :, i, j - d_w]) / 2
            if i >= 0 and j >= crop_size and i < d_h:
                new_image[:, :, i, j] = ru[:, :, i, j - d_w]
            if i >= d_h and j >= 0 and i < crop_size and j < d_w:
                new_image[:, :, i, j] = (lu[:, :, i, j] + ld[:, :, i - d_h, j]) / 2
            if i >= d_h and j >= crop_size and i < crop_size:
                new_image[:, :, i, j] = (ru[:, :, i, j - d_w] + rd[:, :, i - d_h, j - d_w]) / 2
            if i >= crop_size and j >= 0 and j < d_w:
                new_image[:, :, i, j] = ld[:, :, i - d_h, j]
            if i >= crop_size and j >= d_w and j < crop_size:
                new_image[:, :, i, j] = (ld[:, :, i - d_h, j] + rd[:, :, i - d_h, j - d_w]) / 2
            if i >= crop_size and j > crop_size:
                new_image[:, :, i, j] = rd[:, :, i - d_h, j - d_w]

    return new_image.to(device)

def vote_manchenism():
    '''
    Codes needs to be checked again, This method will be open resource soon !
    :return:
    '''
    pass

def thereshold_merge():
    '''
    simple traditional ways
    :return:
    '''
    pass
