import cv2
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt


NEIGHBOR_HOODS_4 = False
OFFSETS_4 = [[0, -1], [-1, 0], [0, 0], [1, 0], [0, 1]]
NEIGHBOR_HOODS_8 = True
OFFSETS_8 = [[-1, -1], [0, -1], [1, -1],
             [-1,  0], [0,  0], [1,  0],
             [-1,  1], [0,  1], [1,  1]]

def visualize(data, filename):
    '''
    :param data:     input is 3d tensor of a image,whose size is (H*W*C)
    :param filename:
    :return:         saved into filename positions
    '''
    assert (len(data.shape) == 3)  # height*width*channels
    # print data
    if data.shape[2] == 1:  # in case it is black and white
        data = np.reshape(data, (data.shape[0], data.shape[1]))
    if np.max(data) > 1:
        img = Image.fromarray(data.astype(np.uint8))  # the image is already 0-255
    else:
        img = Image.fromarray((data * 255).astype(np.uint8))  # the image is between 0-1
    img.save(filename + '.png')
    return img

def group_images(data, per_row):
    '''
    :param data:     both N H W C and N C H W is OK ！
    :param per_row:  every images number in a row
    :return:         images channel last
    '''
    assert data.shape[0]%per_row==0
    if (data.shape[1]==1 or data.shape[1]==3):
        data = np.transpose(data,(0,2,3,1))      # change data format into channel last !
    all_stripe = []
    for i in range(int(data.shape[0]/per_row)):
        stripe = data[i*per_row]
        for k in range(i*per_row+1, i*per_row+per_row):
            stripe = np.concatenate((stripe,data[k]),axis=1)
        all_stripe.append(stripe)
    totimg = all_stripe[0]
    for i in range(1,len(all_stripe)):
        totimg = np.concatenate((totimg,all_stripe[i]),axis=0)
    return totimg

def reorganize(binary_img: np.array):
    index_map = []
    points = []
    index = -1
    rows, cols = binary_img.shape
    for row in range(rows):
        for col in range(cols):
            var = binary_img[row][col]
            if var < 0.5:
                continue
            if var in index_map:
                index = index_map.index(var)
                num = index + 1
            else:
                index = len(index_map)
                num = index + 1
                index_map.append(var)
                points.append([])
            binary_img[row][col] = num
            points[index].append([row, col])
    return binary_img, points

def neighbor_value(binary_img: np.array, offsets, reverse=False):
    binary_img = binary_img.copy()
    rows, cols = binary_img.shape
    label_idx = 0
    rows_ = [0, rows, 1] if reverse == False else [rows-1, -1, -1]
    cols_ = [0, cols, 1] if reverse == False else [cols-1, -1, -1]
    for row in range(rows_[0], rows_[1], rows_[2]):
        for col in range(cols_[0], cols_[1], cols_[2]):
            label = 256
            if binary_img[row][col] < 0.5:
                continue
            for offset in offsets:
                neighbor_row = min(max(0, row+offset[0]), rows-1)
                neighbor_col = min(max(0, col+offset[1]), cols-1)
                neighbor_val = binary_img[neighbor_row, neighbor_col]
                if neighbor_val < 0.5:
                    continue
                label = neighbor_val if neighbor_val < label else label
            if label == 255:
                label_idx += 1
                label = label_idx
            binary_img[row][col] = label
    return binary_img


def Two_Pass(binary_img: np.array, neighbor_hoods):
    if neighbor_hoods == NEIGHBOR_HOODS_4:
        offsets = OFFSETS_4
    elif neighbor_hoods == NEIGHBOR_HOODS_8:
        offsets = OFFSETS_8
    else:
        raise ValueError

    binary_img = neighbor_value(binary_img, offsets, False)
    binary_img = neighbor_value(binary_img, offsets, True)
    return binary_img


def recursive_seed(binary_img: np.array, seed_row, seed_col, offsets, num, max_num=100):
    rows, cols = binary_img.shape
    binary_img[seed_row][seed_col] = num
    for offset in offsets:
        neighbor_row = min(max(0, seed_row+offset[0]), rows-1)
        neighbor_col = min(max(0, seed_col+offset[1]), cols-1)
        var = binary_img[neighbor_row][neighbor_col]
        if var < max_num:
            continue
        binary_img = recursive_seed(binary_img, neighbor_row, neighbor_col, offsets, num, max_num)
    return binary_img


def Seed_Filling(binary_img, neighbor_hoods, max_num=100):
    if neighbor_hoods == NEIGHBOR_HOODS_4:
        offsets = OFFSETS_4
    elif neighbor_hoods == NEIGHBOR_HOODS_8:
        offsets = OFFSETS_8
    else:
        raise ValueError

    num = 1
    rows, cols = binary_img.shape
    for row in range(rows):
        for col in range(cols):
            var = binary_img[row][col]
            if var <= max_num:
                continue
            binary_img = recursive_seed(binary_img, row, col, offsets, num, max_num=100)
            num += 1
    return binary_img


def color_map(img,cutoff=0.5):
    import label
    img_ = img
    ret = np.zeros((img_.shape[0],img_.shape[1],3),dtype=np.uint8)
    # img_ = label(img_, background=0)
    true_id_list = list(np.unique(img_))
    print(len(true_id_list))
    for t in true_id_list[1:]:
        ret[img_==t,0] = random.randint(0,255)
        ret[img_==t,1] = random.randint(0,255)
        ret[img_==t,2] = random.randint(0,255)
    return np.array(ret,dtype=np.uint8)

def instance_seg_results(img_path):
    img = np.asarray(Image.open(img_path))
    binary_img = img
    binary_img = Two_Pass(binary_img, NEIGHBOR_HOODS_8)
    binary_img, points = reorganize(binary_img)
    color_maps_ = color_map(np.array(binary_img))
    return color_maps_

if __name__ == "__main__":
    # img = np.asarray(Image.open('1seg.png'))
    results = instance_seg_results('1seg.png')
    visualize(results, './2')
    pass