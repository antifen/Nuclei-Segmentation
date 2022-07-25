import numpy as np
from PIL import Image

'''
BorderWeight refer to our articles formula 
'''

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

def distance_point(center, point):

    '''
    can be change into other forms of distance as well
    :param center:  a 2-d point (x0, y0)
    :param point:   a 2-d point (x1, y1)
    :return: distance of two points !
    '''

    dis = np.sqrt((point[0]-center[0]) ** 2 + (point[1] - center[1]) ** 2)
    return dis

def my_border_weight(sizes):

    '''
    :param array: (Size, Size)
    :return:      [H, W]
    '''

    epsilong = 1e-16
    W_array = np.ones(shape=(sizes, sizes))
    weight_array = np.zeros_like(W_array)
    H, W = W_array.shape[0], W_array.shape[1]
    total_weight = 0

    # calculate seita parameter
    for i in range (0, H):
        for j in range(0, W):
            min_border = np.minimum(np.minimum(np.minimum(i, j), H-i), W-j)
            border_weight_point = min_border / (min_border + distance_point((H/2, W/2),(i, j)))
            total_weight += border_weight_point

    # calculate current point distance
    for i in range (0, H):
        for j in range(0, W):
            min_border = np.minimum(np.minimum(np.minimum(i, j), H-i), W-j)
            border_weight_point = min_border / (min_border + distance_point((H/2, W/2),(i, j)))
            weight_array[i, j] = border_weight_point * H * W / total_weight

    # change range to  0-1
    for i in range (0, H):
        for j in range(0, W):
            W_array[i,j] = (weight_array[i,j]  - np.min(weight_array) + epsilong) / (np.max(weight_array) - np.min(weight_array))

    return W_array

if __name__ == '__main__':
    '''
    Test our BorderWeight maps and save the maps 
    '''
    visualize(np.expand_dims(my_border_weight(128), axis = 2), './tempt/my_border_weight_map')
    pass