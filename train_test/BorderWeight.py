import numpy as np
from PIL import Image

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


def distance(x,y):
    return  np.sqrt((x[0]-y[0])**2+ (x[1]-y[1])**2)

def get_border_weight(patch_height, patch_width, mode, stride_height=8, stride_width=8, batch_size=4, border = 16):
    loss_weight = np.zeros((patch_height, patch_width))
    center_x = patch_height /2 - 1
    center_y = patch_width / 2 - 1
    if mode == 0:
        return None

    for k in range(0, patch_height):
        for i in range(0, patch_width):
            d1 = np.minimum(k,np.minimum(i,np.minimum(patch_width-i, patch_height-k)))
            d2 = distance((k,i),(center_x,center_y))
            loss_weight[k, i] = d1 / (d1 + d2)         # (0,1)


    max_value = np.max(loss_weight)
    max_value = float(max_value)

    loss_weight = loss_weight/max_value
    loss_weight = np.reshape(loss_weight, (patch_width, patch_height,1))
    result = loss_weight
    print("shape of loss_weight:", result.shape)
    visualize(result,'./loss_weight')
    return result


if __name__ == '__main__':
    get_border_weight(64, 64, 2)
    pass