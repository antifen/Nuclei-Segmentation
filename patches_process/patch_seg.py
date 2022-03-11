from PIL import Image
import numpy as np
import os

def get_color(category):
    # category can be 2, 3, 4
    # Two Category is black-white
    if category == 2:
        return [(0, 0, 0), (255, 255, 255)]
    # Three Category is red - green - blue
    elif category == 3:
        return [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    # Four category is red - green - blue - yellow
    else:
        return [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

def label2rgb(imgs, category):
    if category > 4:
        print("ERROR: at most 4 categories")
        exit()
    assert (len(imgs.shape) == 4)
    result = np.zeros((imgs.shape[0], 3, imgs.shape[2], imgs.shape[3]))
    color = get_color(category)
    for k in range(imgs.shape[0]):
        for i in range(imgs.shape[2]):
            for j in range(imgs.shape[3]):
                for m in range(3):
                    result[k, m, i, j] = color[int(imgs[k, 0, i, j])][m]
    return result


# visualize image (as PIL image, NOT as matplotlib!)
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


def seg_patches_images(imgs, patch_h, patch_w):
    '''
    :param imgs:      the input size is (N*H*W*C)
    :param patch_h:   patch size of height direction
    :param patch_w:   patch size of width  direction
    :return: (N*num_patch_w*num_patch_h, H, W, C) and
              list of many [ (num_patch_w*num_patch_h, H, W, C),...,(num_patch_w*num_patch_h, H, W, C)]
    which represents all patches and one WSI patches in a list ste respectively.
    '''

    assert (imgs.shape[3]==1 or imgs.shape[3]==3)  #check channels
    assert (imgs.shape[1]%patch_h==0)              #check patch_h is integer
    assert (imgs.shape[2]%patch_w==0)              #check patch_w is integer

    num_patch_w = int (imgs.shape[1]/patch_h)      # total patch_w numbers
    num_patch_h = int(imgs.shape[2] /patch_w)      # total patch_h numbers

    # total patch images size is :(N*num_patch_w*num_patch_h, h, w, c)
    cut_images = np.empty(shape=(imgs.shape[0]*num_patch_h*num_patch_w,patch_h ,
                                 patch_w,imgs.shape[3]))
    # one single whole image size is :(num_patch_w*num_patch_h, h, w, c)
    one_single_image = np.empty(shape=(num_patch_w*num_patch_h, patch_h, patch_w, imgs.shape[3]))
    # one_whole_image contains all one single whole image, whose length is N
    one_whole_image = []
    index = 0
    index_s = 0
    for i in range(0,imgs.shape[0]):         # numbers of total images
        for j in range(0, num_patch_h):
            for k in range(0, num_patch_w):
                tempt =imgs[i,j*patch_h:(j+1)*patch_h,
                       k*patch_w: (k+1)*patch_w,:,]
                cut_images[index_s] = tempt       # current tempt' id is i*num_patch_w+j*num_patch_w+k
                one_single_image[index] = tempt
                index = index + 1
                index_s = index_s +1
                # print('............cutting is performing !...........')
        one_whole_image.append(one_single_image) # add current one_single_image into one_whole_image
        index = 0                                # clear one_single_image
        one_single_image =np.zeros(shape=(num_patch_w*num_patch_h, patch_h, patch_w, imgs.shape[3]))
    return cut_images, one_whole_image


def seg_patches_gt (imgs, patch_h, patch_w):
    '''
    :param imgs:      imgs is (N , 1000, 1000 ) bmp format data
    :param patch_h:
    :param patch_w:
    :return:
    '''
    assert (len(imgs.shape)==3)     #  check is 3d tensor
    imgs= np.reshape(imgs, (imgs.shape[0],imgs.shape[1],imgs.shape[2],1))
    return  seg_patches_images(imgs,patch_h, patch_w)


def group_images(imgs, per_row):
    '''

    :param imgs:     # images size is: (N*H*W*C),per_row
    :param per_row:
    :return:         #  output is ( N/ per_row, per_row, c) ,which can be visualization directly
    '''
    height = imgs.shape[1]
    width = imgs.shape[2]
    new_img = np.empty(shape=((imgs.shape[0]//per_row)*height,
                              per_row*width, imgs.shape[3]))
    print('---------',new_img.shape)
    for i in  range(0,imgs.shape[0]//per_row):
        for j in range(0, per_row):
            new_img[height*i:height*(i+1),
            width*j:width*(j+1),:,] = imgs[i*per_row+j,:,:,:,]
    print('-------------------having already been grouped into one big image!---------------------- ')
    return new_img


def load_images(directory_name, channels):
    '''

    :param directory_name:  # read all images of directory into 4d array, which input format is: (N*H*W*C)
    :param channels:
    :return:                # have sorted by numbers, output size is 4d array (N*H*W*C)
    '''
    i = 0
    height = 0
    width  = 0
    for filename in os.listdir(r"./" + directory_name):
        height =  np.asarray(Image.open(directory_name + '/' + filename)).shape[0]
        width  =  np.asarray(Image.open(directory_name + '/' + filename)).shape[1]
        i = i + 1
    j = 0
    if channels==1 or channels==3:                      # check is png or bmp files
        test_group = np.empty(shape=(i, height, width, channels))
    else:
        test_group = np.empty(shape=(i, height, width))

    filenames = os.listdir(r"./" + directory_name)
    filenames.sort(key=lambda x: int(x.split("_")[0]))  # solve the sort question ! the image name must be 123_XXXXX.bmp format
    print('load images sequence is :\n',filenames)
    for filename in filenames:
        img_read = Image.open(directory_name + '/' + filename)
        test_group[j] = np.asarray(img_read)
        j += 1
    print('---------------------have loaded all of images (png format or bmp format)!------------------')
    return test_group

#  my add methods !!!!!!!!!!!!
def change_gtpatch_3ch(get_patches):
    '''
    :param get_patches:   #(N, H, W, 1)
    :return:              #(N, H, W, 3)    the result data can be visualized and every is 0-1
    '''
    print('now gt patch size is:',get_patches.shape)
    new_get_patches =np.empty(shape=(get_patches.shape[0], get_patches.shape[1],get_patches.shape[2],3))
    for i in range(0, get_patches.shape[0]):
        for j in range(0, get_patches.shape[1]):
            for k in range(0, get_patches.shape[2]):
                if get_patches[i,j,k,0] == 0:
                    new_get_patches[i,j,k,0]=1
                if get_patches[i,j,k,0] == 1:
                    new_get_patches[i,j,k,1]=1
                if get_patches[i,j,k,0] == 2:
                    new_get_patches[i,j,k,2]=1
    return new_get_patches

def test_images(patch_h, patch_w,file_directory='training/image',WSI_size=1000):
    new_test_group = load_images(file_directory,channels=3 )
    print(new_test_group.shape)
    cut_images, one_whole_image = seg_patches_images(new_test_group, patch_h, patch_w)
    # print(cut_images.shape)
    # print(len(one_whole_image))
    # print(one_whole_image[1].shape)
    for i in range(0, len(one_whole_image)):
        visualize(group_images(one_whole_image[i], WSI_size//patch_h),'../imgss/images/image'+str(i))


def resuts_compare_gts(patch_h, patch_w,file_directory='training/gt',WSI_size=1000):
    '''
    :return: the segmentation results to take input of (N, 1000, 1000) into rgb results
    '''
    new_gt = load_images(file_directory, channels = -1 )    # gt original size is (1000, 1000)
    new_gt = np.reshape(new_gt, (new_gt.shape[0],1,new_gt.shape[1],new_gt.shape[2]) )
    print(new_gt.shape)                 # get images every pixel values is 0 or 1 or 2 ,total kinds is 3  !
    results = label2rgb(new_gt, 3)      # labeled with 3 kinds of color ! ( label2rgb input is (n,1,h,w))
    results = np.transpose(results,(0,2,3,1))
    print(results.shape)                # (12, 1000, 1000, 3)

    cut_images, one_whole_image = seg_patches_images(results, patch_h, patch_w)
    for i in range(0, len(one_whole_image)):
        visualize(group_images(one_whole_image[i], WSI_size//patch_h),'imgss/gts/gt'+str(i))

    # for i in range(0, results.shape[0]):
    #     visualize(results[i], 'imgss/gts/test_img_gt' + str(i))    # (12, 1000, 1000, 3)
    print('---------------------have preserved the segmented or gt results already !-------------------------')


if __name__ == '__main__':

    # resuts_compare_gts(20, 20,WSI_size=1000)
    test_images(100, 100, file_directory='../MONUCLEI/training/image', WSI_size=1000)
