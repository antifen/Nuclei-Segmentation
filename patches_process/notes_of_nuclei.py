from PIL import Image
import numpy as np
import os

def get_color(category):
    # category can be 2, 3, 4
    # Two Category is black-white
    if category == 2:
        return [(0, 0, 0), (255, 255, 255)]
    # Three Category is red-green-blue
    elif category == 3:
        return [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    # Four category is red-green-blue-yello
    else:
        return [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 0)]


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

# group a set of images row per columns
def group_images(data, per_row):
    assert data.shape[0] % per_row == 0
    assert (data.shape[1] == 1 or data.shape[1] == 3)
    data = np.transpose(data, (0, 2, 3, 1))  # corect format for imshow
    all_stripe = []
    for i in range(int(data.shape[0] / per_row)):
        stripe = data[i * per_row]
        for k in range(i * per_row + 1, i * per_row + per_row):
            stripe = np.concatenate((stripe, data[k]), axis=1)
        all_stripe.append(stripe)
    totimg = all_stripe[0]
    for i in range(1, len(all_stripe)):
        totimg = np.concatenate((totimg, all_stripe[i]), axis=0)
    return totimg

# visualize image (as PIL image, NOT as matplotlib!)
def visualize(data, filename):
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

def read_image_file(directory_name): #
    numfile= 0
    height = 0
    width= 0
    channels= 0
    for filename in os.listdir(r"./" + directory_name):
        # read common images which channel is 1 or 3
        img_read = Image.open(directory_name+'/'+filename)
        assert (np.asarray(img_read).shape[2]==1 or np.asarray(img_read).shape[2]==3)
        height =np.asarray(img_read).shape[0]
        width = np.asarray(img_read).shape[1]
        channels =np.asarray(img_read).shape[2]
        numfile+=1
    data4d = np.empty(shape=(numfile,height,width,channels))
    i=0
    for filename in os.listdir(r"./" + directory_name):
        img_read = Image.open(directory_name + '/' + filename)
        data4d[i]=np.asarray(img_read)
        i+=1
    return data4d

def check_lists():
    arr1 = np.empty(shape=(12, 3, 3, 6))
    arr2 = np.empty(shape=(12, 3, 3, 6))
    arr3 = np.empty(shape=(12, 3, 3, 6))
    lists = []
    lists.append(arr1)
    lists.append(arr2)
    lists.append(arr3)
    print(len(lists))
    print(lists[1].shape)
    return -1

# img_1=Image.open('imgss/2_gt.bmp')
# img_2=np.asarray(img_1)
# img_2=np.reshape(img_2,(1,1,img_2.shape[0],img_2.shape[1]))
# img_3 = label2rgb(img_2, 3)
# visualize(group_images(img_3, 1),'imgss/2_gt2')

images_4d = read_image_file('training/image')   # N*H*W*C
images_4d = np.transpose(images_4d[0:6,:,:,:,],(0,3,1,2))   # N*C*H*W
visualize(group_images(images_4d, 2),'imgss/test_img1')