import numpy as np

def overlap_contour(HEimage, pred):
    # print(HEimage.shape, pred.shape, '-----------------', np.max(HEimage), np.max(pred))
    HEimage = np.transpose(HEimage, (1,2,0))
    new_overlap = HEimage.copy()
    for h in range(0, HEimage.shape[0]):
        for w in range(0, HEimage.shape[1]):
            if pred[h,w,0] >= 255:
                new_overlap[h, w, 0] = 1
                new_overlap[h, w, 1] = 0
                new_overlap[h, w, 2] = 0
            if pred[h,w,1] >= 255:
                new_overlap[h, w, 0] = 0
                new_overlap[h, w, 1] = 1
                new_overlap[h, w, 2] = 0
            if pred[h,w,2] >= 255:
                new_overlap[h, w, 0] = 0
                new_overlap[h, w, 1] = 0
                new_overlap[h, w, 2] = 1
    return new_overlap

def overlap_object(HEimage, pred):
    # print(HEimage.shape, pred.shape, '-----------------')
    HEimage = np.transpose(HEimage, (1, 2, 0))
    new_overlap = HEimage.copy()
    for h in range(0, HEimage.shape[0]):
        for w in range(0, HEimage.shape[1]):
            if pred[h, w, 0] == 255:
                new_overlap[h, w, 0] = 1
                new_overlap[h, w, 1] = 0
                new_overlap[h, w, 2] = 0
            if pred[h, w, 1] == 255:
                new_overlap[h, w, 0] = 0
                new_overlap[h, w, 1] = 1
                new_overlap[h, w, 2] = 0
            if pred[h, w, 2] == 255:
                new_overlap[h, w, 0] = 0
                new_overlap[h, w, 1] = 0
                new_overlap[h, w, 2] = 1
    return new_overlap

def overlap_object_contour(HEimage, pred_contour, pred_object):
    # print(HEimage.shape, pred_contour.shape, '-----------------')
    HEimage = np.transpose(HEimage, (1, 2, 0))
    new_overlap = HEimage.copy()
    for h in range(0, HEimage.shape[0]):
        for w in range(0, HEimage.shape[1]):
            if pred_contour[h, w,:,] == 1:
                new_overlap[h, w, 0] = 0
                new_overlap[h, w, 1] = 1
                new_overlap[h, w, 2] = 0
            if pred_object[h, w,:,] == 1:
                new_overlap[h, w, 0] = 0
                new_overlap[h, w, 1] = 0
                new_overlap[h, w, 2] = 1
    return new_overlap

def overlap_gt_co(HEimage, gt_contour, gt_object):
    # print(HEimage.shape, gt_contour.shape, gt_object.shape,'-----------------')
    HEimage = np.transpose(HEimage, (1, 2, 0))
    new_overlap = HEimage.copy()
    for h in range(0, HEimage.shape[0]):
        for w in range(0, HEimage.shape[1]):
            if gt_contour[0,h, w] == 1:
                new_overlap[h, w, 0] = 0
                new_overlap[h, w, 1] = 1
                new_overlap[h, w, 2] = 0
            if gt_object[0,h, w] == 1:
                new_overlap[h, w, 0] = 0
                new_overlap[h, w, 1] = 0
                new_overlap[h, w, 2] = 1
    return new_overlap

if __name__ == '__main__':

    pass