from PIL import Image
import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment
import torch

'''
some ideas reference the article came from Hover-Net articles 
confused matrix provided as well 
the evaluation metrics includes ACC, Sen, Spe, F1-score(dice), Precision, Recall, AJI, Hausdorff distance, Dice1 and so on
other metrics remain to be added 
the provided metrics can be used in other semantic segmentation scenes as well
'''

def dice_1(pred, true):
    true = np.copy(true)
    pred = np.copy(pred)
    true[true > 0] = 1
    pred[pred > 0] = 1
    inter = true * pred
    denom = true + pred
    return 2.0 * np.sum(inter) / np.sum(denom)

def dice_2(pred, true):
    true = np.copy(true)
    pred = np.copy(pred)
    true_id = list(np.unique(true))
    pred_id = list(np.unique(pred))
    overall_total = 0
    overall_inter = 0
    true_masks = [np.zeros(true.shape)]
    for t in true_id[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)
    pred_masks = [np.zeros(true.shape)]
    for p in pred_id[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)
    for true_idx in range(1, len(true_id)):
        t_mask = true_masks[true_idx]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        try:      # blinly remove background
            pred_true_overlap_id.remove(0)
        except ValueError:
            pass  # just mean no background
        for pred_idx in pred_true_overlap_id:
            p_mask = pred_masks[pred_idx]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            overall_total += total
            overall_inter += inter
    return 2 * overall_inter / overall_total

def iou_score(output, target):
    smooth = 1e-16
    if torch.is_tensor(output):
        output = output.cpu().numpy()
    if torch.is_tensor(target):
        target = target.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    return (intersection + smooth) / (union + smooth)

def matrix_metrics(output, target):
    '''
    :param output:
    :param target:
    :return:     ACC SEN SPE PRECISION RECALL F1
    '''
    smooth = 1e-16
    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.cpu().numpy()
    TP = (output * target).sum()
    TN = ((1-output) * (1-target)).sum()
    FP = (output * (1-target)).sum()
    FN = ((1-output) * target).sum()
    # print(TP, TN, FP, FN)
    return (TP + TN) / (TP + TN + FP + FN),\
           TP / (TP + FN), \
           TN / (TN + FP),\
           TP / (TP + FP), \
           TP / (TP + FN),\
           2/(2 + FP/TP + FN/TP)

def get_fast_aji(true, pred):
    """
    AJI version distributed by MoNuSeg, has no permutation problem but suffered from
    over-penalisation similar to DICE2
    Fast computation requires instance IDs are in contiguous orderding i.e [1, 2, 3, 4]
    not [2, 3, 6, 10]. Please call `remap_label` before hand and `by_size` flag has no
    effect on the result.
    """
    true = np.copy(true)  # we need this ?
    pred = np.copy(pred)
    true = np.array(true, dtype='uint8')
    pred = np.array(pred, dtype='uint8')
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    true_masks = [None, ]
    for t in true_id_list[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)

    pred_masks = [None, ]
    for p in pred_id_list[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)

    # prefill with value
    pairwise_inter = np.zeros([len(true_id_list) - 1,
                               len(pred_id_list) - 1], dtype=np.float64)
    pairwise_union = np.zeros([len(true_id_list) - 1,
                               len(pred_id_list) - 1], dtype=np.float64)

    # caching pairwise
    for true_id in true_id_list[1:]:  # 0-th is background
        t_mask = true_masks[true_id]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:  # ignore
                continue  # overlaping background
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            pairwise_inter[true_id - 1, pred_id - 1] = inter
            pairwise_union[true_id - 1, pred_id - 1] = total - inter
    #
    pairwise_iou = pairwise_inter / (pairwise_union + 1.0e-6)
    # pair of pred that give highest iou for each true, dont care
    # about reusing pred instance multiple times
    paired_pred = np.argmax(pairwise_iou, axis=1)
    pairwise_iou = np.max(pairwise_iou, axis=1)
    # exlude those dont have intersection
    paired_true = np.nonzero(pairwise_iou > 0.0)[0]
    paired_pred = paired_pred[paired_true]
    # print(paired_true.shape, paired_pred.shape)
    overall_inter = (pairwise_inter[paired_true, paired_pred]).sum()
    overall_union = (pairwise_union[paired_true, paired_pred]).sum()
    #
    paired_true = (list(paired_true + 1))  # index to instance ID
    paired_pred = (list(paired_pred + 1))
    # add all unpaired GT and Prediction into the union
    unpaired_true = np.array([idx for idx in true_id_list[1:] if idx not in paired_true])
    unpaired_pred = np.array([idx for idx in pred_id_list[1:] if idx not in paired_pred])
    for true_id in unpaired_true:
        overall_union += true_masks[true_id].sum()
    for pred_id in unpaired_pred:
        overall_union += pred_masks[pred_id].sum()
    #
    aji_score = overall_inter / overall_union
    return aji_score

def hausdorff(pred, true):
    from hausdorff import hausdorff_distance
    '''
    the parameter distance can be changed, such as manhattan or chebyshev or cosine and so on !
    '''
    return  hausdorff_distance(true, pred, distance="euclidean")

def pair_coordinates(setA, setB, radius):
    """Use the Munkres or Kuhn-Munkres algorithm to find the most optimal
    unique pairing (largest possible match) when pairing points in set B
    against points in set A, using distance as cost function.
    Args:
        setA, setB: np.array (float32) of size Nx2 contains the of XY coordinate
                    of N different points
        radius: valid area around a point in setA to consider
                a given coordinate in setB a candidate for match
    Return:
        pairing: pairing is an array of indices
        where point at index pairing[0] in set A paired with point
        in set B at index pairing[1]
        unparedA, unpairedB: remaining poitn in set A and set B unpaired
    """
    # * Euclidean distance as the cost matrix
    pair_distance = scipy.spatial.distance.cdist(setA, setB, metric='euclidean')
    # * Munkres pairing with scipy library
    # the algorithm return (row indices, matched column indices)
    # if there is multiple same cost in a row, index of first occurence
    # is return, thus the unique pairing is ensured
    indicesA, paired_indicesB = linear_sum_assignment(pair_distance)

    # extract the paired cost and remove instances
    # outside of designated radius
    pair_cost = pair_distance[indicesA, paired_indicesB]
    pairedA = indicesA[pair_cost <= radius]
    pairedB = paired_indicesB[pair_cost <= radius]

    pairing = np.concatenate([pairedA[:, None], pairedB[:, None]], axis=-1)
    unpairedA = np.delete(np.arange(setA.shape[0]), pairedA)
    unpairedB = np.delete(np.arange(setB.shape[0]), pairedB)
    return pairing, unpairedA, unpairedB

def Emetrics(mask, preds):
    '''
    :param mask:
    :param preds:  preds are probability maps !
    :return:
    '''
    if mask is None and preds is None:
        mask = np.array(Image.open('./tempt/0prob.png'))   # if parameter is empty, choose a random image for below part !
        preds = np.array(Image.open('./tempt/0seg.png'))
        if np.max(preds) >1:
            preds = preds / 255.
        if np.max(mask) >1:
            mask = mask / 255.
    else:
        mask = np.squeeze(mask, 0)
        preds = np.squeeze(np.squeeze(preds, 3), 0)
    mask = mask.reshape((mask.shape[0], mask.shape[1]))      # (H, W)
    preds = preds.reshape((preds.shape[0], preds.shape[1]))  # (H, W)
    # print(np.max(mask), np.max(preds), mask.shape, preds.shape)
    mask[np.where(mask > 0.5)] = 1
    preds[np.where(preds > 0.5)] = 1
    AJI = get_fast_aji(true=mask, pred=preds)
    dice1 = dice_1(pred=preds, true=mask)
    mds = hausdorff(pred=preds, true=mask)
    ACC, SEN, SPE, Precision, Recall, F1_Score = matrix_metrics(preds, mask)
    print('************************ Image Segmentation Evaluation ************************')
    print(' ACC : {} \n Sen : {} \n Spe : {} \n F1-score(dice) : {} \n Precision : {} \n Recall : {}'
          .format(ACC, SEN, SPE, Precision, Recall, F1_Score))
    print(' AJI : {}\n Hausdorff distance : {}'.format(AJI, mds))
    print(' Dice1 is : {}'.format(dice1))
    print('******************************************************************************')
    return ACC, SEN, SPE, Precision, Recall, F1_Score, AJI, mds, dice1

if __name__ == '__main__':
    Emetrics(None, None)
    pass
