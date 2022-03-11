import sys

sys.path.insert(0, '../src/')
sys.path.insert(0, '../lib/')
sys.path.insert(0, '../pytorch_myself/')
sys.path.insert(0, '../patches_process')
sys.path.insert(0, '../networks')

import numpy as np
import torch
import torch.utils.data as pydata
import torch.optim as optim
from visdom import Visdom
import time

from extract_patches import extract_random_center, gt_to_color, extract_orders, extract_ordered_overlap, \
    paint_border_overlap, recompose_overlap, recompone

from my_untils import read_all_images, test_big_imgs, rgb2gray, clahe_equalized, dataset_normalized, \
    adjust_gamma, group_images, visualize, size_change, test_big_imgs
from base_utils import segment_line, shape_check, value_range_check, net_predict_reshape4d, \
    tensor4d_channels_last, tensor4d_channels_first
from gt_op import gt_to_class


from GAN_RCS_UNet import generator

from Visualizer import Visualizer

size_h = 1000
size_w = 1000
patch_h = 48 * 1 - 16
patch_w = 48 * 1 - 16
stride_h = 3
stride_w = 3
Nimgs = 10000
total_imgs = 20

path_images = '../MONUCLEI/training/image/'
path_gt  = '../MONUCLEI/training/gt/'
path_masks  ='../MONUCLEI/training/mask/'


path_big_group = '../imgss/group1'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

total_epoch = 200
lr_epoch = np.array([150, total_epoch])
lr_value = np.array([0.001, 0.0001])
lr_schedule = np.zeros(total_epoch)
for l in range(len(lr_epoch)):
    if l == 0:
        lr_schedule[0:lr_epoch[l]] = lr_value[l]
    else:
        lr_schedule[lr_epoch[l - 1]:lr_epoch[l]] = lr_value[l]


class PatchDataset(pydata.Dataset):
    def __init__(self, imgs_arr, imgs_label):
        self.imgs_arr = torch.from_numpy(imgs_arr).float()
        self.imgs_label = torch.from_numpy(imgs_label).float()

    def __len__(self):
        return self.imgs_arr.size()[0]

    def __getitem__(self, index):
        self.data_img = self.imgs_arr[index]
        self.data_label = self.imgs_label[index]
        return self.data_img, self.data_label


def my_training_pre():
    all_images = np.empty(shape=(total_imgs, size_h, size_w, 3))
    all_gts = np.empty(shape=(total_imgs, size_h, size_w, 1))
    all_masks = np.empty(shape=(total_imgs, size_h, size_w, 1))
    all_images, call_images = read_all_images(path_images, all_images)
    all_gts, call_gts = read_all_images(path_gt, all_gts)
    all_masks, call_masks = read_all_images(path_masks, all_masks)
    test_big_imgs(all_images, all_gts, all_masks, path_big_group)   # N H W C   and    N C H W

    grey_all_images = rgb2gray(call_images)
    grey_all_images = clahe_equalized(grey_all_images)
    grey_all_images = dataset_normalized(grey_all_images)
    grey_all_images = adjust_gamma(grey_all_images, gamma=0.8)

    # ---enter into patches extract,  (N 1 H W ) and  (N 3 H W)
    patches_train, patches_gt = extract_random_center(grey_all_images, call_gts, Nimgs, patch_h, patch_w)

    re_gt = gt_to_class(patches_gt)
    re_gt = np.reshape(re_gt, (re_gt.shape[0], re_gt.shape[1], patch_h * patch_w))
    re_gt = np.transpose(re_gt, (0, 2, 1))

    # random sort to change suquence !
    index = np.arange(patches_train.shape[0])
    np.random.shuffle(index)

    patches_train = patches_train[index, :, :, :]
    re_gt = re_gt[index, :, :]
    patches_gt = patches_gt[index, :, :]

    train_visualization(patches_train, patches_gt, per_row=5)
    # before dataset method  (B,3,H,W) (B,H*W,2)
    print('check size before patch data method {} and {}'.format(patches_train.shape, re_gt.shape))
    print('is nan check:', np.any(np.isnan(patches_train)), np.any(np.isnan(re_gt)))
    datasets = PatchDataset(patches_train, re_gt)
    train_loader = pydata.DataLoader(datasets, batch_size=2 * 8, shuffle=True, num_workers=0, pin_memory=True)
    return train_loader


def train_visualization(patches_train, patches_gt, per_row=5, name='sample'):
    N_sample = min(patches_train.shape[0], 40)
    visualize(group_images(patches_train[0:N_sample, :, :, :], per_row),
              '../' + 'imgss/' + '' + name + "_input_imgs")
    visualize(group_images(patches_gt[0:N_sample, :, :, :], per_row),
              '../' + 'imgss/' + '' + name + "_input_masks")


def test_train_loader(trainloader):
    running_loss = 0
    #     net = LadderNetv8(inplanes=1)
    net = generator()
    # net = torch.load('last_weights.pkl')
    # net.load_state_dict(torch.load('epoches280weights.pkl'))
    net = net.to(device)
    net.train()
    optimizer = optim.Adam(net.parameters(), lr=lr_value[0], )
    criterion = torch.nn.CrossEntropyLoss()
    #     criterion = LossMulti()
    #     criterion = weighted_entropy()
    corrects = 0.
    print('=====all data has been loaded !====')
    net.train(mode=True)
    v_loss = Visdom(env='loss1')
    viz = Visualizer(env='NAME')
    aix_x = 0

    for epoch in range(0, total_epoch):
        lr = lr_schedule[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print("Learning rate = %4f\n" % lr)

        for index, data in enumerate(trainloader):
            inputs, lables = data
            inputs = inputs.to(device)
            lables = lables.to(device)

            #             print('shape of inputs is {} and lables shape is {} '.format(inputs.size(), lables.size()))
            #             print('shape of inputs is {} and lables shape is {} '.format(torch.max(inputs), torch.max(lables)))
            #             print('shape of inputs is {} and lables shape is {} '.format(inputs, lables))
            #             if index%11 == 3:
            #                 data1 = inputs.permute((0,2,3,1)).numpy()
            #                 data2 = torch.argmax(lables,dim=2).reshape((-1,patch_h,patch_w,1)).numpy()
            #                 train_visualization(data1, data2, per_row=4, name = str(index)+'dataloader_img')

            optimizer.zero_grad()
            lables = torch.argmax(lables, dim=2)
            lables = torch.reshape(lables, (lables.size()[0] * lables.size()[1],))
            outs = net(inputs)
            # print('shape of outs is {} and lables shape is {} '.format(outs.size(), lables.size()))
            #             print(outs)
            loss = criterion(outs, lables)
            #             print(loss,'-----------------------------')
            loss.backward()
            running_loss += loss
            predicted = torch.argmax(outs, dim=1)
            new_preds = predicted.reshape((-1, 1, patch_h, patch_w))
            new_lables = lables.reshape((-1, 1, patch_h, patch_w))
            new_inputs = inputs.reshape((-1, 1, patch_h, patch_w))
            #             print('shape of predicted is {} and lables shape is {},{} '.format(predicted.size(), lables.size(),new_preds.size()))
            results_pred = group_images(new_preds.cpu().numpy(), 4)
            results_lables = group_images(new_lables.cpu().numpy(), 4)
            results_inputs = group_images(new_inputs.cpu().numpy(), 4)
            #             print('results_pred:',results_pred.shape)

            #             viz.img(name='labels', img_=mask[0, :, :, :])
            #             viz.img(name='prediction', img_=pred[0, :, :, :])

            corrects += (predicted == lables).sum().item()
            optimizer.step()
            # print(index)
            if index % 100 == 0 and index != 0:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f acc:%.3f' %
                      (epoch + 1, index / 100, running_loss / (1) / 100, corrects / (lables.size(0)) / (100)))

                time.sleep(1)
                v_loss.line([float(running_loss) / 100], [aix_x], win='training loss', update='append')
                viz.img(name='preds', img_=np.uint8(results_pred.transpose((2, 0, 1)) * 255))
                viz.img(name='lables', img_=np.uint8(results_lables.transpose((2, 0, 1)) * 255))
                viz.img(name='images', img_=np.uint8(results_inputs.transpose((2, 0, 1)) * 255))

                aix_x = aix_x + 1
                corrects = 0.0
                running_loss = 0.0

        corrects = 0.0
        running_loss = 0.0
        if epoch % 20 == 0 and epoch != 0:
            save_path = 'epoches' + str(epoch) + 'weights.pkl'
            torch.save(net.state_dict(), save_path)
    # save last models and weights !
    save_path = 'last_weights.pkl'
    torch.save(net.state_dict(), save_path)


def test_inference(trainloader, save_path):
    print('========================= begain inference phase ====================')
    net_load = generator()
    net_load.load_state_dict(torch.load(save_path))
    net_load.eval()  # use model test module !

    totals, corrects = 0, 0
    with torch.no_grad():
        for index, data in enumerate(trainloader):
            images, labels = data
            lables = torch.argmax(labels, dim=2)
            lables = torch.reshape(lables, (lables.size()[0] * lables.size()[1],))
            outputs = net_load(images)
            print('check output_size : =====>', outputs.size())  # 16 32 32
            _, predicted = torch.max(outputs.data, 1)
            totals += lables.size(0)
            corrects += (predicted == lables).sum().item()
            print('Accuracy of the network on the {} test images: {} %'.format(total_imgs, 100 * corrects / totals))
            pass
    print('---------- have finished test phase process !------------')
    pass


def test_test(save_path):
    all_images = np.empty(shape=(total_imgs, size_h, size_w, 3))
    all_gts = np.empty(shape=(total_imgs, size_h, size_w, 1))
    all_masks = np.empty(shape=(total_imgs, size_h, size_w, 1))
    all_images, call_images = read_all_images(path_images, all_images)
    all_gts, call_gts = read_all_images(path_gt, all_gts)
    all_masks, call_masks = read_all_images(path_masks, all_masks)
    test_big_imgs(all_images, all_gts, all_masks, path_big_group)
    grey_all_images = rgb2gray(call_images)
    grey_all_images = clahe_equalized(grey_all_images)
    grey_all_images = dataset_normalized(grey_all_images)
    grey_all_images = adjust_gamma(grey_all_images, gamma=1.0)

    test_imgs1 = paint_border_overlap(grey_all_images, patch_h, patch_w, stride_h, stride_w, 1)
    test_imgs = extract_ordered_overlap(test_imgs1, patch_h, patch_w, stride_h, stride_w, 1)

    test_lables = paint_border_overlap(call_gts, patch_h, patch_w, stride_h, stride_w, 1)
    test_lables = extract_ordered_overlap(test_lables, patch_h, patch_w, stride_h, stride_w, 1)

    print(test_imgs.shape, test_lables.shape)
    print(np.max(test_imgs), np.max(test_lables))
    test_lables = gt_to_class(test_lables)
    test_lables = np.reshape(test_lables, (test_lables.shape[0], test_lables.shape[1], patch_h * patch_w))
    test_lables = np.transpose(test_lables, (0, 2, 1))

    datasets = PatchDataset(test_imgs, test_lables)
    batch_size = 64 * 2
    test_loader = pydata.DataLoader(datasets, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print('all patches size is {} and all test labels are:{} '.format(test_imgs.shape, test_lables.shape))
    print('========================= begain inference phase ====================')
    net_load = generator()
    net_load.load_state_dict(torch.load(save_path, map_location='cpu'))
    net_load = net_load.to(device)
    net_load.eval()  # use model test eval() !

    totals, corrects = 0, 0
    big_tensor = torch.ones(size=(batch_size * patch_w * patch_h, 2))  # this needs to be changed !
    big_tensor = big_tensor.to(device)
    with torch.no_grad():
        for index, data in enumerate(test_loader, 0):
            images, labels = data
            # print('shape of inputs is {} and lables shape is {} '.format(images.size(), labels.size()))
            images = images.to(device)
            labels = labels.to(device)

            lables = torch.argmax(labels, dim=2)
            lables = torch.reshape(lables, (lables.size()[0] * lables.size()[1],))
            outputs = net_load(images)
            # print('check output_size value range : {}=====> to {}=====> '.format(torch.min(outputs),torch.max(outputs)))  # 16 32 32
            big_tensor = torch.cat([big_tensor, outputs], dim=0)
            predicted = torch.argmax(outputs, 1)
            # print('check predicted value range : {}=====> to {}=====> '.format(torch.min(predicted), torch.max(predicted)))
            print('index is :', index)
            totals += lables.size(0)
            corrects += (predicted == lables).sum().item()
            print('Accuracy of the network on the {} test images: {} %'.format(total_imgs, 100 * corrects / totals))

    print('---------- have finished test phase process !------------')
    print(big_tensor.size())
    img_tensors_pred = big_tensor.cpu().detach().numpy()
    img_tensors_pred = img_tensors_pred[batch_size * patch_w * patch_h:, :, ]
    print('---now img_tensors_pred size is ', img_tensors_pred.shape)
    # img_tensors_pred = np.reshape(img_tensors_pred,(-1,patch_h*patch_w,2))

    arg_predictions = np.argmax(img_tensors_pred, axis=1)
    arg_predictions = np.reshape(arg_predictions, (-1, 1, patch_h, patch_w))
    print('the shape of arg_predictions is: ', arg_predictions.shape)  # shape is N C
    print('value check of arg_predictions is: ()-()', np.max(arg_predictions), np.min(arg_predictions))
    # here insert prediction evaluate codes ...
    #     evaluate_test(test_lables, arg_predictions,legnth)
    pred_imgs = recompose_overlap(arg_predictions, test_imgs1.shape[2], test_imgs1.shape[3],
                                  stride_h, stride_w, loss_weight=None)
    print('value check of pred_imgs is: ()-()', np.max(pred_imgs), np.min(pred_imgs))
    print('new pred_imgs shape is :', pred_imgs.shape)  #
    pred_imgs = pred_imgs[:, :, 0:size_h, 0:size_w]
    print('full image shape is :', pred_imgs.shape)
    print('value check max value is: ', np.max(pred_imgs))
    for i in range(0, pred_imgs.shape[0]):
        visualize(pred_imgs[i].transpose((1, 2, 0)), '../test2/seg_img' + str(i))
    print('==================== end of predicts and picture save ====================')


def evaluate_test(y_true, y_pred, length):
    print('============ first size check ============')
    y_true = np.reshape(np.argmax(y_true, axis=2), (length, -1))
    y_pred = np.reshape(y_pred, (length, -1))
    print(y_true.shape, y_pred.shape)

    '''
    :return:
    '''
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import jaccard_score

    # acc = accuracy_score(y_true, y_pred)
    # precision = precision_score(y_true, y_pred)
    # recall = recall_score(y_true, y_pred)
    # fscore = f1_score(y_true, y_pred)
    # auc = roc_auc_score(y_true, y_pred)

    # print('acc is {} precision is {} recall is {} f1-score is {} and auc is {}'.format(
    #     acc, precision,recall,fscore, auc
    # ))

    total_acc = 0
    total_spe = 0
    total_sen = 0
    total_pre = 0

    for index in range(0, length):
        confusion = confusion_matrix(y_true[index, :], y_pred[index, :])
        print(confusion)
        accuracy = 0
        if float(np.sum(confusion)) != 0:
            accuracy = float(confusion[0, 0] + confusion[1, 1]) / float(np.sum(confusion))
        print("Global Accuracy: " + str(accuracy))
        total_acc += accuracy
        specificity = 0
        if float(confusion[0, 0] + confusion[0, 1]) != 0:
            specificity = float(confusion[0, 0]) / float(confusion[0, 0] + confusion[0, 1])
        print("Specificity: " + str(specificity))
        total_spe += specificity
        sensitivity = 0
        if float(confusion[1, 1] + confusion[1, 0]) != 0:
            sensitivity = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[1, 0])
        print("Sensitivity: " + str(sensitivity))
        total_sen += sensitivity
        precision = 0
        if float(confusion[1, 1] + confusion[0, 1]) != 0:
            precision = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[0, 1])
        print("Precision: " + str(precision))
        total_pre += precision
        # Jaccard similarity index
        # jaccard_index = jaccard_score(y_true, y_pred)
        # print("Jaccard similarity score: " + str(jaccard_index))
        # # F1 score
        # F1_score = f1_score(y_true, y_pred, labels=None, average='binary', sample_weight=None)
        # print("F1 score (F-measure): " + str(F1_score))
    print('average acc of test set is :', total_acc / length)
    print('average acc of test set is :', total_spe / length)
    print('average acc of test set is :', total_sen / length)
    print('average acc of test set is :', total_pre / length)


path_of_weights = 'epoches60weights.pkl'

train_loaders = my_training_pre()
# test_train_loader(train_loaders)
# test_inference(train_loaders,path_of_weights)
# test_test(path_of_weights)

# y_true, y_pred =torch.tensor([1,0,0,1,1,0,1,1,0,1,1,0]), torch.tensor([0,0,1,1,0,1,0,1,1,1,0,0])
# evaluate_test(y_true, y_pred, )
# print('---------------')
# print(torch.tensor([[1,0,0,1,1,0,1,1,0,1,1,0]]).size())
# print(torch.tensor([[1,0,0,1,1,0,1,1,0,1,1,0]])[0,:,].size())