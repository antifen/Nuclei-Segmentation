import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from  BorderWeight import my_border_weight
import warnings
warnings.filterwarnings('ignore')

'''
files to devise different loss strategy !
Reference：
（1）    https://www.aiuai.cn/aifarm1330.html         
         https://blog.csdn.net/just_sort/article/details/104028997
（2）    https://blog.csdn.net/qq_36584673/article/details/117128726  Focal loss
（3）    https://blog.csdn.net/good18Levin/article/details/119656374  GDL loss
'''

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.epsilon = 1e-5

    def forward(self, predict, target):
        assert predict.size() == target.size(), "the size of predict and target must be equal."
        num = predict.size(0)
        pre = predict.view(num, -1)
        tar = target.view(num, -1)
        intersection = (pre * tar).sum(-1).sum()      # multiply flags and labels
        union = (pre + tar).sum(-1).sum()
        score = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)
        return score

class FocalLoss(nn.Module):
    def __init__(self, alpha = 0.5, gamma = 2, logits = False, reduce = True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)    # input sigmoid numbers !
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=2, size_average=True):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重. 当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.255
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """
        super(focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes  # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            # print(" --- Focal_loss alpha = {}, 将对每一类权重进行精细化赋值 --- ".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1  # 如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            # print(" --- Focal_loss alpha = {} ,将对背景类进行衰减,请在目标检测任务中使用 --- ".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)  # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]

        self.gamma = gamma

    def forward(self, preds, labels):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]        [B*N个标签(假设框中有目标)]，[B个标签]
        :return:
        """
        preds = preds.view(-1, preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_logsoft = F.log_softmax(preds, dim=1)
        preds_softmax = torch.exp(preds_logsoft)

        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        self.alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma), preds_logsoft)
        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

class weighted_entropy(nn.Module):
    '''
    pred  : N, C
    label : N, -1
    '''
    def __init__(self, need_soft_max = True, ch=1):
        super(weighted_entropy, self).__init__()
        self.need_soft_max = need_soft_max
        self.ch = ch
        pass

    def forward(self, pred, label):
        if self.ch ==2:
            if self.need_soft_max is True:
                preds = F.softmax(pred, dim=1)
            else:
                preds = pred
            epusi  = 1e-16
            counts = torch.rand(size=(2,))
            counts[0] = label[torch.where(label == 0)].size(0)
            counts[1] = label[torch.where(label == 1)].size(0)
            N = label.size()[0]
            weights = counts[1]
            weights_avg = 1 - weights / N
            loss = weights_avg * torch.log(preds[:,1] + epusi) + (1 - weights_avg) * torch.log(1 - preds[:,1] + epusi)
            loss = - torch.mean(loss)
            return loss
        else:
            epusi = 1e-16
            alpha_cp = 6.2
            counts = torch.rand(size=(2,))
            counts[0] = label[torch.where(label < 0.5)].size(0)
            counts[1] = label[torch.where(label >= 0.5)].size(0)
            N = label.size()[0]
            weights = counts[1]
            weights_avg = 1 - weights / N
            loss = weights_avg * torch.log(pred[:,] + epusi) * label[:,] + \
                   (1 - weights_avg) * torch.log(1 - pred[:,] + epusi) * (1 - label[:,]) * alpha_cp
            loss = - torch.mean(loss)
            return loss

class new_weighted_entropy(nn.Module):
    def __init__(self, ch=1):
        super(new_weighted_entropy, self).__init__()
        self.ch = ch

    def forward(self, pred, label, lebna1=0.1, lebna2 =0.9, alpha = 2.5):
        if self.ch ==2:
            epusi = 1e-16                      # avoid divide by 0
            weights = torch.rand(size=(2,))    # class weights
            weights[0]  = label[torch.where(label == 0)].size(0)  # n samples
            weights[1]  = label[torch.where(label == 1)].size(0)  # p samples
            negative_new = pred[:, 0]
            negative_new = negative_new[torch.where(negative_new >= lebna1)]  # effective negative samples
            positive_new = pred[:, 1]
            positive_new = positive_new[torch.where(label <= lebna2)]         # effective positive samples
            N = label.size()[0]
            beta = 1 - weights[1] / N
            p_r = -beta * torch.mean(torch.log(positive_new + epusi))
            n_r = -alpha*(1-beta) * torch.mean(torch.log(1- negative_new +epusi))
            loss = p_r + n_r
            return loss
        else:                                  # (-1, ) (-1, )
            epusi = 1e-16                      # avoid divide by 0
            threshold = 0.5
            weights = torch.rand(size=(2,))    # class weights
            weights[0]  = label[torch.where(label < 0.5)].size(0)   # n samples
            weights[1]  = label[torch.where(label >= 0.5)].size(0)  # p samples


            new_pred = torch.unsqueeze(pred.reshape((-1,)), dim=0)
            new_label = torch.unsqueeze(label.reshape((-1,)), dim=0)
            maps = torch.cat([new_pred, new_label], dim=0)
            label_p = maps[1,torch.where(maps[0,:,] > threshold)[0]]
            label_n = maps[1, torch.where(maps[0, :, ] <= threshold)[0]]
            negative_new = pred[torch.where(pred <= threshold)]
            positive_new = pred[torch.where(pred > threshold)]

            new_pred_p = torch.unsqueeze(positive_new.reshape((-1,)), dim=0)
            new_label_p = torch.unsqueeze(label_p.reshape((-1,)), dim=0)
            maps_p = torch.cat([new_pred_p, new_label_p], dim=0)
            new_label_p = maps_p[1,torch.where(maps_p[0,:,] <= lebna2)[0]]

            new_pred_n = torch.unsqueeze(negative_new.reshape((-1,)), dim=0)
            new_label_n = torch.unsqueeze(label_n.reshape((-1,)), dim=0)
            maps_n = torch.cat([new_pred_n, new_label_n], dim=0)
            new_label_n = maps_n[1,torch.where(maps_n[0,:,] >= lebna1)[0]]

            negative_new = pred[torch.where(pred<=threshold)]
            negative_new_1 = negative_new[torch.where(negative_new >= lebna1)]  # effective negative samples
            positive_new = pred[torch.where(pred>threshold)]
            positive_new_1 = positive_new[torch.where(positive_new <= lebna2)]  # effective positive samples


            N = label.size()[0]
            beta = weights[1] / N
            # p_r = -beta * (torch.log(positive_new_1 + epusi) * new_label_p + torch.log(1 - positive_new_1 + epusi) * (1 - new_label_p))
            # n_r = -alpha* (1-beta) * (torch.log(1 - negative_new_1 + epusi) * (1 - new_label_n) + torch.log(negative_new_1 + epusi) * new_label_n)
            p_r = -beta * (torch.log(positive_new_1 + epusi) * new_label_p)
            n_r = -alpha* (1-beta) * (torch.log(1 - negative_new_1 + epusi) * (1 - new_label_n))

            loss = torch.mean(p_r) + torch.mean(n_r)
            return loss

class GDL_loss(nn.Module):
    def __init__(self, classes = 2):
        super(GDL_loss, self).__init__()
        self.classes = classes
        self.epusi = 1e-16

    def forward(self,preds, masks):
        # classes = 5
        # outs = preds.permute((0, 2, 3, 1))         # N H W C
        # outs = outs.reshape((-1, outs.size()[3]))  # N*H*W, C
        # masks = masks.reshape((-1,))  # N,1,H,W ===> N,-1

        outs = torch.softmax(preds, dim=1)
        classes = self.classes
        weights = [0] * classes
        epusi = self.epusi

        for j in range(0, classes):
            weights[j] = masks[torch.where(masks == j)].size(0) / masks.reshape((-1,)).size()[0]  # n samples
        outs = torch.cat([outs, masks.reshape((-1, 1))], dim=1)

        fenzi = torch.tensor(data=[0.], device=preds.device)
        fenmu = torch.tensor(data=[0.], device=preds.device)
        cross_entropy = torch.tensor(data=[0.],device=preds.device)

        for i in range(0, classes):
            tempt = outs[torch.where(outs[:, classes] == i)][:, i]
            if tempt.size()[0] > 0:
                fenzi += torch.sum(tempt / weights[i])
                # fenmu += torch.sum((outs[:, i] + weights[i] / 2) / weights[i])
                fenmu += torch.sum(outs[:, i]) + \
                         torch.tensor(data=[tempt.size()[0]] ,device=preds.device) * masks[torch.where(masks == i)].size(0)
                cross_entropy -= torch.sum(torch.log(tempt) / weights[i])

        loss_g = 1 - 2 * fenzi / (fenmu + epusi)
        loss_ce = cross_entropy / masks.reshape((-1,)).size()[0]
        print(loss_g, loss_ce)
        return loss_ce

def loss_ce_ds(preds, masks, criterion, selected_mode = 2):
    criterion_v = criterion
    alpha = 0.1
    l1 = criterion_v(preds[0], masks) + alpha * DiceLoss()(preds[0], masks)  # ce joint dice !
    l2 = criterion_v(preds[1], masks) + alpha * DiceLoss()(preds[1], masks)  # ce joint dice !
    l3 = criterion_v(preds[2], masks) + alpha * DiceLoss()(preds[2], masks)  # ce joint dice !
    l4 = criterion_v(preds[3], masks) + alpha * DiceLoss()(preds[3], masks)  # ce joint dice !
    l5 = criterion_v(preds[4], masks) + alpha * DiceLoss()(preds[4], masks)  # ce joint dice !
    loss = 0.2/3.0 * l1 + 0.4/3.0  * l2  + 0.6/3.0 * l3  + 0.8/3.0  * l4 + 1.0/3.0 * l5
    return loss

def loss_ce_ds_single(preds, masks, criterion):
    alpha = 0.1
    losses = nn.BCELoss()(preds, masks) + alpha * DiceLoss()(preds, masks)  # ce joint dice !
    # print(losses,'--------------')
    return losses

def new_border_loss(pred, mask,boder_weight, device):
    # print(boder_weight.shape, 'loss functions !')
    epusi = 1e-16
    counts = torch.rand(size=(2,))
    counts[0] = mask[torch.where(mask < 0.5)].size(0)
    counts[1] = mask[torch.where(mask >= 0.5)].size(0)
    boder_weight = torch.tensor(boder_weight, device=device)
    loss = boder_weight[:, ] * (torch.log(pred[:, ] + epusi) * mask[:, ] + torch.log(1 - pred[:, ] + epusi) * (1 - mask[:, ]))
    losses = - torch.mean(loss) + DiceLoss()(pred, mask)
    return losses

if __name__ == '__main__':
    labels = torch.tensor([0, 1, 1, 0, 1, 1])
    pred = torch.tensor([[-0.3, 0.7], [0.3, 0.7], [0.3, 0.7], [0.4, 0.6], [0.3, 0.7], [0.3, -0.7]])
    pred2 = torch.tensor([0.3, 0.7, 0.6, 0.2, 0.5, 0.9])
    # print(weighted_entropy(need_soft_max = True)(pred,labels))
    print(DiceLoss()(pred2, labels))
    print(focal_loss()(pred, labels))
    # print(nn.CrossEntropyLoss()(pred, labels))
    print(FocalLoss()(pred2, torch.tensor(labels, dtype=torch.float32, device=labels.device)))
    pass