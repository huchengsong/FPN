import torch
import torch.nn as nn
from torch.autograd import Variable


def fast_rcnn_loss(roi_score, roi_cls_loc, gt_roi_loc, gt_roi_label, roi_sigma):
    """
    calculate fast rcnn class and regression loss
    :param roi_score: (N, num_classes), pytorch cuda Variable
    :param roi_cls_loc: (N, 4 * num_classes), pytorch cuda Variable
    :param gt_roi_loc: (N, 4), pytorch tensor
    :param gt_roi_label: (N, ), pytorch tensor
    :param roi_sigma: sigma for smooth l1 loss
    :return: pytorch Variable: cls_loss, loc_loss
    """

    num_roi = roi_cls_loc.size()[0]
    roi_cls_loc = roi_cls_loc.view(num_roi, -1, 4)
    roi_loc = roi_cls_loc[torch.arange(0, num_roi).long().cuda(), gt_roi_label]

    # regression loss
    gt_roi_loc = Variable(gt_roi_loc)
    gt_roi_label = Variable(gt_roi_label)
    pos_mask = Variable(torch.cuda.FloatTensor(num_roi, 4).fill_(0))
    pos_mask[(gt_roi_label > 0).view(-1, 1).expand_as(pos_mask)] = 1
    loc_loss = _smooth_l1_loss(roi_loc, gt_roi_loc, pos_mask, roi_sigma)
    loc_loss = loc_loss/(gt_roi_label >= 0).sum().float()

    # class loss
    cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label)

    return cls_loss, loc_loss


def _smooth_l1_loss(x, gt, mask, sigma):
    """
    retrun smooth l1 loss
    :param x: [N, K], troch Variable
    :param gt: [N, K], troch Variable
    :param mask: [N, K], troch Variable
    :param sigma: constant
    :return: loss
    """
    sigma2 = sigma ** 2
    diff = mask * (x - gt)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    flag = Variable(flag)
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    loss = y.sum()
    return loss
