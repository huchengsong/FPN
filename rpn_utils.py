import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from box_parametrize import box_parameterize_gpu
from bbox_IoU import bbox_IoU_gpu

from configure import Config


def generate_training_anchors(roi, gt_bbox, gt_label, num_sample=Config.roi_num_sample,
                              pos_ratio=Config.roi_pos_ratio, pos_iou_thresh=Config.roi_pos_iou_thresh,
                              neg_iou_thresh_hi=Config.roi_neg_iou_thresh_hi,
                              neg_iou_thresh_lo=Config.roi_neg_iou_thresh_lo,
                              loc_mean=Config.loc_normalize_mean,
                              loc_std=Config.loc_normalize_std):
    """
    generate ground truth label and location for sampled proposals
    :param roi: (N, 4) pytorch tensor; region of interest from rpn proposal
    :param gt_bbox: (K, 4) ndarray; ground truth bounding box an image
    :param gt_label: (K, ) ndarray; ground truth label for each bounding box; range[1, class_number]
    :param num_sample: number of sampled roi
    :param pos_ratio: ratio of positive samples in the output
    :param pos_iou_thresh: positive iou threshold
    :param neg_iou_thresh_hi: negative iou threshold low end
    :param neg_iou_thresh_lo: negative iou threshold high end
    :param loc_mean: list (4, )
    :param loc_std: list (4, )
    :return: pytorch tensor: sampled_roi, gt_loc, gt_label
    """
    gt_bbox = torch.from_numpy(gt_bbox).cuda()
    gt_label = torch.from_numpy(gt_label).long().cuda()

    # add gt_box to roi
    roi = torch.cat((roi, gt_bbox), 0)

    iou_matrix = bbox_IoU_gpu(gt_bbox, roi)
    max_iou, roi_gt_assignment = iou_matrix.max(dim=0)

    gt_roi_label = gt_label[roi_gt_assignment]

    # sample positive roi and get roi index
    max_num_pos_roi = int(pos_ratio * num_sample)
    pos_index = torch.nonzero(max_iou >= pos_iou_thresh).squeeze_()
    num_pos_roi = int(min(max_num_pos_roi, len(pos_index)))
    if num_pos_roi > 0:
        pos_index = pos_index[torch.randperm(len(pos_index))[:num_pos_roi].cuda()]

    # sample negative roi and get roi index
    neg_index = torch.nonzero((max_iou < neg_iou_thresh_hi) &
                              (max_iou >= neg_iou_thresh_lo)).squeeze_()
    max_num_neg_roi = num_sample - num_pos_roi
    num_neg_roi = int(min(max_num_neg_roi, len(neg_index)))
    if num_neg_roi > 0:
        neg_index = neg_index[torch.randperm(len(neg_index))[:num_neg_roi].cuda()]

    # get sampled rois and their labels
    keep_index = torch.cat((pos_index, neg_index))
    gt_roi_label = gt_roi_label[keep_index]
    gt_roi_label[num_pos_roi:] = 0
    sampled_roi = roi[keep_index]

    # get parameterized roi
    gt_roi_loc = box_parameterize_gpu(gt_bbox[roi_gt_assignment[keep_index]], sampled_roi)

    # normalize
    loc_mean = torch.cuda.FloatTensor(loc_mean)
    loc_std = torch.cuda.FloatTensor(loc_std)
    gt_roi_loc = (gt_roi_loc - loc_mean) / loc_std

    return sampled_roi, gt_roi_loc, gt_roi_label


def generate_anchor_loc_label(anchor, gt_bbox, img_size,
                              num_sample=Config.rpn_num_sample, pos_iou_thresh=Config.rpn_pos_iou_thresh,
                              neg_iou_thresh=Config.rpn_neg_iou_thresh, pos_ratio=Config.rpn_pos_ratio):
    """
    generate ground truth loc and label for anchors
    :param anchor: (N, 4) pytorch tensor
    :param gt_bbox: (K, 4) ndrray
    :param img_size: image size [H, W]
    :param num_sample: number of output samples
    :param pos_iou_thresh: threshold for positive anchors
    :param neg_iou_thresh: threshold for negative anchors
    :param pos_ratio: ratio of positive samples in output anchors
    :return: pytorch tensor: anchor_labels (N, ), anchor_gt_parameterized (N, 4)
    """
    num_anchors = anchor.size()[0]
    gt_bbox = torch.from_numpy(gt_bbox).cuda()

    # # cross-boundary anchors will not be used
    # ind_inside_img = torch.nonzero(
    #     (anchor[:, 0] >= 0) &
    #     (anchor[:, 1] >= 0) &
    #     (anchor[:, 2] <= img_size[0]) &  # height
    #     (anchor[:, 3] <= img_size[1])  # width
    # ).squeeze_()
    # selected_anchors = anchor[ind_inside_img, :]
    # labels = torch.cuda.LongTensor(len(ind_inside_img)).fill_(-1)

    # TODO: test clipping anchors
    anchor[:, 0].clamp_(0, img_size[0])
    anchor[:, 2].clamp_(0, img_size[0])
    anchor[:, 1].clamp_(0, img_size[1])
    anchor[:, 3].clamp_(0, img_size[1])
    selected_anchors = anchor
    labels = torch.cuda.LongTensor(selected_anchors.size(0)).fill_(-1)
    # TODO: the above is for testing

    iou_matrix = bbox_IoU_gpu(gt_bbox, selected_anchors)
    max_iou_each_anchor, ind_max_each_anchor = iou_matrix.max(dim=0)

    # set 1 to positive anchors
    labels[max_iou_each_anchor >= pos_iou_thresh] = 1
    # set 0 to negative anchors
    labels[max_iou_each_anchor <= neg_iou_thresh] = 0
    # set 1 to the anchors that have the highest IoU with a certain ground-truth box
    max_iou_with_gt, _ = iou_matrix.max(dim=1)
    ind_max_iou_with_gt = torch.nonzero(iou_matrix.t() == max_iou_with_gt)[:, 0]
    labels[ind_max_iou_with_gt] = 1

    # if positive anchors are too many, reduce the positive anchor number
    num_pos_sample = int(pos_ratio * num_sample)
    ind_positive_anchor = torch.nonzero(labels == 1).squeeze_()
    num_positive_anchor = len(ind_positive_anchor)
    if num_positive_anchor > num_pos_sample:
        disable_inds = \
            ind_positive_anchor[torch.randperm(num_positive_anchor)[:num_positive_anchor - num_pos_sample].cuda()]
        labels[disable_inds] = -1

    # if negative anchors are too many, reduce the negative anchor number
    # if positive anchors are not enough, pad with negative anchors
    num_neg_sample = num_sample - len(torch.nonzero(labels == 1).squeeze_())
    ind_negative_anchor = torch.nonzero(labels == 0).squeeze_()
    num_negative_anchor = len(ind_negative_anchor)
    if num_negative_anchor > num_neg_sample:
        disable_inds = \
            ind_negative_anchor[torch.randperm(num_negative_anchor)[:num_negative_anchor - num_neg_sample].cuda()]
        labels[disable_inds] = -1

    gt_box_parameterized = box_parameterize_gpu(
        gt_bbox[ind_max_each_anchor, :], selected_anchors)

    # anchor_labels = _unmap(labels, num_anchors, ind_inside_img, fill=-1)
    # anchor_gt_parameterized = _unmap(gt_box_parameterized, num_anchors, ind_inside_img, fill=0)

    # TODO: this is for testing
    anchor_labels = labels
    anchor_gt_parameterized = gt_box_parameterized
    # TODO: above is for testing
    return anchor_labels, anchor_gt_parameterized


def _unmap(data, num_original_data, index_of_data, fill=-1):
    """
    :param data: torch tensor, data to be unmaped to original size
    :param num_original_data: original_matrix.shape[0]
    :param index_of_data: index of data in original matrix
    :param fill: number to be filled in unmaped matrix
    :return: torch tensor, an unmaped matrix
    """
    if len(list(data.size())) == 1:
        ret = torch.cuda.LongTensor(num_original_data)
        ret.fill_(fill)
        ret[index_of_data] = data
    else:
        ret_shape = list(data.size())
        ret_shape[0] = num_original_data
        ret = torch.cuda.FloatTensor(ret_shape[0], ret_shape[1])
        ret.fill_(fill)
        ret[index_of_data, 0:4] = data
    return ret


def rpn_loss(rpn_score, rpn_loc, gt_rpn_loc, gt_rpn_label, rpn_sigma):
    """
    return rpn loss
    :param rpn_score: (N, num_class), pytroch Variable
    :param rpn_loc: (N, 4), pytroch Variable
    :param gt_rpn_loc: (N, 4), pytroch tensor
    :param gt_rpn_label: (N,), pytroch tensor, range(0, num_class)
    :param rpn_sigma: sigma for  smooth l1 loss
    :return: cls_loss, loc_loss
    """
    gt_rpn_loc = Variable(gt_rpn_loc)
    gt_rpn_label = Variable(gt_rpn_label)

    # rpn loc loss
    mask = Variable(torch.cuda.FloatTensor(gt_rpn_loc.size()).fill_(0))
    mask[(gt_rpn_label > 0).view(-1, 1).expand_as(mask)] = 1
    loc_loss = _smooth_l1_loss(rpn_loc, gt_rpn_loc, mask, rpn_sigma)

    # normalize by the number of positive and negative rois
    loc_loss = loc_loss / (gt_rpn_label >= 0).float().sum()

    # rpn cls loss
    # nn.CrossEntropy includes LogSoftMax and NLLLoss in one single function
    cls_loss = nn.CrossEntropyLoss(ignore_index=-1)(rpn_score, gt_rpn_label)

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
