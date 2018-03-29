from collections import defaultdict
import numpy as np

from bbox_IoU import bbox_IoU
from configure import Config


def calc_map(bboxes, labels, scores, gt_bboxes, gt_labels, gt_difficult, iou_thresh=0.5, use_07_metric=False):
    """
    Calculate average precisions
    :param bboxes: list of N ndarray (K, 4)
    :param labels: list of N ndarray (K, )
    :param scores: list of N ndarray (K, )
    :param gt_bboxes: list of N ndarray (J, 4)
    :param gt_labels: list of N ndarray (J, )
    :param gt_difficult: list of N ndarray (J, )
    :param iou_thresh: threshold
    :param use_07_metric: whether use VOC2007 metric
    :return:
    """

    prec, rec = calc_precision_recall(bboxes, labels, scores, gt_bboxes, gt_labels, gt_difficult, iou_thresh)
    ap = calc_ap(prec, rec, use_07_metric)
    return ap, np.nanmean(ap)


def calc_precision_recall(bboxes, labels, scores, gt_bboxes,
                          gt_labels, gt_difficult, iou_thresh=0.5,
                          num_class=Config.num_class):
    """
    return precision and recall for each class. N: number of images
    :param bboxes: list of N (K, 4) ndarray
    :param labels: list of N (K, ) ndarray
    :param scores: list of N  (K, ) ndarray
    :param gt_bboxes: list of N (J, 4) ndarray
    :param gt_labels: list of N (J, ) ndarray
    :param gt_difficult: list of N ndarray (J, )
    :param iou_thresh: threshold
    :param num_class: classes
    :return: prec, rec: list (num_class, ), with cumulative recall and precision for each class
    """
    n_pos = defaultdict(int)
    score = defaultdict(list)
    match = defaultdict(list)

    # repeat over N images
    for i in range(len(bboxes)):
        boxes_i = bboxes[i]
        labels_i = labels[i]
        scores_i = scores[i]
        gt_boxes_i = gt_bboxes[i]
        gt_labels_i = gt_labels[i]

        if gt_difficult is None:
            gt_difficult_i = np.zeros(len(gt_labels_i), dtype=bool)
        else:
            gt_difficult_i = gt_difficult[i]

        # go through each class id in prediction and gt labels
        for l in np.unique(np.concatenate((gt_labels_i, labels_i))):
            pred_l = labels_i == l
            pred_box_l = boxes_i[pred_l]
            pred_score_l = scores_i[pred_l]

            # sort from largest score to smallest score
            order = pred_score_l.argsort()[::-1]
            pred_box_l = pred_box_l[order]
            pred_score_l = pred_score_l[order]

            # selected ground truth bounding box for class l
            gt_l = gt_labels_i == l
            gt_box_l = gt_boxes_i[gt_l]
            gt_difficult_l = gt_difficult_i[gt_l]

            # add the number of gt bbox for class l
            n_pos[l] += np.sum(gt_difficult_l == 0)

            # there is no prediction box for class l
            if len(pred_box_l) == 0:
                continue
            # there is no gt box for class l
            if len(gt_box_l) == 0:
                # record the score for class l
                score[l].extend(pred_score_l)
                match[l].extend([0] * len(pred_score_l))
                continue
            # record the score for class l
            score[l].extend(pred_score_l)
            # calculate iou
            iou = bbox_IoU(gt_box_l, pred_box_l)
            # assign index with iou smaller than iou_thresh to -1
            gt_index = iou.argmax(axis=0)
            gt_index[iou.max(axis=0) < iou_thresh] = -1

            select = np.zeros(len(gt_box_l), dtype=np.bool)
            for gt_idx in gt_index:
                if gt_idx >= 0:
                    if gt_difficult_l[gt_idx]:
                        match[l].append(-1)
                    else:
                        if not select[gt_idx]:
                            match[l].append(1)
                        else:
                            match[l].append(0)
                        # if two bounding box are predicted
                        # for the same gt box, the second one
                        # will be assigned as 0
                        select[gt_idx] = True
                else:
                    match[l].append(0)

    prec = [None] * num_class
    rec = [None] * num_class
    for l in n_pos.keys():
        score_l = np.array(score[l])
        match_l = np.array(match[l], dtype=np.int8)

        order = score_l.argsort()[::-1]
        match_l = match_l[order]

        # return the cumulative sum of true positive and false positive
        tp = np.cumsum(match_l == 1)
        fp = np.cumsum(match_l == 0)

        prec[l] = tp / (fp + tp)

        if n_pos[l] > 0:
            rec[l] = tp / n_pos[l]

    return prec, rec


def calc_ap(prec, rec, use_07_metric):
    num_classes = len(prec)
    ap = np.empty(num_classes)
    for i in range(num_classes):
        if prec[i] is None or rec[i] is None:
            ap[i] = np.nan
            continue

        if use_07_metric:
            # 11 point metric
            ap[i] = 0
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec[i] >= t) == 0:
                    p = 0
                else:
                    p = np.max(np.nan_to_num(prec[i])[rec[i] >= t])
                ap[i] += p / 11
        else:
            mean_prec = np.concatenate(([[0], prec[i], [0]]))
            mean_rec = np.concatenate(([[0], rec[i], [1]]))
            mean_prec = np.maximum.accumulate(mean_prec[::-1])[::-1]
            index = np.where(mean_rec[1:] != mean_rec[:-1])[0]
            ap[i] = np.sum((mean_rec[index + 1] - mean_rec[index]) * mean_prec[index + 1])
    return ap