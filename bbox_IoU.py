import numpy as np
import torch
from numba import jit, float32


@jit([float32[:](float32[:], float32[:])])
def bbox_IoU(boxes, query_boxes):
    """
    :param boxes: (N, 4) ndarray
    :param query_boxes: (K, 4) ndarray
    :return: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    IoU = np.zeros((N, K), dtype=np.float32)
    query_box_area = (query_boxes[:, 2] - query_boxes[:, 0] + 1) *\
                     (query_boxes[:, 3] - query_boxes[:, 1] + 1)
    boxes_area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)

    for k in range(K):
        for n in range(N):
            intersection_h = (
                    min(boxes[n, 2], query_boxes[k, 2]) -
                    max(boxes[n, 0], query_boxes[k, 0]) + 1
            )
            if intersection_h > 0:
                intersection_w = (
                        min(boxes[n, 3], query_boxes[k, 3]) -
                        max(boxes[n, 1], query_boxes[k, 1]) + 1
                )
                if intersection_w > 0:
                    intersection_area = intersection_w * intersection_h
                    union_area = (boxes_area[n] + query_box_area[k] - intersection_area)
                    IoU[n, k] = intersection_area / union_area
    return IoU


def bbox_IoU_gpu(boxes, query_boxes):
    """
    :param boxes: (N, 4) pytorch tensor
    :param query_boxes: (K, 4) pytorch tensor
    :return: (N, K) pytroch tensor of overlap between boxes and query_boxes
    """
    N = boxes.size()[0]
    K = query_boxes.size()[0]
    query_box_area = (query_boxes[:, 2] - query_boxes[:, 0] + 1) *\
                     (query_boxes[:, 3] - query_boxes[:, 1] + 1)
    query_box_area = query_box_area.expand(N, K)
    boxes_area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    boxes_area = boxes_area.expand(K, N).t()

    intersection_h_1, _ = torch.min(torch.stack((boxes[:, 2].expand(K, N).t(),
                                                 query_boxes[:, 2].expand(N, K)), dim=0), dim=0)
    intersection_h_2, _ = torch.max(torch.stack((boxes[:, 0].expand(K, N).t(),
                                                 query_boxes[:, 0].expand(N, K)), dim=0), dim=0)
    intersection_h = intersection_h_1 - intersection_h_2 + 1
    intersection_h.clamp_(min=0)

    intersection_w_1, _ = torch.min(torch.stack((boxes[:, 3].expand(K, N).t(),
                                                 query_boxes[:, 3].expand(N, K)), dim=0), dim=0)
    intersection_w_2, _ = torch.max(torch.stack((boxes[:, 1].expand(K, N).t(),
                                                 query_boxes[:, 1].expand(N, K)), dim=0), dim=0)
    intersection_w = intersection_w_1 - intersection_w_2 + 1
    intersection_w.clamp_(min=0)

    intersection_area = intersection_w * intersection_h
    union_area = boxes_area + query_box_area - intersection_area
    IoU = intersection_area / union_area
    return IoU
