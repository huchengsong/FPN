import numpy as np
import torch


def generate_base_anchors(base_size, ratios, scales):
    ratios = np.array(ratios)
    scales = np.array(scales)

    target_size = np.tile(base_size * scales, len(ratios))
    target_h = target_size * np.repeat(np.sqrt(ratios), len(scales))
    target_w = target_size / np.repeat(np.sqrt(ratios), len(scales))

    anchors = np.transpose([-(target_h-1)/2, -(target_w-1)/2, (target_h-1)/2, (target_w-1)/2])
    return anchors


def anchor_proposals(feature_height, feature_width, stride, anchor_base):
    """
    return anchor proposals for an image
    :param feature_height: height of feature map
    :param feature_width: width of feature map
    :param stride: stride on images
    :param anchor_base: (K, 4), ndarray, anchors base at each location
    :return: (feature_height, feature_width, num_anchor_base, 4), pytroch tensor, anchor proposals
    """
    anchor_base = torch.from_numpy(anchor_base).float().cuda()
    num_base = anchor_base.size()[0]
    anchor_x_shift = torch.arange(0, feature_width) * stride + stride / 2
    anchor_x_shift = anchor_x_shift.float().cuda()
    anchor_y_shift = torch.arange(0, feature_height) * stride + stride / 2
    anchor_y_shift = anchor_y_shift.float().cuda()

    anchor_x_shift = anchor_x_shift.expand(feature_height, num_base, -1).permute(0, 2, 1)
    anchor_y_shift = anchor_y_shift.expand(feature_width, num_base, -1).permute(2, 0, 1)
    anchor_centers = torch.stack((anchor_y_shift, anchor_x_shift,
                                  anchor_y_shift, anchor_x_shift), dim=3)
    anchor_base = anchor_base.expand(feature_height, feature_width, -1, -1)
    anchors = anchor_centers + anchor_base
    return anchors


if __name__ == '__main__':
    anchors = generate_base_anchors(16, [0.5, 1., 2.], [8, 16, 32])
    print(anchors)
    print((anchors[:, 3] - anchors[:, 1] + 1) * (anchors[:, 2] - anchors[:, 0] + 1))
    print((anchors[:, 3] - anchors[:, 1] + 1) / (anchors[:, 2] - anchors[:, 0] + 1))
