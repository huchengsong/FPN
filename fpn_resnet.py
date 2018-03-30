import torch
from torch.autograd import Variable
import torch.nn as nn

from fpn import FPN
from resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from generate_base_anchors import generate_base_anchors, anchor_proposals
from configure import Config
from box_parametrize import box_deparameterize_gpu
from non_maximum_suppression import non_maximum_suppression_rpn


def initialize_params(x, mean=0, stddev=0.01):
    x.weight.data.normal_(mean, stddev)
    x.bias.data.zero_()


def create_rpn_proposals(locs, scores, anchors, img_size):
    """
    create rpn proposal based on rpn result
    :param locs: (N, 4), pytorch tensor of RPN prediction
    :param scores: (N, ), pytorch tensor of RPN prediction
    :param anchors: (N, 4), pytorch tensor
    :param img_size: [height, width]
    :return: (K, 4), pytorch tensor, rpn proposals
    """
    nms_thresh = Config.nms_thresh
    num_pre_nms = Config.num_pre_nms
    num_post_nms = Config.num_post_nms
    min_size = Config.min_size
    img_h = img_size[0]
    img_w = img_size[1]

    # loc de-parameterize
    rois = box_deparameterize_gpu(locs, anchors)

    # take top num_pre_nms rois and scores
    _, order = torch.sort(scores, descending=True)
    order = order[:num_pre_nms]
    rois = rois[order, :]
    scores = scores[order]

    # clip bbox to image size
    rois[rois < 0] = 0
    rois[:, 0][rois[:, 0] > img_h] = img_h
    rois[:, 1][rois[:, 1] > img_w] = img_w
    rois[:, 2][rois[:, 2] > img_h] = img_h
    rois[:, 3][rois[:, 3] > img_w] = img_w

    # remove boxes with size smaller than threshold
    height = rois[:, 2] - rois[:, 0]
    width = rois[:, 3] - rois[:, 1]
    keep = torch.nonzero((height >= min_size) & (width >= min_size))[:, 0]
    rois = rois[keep, :].contiguous()
    scores = scores[keep].contiguous()

    # nms
    _, roi_selected = non_maximum_suppression_rpn(rois, nms_thresh, scores, num_post_nms)

    return roi_selected


class ROIAlign(nn.Module):
    def __init__(self, pool_out_size, sub_sample=2):
        super(ROIAlign, self).__init__()
        self.pool_out_size = pool_out_size
        self.sub_sample = sub_sample

    def forward(self, features, rois, img_size):
        """
        bilinear interpolation
        :param features: pytorch Variable (1, C, H, W)
        :param rois: pytorch tensor, (N, 4)
        :param img_size: [H, W]
        :param pool_out_size: size after ROI align, [H_out, W_out]
        :param sub_sample: subsample number along an axis in a bin
        :return: (N, C, H_out, W_out)
        """
        feature_size = list(features.size())
        img_size = torch.cuda.FloatTensor(img_size)
        feature_hw = torch.cuda.FloatTensor(feature_size[2:])
        rois_in_features = rois * (feature_hw.repeat(2) - 1) / (img_size.repeat(2) - 1)
        h_step = ((rois_in_features[:, 2] - rois_in_features[:, 0]) / (self.pool_out_size[0] * self.sub_sample))[:, None]
        w_step = ((rois_in_features[:, 3] - rois_in_features[:, 1]) / (self.pool_out_size[1] * self.sub_sample))[:, None]
        y_shift = torch.arange(0, self.pool_out_size[0] * self.sub_sample).cuda().expand(rois.size(0), -1) * h_step + \
                h_step / 2 + rois_in_features[:, 0][:, None]
        x_shift = torch.arange(0, self.pool_out_size[1] * self.sub_sample).cuda().expand(rois.size(0), -1) * w_step + \
                w_step / 2 + rois_in_features[:, 1][:, None]
        y_shift = y_shift.expand(self.pool_out_size[1] * self.sub_sample, -1, -1).permute(1, 2, 0)
        x_shift = x_shift.expand(self.pool_out_size[0] * self.sub_sample, -1, -1).permute(1, 0, 2)

        centers = torch.stack((y_shift, x_shift), dim=3)
        centers = centers.contiguous().view(-1, 2)  # (N, H, W, 2) -> (N*H*W, 2)

        # bilinear interpolation
        loc_y = Variable(torch.frac(centers[:, 0].expand(feature_size[0], feature_size[1], -1)))
        loc_x = Variable(torch.frac(centers[:, 1].expand(feature_size[0], feature_size[1], -1)))

        ind_left = torch.floor(centers[:, 1]).long()
        ind_right = torch.ceil(centers[:, 1]).long()
        ind_up = torch.floor(centers[:, 0]).long()
        ind_down = torch.ceil(centers[:, 0]).long()

        # print(ind_left, ind_right, ind_up, ind_down)
        pre_pool = features[:, :, ind_up, ind_left] * (1 - loc_y) * (1 - loc_x) + \
                   features[:, :, ind_down, ind_left] * loc_y * (1 - loc_x) + \
                   features[:, :, ind_up, ind_right] * (1 - loc_y) * loc_x + \
                   features[:, :, ind_down, ind_right] * loc_y * loc_x

        pre_pool = pre_pool.view(feature_size[0] * feature_size[1], rois.size()[0],
                                 self.pool_out_size[0] * self.sub_sample,
                                 self.pool_out_size[1] * self.sub_sample).permute(1, 0, 2, 3)
        max_pool = nn.MaxPool2d(kernel_size=self.sub_sample, stride=self.sub_sample, padding=0)
        post_pool = max_pool(pre_pool)

        return post_pool


def pyramid_roi_pooling(feature_list, rois, img_size, pool_out_size):
    """
    roi pooling for each pyramid level
    :param feature_list: list of (1, C, Hn, Wn) pytorch Variable, [p2, p3, p4, p5, p6]
    :param rois: pytorch tensor, (N, 4)
    :param pool_out_size: size after ROI align, [H_out, W_out]
    :param img_size: [H, W]
    :return: pytorch Variable (N, C, H_out, W_out)
    """
    from roi_module import RoIPooling2D

    roi_align = ROIAlign(pool_out_size, sub_sample=2)

    h = rois[:, 2] - rois[:, 0] + 1
    w = rois[:, 3] - rois[:, 1] + 1
    roi_level = torch.log(torch.sqrt(h * w) / 224.0) / 0.693147  # divide by log2
    roi_level = torch.floor(roi_level + 4)
    roi_level[roi_level < 2] = 2
    roi_level[roi_level > 5] = 5

    roi_pool = []
    box_level = []
    for i, l in enumerate(range(2, 6)):
        if torch.sum(roi_level == l) == 0:
            continue
        idx_l = torch.nonzero(roi_level == l).squeeze()
        box_level.append(idx_l)
        pool_l = roi_align(feature_list[i], rois[idx_l], img_size)

        # using roi pooling (old method)
        # # TODO: write code to replace this
        # spatial_scale = feature_list[i].size(2) / img_size[0]
        # roi_pooling = RoIPooling2D(pool_out_size[0], pool_out_size[1], spatial_scale)
        # sub_rois = rois[idx_l]
        # roi_indices = torch.cuda.FloatTensor(sub_rois.size(0)).fill_(0)
        # indices_and_rois = torch.stack([roi_indices, sub_rois[:, 0], sub_rois[:, 1], sub_rois[:, 2], sub_rois[:, 3]],
        #                                dim=1)
        # indices_and_rois = Variable(indices_and_rois[:, [0, 2, 1, 4, 3]]).contiguous()
        # pool_l = roi_pooling(feature_list[i], indices_and_rois)
        roi_pool.append(pool_l)

    roi_pool = torch.cat(roi_pool, 0)
    box_level = torch.cat(box_level, 0)
    _, order = torch.sort(box_level)
    roi_pool = roi_pool[order]

    return roi_pool


class FPNResNet(FPN):
    def __init__(self, num_layers=50, pretrained=True,
                 num_classes=Config.num_class, ratios=Config.ratios,
                 scales=Config.scales, stride=Config.stride):
        if num_layers == 18:
            model = resnet18(pretrained=pretrained)
        elif num_layers == 34:
            model = resnet34(pretrained=pretrained)
        elif num_layers == 50:
            model = resnet50(pretrained=pretrained)
        elif num_layers == 101:
            model = resnet101(pretrained=pretrained)
        elif num_layers == 152:
            model = resnet152(pretrained=pretrained)
        else:
            raise Exception('layer number must be one of 18, 34, 50, 101, 152')

        rpn = RPN(256, 512, ratios, scales, stride)
        head = ROIHead(num_classes, [7, 7])

        super(FPNResNet, self).__init__(model, rpn, head, num_classes)


class RPN(nn.Module):
    def __init__(self, in_channel, out_channel, ratios, scales, stride):
        super(RPN, self).__init__()
        self.stride = stride
        self.scales = scales
        self.ratios = ratios
        self.num_anchor_base = len(self.ratios) * len(self.scales)

        # network
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1)
        self.score = nn.Conv2d(out_channel, self.num_anchor_base * 2, 1, stride=1, padding=0)
        self.loc = nn.Conv2d(out_channel, self.num_anchor_base * 4, 1, stride=1, padding=0)
        self.softmax = nn.Softmax(dim=1)
        initialize_params(self.conv, 0, 0.01)
        initialize_params(self.score, 0, 0.01)
        initialize_params(self.loc, 0, 0.01)

    def forward(self, features, img_size):
        """
        forward function of RPN
        :param features: list of pytroch Variable (1, C, Hn, Wn), [p2, p3, p4, p5, p6]
        :param img_size: [H, W]
        :return: torch Variable: rpn_locs, (N, 4),
                                 rpn_scores, (N, 2)
                torch tensors: rois, (K, 4),
                         anchors (L, 4)
        """
        rpn_locs_list = []
        rpn_scores_list = []
        anchors_list = []
        for i, x in enumerate(features):
            feature_h, feature_w = x.size()[2:4]
            anchor_base = generate_base_anchors(self.stride[i], self.ratios, self.scales)
            anchors = anchor_proposals(feature_h, feature_w, self.stride[i], anchor_base).view(-1, 4)
            x = self.relu(self.conv(x))
            rpn_locs = self.loc(x)
            rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(-1, 4)

            rpn_scores = self.score(x)
            rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous().view(-1, 2)

            rpn_locs_list.append(rpn_locs)
            rpn_scores_list.append(rpn_scores)
            anchors_list.append(anchors)

        rpn_locs_list = torch.cat(rpn_locs_list, 0)
        rpn_scores_list = torch.cat(rpn_scores_list, 0)
        anchors_list = torch.cat(anchors_list, 0)
        rois = create_rpn_proposals(rpn_locs_list.data, rpn_scores_list[:, 1].data, anchors_list, img_size)

        return rpn_locs_list, rpn_scores_list, rois, anchors_list


class ROIHead(nn.Module):
    def __init__(self, num_class, pool_size):
        super(ROIHead, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(12544, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU()
        )
        self.cls_loc = nn.Linear(1024, num_class * 4)
        self.score = nn.Linear(1024, num_class)
        initialize_params(self.cls_loc, 0, 0.001)
        initialize_params(self.score, 0, 0.01)

        self.num_class = num_class
        self.pool_size = pool_size

    def forward(self, x, rois, img_size):
        """
        retrun class and location prediction for each roi
        :param x: pytorch Variable, extracted feature list, (1, C, Hn, Wn), [p2, p3, p4, p5, p6]
        :param rois: pytorch tensor, rois generated from rpn proposals, (N, 4)
        :param img_size: image size [H, W]
        :return: pytorch Variable, roi_cls_locs, roi_scores
        """
        # replaced original roi_pooling with roi_align
        pool_result = pyramid_roi_pooling(x, rois, img_size, self.pool_size)
        pool_result = pool_result.view(pool_result.size(0), -1).contiguous()

        fc = self.classifier(pool_result)
        roi_cls_locs = self.cls_loc(fc)
        roi_scores = self.score(fc)

        return roi_cls_locs, roi_scores
