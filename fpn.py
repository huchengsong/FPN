import torch
import torch.nn.functional as F
import torch.nn as nn
from configure import Config

from box_parametrize import box_deparameterize_gpu
from non_maximum_suppression import non_maximum_suppression_roi


def upsample_add(x, y):
    # upsample x and then add y
    _, _, H, W = y.size()
    return F.upsample(x, size=(H, W), mode='bilinear') + y


class FPN(nn.Module):
    def __init__(self, model, rpn, head, num_classes):
        super(FPN, self).__init__()
        self.num_classes = num_classes
        self.model = model
        self.rpn = rpn
        self.head = head
        self.optimizer = None

        self.layer0 = nn.Sequential(self.model.conv1, self.model.bn1,
                                    self.model.relu, self.model.maxpool)
        self.layer1 = nn.Sequential(self.model.layer1)
        self.layer2 = nn.Sequential(self.model.layer2)
        self.layer3 = nn.Sequential(self.model.layer3)
        self.layer4 = nn.Sequential(self.model.layer4)

        # top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)

        # smooth layer
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # lateral layer
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        self.maxpool2d = nn.MaxPool2d(2, stride=2)

    def get_optimizer(self, lr):
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params.append({'params': [value], 'lr': lr * 2, 'weight_decay': 0})
                else:
                    params.append({'params': [value], 'lr': lr, 'weight_decay': Config.weight_decay})
        self.optimizer = torch.optim.SGD(params, momentum=0.9)

    def extractor(self, img_tensor):
        # Bottom-up
        c1 = self.layer0(img_tensor)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # Top-down
        p5 = self.toplayer(c5)
        p4 = upsample_add(p5, self.latlayer1(c4))
        p4 = self.smooth1(p4)
        p3 = upsample_add(p4, self.latlayer2(c3))
        p3 = self.smooth2(p3)
        p2 = upsample_add(p3, self.latlayer3(c2))
        p2 = self.smooth3(p2)

        # p6
        p6 = self.maxpool2d(p5)

        features = [p2, p3, p4, p5, p6]

        return features

    def forward(self, features, img_size):
        _, _, rois, _ = self.rpn(features, img_size)
        roi_cls_locs, roi_scores = self.head(features, rois, img_size)

        return roi_cls_locs, roi_scores, rois

    def predict(self, img_tensor):
        """
        bounding box prediction
        :param img_tensor: preprocessed image tensor
        :param iou_thresh: iou threshold for nms
        :param score_thresh: score threshold
        :return: ndarray: label (N, ), score (N, ), box (N, 4)
        """
        # self.eval() set the module in evaluation mode: self.train(False)
        self.eval()

        # store training parameters
        train_num_pre_nms = Config.num_pre_nms
        train_num_post_nms = Config.num_post_nms

        score_thresh = Config.score_thresh
        iou_thresh = Config.iou_thresh
        loc_mean = Config.loc_normalize_mean
        loc_std = Config.loc_normalize_std

        # set parameters for evaluation
        Config.num_pre_nms = 6000
        Config.num_post_nms = 1000

        img_size = list(img_tensor.size()[2:4])

        roi_cls_loc, roi_scores, rois = self(img_tensor)
        roi_scores = nn.Softmax(dim=1)(roi_scores).data
        roi_cls_loc = roi_cls_loc.view(-1, 4).data

        # de-normalize
        loc_mean = torch.cuda.FloatTensor(loc_mean)
        loc_std = torch.cuda.FloatTensor(loc_std)
        roi_cls_loc = roi_cls_loc * loc_std + loc_mean

        rois = rois.view(-1, 1, 4).repeat(1, self.num_class, 1).view(-1, 4)
        cls_bbox = box_deparameterize_gpu(roi_cls_loc, rois)

        # clip bounding boxes
        cls_bbox[:, [0, 2]] = cls_bbox[:, [0, 2]].clamp(0, img_size[0])
        cls_bbox[:, [1, 3]] = cls_bbox[:, [1, 3]].clamp(0, img_size[1])
        cls_bbox = cls_bbox.view(-1, self.num_class * 4)

        box, score, label = non_maximum_suppression_roi(roi_scores, cls_bbox, range(1, Config.num_class),
                                                        score_thresh=score_thresh, iou_thresh=iou_thresh)
        self.train()

        # restore parameter for training
        Config.num_pre_nms = train_num_pre_nms
        Config.num_post_nms = train_num_post_nms

        return box, score, label
