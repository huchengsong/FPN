import numpy as np
from torch import nn
import torch
from rpn_utils import generate_anchor_loc_label, generate_training_anchors

from convert_label import text_to_num
from rpn_utils import rpn_loss
from fast_rcnn_utils import fast_rcnn_loss
from configure import Config


class FasterRCNNTrainer(nn.Module):
    def __init__(self, fpn_resnet, rpn_sigma=Config.rpn_sigma, roi_sigma=Config.roi_sigma):
        super(FasterRCNNTrainer, self).__init__()
        self.fpn_resnet = fpn_resnet
        self.rpn_sigma = rpn_sigma
        self.roi_sigma = roi_sigma
        self.optimizer = self.fpn_resnet.optimizer

        self.loc_normalize_mean = Config.loc_normalize_mean
        self.loc_normalize_std = Config.loc_normalize_std

    def forward(self, img_tensor, img_info):
        img_size = img_info['img_size']
        features = self.fpn_resnet.extractor(img_tensor)

        gt_bbox = np.array(img_info['objects'])[:, 1:5].astype(np.float32)
        gt_label = np.array(img_info['objects'])[:, 0]
        gt_label = text_to_num(gt_label)

        # RPN loss
        rpn_locs, rpn_scores, rois, anchors = self.fpn_resnet.rpn(features, img_size)
        gt_rpn_label, gt_rpn_loc = generate_anchor_loc_label(anchors, gt_bbox, img_size)
        rpn_cls_loss, rpn_loc_loss = rpn_loss(rpn_scores, rpn_locs,
                                              gt_rpn_loc, gt_rpn_label,
                                              self.rpn_sigma)

        # print('rpn_cls_loss', rpn_cls_loss.data.cpu().numpy(),
        #       'rpn_loc_loss', rpn_loc_loss.data.cpu().numpy())

        # generate proposals from rpn rois
        sampled_roi, gt_roi_loc, gt_roi_label = generate_training_anchors(rois, gt_bbox, gt_label)

        # ROI loss
        roi_cls_loc, roi_score = self.fpn_resnet.head(features, sampled_roi, img_size)
        roi_cls_loss, roi_loc_loss = fast_rcnn_loss(roi_score, roi_cls_loc,
                                                    gt_roi_loc, gt_roi_label,
                                                    self.roi_sigma)

        # print('roi_cls_loss', roi_cls_loss.data.cpu().numpy(),
        #       'roi_loc_loss', roi_loc_loss.data.cpu().numpy())

        return rpn_cls_loss + rpn_loc_loss + roi_cls_loss + roi_loc_loss

    def train_step(self, img_tensor, img_info):
        self.optimizer.zero_grad()
        loss = self.forward(img_tensor, img_info)
        # print('total loss', loss.data.cpu().numpy())
        loss.backward()
        self.optimizer.step()

    def save(self, save_path):
        save_dict = dict()
        save_dict['model'] = self.fpn_resnet.state_dict()
        save_dict['optimizer'] = self.optimizer.state_dict()
        torch.save(save_dict, save_path)
        print('model saved as ' + save_path)

    def load(self, load_path):
        state_dict = torch.load(load_path)
        self.fpn_resnet.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        print('model_loaded from ' + load_path)

    def scale_lr(self, decay):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= decay
        print('learning rate changed; decay = ', decay)
