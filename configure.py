class Config:
    num_class = 21
    class_key = ['background', 'aeroplane', 'bicycle', 'bird',
                 'boat', 'bottle', 'bus', 'car',
                 'cat', 'chair', 'cow', 'diningtable',
                 'dog', 'horse', 'motorbike', 'person',
                 'pottedplant', 'sheep', 'sofa', 'train',
                 'tvmonitor']

    img_box_dict = '../VOCdevkit/img_box_dict.npy'
    img_min_size = 600  # image resize
    img_max_size = 1000  # image resize

    # anchor params
    ratios = [0.5, 1., 2.]
    scales = [8]
    stride = [4, 8, 16, 32, 64]

    # sigma for l1_smooth_loss
    rpn_sigma = 3.
    roi_sigma = 1.

    # param for optimizer
    weight_decay = 0.0005
    lr_decay = 0.1
    lr = 0.0001  # 0.001

    # loc mean and std
    loc_normalize_mean = [0., 0., 0., 0.]
    loc_normalize_std = [0.1, 0.1, 0.2, 0.2]

    # param for creating rpn proposals
    nms_thresh = 0.7
    num_pre_nms = 12000
    num_post_nms = 2000
    min_size = 16

    # params for creating rois for fast-rcnn training
    roi_num_sample = 128
    roi_pos_ratio = 0.25
    roi_pos_iou_thresh = 0.5
    roi_neg_iou_thresh_hi = 0.5
    roi_neg_iou_thresh_lo = 0.0

    # params for rpn training
    rpn_num_sample = 256
    rpn_pos_iou_thresh = 0.7
    rpn_neg_iou_thresh = 0.3
    rpn_pos_ratio = 0.5

    # training
    epoch = 14

    # predict param
    score_thresh = 0.05
    iou_thresh = 0.3

    # evaluation
    eval_num = float("inf")

    # pyramid levels to be used
    rpn_pyramid_levels = [4, 5, 6]
    roi_pyramid_levels = [2, 3, 4, 5]


