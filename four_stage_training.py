from tqdm import tqdm
import torch

from fpn_resnet import FPNResNet
from rescale_image import rescale_image
from voc_parse_xml import voc_generate_img_box_dict
from train import train
from train import evaluation
from train import create_img_tensor
import numpy as np

def four_stage_training(epochs=[1, 1, 1, 1]):
    # train
    xml_dir = '../VOCdevkit2007/VOC2007/Annotations'
    img_dir = '../VOCdevkit2007/VOC2007/JPEGImages'
    img_box_dict = voc_generate_img_box_dict(xml_dir, img_dir)

    # train rpn
    print('stage 1:')
    train(epochs[0], img_box_dict, pretrained_model=None, save_path='first_stage.pt',
          rpn_rois=None, train_rpn=True, train_rcnn=False, validate=False,
          lock_grad_for_rpn=False, lock_grad_for_rcnn=False)

    # generate rois
    fpn_resnet = FPNResNet().cuda()
    state_dict = torch.load('first_stage.pt')
    fpn_resnet.load_state_dict(state_dict['model'])
    roi_proposals = {}
    for i, [img_dir, img_info] in tqdm(enumerate(img_box_dict.items())):
        img, img_info = rescale_image(img_dir, img_info, flip=False)
        img_size = list(img_info['img_size'])
        img_tensor = create_img_tensor(img)
        features = fpn_resnet.extractor(img_tensor)
        _, _, rois, _ = fpn_resnet.rpn(features, img_size)
        roi_proposals[img_dir] = rois.cpu().numpy()
    np.save('roi_proposals', roi_proposals)

    roi_proposals = np.load('roi_proposals.npy')[()]
    # train rcnn
    print('stage 2:')
    train(epochs[1], img_box_dict, pretrained_model=None, save_path='second_stage.pt',
          rpn_rois=roi_proposals, train_rpn=False, train_rcnn=True, validate=False,
          lock_grad_for_rpn=False, lock_grad_for_rcnn=False)

    # use model from the second stage and only train rpn head
    print('stage 3:')
    train(epochs[2], img_box_dict, pretrained_model='second_stage.pt', save_path='third_stage.pt',
          rpn_rois=None, train_rpn=True, train_rcnn=False, validate=False,
          lock_grad_for_rpn=True, lock_grad_for_rcnn=False)

    # use model from the third stage and only train rcnn head
    print('stage 4:')
    train(epochs[3], img_box_dict, pretrained_model='third_stage.pt', save_path='final_stage.pt',
          rpn_rois=None, train_rpn=False, train_rcnn=True, validate=False,
          lock_grad_for_rpn=False, lock_grad_for_rcnn=True)

    # test
    xml_dir = '../VOCtest2007/VOC2007/Annotations'
    img_dir = '../VOCtest2007/VOC2007/JPEGImages'
    test_dict = voc_generate_img_box_dict(xml_dir, img_dir)
    fpn_resnet = FPNResNet().cuda()
    state_dict = torch.load('final_stage.pt')
    fpn_resnet.load_state_dict(state_dict['model'])
    mAP = evaluation(test_dict, fpn_resnet)
    print(mAP)


if __name__ == '__main__':
    four_stage_training(epochs=[8, 8, 8, 8])
