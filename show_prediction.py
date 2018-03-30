import cv2
import os
import torch
from fpn_resnet import FPNResNet

from configure import Config
from train import create_img_tensor

folder_dir = '../VOCtest2007/VOC2007/JPEGImages/'


def show(num):
    # change score_thresh for visualization
    Config.score_thresh = 0.7
    key = Config.class_key
    # load model
    faster_rcnn = FPNResNet().cuda()
    state_dict = torch.load('faster_rcnn_model.pt')
    faster_rcnn.load_state_dict(state_dict['model'])

    for i in range(num):
        for file in os.listdir(folder_dir):
            filename = os.fsdecode(file)
            img_dir = folder_dir + filename
            img = cv2.imread(img_dir)
            img = process_image(img)
            img_tensor = create_img_tensor(img)
            box, score, label = faster_rcnn.predict(img_tensor)
            for i, b in enumerate(box):
                ymin, xmin, ymax, xmax = [int(j) for j in b]
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
                cv2.putText(img,
                            key[label[i]] + str(score[i]),
                            (xmin, ymin),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255))
            cv2.imshow('image', img[:, :, [2, 1, 0]])
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def process_image(image):
    min_size = Config.img_min_size
    max_size = Config.img_max_size
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height = image.shape[0]
    width = image.shape[1]
    if height >= width:
        scale_ratio = min_size / width
        width_rescaled = min_size
        height_rescaled = int(height * scale_ratio)
        if height_rescaled > max_size:
            scale_ratio = max_size / height
            height_rescaled = max_size
            width_rescaled = int(width * scale_ratio)
        rescaled_img = cv2.resize(image, (width_rescaled, height_rescaled))

    if height < width:
        scale_ratio = min_size / height
        height_rescaled = min_size
        width_rescaled = int(width * scale_ratio)
        if width_rescaled > max_size:
            scale_ratio = max_size / width
            width_rescaled = max_size
            height_rescaled = int(height * scale_ratio)
        rescaled_img = cv2.resize(image, (width_rescaled, height_rescaled))

    return rescaled_img


if __name__ == '__main__':
    show(100)