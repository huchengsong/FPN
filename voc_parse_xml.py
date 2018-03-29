import xml.etree.ElementTree as ET
import os 
import numpy as np


def voc_generate_img_box_dict(xml_dir, img_dir):
    """
    :param xml_dir: xml file directory
    :param img_dir: image file directory
    :return: a dictionary {img_dir: {'image_size': [], 'object': [name, ymin, xmin, ymax, xmax]}}
    """

    img_box_dict = {}
    for xml in os.listdir(xml_dir):
        if xml.endswith('.xml'):
            xml_path = os.path.join(xml_dir, xml)
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # get img path from xml file and store in the dict
            img_path = img_dir+'/' + root.find('filename').text
            img_box_dict[img_path] = {}

            width = int(root.find('size').find('width').text)
            height = int(root.find('size').find('height').text)
            img_box_dict[img_path]['img_size'] = [height, width]
            img_box_dict[img_path]['objects'] = []
            img_box_dict[img_path]['difficult'] = []

            for obj in root.iter('object'):
                img_box_dict[img_path]['difficult'].append(bool(int(obj.find('difficult').text)))
                name = obj.find('name').text
                bndbox = obj.find('bndbox')
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)
                img_box_dict[img_path]['objects'].append([name, ymin, xmin, ymax, xmax])
    return img_box_dict


if __name__ == '__main__':
    XML_DIR = '../VOCdevkit/VOC2012/Annotations'
    IMG_DIR = '../VOCdevkit/VOC2012/JPEGImages'
    SAVE_DIR = '../VOCdevkit/img_box_dict.npy'
    img_box_dict = voc_generate_img_box_dict(XML_DIR, IMG_DIR)
    np.save(SAVE_DIR, img_box_dict)
