import numpy as np
from configure import Config

KEY = Config.class_key


def text_to_num(label_text):
    """
    conver text label to number label
    :param label_text: np array (N,)
    :return: np array (N,)
    """
    label_num = [KEY.index(i) for i in label_text]

    return np.array(label_num)


def num_to_text(label_num):
    """
    conver number label to text label
    :param label_num: np array (N,)
    :return: np array (N,)
    """
    label_text = [KEY[i] for i in label_num]

    return np.array(label_text)
