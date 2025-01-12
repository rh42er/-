import cv2
import numpy as np


def tensor2cvimg(src):
    """
    将张量转换为OpenCV格式的图像。
    :param src: 输入的张量
    :return: OpenCV格式的图像（numpy数组，uint8类型，BGR通道顺序，(H, W, C)维度顺序）
    """
    out = src.copy() * 255
    out = out.transpose((1, 2, 0)).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def cvimg2tensor(src):
    """
    将OpenCV格式的图像转换为张量。
    :param src: 输入的OpenCV格式图像
    :return: 张量（numpy数组，float64类型，RGB通道顺序，(C, H, W)维度顺序，像素值归一化到[0, 1]）
    """
    out = src.copy()
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    out = out.transpose((2, 0, 1)).astype(np.float64)
    return out / 255

