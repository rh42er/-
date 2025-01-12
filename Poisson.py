import numpy as np
import scipy.sparse
import cv2
import pyamg


def prepare_mask(mask):
    """
    对掩码数组进行预处理，将opencv.imread读取的可能的uint64类型及多通道情况转换为单通道二值化的uint8类型掩码。
    :param mask: 输入的掩码图像
    :return: 处理后的二值化掩码
    """
    if len(mask.shape) == 3:
        result = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if np.sum(mask[i, j]) > 0:
                    result[i, j] = 1
                else:
                    result[i, j] = 0
        mask = result
    elif mask.dtype!= np.uint8:
        mask = (mask > 0).astype(np.uint8)
    return mask


def blend(img_target, img_source, img_mask, offset=(0, 0)):
    """
    使用泊松融合算法将源图像融合到目标图像的指定区域。
    :param img_target: 目标图像
    :param img_source: 源图像
    :param img_mask: 掩码图像，指定融合区域
    :param offset: 源图像在目标图像上的偏移量
    :return: 融合后的目标图像
    """
    # 计算需要融合的区域
    region_source = (
        max(-offset[0], 0),
        max(-offset[1], 0),
        min(img_target.shape[0] - offset[0], img_source.shape[0]),
        min(img_target.shape[1] - offset[1], img_source.shape[1])
    )
    region_target = (
        max(offset[0], 0),
        max(offset[1], 0),
        min(img_target.shape[0], img_source.shape[0] + offset[0]),
        min(img_target.shape[1], img_source.shape[1] + offset[1])
    )
    region_size = (region_source[2] - region_source[0], region_source[3] - region_source[1])

    # 裁剪并归一化掩码图像
    img_mask = img_mask[region_source[0]:region_source[2], region_source[1]:region_source[3]]
    img_mask = prepare_mask(img_mask)
    img_mask = img_mask.astype(bool)

    # 创建系数矩阵
    A = scipy.sparse.identity(np.prod(region_size), format='lil')
    for y in range(region_size[0]):
        for x in range(region_size[1]):
            if img_mask[y, x]:
                index = x + y * region_size[1]
                A[index, index] = 4
                if index + 1 < np.prod(region_size):
                    A[index, index + 1] = -1
                if index - 1 >= 0:
                    A[index, index - 1] = -1
                if index + region_size[1] < np.prod(region_size):
                    A[index, index + region_size[1]] = -1
                if index - region_size[1] >= 0:
                    A[index, index - region_size[1]] = -1
    A = A.tocsr()

    # 创建用于b的泊松矩阵
    P = pyamg.gallery.poisson(img_mask.shape)

    # 对每个颜色通道进行处理
    for num_layer in range(img_target.shape[2]):
        # 获取子图像
        t = img_target[region_target[0]:region_target[2], region_target[1]:region_target[3], num_layer]
        s = img_source[region_source[0]:region_source[2], region_source[1]:region_source[3], num_layer]
        t = t.flatten()
        s = s.flatten()

        # 创建b
        b = P * s
        for y in range(region_size[0]):
            for x in range(region_size[1]):
                if not img_mask[y, x]:
                    index = x + y * region_size[1]
                    b[index] = t[index]

        # 求解Ax = b
        x = pyamg.solve(A, b, verb=False, tol=1e - 10)

        # 将结果赋值给目标图像
        x = np.reshape(x, region_size)
        x = np.clip(x, 0, 255)
        x = x.astype(img_target.dtype)
        img_target[region_target[0]:region_target[2], region_target[1]:region_target[3], num_layer] = x

    return img_target
