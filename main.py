import argparse
import os
import torch
from torch.legacy import nn
from torch.legacy.nn.Sequential import Sequential
import cv2
import numpy as np
from torch.utils.serialization import load_lua
import torchvision.utils as vutils
from utils import *
from poissonblending import prepare_mask, blend

# 设置命令行参数
def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='none', help='输入图像路径')
    parser.add_argument('--mask', default='none', help='掩码图像路径')
    parser.add_argument('--model_path', default='completionnet_places2.t7', help='预训练模型路径')
    parser.add_argument('--gpu', default=False, action='store_true', help='是否使用GPU')
    parser.add_argument('--postproc', default=False, action='store_true', help='是否禁用后处理')
    return parser.parse_args()

# 加载模型及数据均值
def load_model(model_path):
    data = load_lua(model_path)
    model = data.model
    model.evaluate()
    datamean = data.mean
    return model, datamean

# 加载并预处理数据
def load_and_preprocess_data(opt, datamean):
    input_img = cv2.imread(opt.input)
    I = torch.from_numpy(cvimg2tensor(input_img)).float()
    if opt.mask!= 'none':
        input_mask = cv2.imread(opt.mask)
        M = torch.from_numpy(cv2.cvtColor(input_mask, cv2.COLOR_BGR2GRAY) / 255).float()
        M[M <= 0.2] = 0.0
        M[M > 0.2] = 1.0
        M = M.view(1, M.size(0), M.size(1))
    else:
        M = torch.FloatTensor(1, I.size(1), I.size(2)).fill_(0)
        # 随机生成孔洞作为掩码
        nHoles = np.random.randint(1, 4)
        for _ in range(nHoles):
            mask_w = np.random.randint(32, 128)
            mask_h = np.random.randint(32, 128)
            px = np.random.randint(0, I.size(2) - mask_w)
            py = np.random.randint(0, I.size(1) - mask_h)
            M[:, py:py + mask_h, px:px + mask_w] = 1
    # 图像归一化
    for i in range(3):
        I[i, :, :] = I[i, :, :] - datamean[i]
    M_3ch = torch.cat((M, M, M), 0)
    Im = I * (M_3ch * (-1) + 1)
    return I, M, M_3ch, Im

# 准备模型输入
def prepare_model_input(Im, M):
    input_tensor = torch.cat((Im, M), 0)
    input_tensor = input_tensor.view(1, input_tensor.size(0), input_tensor.size(1), input_tensor.size(2)).float()
    return input_tensor

# 执行推理
def perform_inference(model, input_tensor, opt):
    if opt.gpu:
        model.cuda()
        input_tensor = input_tensor.cuda()
    res = model.forward(input_tensor)[0].cpu()
    return res

# 生成输出图像
def generate_output_image(I, res, M_3ch, datamean):
    # 图像反归一化
    for i in range(3):
        I[i, :, :] = I[i, :, :] + datamean[i]
    out = res.float() * M_3ch.float() + I.float() * (M_3ch * (-1) + 1).float()
    return out

# 后处理图像（泊松融合）
def postprocess_image(out, opt, input_img, input_mask):
    if opt.postproc:
        target = input_img
        source = tensor2cvimg(out.numpy())
        mask = input_mask
        out = blend(target, source, mask, offset=(0, 0))
        out = torch.from_numpy(cvimg2tensor(out))
    return out

# 保存输出图像
def save_output_image(out):
    vutils.save_image(out, 'out.png', normalize=True)

def main():
    opt = setup_args()
    model, datamean = load_model(opt.model_path)
    I, M, M_3ch, Im = load_and_preprocess_data(opt, datamean)
    input_tensor = prepare_model_input(Im, M)
    res = perform_inference(model, input_tensor, opt)
    out = generate_output_image(I, res, M_3ch, datamean)
    out = postprocess_image(out, opt, input_img, input_mask)
    save_output_image(out)
    print('Done')

if __name__ == "__main__":
    main()