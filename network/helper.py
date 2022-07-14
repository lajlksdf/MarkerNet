import math
import os
import uuid

import cv2
import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn as nn
from torch.nn import functional as F
from torch.nn.init import trunc_normal_
from torchvision import utils as vutils
from torchvision.transforms import transforms

loader = transforms.Compose([
    transforms.ToTensor()
])
unloader = transforms.ToPILImage()


def tensor2img(image, path=None):
    image = unloader(image)
    if path:
        image.save(path)
    return image


def img2tensor(img):
    return loader(img)


def save_tensor(img, file):
    vutils.save_image(img, file)


def torch_resize(image, size=224):
    image = img2tensor(image).unsqueeze(0)
    image = F.interpolate(image, size=size, mode='bilinear', align_corners=True)
    return image


def tensor_resize(image, size=224):
    image = F.interpolate(image, size=size, mode='bilinear', align_corners=True)
    return image


def rgb2gray(rgb):
    b, g, r = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    gray = torch.unsqueeze(gray, 1)
    return gray


def gray2rgb4d(gray):
    b, c, h, w = gray.shape
    gray = gray[:, 0, :, :]
    src_new = torch.randn([b, 3, h, w])
    src_new[:, 2, :, :] = gray
    src_new[:, 1, :, :] = gray * 0.5
    src_new[:, 0, :, :] = gray * 3
    return src_new


def init_m(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        fan_out //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()


def compress(file, quality=85):
    '''
    compress image
    :param file: image file
    :param quality: 85 or 70
    :return: compressed image
    '''
    outfile = str(uuid.uuid1()) + '.jpg'
    im = Image.open(file)
    im.save(outfile, quality=quality)
    return outfile


def cal_compress(outfile, is_path=True):
    if is_path:
        im = Image.open(outfile)
        sz = os.path.getsize(outfile)
    else:
        im = outfile
        outfile = str(uuid.uuid1()) + '.jpg'
        im.save(outfile, quality=100)
        sz = os.path.getsize(outfile)
    h, w = im.size
    return h * w // sz


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def unnorm(self, tensor):
        """
        Args:
        :param tensor: tensor image of size (B,C,H,W) to be un-normalized
        :return: UnNormalized image
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def process_mask(file, bin=False):
    if bin:
        gray = file
    else:
        img = cv2.imread(file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_MASK)
    # noise removal
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    sure_bg = cv2.dilate(opening, kernel, iterations=2)  # sure background area
    sure_op = cv2.erode(opening, kernel, iterations=2)  # sure foreground area
    sure_fg = cv2.erode(sure_bg, kernel, iterations=2)  # sure foreground area
    # cv2.imshow('1', to_binary(sure_fg))
    # cv2.imshow('2', to_binary(sure_op))
    # cv2.imshow('gray', gray)
    # cv2.waitKey(-1)
    return to_binary(sure_fg), to_binary(sure_op)


def get_binary_img(file, bin=False):
    if bin:
        img = file
    else:
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    bin_img = np.zeros(shape=(img.shape), dtype=np.uint8)
    h = img.shape[0]
    w = img.shape[1]
    for i in range(h):
        for j in range(w):
            bin_img[i][j] = 255 if img[i][j] > 85 else 0
    # cv2.imshow('bin_img', bin_img)
    bin_img = cv2.blur(bin_img, (5, 5))
    # cv2.waitKey(-1)
    return process_mask(bin_img, True)


def to_binary(img):
    bin_img = np.zeros(shape=(img.shape), dtype=np.uint8)
    h = img.shape[0]
    w = img.shape[1]
    for i in range(h):
        for j in range(w):
            bin_img[i][j] = 255 if img[i][j] > 120 else 0
    return bin_img


def fill_gray(file):
    img = cv2.imread(file)
    img = cv2.GaussianBlur(img, (7, 7), 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    process_mask(img, True)


def read_data(_dir, files, loader_, mask=False):
    tensors = []
    for f in files:
        _f = os.path.join(_dir, f)
        im = Image.open(_f)
        if mask:
            tensor = to_mask_tensor(im).unsqueeze(0)
        else:
            tensor = tensor_resize(loader_(im).unsqueeze(0))
        tensors.append(tensor)
        im.close()
    data = torch.cat(tensors, dim=0)
    return data


def to_mask_tensor(img, image_size=224):
    img = img.convert('L')
    img = img.resize((image_size, image_size))
    img = np.asarray(img, dtype=np.int32)
    img[img < 50] = 0.0
    img[img > 0] = 1.0
    mask = torch.from_numpy(img.astype(np.float32))
    return torch.unsqueeze(mask, 0)


def to_mask_tensor_cv2_gray(img, image_size=224):
    img = cv2.resize(img, (image_size, image_size))
    img[img < 50] = 0.0
    img[img > 0] = 1.0
    mask = torch.from_numpy(img.astype(np.float32))
    return torch.unsqueeze(mask, 0)


def to_mask_tensor_cv2(img, image_size=224):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return to_mask_tensor_cv2_gray(img, image_size)


def cal_iou(mask_data, outs: Tensor):
    b, t, c, h, w = outs.shape
    iou1, iou2, count = 0, 0, 0
    for i in range(b):
        for j in range(t):
            count += 1
            out = outs[i][j].cpu().clone().detach()
            mask = mask_data[i][j].clone().detach()
            out = out.mul(255).byte().numpy().transpose((1, 2, 0))
            p1, p2 = get_binary_img(out, True)
            iou1 += mask_iou(mask, p1 // 255)
            iou2 += mask_iou(mask, p2 // 255)
    return iou1 / count, iou2 / count


def cal_iou_image(mask_data, outs: Tensor):
    b, c, h, w = outs.shape
    iou1, iou2, count = 0, 0, 0
    for i in range(b):
        count += 1
        out = outs[i].cpu().clone().detach()
        mask = mask_data[i].clone().detach()
        out = out.mul(255).byte().numpy().transpose((1, 2, 0))
        p1, p2 = get_binary_img(out, True)
        iou1 += mask_iou(mask, p1 // 255)
        iou2 += mask_iou(mask, p2 // 255)
    return iou1 / count, iou2 / count


def mask_iou(mask1, mask2):
    area1 = mask1.sum()
    area2 = mask2.sum()
    inter = ((mask1 + mask2) == 2).sum()
    m_iou = inter / (area1 + area2 - inter)
    m_iou = round(m_iou.item(), 3)
    return 0 if math.isnan(m_iou) else m_iou


def f1_score(premask, groundtruth):
    # 二值分割图是一个波段的黑白图，正样本值为1，负样本值为0
    # 通过矩阵的逻辑运算分别计算出tp,tn,fp,fn
    seg_inv, gt_inv = np.logical_not(premask), np.logical_not(groundtruth)
    true_pos = float(np.logical_and(premask, groundtruth).sum())  # float for division
    false_pos = np.logical_and(premask, gt_inv).sum()
    false_neg = np.logical_and(seg_inv, groundtruth).sum()

    return 2 * true_pos / (2 * true_pos + false_pos + false_neg + 1e-6)


def cal_iou_image_f1(mask_data, outs: Tensor):
    b, c, h, w = outs.shape
    f1, f2, count = 0, 0, 0
    for i in range(b):
        count += 1
        out = outs[i].cpu().clone().detach()
        mask = mask_data[i].clone().detach()
        out = out.mul(255).byte().numpy().transpose((1, 2, 0))
        p1, p2 = get_binary_img(out, True)
        f1 += f1_score(p1 // 255,mask)
        f2 += f1_score(p2 // 255, mask)
    return f1 / count, f2 / count