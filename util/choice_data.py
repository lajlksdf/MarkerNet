import json
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms import transforms

loader = transforms.Compose([
    transforms.ToTensor()
])


def check(m_dir, th=0.25):
    means, count = 0, 0
    for f in os.listdir(m_dir):
        img = Image.open(os.path.join(m_dir, f))
        mask = loader(img).cuda()
        means += torch.flatten(mask).mean()
        count += 1
    # print(means * 100 / count)
    return means * 100 / count < th


def delete_dir(dir):
    print(dir)
    rm_cmd = f'rm -rf {dir}'
    os.system(rm_cmd)


def walk_dir(path=r'C:\Users\Administrator\Desktop\manual'):
    mask = os.path.join(path, 'mask')
    fake = os.path.join(path, 'fake')
    for m in os.listdir(mask):
        m_dir = os.path.join(mask, m)
        f_dir = os.path.join(fake, m)
        if check(m_dir):
            delete_dir(m_dir)
            delete_dir(f_dir)


if __name__ == '__main__':
    walk_dir(sys.argv[1])
