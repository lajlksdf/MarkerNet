import os
import random

import cv2

from dataset.Base import BaseDataset


class CoverDataset(BaseDataset):

    def __init__(self, cfg):
        self.length = 100
        super(CoverDataset, self).__init__(cfg)

    def __getitem__(self, idx):
        idx += 1
        fake_dir = os.path.join(self.cfg.set_path, 'image')
        mask_dir = os.path.join(self.cfg.set_path, 'mask')
        src = os.path.join(fake_dir, f'{idx}.tif')
        fake = os.path.join(fake_dir, f'{idx}t.tif')
        mask = os.path.join(mask_dir, f'{idx}forged.tif')
        c_ratio = [cv2.IMWRITE_JPEG_QUALITY, random.randint(60, 100)]
        op = random.randint(-1, 100)
        fake_data = self.read_image(fake, c_ratio, False, op).squeeze(0)
        mask_data = self.read_image(mask, c_ratio, True, op).squeeze(0)
        return str(idx), fake_data, mask_data, src, fake
