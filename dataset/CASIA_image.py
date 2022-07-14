import os
import random

import cv2

from dataset.Base import BaseDataset


class CASIADataset(BaseDataset):

    def __init__(self, cfg):
        super().__init__(cfg)

    def __getitem__(self, idx):
        fake_dir = os.path.join(self.cfg.set_path, 'fake')
        mask_dir = os.path.join(self.cfg.set_path, 'mask')
        file = self.data[idx]
        c_ratio = [cv2.IMWRITE_JPEG_QUALITY, random.randint(60, 100)]
        op = random.randint(-1, 100)
        fake = os.path.join(fake_dir, file.replace('_gt','').replace('png','jpg'))
        mask = os.path.join(mask_dir, file)
        fake_data = self.read_image(fake, c_ratio, False, op).squeeze(0)
        mask_data = self.read_image(mask, c_ratio, True, op).squeeze(0)

        return str(idx), fake_data, mask_data, fake, fake

    def _load_data(self):
        mask_dir = os.path.join(self.cfg.set_path, 'mask')
        self.data = os.listdir(mask_dir)
        self.length = len(self.data)
