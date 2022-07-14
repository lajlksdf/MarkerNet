import os
import random

from dataset.Base import BaseDataset, DataItem


class FFDataset(BaseDataset):
    test_compress = ['raw']
    train_compress = ['c23', 'c40']
    mask_listdir = ['neuraltextures']

    # mask_listdir = ['face2face', 'faceswap', 'deepfakes', 'neuraltextures']
    def __init__(self, cfg):
        super().__init__(cfg=cfg)

    def _load_data(self):
        start = 0
        self.cfg.set_path = os.path.join(self.cfg.set_path, self.cfg.mode)
        item_path = self.cfg.set_path
        compresses = self.train_compress if self.cfg.mode == self.cfg.TRAIN else self.test_compress
        if os.path.isdir(item_path):
            src_dir = os.path.join(item_path, 'src')
            fake_dir = os.path.join(item_path, 'fake')
            mask_dir = os.path.join(item_path, 'mask')
            for item in os.listdir(src_dir):
                src = os.path.join(src_dir, item)
                label = item
                fakes, masks = [], []
                for cls in self.mask_listdir:
                    for c in compresses:
                        fake = os.path.join(fake_dir, cls, c, item)
                        fakes.append(fake)
                        mask = os.path.join(mask_dir, cls, item)
                        masks.append(mask)
                data_item = DataItem(src, label, start, masks, fakes)
                start = data_item.end
                self.data.append(data_item)
        self.count(start)
