import os
import random

from dataset.Base import BaseDataset, DataItem


class SplicingDataset(BaseDataset):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)

    def __getitem__(self, index):
        files, video_data = self.getitem(index)
        video_data: DataItem = video_data
        i = random.randint(-3, 100)
        fake_data = self.read_data(video_data.fake_dir, files, op=i)
        mask_data = self.read_data(video_data.mask_dir, files, op=i, mask=True)
        fake_file = os.path.join(video_data.fake_dir, files[0])
        src_file = os.path.join(video_data.src_dir, files[0])
        return video_data.label, fake_data, mask_data, src_file, fake_file

    def _load_data(self):
        start = 0
        self.cfg.set_path = os.path.join(self.cfg.set_path, self.cfg.mode)
        item_path = os.path.abspath(self.cfg.set_path)
        if os.path.isdir(item_path):
            fake_dir = os.path.join(item_path, 'fake')
            mask_dir = os.path.join(item_path, 'mask')
            listdir = sorted(os.listdir(fake_dir))
            for _f in listdir:
                label = _f
                mask = os.path.join(mask_dir, _f)
                fake = os.path.join(fake_dir, _f)
                data_item = DataItem(fake, label, start, mask, fake)
                start = data_item.end
                self.data.append(data_item)
        self.count(start)
