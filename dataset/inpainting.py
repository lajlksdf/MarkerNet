import os

from dataset.Base import BaseDataset, DataItem


class InpaintingDataset(BaseDataset):
    train_methods = ['FGVC', 'STTN', 'OPN', 'DFGVI', 'DVI']
    test_methods = ['CPNET', ]

    def __init__(self, cfg):
        super(InpaintingDataset, self).__init__(cfg=cfg)

    def _load_data(self):
        start = 0
        methods = self.train_methods if self.cfg.mode == self.cfg.TRAIN else self.test_methods
        item_path = os.path.abspath(self.cfg.set_path)
        if os.path.isdir(item_path):
            src_dir = os.path.join(item_path, 'src')
            fake_dir = os.path.join(item_path, 'fake')
            mask_dir = os.path.join(item_path, 'mask')
            listdir = sorted(os.listdir(src_dir))
            for _f in listdir:
                label = _f
                mask = os.path.join(mask_dir, _f)
                src = os.path.join(src_dir, _f)
                fakes, masks = [], []
                for fake_ in methods:
                    fake = os.path.join(fake_dir, fake_, _f)
                    fakes.append(fake)
                    masks.append(mask)
                data_item = DataItem(src, label, start, masks, fakes)
                start = data_item.end
                self.data.append(data_item)
        self.count(start)
