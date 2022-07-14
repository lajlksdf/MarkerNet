import os

from dataset.Base import BaseDataset, DataItem


class SplicingDatasetTL(BaseDataset):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)

    def _load_data(self):
        start = 0
        self.cfg.set_path = os.path.join(self.cfg.set_path, self.cfg.mode)
        item_path = os.path.abspath(self.cfg.set_path)
        if os.path.isdir(item_path):
            src_dir = os.path.join(item_path, 'src')
            fake_dir = os.path.join(item_path, 'fake')
            mask_dir = os.path.join(item_path, 'mask')
            if self.cfg.mode == self.cfg.TRAIN:
                listdir = sorted(os.listdir(fake_dir))
                for cls in listdir:
                    mask_ = os.path.join(mask_dir, cls)
                    fake_ = os.path.join(fake_dir, cls)
                    for _f in os.listdir(fake_):
                        mask = os.path.join(mask_, _f)
                        fake = os.path.join(fake_, _f)
                        # Dataset does not retain original video
                        data_item = DataItem(mask, cls, start, [mask], [fake])
                        start = data_item.end
                        self.data.append(data_item)
            else:
                listdir = sorted(os.listdir(fake_dir))
                for _f in listdir:
                    label = _f
                    mask = os.path.join(mask_dir, _f)
                    fake = os.path.join(fake_dir, _f)
                    src = os.path.join(src_dir, _f)
                    data_item = DataItem(src, label, start, [mask], [fake])
                    start = data_item.end
                    self.data.append(data_item)
        self.count(start)
