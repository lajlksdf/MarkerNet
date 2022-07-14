import os

from dataset.Video import BaseVideoDataset, VideoDataItem


class FFVideoDataset(BaseVideoDataset):
    test_compress = ['c40']
    train_compress = ['c23', 'c40']
    # datasets = ['Deepfakes', 'NeuralTextures', 'FaceSwap', 'Face2Face']

    datasets = ['NeuralTextures']

    def __init__(self, cfg):
        super().__init__(cfg=cfg)

    def _load_data(self):
        item_path = self.cfg.set_path
        compresses = self.train_compress if self.cfg.mode == self.cfg.TRAIN else self.test_compress
        self.load_video(compresses, item_path)

    def load_video(self, compresses, item_path, mask=True):
        start = 0
        src_dir = os.path.join(item_path, 'original_sequences')
        fake_dir = os.path.join(item_path, 'manipulated_sequences')
        for cls in self.datasets:
            for c in compresses:
                fake_dir_cls = os.path.join(fake_dir, cls, c, 'videos')
                mask_dir_cls = os.path.join(fake_dir, cls, 'masks', 'videos')
                src_ = os.path.join(src_dir, 'c40', 'videos')
                for item in sorted(os.listdir(fake_dir_cls)):
                    if not item.endswith('.mp4'):
                        continue
                    fakes, masks = [], []
                    fake_video = os.path.join(fake_dir_cls, item)
                    fakes.append(fake_video)
                    if mask:
                        mask = os.path.join(mask_dir_cls, item)
                        masks.append(mask)
                    src_video = str(item).split('_')[0]
                    src = os.path.join(src_, src_video + '.mp4')
                    data_item = VideoDataItem(src, item, start, masks, fakes)
                    start = data_item.end
                    self.data.append(data_item)
        self.count(start)
