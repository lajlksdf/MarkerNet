import os

from dataset.Video import VideoDataItem, BaseVideoDataset

train_compresses = ['c23']
test_compresses = ['c23']


class DFDVideoDataset(BaseVideoDataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _load_data(self):
        item_path = self.cfg.set_path
        compresses = train_compresses if self.cfg.mode == self.cfg.TRAIN else test_compresses
        start = 0
        src_dir = os.path.join(item_path, 'src', 'c23', 'videos')
        fake_dir = os.path.join(item_path, 'fake')
        mask_dir = os.path.join(item_path, 'masks', 'videos')
        for item in sorted(os.listdir(src_dir)):
            src = os.path.join(src_dir, item)
            label = item.replace('.mp4', '')
            fakes, masks = [], []
            for c in compresses:
                fake_video_dir = os.path.join(fake_dir, c, 'videos', label)
                for f in os.listdir(fake_video_dir):
                    fake_video = os.path.join(fake_video_dir, f)
                    fakes.append(fake_video)
                    mask_video_dir = os.path.join(mask_dir, f)
                    masks.append(mask_video_dir)
            if len(fakes) != 0:
                data_item = VideoDataItem(src=src, fake=fakes, mask=masks, label=label, start=start)
                start = data_item.end
                self.data.append(data_item)
        self.count(start)

