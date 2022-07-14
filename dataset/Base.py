import os
import random
import time
from abc import ABCMeta

import numpy as np
import torch
from PIL import Image, ImageFilter
import cv2
from torch.utils import data as tud
from torch.utils.data import Dataset

from network.helper import tensor_resize, tensor2img, to_mask_tensor
from util.logUtil import logger


class BaseTrainItem:
    idx = None

    def __init__(self):
        super().__init__()
        pass


class TrainCache:
    def __init__(self, size):
        self.cache = {}
        self.size = size
        self.finished = False

    def put(self, idx, item: BaseTrainItem):
        self.cache[idx] = item
        item.idx = idx

    def is_stop(self):
        return len(self.cache) > self.size

    def next_data(self):
        return self.cache.popitem()

    def has_item(self):
        return len(self.cache) > 0

    def finish(self):
        self.finished = True


class DataItem(object):
    def __init__(self, src, label, start, mask=None, fake=None):
        self.label = label
        self.start = start
        self.src_dir = src
        self.mask_dir = mask
        self.fake_dir = fake
        print(f'{src}-{mask}-{fake}')
        self.init_files()
        self.end = self.start + len(self.files)

    def init_files(self):
        self.files = sorted(os.listdir(self.src_dir))
        if isinstance(self.fake_dir, list):
            for fake_dir in self.fake_dir:
                self.set_files(fake_dir)
            for mask_dir in self.mask_dir:
                self.set_files(mask_dir)
        else:
            self.set_files(self.fake_dir)
            self.set_files(self.mask_dir)

    def set_files(self, dir_):
        files = sorted(os.listdir(dir_))
        if len(files) < len(self.files):
            self.files = files


class BaseDataset(Dataset, metaclass=ABCMeta):
    def __len__(self):
        return self.length

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.data = []
        self._load_data()
        logger.info(f'Dataset,loading.{self.length}-{self.cfg.NUM_FRAMES}-{self.cfg.BATCH_SIZE}-{cfg}')

    def _load_data(self):
        pass

    def __getitem__(self, index):
        files, video_data = self.getitem(index)
        idx = random.randint(0, 100) % len(video_data.fake_dir)
        i = random.randint(-3, 100)

        mask_data = self.read_data(video_data.mask_dir[idx], files, mask=True, op=i)
        fake_data = self.read_data(video_data.fake_dir[idx], files, op=i)
        return video_data.label, fake_data, mask_data, \
               os.path.join(video_data.src_dir, files[0]), os.path.join(video_data.fake_dir[idx], files[0])

    def count(self, count):
        self.length = count // self.cfg.FRAMES_STEP

    def _get_files(self, idx):
        video_data, start, end, size = None, 0, 0, self.cfg.NUM_FRAMES
        for e in self.data:
            video_data: DataItem = e
            # global length
            start = idx * self.cfg.FRAMES_STEP
            if start < video_data.end:
                end = start + size
                if end > video_data.end:
                    # item length
                    start = video_data.end - size
                    end = video_data.end
                break
        start = start - video_data.start
        end = end - video_data.start
        files = video_data.files[start:end]
        return files, video_data

    def getitem(self, index):
        files, video_data = self._get_files(index)
        while True:
            try:
                assert len(files) == self.cfg.NUM_FRAMES, 'Inconsistent data length'
                return files, video_data
            except BaseException as e:
                logger.error('dir:{} ERROR:{}'.format(video_data.src_dir, e))
                index = random.randint(0, self.length)
                files, video_data = self._get_files(index)
                continue

    def read_data(self, _dir, files, op=0, mask=False):
        tensors, c_ratio = [], [cv2.IMWRITE_JPEG_QUALITY, random.randint(60, 100)]
        for f in files:
            _f = os.path.join(_dir, f)
            tensor = self.read_image(_f, c_ratio, mask, op)
            tensors.append(tensor)
        data = torch.cat(tensors, dim=0)
        if self.cfg.image_based:
            data = torch.squeeze(data, dim=0)
        return data

    def read_image(self, _f, c_ratio, mask, op):
        if self.cfg.mode == self.cfg.TEST:
            op = 0
        if op % 3 == 1:
            frame = cv2.imread(_f)
            msg = cv2.imencode(".jpg", frame, c_ratio)[1]
            msg = (np.array(msg)).tobytes()
            frame = cv2.imdecode(np.frombuffer(msg, np.uint8), cv2.IMREAD_COLOR)
            im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            im = Image.open(_f)
        if op % 5 == 1:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        if mask:
            tensor = to_mask_tensor(im, self.cfg.IMAGE_SIZE).unsqueeze(0)
        else:
            if op % 9 == 5:
                im = im.filter(ImageFilter.DETAIL)
            elif op % 9 == 1:
                im = im.filter(ImageFilter.GaussianBlur)
            elif op % 9 == 2:
                im = im.filter(ImageFilter.BLUR)
            elif op % 9 == 3:
                im = im.filter(ImageFilter.MedianFilter)
            tensor = tensor_resize(self.cfg.loader(im).unsqueeze(0), self.cfg.IMAGE_SIZE)
        im.close()
        return tensor


def get_dataloader(dataset, cfg, num_workers=4):
    dataloader = tud.DataLoader(dataset=dataset,
                                num_workers=num_workers,
                                batch_size=cfg.BATCH_SIZE, shuffle=cfg.shuffle,
                                )
    return dataloader


def load_cache(dataloader, train_cache: TrainCache, item):
    idx = 0
    for _, values in enumerate(dataloader):
        cache = item(*values)
        train_cache.put(idx, cache)
        idx += 1
        while train_cache.is_stop():
            time.sleep(1)
    time.sleep(10)
    train_cache.finish()


if __name__ == '__main__':
    im = Image.open('../1.png')
    im = to_mask_tensor(im)
    im = tensor2img(im)
    im.show()
