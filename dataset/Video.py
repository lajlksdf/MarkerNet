import os.path
import random
from abc import ABCMeta

import numpy as np
import torch
from PIL import Image
import cv2

from config import BaseConfig
from dataset.Base import BaseDataset
from network.helper import to_mask_tensor_cv2, torch_resize
from util.logUtil import logger


class VideoDataItem(object):
    avg, count = 0, 0

    def __init__(self, src, label, start, mask=None, fake=None):
        self.label = label
        self.start = start
        self.src_dir = src
        self.mask_dir = mask
        self.fake_dir = fake
        self.init_videos()
        logger.info(f'{self.avg}-{self.count}-{self.avg/self.count}')

    def init_videos(self):
        min_frame = self.count_frame(-1, self.src_dir)
        for fake_dir in self.fake_dir:
            min_frame = self.count_frame(min_frame, fake_dir)
        for mask_dir in self.mask_dir:
            min_frame = self.count_frame(min_frame, mask_dir)
        if BaseConfig.Evaluation:
            min_frame = min(20, min_frame)  # test too slowly, so just test 20 frames.
        self.min_frame = min_frame
        self.end = self.start + self.min_frame
        logger.info(f'{self.min_frame}-{self.label}-{self.fake_dir[0]}')

    def count_frame(self, min_frame, video):
        if not os.path.exists(video):
            return min_frame
        mask_cap = cv2.VideoCapture(video)
        frame_count = int(mask_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        mask_cap.release()
        if min_frame == -1:
            return frame_count
        VideoDataItem.avg += frame_count
        VideoDataItem.count += 1
        return min(min_frame, frame_count)


class BaseVideoDataset(BaseDataset, metaclass=ABCMeta):
    def __len__(self):
        return self.length

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.video = True

    def _load_data(self):
        pass

    def __getitem__(self, index):
        start, video_data = self._get_video(index)
        while True:
            try:
                return self.getVideo(start, video_data)
            except Exception as e:
                index = random.randint(0, self.length)
                start, video_data = self._get_video(index)
                logger.error('dir:{} ERROR:{}'.format(video_data.src_dir, e))
                continue

    def getVideo(self, start, video_data):
        idx = random.randint(0, 100) % len(video_data.fake_dir)
        i = random.randint(-3, 100)

        mask_data, _ = self.read_video(video_data.mask_dir[idx], start, mask=True, op=i, file_prefix='mask')
        fake_data, fake_file = self.read_video(video_data.fake_dir[idx], start, op=i, file_prefix='fake')
        _, src_file = self.read_video(video_data.src_dir, start, op=0, file_prefix='src')
        return video_data.label, fake_data, mask_data, src_file, fake_file

    def _get_video(self, idx):
        video_data, start, end, size = None, 0, 0, self.cfg.NUM_FRAMES
        for e in self.data:
            video_data: VideoDataItem = e
            start = idx * self.cfg.FRAMES_STEP
            if start < video_data.end:
                if end > video_data.end:
                    start = video_data.end - size
                break
        start = start - video_data.start
        return start, video_data

    def read_video(self, video, start, op=0, mask=False, file_prefix='fake'):
        if self.cfg.mode == self.cfg.TEST:
            op = 0
        cap = cv2.VideoCapture(video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        tensors, frames, count, c_ratio, flip = [], [], 0, [cv2.IMWRITE_JPEG_QUALITY,
                                                            random.randint(60, 100)], random.randint(-1, 1)
        outfile = None
        video_name = os.path.basename(video).replace('.mp4', '')
        while cap.isOpened() and count < self.cfg.NUM_FRAMES:
            ret, frame = cap.read()
            if not ret:
                return None, None
            if not outfile:
                outfile = f'images/{video_name}_{file_prefix}_{start + count}.jpg'
                cv2.imwrite(outfile, frame)
                outfile = os.path.abspath(outfile)
            count += 1
            if mask:
                tensor = to_mask_tensor_cv2(frame, self.cfg.IMAGE_SIZE)
            else:
                if op % 3 == 1:
                    msg = cv2.imencode(".jpg", frame, c_ratio)[1]
                    msg = (np.array(msg)).tobytes()
                    frame = cv2.imdecode(np.frombuffer(msg, np.uint8), cv2.IMREAD_COLOR)
                if op % 6 == 5:
                    frame = cv2.detailEnhance(frame, sigma_s=80, sigma_r=0.3)
                elif op % 6 == 1:
                    frame = cv2.GaussianBlur(frame, (5, 5), 0)
                elif op % 6 == 2:
                    frame = cv2.flip(frame, flip)
                # elif op % 6 == 3:
                #     frame = cv2.medianBlur(frame, 5)
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                tensor = torch_resize(image)
            tensors.append(tensor)
        cap.release()
        data = torch.cat(tensors, dim=0)
        return data, outfile
