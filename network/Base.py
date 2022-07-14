import _thread
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from dataset.Base import BaseTrainItem, TrainCache, load_cache
from util.logUtil import logger


def resize(x: Tensor, size):
    return F.interpolate(x, size=size, mode="bilinear", align_corners=True)


class BaseTrainer:
    def __init__(self, cfg):
        # base
        self.opt = None
        self.cfg = cfg
        self.setup()

    def init(self):
        pass

    def save(self, path):
        pass

    def train(self, item: BaseTrainItem):
        pass

    def eval(self, item: BaseTrainItem):
        pass

    def test(self, item: BaseTrainItem):
        pass

    def finish(self):
        pass

    @staticmethod
    def optimizer_sgd(m: nn.Module, learning_rate=1e-3):
        opt = torch.optim.SGD(filter(lambda p: p.requires_grad, m.parameters()), lr=learning_rate)
        return opt

    def multi_init(self, net, path):
        if self.cfg.IS_DISTRIBUTION and torch.cuda.device_count() > 1:
            logger.info('MULTI_CUDA:rank:{}'.format(self.cfg.rank))
            net = nn.DataParallel(net)
        if os.path.exists(path):
            logger.info('loading model:{}'.format(path))
            map_location = 'cuda:%d' % self.cfg.rank
            net.load_state_dict(torch.load(path, map_location=map_location))
        net = net.to(self.device)
        return net

    def setup(self):
        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(1)
        if torch.cuda.device_count() > 1:
            self.device = self.cfg.rank
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(trainer: BaseTrainer, dataloader, testloader, item_cls, cfg):
    for epoch in range(cfg.EPOCH):
        train_cache = TrainCache(size=8)
        _thread.start_new_thread(load_cache, (dataloader, train_cache, item_cls))
        test_cache = TrainCache(size=1)
        _thread.start_new_thread(load_cache, (testloader, test_cache, item_cls))
        while not train_cache.finished:
            if train_cache.has_item():
                idx, item = train_cache.next_data()
                loss = trainer.train(item)
                if idx % 10 == 0 and test_cache.has_item():
                    time.sleep(4)
                    if idx % 100 == 0:
                        trainer.save(f'{cfg.checkpoint}/{epoch}-{idx}')
                        time.sleep(100)
                    _, item = test_cache.next_data()
                    loss_ = trainer.eval(item)
                    logger.info(f" Epoch:{epoch}/{idx}, Train-loss:{'%.6f' % loss}, Test-loss:{'%.6f' % loss_}")
                idx += 1


def test(trainer: BaseTrainer, testloader, item_cls):
    for idx, values in enumerate(testloader):
        item = item_cls(*values)
        item.idx = idx
        if idx % 10 == 0:
            time.sleep(50)
        trainer.test(item)
    trainer.finish()
