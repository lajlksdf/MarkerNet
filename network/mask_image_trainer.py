import os

from dataset.Base import BaseTrainItem
from network.helper import cal_iou_image_f1
from util.logUtil import logger

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch

from network.Base import BaseTrainer
from network.masknet import MaskImageNet
from util import figureUtil


class MaskTrainItem(BaseTrainItem):
    def __init__(self, label, fake_data, mask_data, src_files, fake_files):
        super().__init__()
        self.label = label
        self.fake_data = fake_data
        self.mask_data = mask_data
        self.src_files = src_files
        self.fake_files = fake_files


class MaskImageTrainer(BaseTrainer):
    low_iou = []
    iou1_, iou2_, count = 0, 0, 0

    def __init__(self, pretrained, cfg, train=True):
        super().__init__(cfg)
        self.net_g = MaskImageNet(cfg=cfg)
        self.pretrained = pretrained
        self.net_g = self.multi_init(self.net_g, self.pretrained)
        if train:
            self.loss_d = torch.nn.MSELoss()  # fn.mask_loss
            # self.opt = self.optimizer_sgd(self.net_g, self.cfg.base_lr)
            # self.scheduler_g = ExponentialLR(self.opt, gamma=0.98)
            self.opt = torch.optim.Adam(filter(lambda p: p.requires_grad, self.net_g.parameters()),
                                        lr=self.cfg.base_lr)

    def save(self, path):
        if self.cfg.IS_DISTRIBUTION and self.cfg.rank == 0:
            torch.save(self.net_g.module.state_dict(), path + "-net_g.pth")
        else:
            torch.save(self.net_g.state_dict(), path + "-net_g.pth")

    def train(self, item: MaskTrainItem):
        self.net_g.train()
        images, out_aux = self.net_g(item.fake_data.to(self.device))
        loss = self.loss_d(images.squeeze(), item.mask_data.squeeze().to(self.device))
        if item.idx % 100 == 0:
            figureUtil.mask_result(images.detach().cpu().squeeze(dim=2), item.mask_data.squeeze(dim=2),
                                   item.src_files, item.fake_files, f'tmp/train-{item.idx}.jpg')
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        # self.scheduler_g.step()
        return loss

    def eval(self, item: MaskTrainItem):
        self.net_g.eval()
        with torch.no_grad():
            images, out_aux = self.net_g(item.fake_data.to(self.device))
            figureUtil.image_result(images.detach().cpu().squeeze(dim=2), item.mask_data.squeeze(dim=2),
                                    out_aux.detach().cpu(), item.fake_files,
                                   f'tmp/{self.cfg.type}--{item.idx}.jpg')
            loss = self.loss_d(images.squeeze(), item.mask_data.squeeze().to(self.device))
            return loss

    def test(self, item: MaskTrainItem, net=None):
        self.net_g.eval()
        with torch.no_grad():
            masks, out_aux = self.net_g(item.fake_data.to(self.device))
            iou1, iou2 = cal_iou_image_f1(item.mask_data, masks)
            self.iou1_ += iou1
            self.iou2_ += iou2
            self.count += 1
            figureUtil.image_result(masks.detach().cpu().squeeze(dim=2), item.mask_data.squeeze(dim=2),
                                   out_aux.detach().cpu(), item.fake_files,
                                   f'test/{self.cfg.type}-{item.idx}.jpg')
            # if iou1 < 0.5:
            #     self.low_iou.extend(list(item.label))
            #     self.low_iou = list(set(self.low_iou))
            #     logger.info(f'-{self.cfg.type}-{iou1}-{iou2}-{self.pretrained}')

    def finish(self):
        logger.info(
            f'final-{self.cfg.type}-{self.iou1_ / self.count}-{self.iou2_ / self.count}-{self.pretrained}-{self.low_iou}')
