import argparse

import torch.multiprocessing as mp

from config import MaskConfig, SplicingMaskConfig, ImageSplicingMaskConfig
from dataset.Base import get_dataloader
from dataset.CASIA_image import CASIADataset
from dataset.Converage_image import CoverDataset
from dataset.DFD_Video import DFDVideoDataset
from dataset.ff_video import FFVideoDataset
from dataset.inpainting import InpaintingDataset
from dataset.splicingtl import SplicingDatasetTL
from network.Base import train
from network.mask_image_trainer import MaskImageTrainer
from network.mask_trainer import MaskTrainer, MaskTrainItem

mp.set_sharing_strategy('file_system')
choices = {
    'splicing': [SplicingDatasetTL, MaskTrainer, SplicingMaskConfig, MaskTrainItem],
    'cover': [CoverDataset, MaskImageTrainer, ImageSplicingMaskConfig, MaskTrainItem],
    'casia': [CASIADataset, MaskImageTrainer, ImageSplicingMaskConfig, MaskTrainItem],
    'FF': [FFVideoDataset, MaskTrainer, MaskConfig, MaskTrainItem],
    'dfd': [DFDVideoDataset, MaskTrainer, MaskConfig, MaskTrainItem],
    'inpainting': [InpaintingDataset, MaskTrainer, SplicingMaskConfig, MaskTrainItem],
}


def main(local_rank, args_, data_type):
    Dataset, Trainer, cfg, Item = data_type[0], data_type[1], data_type[2], data_type[3]

    train_cfg = cfg(cfg.TRAIN, args_.set_path,  local_rank)
    train_cfg.type = args_.type
    dataset = Dataset(cfg=train_cfg)
    dataloader = get_dataloader(dataset=dataset, cfg=train_cfg, num_workers=0)

    test_cfg = cfg(cfg.TEST, args_.set_path,  local_rank)
    test_cfg.type = args_.type
    test_cfg.FRAMES_STEP = test_cfg.NUM_FRAMES
    dataset = Dataset(cfg=test_cfg)
    test_loader = get_dataloader(dataset=dataset, cfg=test_cfg, num_workers=0)

    train(
        trainer=Trainer(pretrained=args_.pretrained, cfg=train_cfg),
        dataloader=dataloader,
        testloader=test_loader,
        item_cls=Item,
        cfg=train_cfg
    )


parser = argparse.ArgumentParser()
parser.add_argument('--set_path', type=str, default=r'D:/dataset/Coverage')
parser.add_argument('--pretrained', type=str, default=r'./checkpoint/1-0-net_g.pth')
parser.add_argument('--local_rank', type=int, default=0)
# parser.add_argument('--master_addr', type=str, default='127.0.0.1')
# parser.add_argument('--master_port', type=str, default="12345")
parser.add_argument('--type', type=str, default='casia')
parser.add_argument('--dist', type=bool, default=False)
if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    conf = choices[args.type][2]
    main(args.local_rank, args, choices[args.type])
