import argparse
import os

import cv2
import torch
from PIL import Image
from torch import Tensor

from config import SplicingMaskConfig, MaskConfig
from network.helper import tensor_resize, tensor2img, gray2rgb4d
from network.masknet import MaskNet
from util.logUtil import logger


def test(net, tensors, frames, res_file, device):
    data = torch.cat(tensors, dim=0)
    v, frames = data.unsqueeze(0), frames
    data = v.to(device)
    out, out_aux = net(data)
    out, out_aux = out.cpu().detach(), out_aux.cpu().detach()
    logger.info(f'Writing {res_file} ！')
    write_result_pic(frames, out[0], gray2rgb4d(out_aux[0]), res_file)


def load_model(path, cfg, device):
    net = MaskNet(cfg=cfg)
    net.load_state_dict(torch.load(path, map_location=device))
    net.eval()
    return net


def predict(video_path, pretrained, device, cfg=MaskConfig):
    iou_total, total = 0, 0
    net = load_model(pretrained, cfg, device)
    net.to(device)
    video_cls = os.path.dirname(video_path)
    video_cls = os.path.dirname(video_cls)
    video_cls = os.path.basename(video_cls)
    model_name = os.path.basename(pretrained)

    with torch.no_grad():
        for video_name in sorted(os.listdir(video_path)):
            if not video_name.endswith('.mp4'):
                continue
            video = os.path.join(video_path, video_name)
            cap = cv2.VideoCapture(video)
            # frame_count = frame_count // cfg.NUM_FRAMES * cfg.NUM_FRAMES
            frame_count = cfg.NUM_FRAMES
            tensors, frames, count = [], [], 0
            while cap.isOpened() and count < frame_count:
                ret, frame = cap.read()
                if ret:
                    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    frames.append(image)
                    tensor = tensor_resize(cfg.loader(image).unsqueeze(0), cfg.IMAGE_SIZE)
                    tensors.append(tensor)
                    count += 1
                    if len(tensors) == cfg.NUM_FRAMES:
                        res_file = os.path.join(out_dir, video_cls, f'{video_name}_{model_name}_{count}.jpg')
                        test(net, tensors, frames, res_file, device)
                        tensors.clear()
                        frames.clear()
                        total += 1
            cap.release()


def write_result_pic(images: [], out: Tensor, aux: Tensor, name):
    B = len(images)
    b, t, h, w = out.shape
    aux = tensor_resize(aux, h)
    new_img = Image.new('RGB', (h * 3, w * b))
    for i in range(B):
        img = images[i].resize((h, w))
        new_img.paste(img, box=(0, i*w))
        img = tensor2img(out[i])
        new_img.paste(img, box=(1 * h, i*w))
        img = tensor2img(aux[i])
        new_img.paste(img, box=(2 * h, i*w))
    path = os.path.dirname(name)
    if not os.path.exists(path):
        os.makedirs(path)
    new_img.save(name)


def main(device, file_path, cfg, pretrained):
    device = torch.device(device)
    video_name_list = sorted(os.listdir(file_path))
    if len(video_name_list) < 1:
        return
    print(f'Starting run predict of object addition {file_path} ！')
    predict(file_path, pretrained, device, cfg)


out_dir = os.path.join(os.getcwd(), 'result_images')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

parser = argparse.ArgumentParser()
parser.add_argument('--file_path', type=str, default=r'./videos1')
parser.add_argument('--pretrained', type=str, default=r'/home/adminis/ppf/dfs/model/vstl/vstl.pth')
parser.add_argument('--local_rank', type=int, default=0)
if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    conf = MaskConfig(MaskConfig.TEST, '', args.local_rank)
    main(args.local_rank, args.file_path, conf, args.pretrained)
