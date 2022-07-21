import os
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from mpl_toolkits.mplot3d import Axes3D
from torch import Tensor

from network.helper import tensor2img, tensor_resize, gray2rgb4d

matplotlib.use('Agg')
import matplotlib.patches as mpatches


def draw_figure(title, p1, p2: [], legends: [], name='Draw.jpg', x='x', y='y', markers=None):
    plt.figure('Draw')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.ylim([0,1])
    plt.title(title)
    for i in range(len(p2)):
        m = markers[i] if markers else 'o'
        p = p2[i]
        plt.plot(p1, p, marker=m, markersize=3)
    plt.legend(legends)
    plt.draw()
    plt.savefig(name)
    plt.close()


def draw3d(x, y, z, title, name):
    plt.title(title)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='rainbow')
    plt.draw()
    plt.pause(10)
    plt.savefig(name)
    plt.close()


def mask_result(g_tensor: Tensor, mask: Tensor, src_files: [], fake_files: [], name):
    g_tensor = g_tensor.detach().cpu()
    b, t, h, w = g_tensor.shape
    B = 4
    images = Image.new('RGB', (w * B, h * b))
    for j in range(B * b):
        if j % B == 0:
            img = Image.open(src_files[j // B])
            img = img.resize((h, w))
        elif j % B == 1:
            img = Image.open(fake_files[j // B])
            img = img.resize((h, w))
        elif j % B == 2:
            img = tensor2img(mask[j // B][0])
        else:
            img = tensor2img(g_tensor[j // B][0])
        images.paste(img, box=((j % B) * w, (j // B) * h))
    path = os.path.dirname(name)
    if not os.path.exists(path):
        os.makedirs(path)
    images.save(name)


def image_result(g_tensor: Tensor, mask: Tensor, out: Tensor, fake_files: [], name):
    g_tensor = g_tensor.detach().cpu()
    b, t, h, w = g_tensor.shape
    out = tensor_resize(out, h)
    out = gray2rgb4d(out)
    B = 4
    images = Image.new('RGB', (w * B, h * b))
    for j in range(B * b):
        if j % B == 0:
            img = Image.open(fake_files[j // B])
            img = img.resize((h, w))
        elif j % B == 1:
            img = tensor2img(mask[j // B])
        elif j % B == 2:
            img = tensor2img(g_tensor[j // B])
        else:
            img = tensor2img(out[j // B])
        images.paste(img, box=((j % B) * w, (j // B) * h))
    path = os.path.dirname(name)
    if not os.path.exists(path):
        os.makedirs(path)
    images.save(name)


def merge_pic2(g_tensor: Tensor, src_files: [], name, label, font=None, fillColor="#0000ff"):
    # print(f'{g_tensor.shape}-{mask.shape}-{src.shape}')
    g_tensor = g_tensor.detach().cpu()
    g_tensor = tensor_resize(g_tensor)
    b, c, h, w = g_tensor.shape
    img = Image.open(src_files[0])
    images = Image.new('RGB', (w * (c + 3), h * b))
    draw = ImageDraw.Draw(images)
    for i in range(b):
        img = tensor2img(g_tensor[i])
        images.paste(img, box=(0, i * h))
        img = Image.open(src_files[i])
        img = img.resize((h, w))
        images.paste(img, box=(w, i * h))
        for j in range(c):
            img = tensor2img(g_tensor[i][j])
            images.paste(img, box=(w * (2 + j), i * h))
        draw.text((w * (c + 2) + 10, h * i + 10), str(label[i]), font=font, fill=fillColor)
    path = os.path.dirname(name)
    if not os.path.exists(path):
        os.makedirs(path)
    images.save(name)


def merge_video2(g_tensor: Tensor, tensors: Tensor, name, labels, fillColor="#0000ff"):
    # print(f'{g_tensor.shape}-{mask.shape}-{src.shape}')
    g_tensor = g_tensor.detach().cpu()
    g_tensor = tensor_resize(g_tensor)
    b, c, h, w = g_tensor.shape
    images = Image.new('RGB', (w * c, h * b))
    for i in range(b):
        img = tensor2img(g_tensor[i])
        images.paste(img, box=(0, i * h))
        img = tensor2img(tensors[i])
        images.paste(img, box=(w, i * h))
        for j in range(c):
            img = tensor2img(g_tensor[i][j])
            images.paste(img, box=(w * (2 + j), i * h))
    path = os.path.dirname(name)
    if not os.path.exists(path):
        os.makedirs(path)
    images.save(name)


font = None


# ImageFont.truetype(
#     font='Ubuntu-B.ttf',
#     size=40
# )


def dfs_images(g_tensor: Tensor, src_files: [], fake_files: [], name, labels=None, fillColor="red"):
    g_tensor = g_tensor.detach().cpu()
    g_tensor = tensor_resize(g_tensor)
    b, c, h, w = g_tensor.shape
    images = Image.new('RGB', (w * (c + 4), h * b))
    draw = ImageDraw.Draw(images)
    for i in range(b):
        img = Image.open(src_files[i])
        img = img.resize((h, w))
        images.paste(img, box=(0, i * h))
        img = Image.open(fake_files[i])
        img = img.resize((h, w))
        images.paste(img, box=(w, i * h))
        for j in range(c):
            img = tensor2img(g_tensor[i][j])
            images.paste(img, box=(w * (2 + j), i * h))
        draw.text((w * (c + 2) + 10, h * i), str(labels[i]), font=font, fill=fillColor)
    path = os.path.dirname(name)
    if not os.path.exists(path):
        os.makedirs(path)
    images.save(name)


def merge_pic3(g_tensor: Tensor, mask: Tensor, src: Tensor, name):
    b, t, c, h, w = g_tensor.shape
    img = tensor2img(g_tensor[0][0])
    images = Image.new(img.mode, (w * t, h * b * 3))
    for j in range(b * 3):
        for i in range(t):
            if j % 3 == 0:
                tensor = g_tensor[j // 3][i]
            elif j % 3 == 1:
                tensor = mask[j // 3][i]
            else:
                tensor = src[j // 3][i]
            img = tensor2img(tensor)
            images.paste(img, box=(i * w, j * h))
    path = os.path.dirname(name)
    if not os.path.exists(path):
        os.makedirs(path)
    images.save(name)


def save_img(g_tensor: Tensor, name=f"{time.time()}.png"):
    b, c, h, w = g_tensor.shape
    g_tensor = g_tensor.clone().detach().cpu()
    img = tensor2img(g_tensor[0][0])
    images = Image.new(img.mode, (w * b, h * c))
    for j in range(b):
        for i in range(c):
            img = tensor2img(g_tensor[j][i])
            images.paste(img, box=(i * w, j * h))
    path = os.path.dirname(name)
    if not os.path.exists(path):
        os.makedirs(path)
    images.save(name)


def unnormalized(img, std, mu):
    img = img * std + mu
    return img


def merge_pic(g_tensor: Tensor, mask: Tensor, name):
    b, t, h, w = g_tensor.shape
    g_tensor = g_tensor.detach().cpu()
    img = tensor2img(g_tensor[0][0])
    images = Image.new(img.mode, (w * t, h * b * 2))
    for j in range(b * 2):
        for i in range(t):
            img = tensor2img(g_tensor[j // 2][i] if j % 2 == 0 else mask[j // 2][i])
            images.paste(img, box=(i * w, j * h))
    path = os.path.dirname(name)
    if not os.path.exists(path):
        os.makedirs(path)
    images.save(name)


def draw_test_iou():
    spatial_itrs = [
        0.377939, 0.175554, 0.160021, 0.170534, 0.133934, 0.170337,
        0.125127, 0.117704, 0.117576, 0.111584,
        0.113190, 0.109101, 0.116783, 0.106742, 0.106576, 0.106516, 0.103445, 0.105428, 0.104545, 0.103513,
        0.101325, 0.105339, 0.102349, 0.101941,
        0.104469, 0.100848, 0.104917, 0.102321, 0.104615, 0.103508, 0.102890
    ]
    frequency_itrs = [
        0.153049, 0.228146, 0.182763, 0.168356, 0.162630, 0.155650,
        0.151629, 0.152788, 0.145427, 0.143250,
        0.141120, 0.192455, 0.136601, 0.132945, 0.128647, 0.127506, 0.129074, 0.124227, 0.124868, 0.123511,
        0.117832, 0.119272, 0.117805, 0.118620,
        0.112415, 0.116589, 0.114334, 0.112984, 0.112085, 0.110506, 0.113182
    ]
    all_itrs = [
        0.201412, 0.197401, 0.175303, 0.171267, 0.162487, 0.158481,
        0.152193, 0.148089, 0.147598, 0.141338,
        0.140728, 0.138679, 0.136331, 0.131810, 0.129510, 0.129564, 0.125561, 0.125446, 0.123413, 0.121547,
        0.120308, 0.117966, 0.119461, 0.118870,
        0.114740, 0.115841, 0.113660, 0.111409, 0.113220, 0.109614, 0.109018
    ]
    dises = []
    for i in range(len(spatial_itrs)):
        dises.append(i)
    params = [spatial_itrs, frequency_itrs, all_itrs]
    legends = ['Spatial', 'Frequency', 'Twin-bottleneck']
    draw_figure('', dises, params, legends,
                x='Training Epoch', y='MSE loss', name='test_iou.png')


def draw_test_acc():
    mIoU = [0.58, 0.652, 0.734]
    dises = ['Frequency', 'Spatial', 'Twin-Bottleneck']
    for i in range(len(dises)):
        plt.bar(dises[i], mIoU[i], width=0.382)
    plt.savefig('bar.png')
    plt.close()


def analyze_df_c():
    legends = ['raw', 'c23', 'c40']
    raw = [0.867, 0.698, 0.644, 0.701, 0.477]
    c23 = [0.866, 0.698, 0.644, 0.700, 0.477]
    c40 = [0.866, 0.696, 0.640, 0.699, 0.464]

    params = [raw, c23, c40]
    dises = ['DF', 'F2F', 'FS', 'NT', 'DFD']
    draw_figure('', dises, params, legends, x='', y='mIoU',
                name='cdf.png')


def analyze_f2f_c():
    legends = ['raw', 'c23', 'c40']
    raw = [0.690, 0.885, 0.745, 0.878, 0.449]
    c23 = [0.689, 0.884, 0.732, 0.877, 0.449]
    c40 = [0.684, 0.871, 0.774, 0.871, 0.438]

    params = [raw, c23, c40]
    dises = ['DF', 'F2F', 'FS', 'NT', 'DFD']
    draw_figure('', dises, params, legends, x='', y='mIoU',
                name='cf2f.png')


def analyze_fs_c():
    legends = ['raw', 'c23', 'c40']
    raw = [0.646, 0.792, 0.864, 0.790, 0.384]
    c23 = [0.646, 0.791, 0.864, 0.790, 0.384]
    c40 = [0.646, 0.789, 0.856, 0.788, 0.374]

    params = [raw, c23, c40]
    dises = ['DF', 'F2F', 'FS', 'NT', 'DFD']
    draw_figure('', dises, params, legends, x='', y='mIoU',
                name='cfs.png')


def analyze_nt_c():
    legends = ['raw', 'c23', 'c40']
    raw = [0.685, 0.884, 0.743, 0.881, 0.348]
    c23 = [0.684, 0.883, 0.742, 0.880, 0.348]
    c40 = [0.680, 0.877, 0.736, 0.877, 0.332]

    params = [raw, c23, c40]
    dises = ['DF', 'F2F', 'FS', 'NT', 'DFD']
    draw_figure('', dises, params, legends, x='', y='mIoU',
                name='cnt.png')


def analyze_dfd_c():
    legends = ['raw', 'c23', 'c40']
    raw = [0.657, 0.469, 0.419, 0.503, 0.686]
    c23 = [0.660, 0.443, 0.443, 0.503, 0.686]
    c40 = [0.656, 0.512, 0.492, 0.514, 0.686]

    params = [raw, c23, c40]
    dises = ['DF', 'F2F', 'FS', 'NT', 'DFD']
    draw_figure('', dises, params, legends, x='', y='mIoU',
                name='cdfd.png')


def analyze_vi():
    legends = ['Ours', 'FAST', 'VIDNet']
    Ours = [0.65, 0.64, 0.75]
    FAST = [0.63, 0.32, 0.22]
    VIDNet = [0.57, 0.39, 0.25]

    params = [Ours, FAST, VIDNet]
    dises = ['DVI', 'OPN', 'CPNET']
    draw_figure('', dises, params, legends, x='', y='mIoU',
                name='vi.png')


def analyze_c():
    legends = ['raw', 'c23', 'c40']
    raw = [0.867, 0.885, 0.864,  0.881]
    c23 = [0.866, 0.884, 0.864, 0.880]
    c40 = [0.866, 0.871, 0.856, 0.877]

    params = [raw, c23, c40]
    dises = ['DF', 'F2F', 'FS', 'NT']
    draw_figure('', dises, params, legends, x='', y='mIoU',
                name='c.png')


if __name__ == '__main__':
    analyze_vi()
