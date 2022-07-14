import os
import sys

import cv2

if __name__ == '__main__':
    path = sys.argv[1]
    output = sys.argv[2]
    mp4_list = []
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.mp4'):
                video = os.path.join(root, file)
                cap = cv2.VideoCapture(video)
                if cap.isOpened():  # 当成功打开视频时cap.isOpened()返回True,否则返回False
                    rate = cap.get(cv2.CAP_PROP_FPS)  # 帧速率
                    FrameNumber = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # 视频文件的帧数
                    if FrameNumber < 1000:
                        count += 1
                        cmd = f'cp {video} {output}/'
                        os.system(cmd)
                    else:
                        duration = FrameNumber / rate
                        for i in range(int(duration // 15 + 1)):
                            count += 1
                            seconds = '%02d' % int(i * 15 % 60)
                            minutes = '%02d' % int(i * 15 / 60)
                            cmd = f'ffmpeg -ss 00:{minutes}:{seconds} -t 15 -i {video} {output}/{count}.mp4'
                            os.system(cmd)


def rm_frames_less4(path):
    for dir_ in os.listdir(path):
        dir_ = os.path.join(path, dir_)
        if len(os.listdir(dir_)) < 4:
            cmd = f'rm -rf {dir_}'
            print(cmd)
            os.system(cmd)
