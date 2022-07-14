import codecs
import csv
import json
import os
import sys



def map2csv(map_, csv_path):
    csv_file = open(csv_path, 'w', encoding='utf-8')
    for k, v in map_.items():
        csv_file.write('{}\t{}\n'.format(k, v))
    csv_file.close()


def mv_file(filename, file_dir, outpath):
    try:
        file = open(filename)
        while 1:
            line = file.readline()
            if not line:
                break
            item = line.split(",")[0]
            idx = line.split(",")[1].replace('\n', '').replace('\r', '')
            out_file = os.path.join(outpath, str(idx))
            src_file = os.path.join(file_dir, item)
            if not os.path.exists(out_file):
                os.makedirs(out_file)
            cmd = 'mv {} {}'.format(src_file, out_file)
            print(cmd)
            os.system(cmd)
        file.close()
    except IOError as err:
        print("I/O error({0})".format(err))


def fill(filename, file_dir, outpath):
    result_map = {}
    file = open('f3net.csv')
    while 1:
        line = file.readline()
        if not line:
            break
        item = line.split("\t")[0]
        idx = line.split("\t")[1]
        result_map[item] = idx
    file.close()
    for fid in os.listdir('/workspace/data/data2/testData'):
        if not result_map.__contains__(fid):
            result_map[fid] = 0
    csv_file = open('result2.csv', 'a',newline='', encoding='utf-8')
    for k, v in result_map.items():
        csv_file.write('{}\t{}\n'.format(k, v))
    csv_file.close()


if __name__ == '__main__':
    mv_file(sys.argv[1], sys.argv[2], sys.argv[3])
