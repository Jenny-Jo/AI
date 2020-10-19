import shutil
import os
import numpy as np
from tkinter import Tcl
import argparse

parser = argparse.ArgumentParser(description='make data_list_file [Train / test]')
parser.add_argument('-p', '--path', default='./gender', type=str, help='dataset 폴더 경로')
parser.add_argument('-r', '--ratio', default=0.2, type=float, help='전체 이미지의 몇 퍼센트를 test 이미지로 사용할 것인가 ')
args = parser.parse_args()

path = args.path

category = os.listdir(path)
print(len(category))

min_n = 0
max_n = 0
for cate in category:
    images = os.listdir(os.path.join(path, cate))

    if min_n == 0:
        min_n = len(images)
    max_n = max([max_n, len(images)])
    min_n = min([min_n, len(images)])

print('max :', max_n)
print('min :', min_n)

val_size = int(min_n * args.ratio)
print(val_size)
if val_size == 0:
    val_size = 1

for cate in category:
    os.makedirs(os.path.join(path, 'dataset', 'train', cate))
    os.makedirs(os.path.join(path, 'dataset', 'test', cate))

    # print(os.path.join(path, cate))

    images = os.listdir(os.path.join(path, cate))
    images = Tcl().call('lsort', '-dict', images)  # sort

    for image in images[:-val_size]:
        shutil.copy(os.path.join(path, cate, image), os.path.join(path, 'dataset', 'train', cate, image))
    for image in images[-val_size:]:
        shutil.copy(os.path.join(path, cate, image), os.path.join(path, 'dataset', 'test', cate, image))

print('------Done------')