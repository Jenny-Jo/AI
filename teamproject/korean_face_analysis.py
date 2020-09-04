import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm
from glob import glob
from sklearn.model_selection import train_test_split
import csv
def load_images_from_folder(filename):
    img = cv2.imread(filename, cv2.IMREAD_COLOR)  # 컬러사진을 이용한다.
    img = img[10:120, 20:150]  # 이미지를 얼굴부분만 나오게 잘라준다.
    return img
def search(dirname):
    filenames = os.listdir(dirname)
    b=[]
    v=[]
    n=[]
    x=[]
    Image = []
    for i in filenames:
        if i == 'S001':
            x.append(os.path.join('{}'.format(dirname), i))
        else:
            continue
        for dirname2 in x:
            l = os.listdir('{}'.format(dirname2))
            for i in l:
                if i == 'L1':
                    b.append(os.path.join('{}'.format(dirname2), i))
                elif i == 'L2':
                    b.append(os.path.join('{}'.format(dirname2), i))
                # elif i == 'L3':
                #     b.append(os.path.join('{}'.format(dirname2), i))
                # elif i == 'L4':
                #     b.append(os.path.join('{}'.format(dirname2), i))
                else:
                    continue
                for dirname3 in b:
                    e = os.listdir('{}'.format(dirname3))
                    for i in e:
                        if i == 'E01':
                            v.append(os.path.join('{}'.format(dirname3), i))
                        elif i == 'E02':
                            v.append(os.path.join('{}'.format(dirname3), i))
                        else:
                            continue
                        for dirname4 in v:
                            c = os.listdir('{}'.format(dirname4))
                            for i in c:
                                if i == 'C7.jpg':
                                    n.append(os.path.join('{}'.format(dirname4), i))
                                # elif i == 'C6.jpg':
                                #     n.append(os.path.join('{}'.format(dirname4), i))
                                # elif i == 'C8.jpg':
                                #     n.append(os.path.join('{}'.format(dirname4), i))
                                else:
                                    continue
                                for i in tqdm(n):
                                    I = load_images_from_folder(i)

                                    Image.append(I)

    f = np.array(Image)
    return np.save('teamproject/save/Image2.npy', arr=f)

search('C:/Users/bitcamp/Downloads/Middle_Resolution/19062421')

