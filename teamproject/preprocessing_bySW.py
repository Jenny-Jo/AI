import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm
from glob import glob
from sklearn.model_selection import train_test_split
import csv

#Load-------------------------------------------------------------------------------------------------------------------
def load_images_from_folder(folder):
   images=[]
   for filename in tqdm(folder):
       img=cv2.imread(filename,cv2.IMREAD_COLOR) #컬러사진을 이용한다.
       img=img[15:200,30:300] #이미지를 얼굴부분만 나오게 잘라준다.
       images.append(img)
   images=np.array(images)
   return images


#특정데이터 추출
a=[]
def load(t):
   for root, dirs, files in os.walk(t):
       for fname in dirs:
           full_dirs = os.path.join(root, fname)

           full = glob('{}/*.jpg'.format(full_dirs))
           if full != []:

               T=full
               try:
                   u = load_images_from_folder(T)
                   a.append(u)
               except:
                   print(T)

   u=np.asarray(a)
   return u

x=load('teamproject/test/data/E02')
#저장한다.
np.save('teamproject/test/data/E02FacesFeature.npy',arr=x)
