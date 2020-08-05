import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

def aidemy_imshow(name, img):
    b,g,r = cv2.split(img)
    img = cv2.merge([r,g,b])
    plt.imshow(img)
    plt.show()

cv2.imshow = aidemy_imshow

img = cv2.imread('teamproject/test/E01/C7.jpg')
cv2.imshow('sample',img)

print('end')