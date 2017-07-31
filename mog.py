# coding=utf-8
"""
Created on 上午11:29 17-7-31
高斯混合模型--MOG
@author: zyl
"""
import numpy as np
import cv2

cap = cv2.VideoCapture('video/a.mp4')
mog = cv2.createBackgroundSubtractorMOG2()

while(True):
    ret, frame = cap.read()
    fgmask = mog.apply(frame)

    cv2.imshow('frame', fgmask)
    cv2.imwrite('result/background_knn.jpg', fgmask)
    if cv2.waitKey(1000/12) & 0xff:
        break
cap.release()
cv2.destroyAllWindows()