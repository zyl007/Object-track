# coding=utf-8
"""
Created on 上午11:41 17-7-31

@author: zyl
"""
import cv2
import numpy as np

bs = cv2.createBackgroundSubtractorKNN(detectShadows=True)
camera = cv2.VideoCapture('video/a.mp4')

while True:
    ret, frame = camera.read()
    fgmask = bs.apply(frame)
    th = cv2.threshold(fgmask.copy(), 244, 255, cv2.THRESH_BINARY)[1]
    dilated = cv2.dilate(th,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)),
                         iterations=2)
    # 获取目标图像轮廓
    image, contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) > 1600:
            (x,y,w,h)=cv2.boundingRect(c)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,255,0), 2)
    cv2.imshow('fgmask', fgmask)
    cv2.imshow('thresh', th)
    cv2.imshow('detection', frame)
    if cv2.waitKey(1000/12) & 0xff==ord('q'):
        break
camera.release()
cv2.destroyAllWindows()