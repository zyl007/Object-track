# coding=utf-8
"""
Created on 下午2:12 17-7-31
均值漂移,利用颜色信息
@author: zyl
"""

import numpy as np
import cv2

cap = cv2.VideoCapture('video/a.mp4')
ret, frame = cap.read()
# 划定感兴趣的区域ROI
r,h,c,w = 500, 300, 1100, 200
track_window = (c,r,w,h)

roi = frame[r:r+h, c:c+w]
# 获取HSV(色调，饱和度，亮度)
hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((100., 30., 32.)),
                   np.array((180., 120., 255.)))
# 计算图像的彩色直方图，x是色彩值， y是色彩值的像素数量
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
# 归一化
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
# 均值漂移的停止条件 迭代10次或中心点漂移一次
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    ret, frame = cap.read()

    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # 图像中每个像素点属于原图的概率
        dst = cv2.calcBackProject([hsv],[0], roi_hist, [0, 180], 1)

        # meanShift
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        # draw it on image
        x,y,w,h = track_window
        img2 = cv2.rectangle(frame, (x,y), (x+w, y+h), 255, 2)
        cv2.imshow("img2", img2)
        k = cv2.waitKey(60) & 0xff
        if k==27:
            break
    else:
        break
cv2.destroyAllWindows()
cap.release()