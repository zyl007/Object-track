# coding=utf-8
"""
Created on 下午3:53 17-7-31

@author: zyl
"""
import numpy as np
import cv2

cap = cv2.VideoCapture('video/a.mp4')
# take first frame
ret, frame  = cap.read()

# 窗口初始化设置
r,h,c,w = 300, 200, 400, 300
track_window = (c, r, w, h)

roi = frame[r:r+h, c:c+w]
hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((100., 30., 32.)), np.array((180., 120., 255.)))
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while 1:
    ret, frame = cap.read()
    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0,180], 1)
        # CamShift 调用
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        img2 = cv2.polylines(frame, [pts], True, 255, 2)
        #  调用完毕

        cv2.imshow("img2", img2)
        k = cv2.waitKey(60) & 0xff
        if k==27:
            break
    else:
        break

cv2.destroyAllWindows()
cap.release()