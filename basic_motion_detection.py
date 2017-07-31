# coding=utf-8
"""
Created on 上午10:40 17-7-31
打开默认摄像头获取视频图像， 比较每一帧图像和背景的差异 捕获运动行为
@author: zyl
"""
import cv2
import numpy as np


camera = cv2.VideoCapture("video/a.mp4")

es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,4))
kernel = np.ones((5,5), np.uint8)
background = None

# 写入检测视频
fps = camera.get(cv2.CAP_PROP_FPS)
size = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
videoWriter = cv2.VideoWriter('result/basic_motion_detection.avi',
                              cv2.VideoWriter_fourcc('I', '4', '2', '0'),
                              fps, size)

while (True):
    ret, frame = camera.read()
    # 第一帧作为背景
    if background is None:
        background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        background = cv2.GaussianBlur(background, (21,21), 0)
        continue

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (21,21), 0)

    # 比较每一帧图像与背景的差值
    print(background.shape, gray_frame.shape)
    diff = cv2.absdiff(background, gray_frame)
    diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
    diff = cv2.dilate(diff, es, iterations=2)
    # 计算图像中目标的轮廓
    image, cnts, hierarchy = cv2.findContours(diff.copy(),
                                              cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
        if cv2.contourArea(c) < 1500:
            continue
        # 计算边界框
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow("contours", frame)
    videoWriter.write(frame)
    cv2.imshow("dif", diff)
    # cv2.imwrite('didff.jpg', diff)
    if cv2.waitKey(1000/12) &0xff == ord('q'):
        break
cv2.destroyAllWindows()
camera.release()