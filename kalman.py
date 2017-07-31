# coding=utf-8
"""
Created on 下午4:25 17-7-31

@author: zyl
"""
import cv2
import numpy as np

measurements = []
predictions = []

frame = np.zeros((1000, 1000, 3), np.uint8)
last_measurement = current_measurement = np.array((2,1), np.float32)
last_prediction = current_prediction = np.zeros((2,1), np.float32)

def mouse_move(event, x, y, s, p):
    global frame, current_measurement, measurements, last_measurement, last_prediction, current_prediction

    last_prediction = current_prediction
    last_measurement = current_measurement
    current_measurement = np.array([[np.float32(x)], [np.float32(y)]])
    kalman.correct(current_measurement)
    current_prediction = kalman.predict()

    lmx, lmy = last_measurement[0], last_measurement[1]
    cmx, cmy = current_measurement[0], current_measurement[1]
    lpx, lpy = last_prediction[0], last_prediction[1]
    cpx, cpy = current_prediction[0], current_prediction[1]
    # BGR
    cv2.line(frame, (lmx, lmy), (cmx, cmy), (0,100,0)) # 绿色 测量值
    cv2.line(frame, (lpx, lpy), (cpx, cpy), (0,0,200)) # 红色 预测值

cv2.namedWindow("kalman_tracker")
cv2.setMouseCallback("kalman_tracker", mouse_move)

kalman = cv2.KalmanFilter(4,2,1)
kalman.measurementMatrix = np.array([[1,0,0,0], [0,1,0,0]], np.float32)
kalman.transitionMatrix = np.array([[1,0,1,0], [0,1,0,1], [0,0,1,0], [0,0,0,1]], np.float32)
kalman.processNoiseCov = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]], np.float32)

while True:
    cv2.imshow("kalman_tracker", frame)
    if (cv2.waitKey(30)&0xff) == 27:
        break
    if (cv2.waitKey(30)&0xff) == ord('q'):
        cv2.imwrite("kalman.jpg", frame)
        break
cv2.destroyAllWindows()
