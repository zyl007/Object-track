# coding=utf-8
"""
Created on 下午5:36 17-7-31
为每一个行人创建一个kalman滤波器进行跟踪
@author: zyl
"""
import cv2
import numpy as np
import os.path as path
import argparse

font = cv2.FONT_HERSHEY_SIMPLEX
# 设置命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("-a", "--algorithm",
    help = "m (or nothing) for meanShift and c for camshift")
args = vars(parser.parse_args())

# 计算中心点
def center(points):
    """calculates centroid of a given matrix"""
    x = (points[0][0] + points[1][0] + points[2][0] + points[3][0]) / 4
    y = (points[0][1] + points[1][1] + points[2][1] + points[3][1]) / 4
    return np.array([np.float32(x), np.float32(y)], np.float32)


class Pedestrian():
    """
    每一个Pedestrian 包含ROI ID Kalman filter
    """
    def __init__(self, id, frame, track_window):
        """
        利用跟踪窗口坐标初始化行人对象
        :param id:
        :param frame:
        :param track_window:
        """
        # 设置ROI
        self.id = int(id)
        x,y,w,h = track_window
        self.track_window = track_window
        self.roi = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2HSV)
        roi_hist = cv2.calcHist([self.roi], [0], None, [16], [0, 180])
        self.roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

        # 设置kalman
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1,0,0,0], [0,1,0,0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1,0,1,0], [0,1,0,1], [0,0,1,0], [0,0,0,1]])
        self.kalman.processNoiseCov = np.array([[1,0,0,0], [0,1,0,1], [0,0,1,0], [0,0,0,1]], np.float32)*0.03

        # 记录位置
        self.measurement = np.array((2,1), np.float32)
        self.prediction = np.array((2,1), np.float32)
        self.term_crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10, 1)
        self.center = None
        self.update(frame)

    def __del__(self):
        print("Pedestrian %d destroyed"%self.id)

    def update(self, frame):
        print("updating %d " % self.id)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        back_project = cv2.calcBackProject([hsv], [0], self.roi_hist, [0, 180], 1)

        # 使用meanShift or CAMShift 跟踪行人， kalman进行修正
        if args.get('algorithm') == "c":
            ret, self.track_window = cv2.CamShift(back_project, self.track_window, self.term_crit)
            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)
            self.center = center(pts)
            cv2.polylines(frame, [pts], True, 255, 1)

        if not args.get('algorithm') or args.get("algorithm") == 'm':
            ret, self.track_window = cv2.meanShift(back_project, self.track_window, self.term_crit)
            x,y,w,h = self.track_window
            self.center = center([[x,y], [x+w, y], [x, y+h], [x+w, y+h]])
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,255,0), 1)
        # kalman修正，并预测
        self.kalman.correct(self.center)
        predict = self.kalman.predict()
        cv2.circle(frame, (int(predict[0]), int(predict[1])), 4, (0,255,0), -1)

        # fake shadow
        cv2.putText(frame, "ID: %d -> %s "%(self.id, self.center),(11, (self.id + 1)*25 + 1),
                    font, 0.6, (0,0,0),1,cv2.LINE_AA)
        # actual info
        cv2.putText(frame, "ID: %d -> %s"%(self.id, self.center), (10, (self.id + 1)*25),
                    font, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

def main():
    # camera = cv2.VideoCapture(path.join(path.dirname(__file__), "traffic.flv"))
    camera = cv2.VideoCapture("/home/knight/PycharmProjects/Object-track/video/768x576.avi")
    # camera = cv2.VideoCapture(path.join(path.dirname(__file__), "..", "movie.mpg"))
    # camera = cv2.VideoCapture(0)
    history = 20
    # KNN background subtractor
    bs = cv2.createBackgroundSubtractorKNN(detectShadows=True)
    bs.setHistory(history)

    # MOG subtractor
    # bs = cv2.bgsegm.createBackgroundSubtractorMOG(history = history)
    # bs.setHistory(history)

    # GMG
    # bs = cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames = history)

    cv2.namedWindow("surveillance")
    pedestrians = {}
    firstFrame = True
    frames = 0
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
    while True:
        print(" -------------------- FRAME %d --------------------" % frames)
        grabbed, frame = camera.read()
        if (grabbed is False):
            print "failed to grab frame."
            break

        fgmask = bs.apply(frame)

        # this is just to let the background subtractor build a bit of history
        if frames < history:
            frames += 1
            continue

        th = cv2.threshold(fgmask.copy(), 127, 255, cv2.THRESH_BINARY)[1]
        th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
        dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 3)), iterations=2)
        image, contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        counter = 0
        for c in contours:
            if cv2.contourArea(c) > 500:
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                # only create pedestrians in the first frame, then just follow the ones you have
                if firstFrame is True:
                    pedestrians[counter] = Pedestrian(counter, frame, (x, y, w, h))
                counter += 1

        for i, p in pedestrians.iteritems():
            p.update(frame)

        firstFrame = False
        frames += 1

        cv2.imshow("surveillance", frame)
        out.write(frame)
        if cv2.waitKey(110) & 0xff == 27:
            break
    out.release()
    camera.release()

if __name__ == '__main__':
    main()