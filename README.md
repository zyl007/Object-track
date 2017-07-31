# 关于

+ 运动检测(`basic_motion_detection.py`)
+ 目标检测(`background_knn.py`, `mog.py`)
+ 目标跟踪(`CAMShift.py`, `meanShift.py`,`kalman.py`)
+ 一个基于行人跟踪的demo
    - 检查第一帧
    - 检查后面的输入帧，从场景的开始通过背景分割器来识别场景中的行人
    - 为每个行人建立ROI，并利用Kalman/CAMShift来跟踪行人ID
    - 检查下一帧是否有进入场景的新行人


# 参考资料

+ 《opencv3计算机视觉：Python语言实现》
+ https://zh.wikipedia.org/wiki/%E5%8D%A1%E5%B0%94%E6%9B%BC%E6%BB%A4%E6%B3%A2
+ https://github.com/techfort/pycv/