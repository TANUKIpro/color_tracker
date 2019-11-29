# -*- coding: utf-8 -*-
#http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html

import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from scipy.optimize import curve_fit

#+------trackbar--------+#
t_init = True            #
#+----------------------+#
#+-----MedianBlur-------+#
MB = True                #
#+----------------------+#
#+-----fill_holes-------+#
#fill_holes = False       #
#+----------------------+#
#+-------opening--------+#
opening = True           #
#+----------------------+#
#+-------closing--------+#
closing = True           #
#+----------------------+#
#+----特定色の消去------+#
ColorErase = False        #
#+----------------------+#

#動画ファイルのパス
videofile_path = "20191122/nihongi_n_n1.mp4"

#5x5のカーネル
kernel = np.ones((8,8),np.uint8)
'''
kernel = np.array([[1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1]])
'''
#トラックバーの初期設定
def callback(x):
    pass

def Trackbars_init():
    cv2.namedWindow('value')
    cv2.createTrackbar('H_Hue','value',179,179,callback)
    cv2.createTrackbar('L_Hue','value',0,179,callback)

    cv2.createTrackbar('H_Saturation','value',255,255,callback)
    cv2.createTrackbar('L_Saturation','value',0,255,callback)

    cv2.createTrackbar('H_Value','value',255,255,callback)
    cv2.createTrackbar('L_Value','value',0,255,callback)

def color_detect(hsv, img):
    hsv_min = np.array([15,127,0])
    hsv_max = np.array([240,255,255])
    mask = cv2.inRange(hsv, hsv_min, hsv_max)

    return mask

#ブロブ解析
def analysis_blob(binary_img):
    label = cv2.connectedComponentsWithStats(binary_img)
    n = label[0] - 1
    data = np.delete(label[2], 0, 0)
    center = np.delete(label[3], 0, 0)
    maxblob = {}
    if len(data[:, 4]) is 0:
        max_index = None
        maxblob["center"] = [0, 0]
    else:
        max_index = np.argmax(data[:, 4])

        maxblob["upper_left"] = (data[:, 0][max_index], data[:, 1][max_index]) # 左上座標
        maxblob["width"] = data[:, 2][max_index]
        maxblob["height"] = data[:, 3][max_index]
        maxblob["area"] = data[:, 4][max_index]
        maxblob["center"] = center[max_index]

    return data, center, maxblob

#データ出力
def data_plot(data):
    data_np = np.array(data)
    t = data_np[:,0]
    x = data_np[:,1]
    y = data_np[:,2]

    print(t, x, y)

    plt.rcParams["font.family"] = "Times New Roman"
    plt.plot(t, x, "r-", label="x")
    plt.plot(t, y, "b-", label="y")
    plt.xlabel("Frame [num]", fontsize=16)
    plt.ylabel("Position[px]", fontsize=16)
    plt.grid()

    plt.legend(loc=1, fontsize=16)
    plt.show()

#トラックバーからコールバックを受け取り、値を返す
def trackbars():
    lowH = cv2.getTrackbarPos('L_Hue', 'value')
    highH = cv2.getTrackbarPos('H_Hue', 'value')

    lowS = cv2.getTrackbarPos('L_Saturation', 'value')
    highS = cv2.getTrackbarPos('H_Saturation', 'value')

    lowV = cv2.getTrackbarPos('L_Value', 'value')
    highV = cv2.getTrackbarPos('H_Value', 'value')

    return (lowH, lowS, lowV, highH, highS, highV)
#======================================================================
def median_blar(image, size):
    _image = cv2.medianBlur(image, size)
    return _image

def Fill_Holes(image):
    _image = ndimage.binary_fill_holes(image).astype(int) * 255
    return _image

def Opening(image):
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return opening

def Closing(image):
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return closing

def color_eraser(image, RGBarray):
    pass
#======================================================================

def resize_image(img, dsize, X, Y):
    re_image = cv2.resize(img, dsize, fx=X, fy=Y)
    return re_image

#メイン
def main():
    data = []
    cap = cv2.VideoCapture(videofile_path)
    start = time.time()
    if t_init is True:
        Trackbars_init()
    n = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if frame is None:
            print("frame is None")
            break
        #画像の認識範囲を狭めるためのトリミング操作
        h, w = frame.shape[:2]
        frame = frame[80:h, 0:w]

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        #"""
        Lh, Ls, Lv = trackbars()[:3]
        Hh, Hs, Hv = trackbars()[3:]
        """
        Lh, Ls, Lv = (40, 40, 109)
        Hh, Hs, Hv = (121, 255, 255)
        """
        hsv_min = np.array([Lh,Ls,Lv])
        hsv_max = np.array([Hh,Hs,Hv])

        #print( "H:{0} - {1}\nS:{2} - {3}\nV:{4} - {5}\n-------------"
            #.format(Lh, Hh, Ls, Hs, Lv, Hv))

        mask = cv2.inRange(hsv, hsv_min, hsv_max)

        if MB is True: mask = median_blar(mask, 3)
        #if fill_holes is True: mask = Fill_Holes(mask)
        if opening is True: mask = Opening(mask)
        if closing is True: mask = Closing(mask)
        if ColorErase is True: mask = color_eraser(mask, None)

        _, center, maxblob = analysis_blob(mask)
        #print("targetnum:",len(center))
        for i in center:
            cv2.circle(frame, (int(i[0]), int(i[1])), 10, (255, 0, 0),
                    thickness=-3, lineType=cv2.LINE_AA)

        center_x = int(maxblob["center"][0])
        center_y = int(maxblob["center"][1])

        #print(center_x, center_y)

        cv2.circle(frame, (center_x, center_y), 30, (0, 200, 0),
                  thickness=3, lineType=cv2.LINE_AA)

        #data.append([time.time() - start, center_x, center_y])
        data.append([n, center_x, center_y])
        #print(time.time() - start, center_x, center_y)

        re_frame=resize_image(frame, None, .4, .4)
        cv2.imshow("Frame", re_frame)
        mask = resize_image(mask, None, .4, .4)
        cv2.imshow("image_mask", mask)
        n += 1
        print(n)
        print("----------")

        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    data_plot(data)

if __name__ == '__main__':
    main()
