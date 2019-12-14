# -*- coding: utf-8 -*-

import sys
try:
    py_path = sys.path
    ros_CVpath = '/opt/ros/kinetic/lib/python2.7/dist-packages'
    if py_path[3] == ros_CVpath:
        print("INFO : ROS and OpenCV are competing")
        sys.path.remove(py_path[3])
except: pass

import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
try:
    import scipy.ndimage as ndimage
    from scipy.optimize import curve_fit
except: pass

#setup global
image_frag = True

class Mouse:
    def __init__(self, window_name, cp_image, org_image):
        self.cp_image     = cp_image
        self.org_image    = org_image
        self.event_call   = 0
        self.ClickedPoint = [None, None, None, None]
        
        self.mouseEvent   = {"x":None, "y":None, "event":None, "flags":None}
        #cv2.setMouseCallback(window_name, self.__NORMALCallBack, None)
        cv2.setMouseCallback(window_name, self.__CallBack, None)

    def __NORMALCallBack(self, eventType, x, y, flags, userdata):
        self.mouseEvent["x"] = x
        self.mouseEvent["y"] = y
        self.mouseEvent["event"] = eventType
        self.mouseEvent["flags"] = flags
    
    def __CallBack(self, eventType, x, y, flags, userdata):
        self.mouseEvent["x"] = x
        self.mouseEvent["y"] = y
        self.mouseEvent["event"] = eventType
        self.mouseEvent["flags"] = flags

        if self.mouseEvent["event"] == cv2.EVENT_LBUTTONDOWN:
            self.event_call += 1
            M_x = self.mouseEvent["x"]
            M_y = self.mouseEvent["y"]
            print(M_x, M_y)
            global image_frag
            
            if self.event_call == 1:
                image_frag = True
                self.ClickedPoint[0] = M_x
                self.ClickedPoint[1] = M_y
                
                cv2.circle(self.cp_image, (self.ClickedPoint[0], self.ClickedPoint[1]), 5, (0, 0, 255), -1)
                cv2.putText(self.cp_image, "START", (self.ClickedPoint[0], self.ClickedPoint[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
            
            elif self.event_call == 2:
                self.ClickedPoint[2] = M_x
                self.ClickedPoint[3] = M_y
                
                cv2.circle(self.cp_image, (self.ClickedPoint[2], self.ClickedPoint[3]), 5, (0, 255, 0), -1)
                cv2.putText(self.cp_image, "GOAL", (self.ClickedPoint[2], self.ClickedPoint[3]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
                
                #START --> GOAL までの線を引っ張る
                cv2.line(self.cp_image, (self.ClickedPoint[0], self.ClickedPoint[1]), (self.ClickedPoint[2], self.ClickedPoint[3]),(255, 0, 0), 2)
                
            elif self.event_call >= 2:
                image_frag = False
                self.event_call = 0
                self.cp_image = self.org_image
                print(self.ClickedPoint)
                self.ClickedPoint = [None, None, None, None]
                
    def getData(self):
        return self.mouseEvent

    def getEvent(self):
        return self.mouseEvent["event"]

    def getFlags(self):
        return self.mouseEvent["flags"]

    def posXY(self):
        x = self.mouseEvent["x"]
        y = self.mouseEvent["y"]
        return (x, y)


class HSV_supporter:
    def __init__(self):
        self.t_init     = False
        self.MB         = True
        self.fill_holes = False #is not working
        self.opening    = True
        self.closing    = True
        self.ColorErase = False

        self.kernel = np.ones((8,8),np.uint8)
        
        self.window_name0 = "Serection"
        self.window_name1 = "Frame"

    #トラックバーの初期設定
    def TBcallback(self, x):
        pass

    def Trackbars_init(self):
        cv2.namedWindow('value')
        cv2.createTrackbar('H_Hue','value',179,179,self.TBcallback)
        cv2.createTrackbar('L_Hue','value',0,179,self.TBcallback)

        cv2.createTrackbar('H_Saturation','value',255,255,self.TBcallback)
        cv2.createTrackbar('L_Saturation','value',0,255,self.TBcallback)

        cv2.createTrackbar('H_Value','value',255,255,self.TBcallback)
        cv2.createTrackbar('L_Value','value',0,255,self.TBcallback)

    def color_detect(self, hsv, img):
        hsv_min = np.array([15,127,0])
        hsv_max = np.array([240,255,255])
        mask = cv2.inRange(hsv, hsv_min, hsv_max)

        return mask

    #ブロブ解析
    def analysis_blob(self, binary_img):
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

            maxblob["upper_left"] = (data[:, 0][max_index], data[:, 1][max_index])
            maxblob["width"] = data[:, 2][max_index]
            maxblob["height"] = data[:, 3][max_index]
            maxblob["area"] = data[:, 4][max_index]
            maxblob["center"] = center[max_index]

        return data, center, maxblob

    #データ出力
    def data_plot(self, data):
        data_np = np.array(data)
        if len(data_np) <= 0:
            print("too many indices for array")
            f = 0
            x = 0
            y = 0
        else:
            f = data_np[:,0]
            x = data_np[:,1]
            y = data_np[:,2]

        #xの勾配dxを求める
        dx = np.gradient(x)

        plt.rcParams["font.family"] = "Times New Roman"

        #1つ目のグラフ描画
        plt.subplot(1,2,1)
        plt.plot(f, x, "r-", label="x")
        plt.plot(f, y, "b-", label="y")
        plt.xlabel("Frame [num]", fontsize=16)
        plt.ylabel("Position[px]", fontsize=16)
        plt.grid(True)

        #2つ目のグラフ描画
        plt.subplot(1,2,2)
        plt.plot(f, dx, "g-", label="dx")
        plt.xlabel("Frame [num]", fontsize=16)
        plt.ylabel("Position[px]", fontsize=16)
        plt.grid(True)

        plt.legend(loc=1, fontsize=16)
        plt.show()

    #トラックバーからコールバックを受け取り、値を返す
    def trackbars(self):
        lowH = cv2.getTrackbarPos('L_Hue', 'value')
        highH = cv2.getTrackbarPos('H_Hue', 'value')

        lowS = cv2.getTrackbarPos('L_Saturation', 'value')
        highS = cv2.getTrackbarPos('H_Saturation', 'value')

        lowV = cv2.getTrackbarPos('L_Value', 'value')
        highV = cv2.getTrackbarPos('H_Value', 'value')

        return (lowH, lowS, lowV, highH, highS, highV)
    #======================================================================
    def median_blar(self, image, size):
        _image = cv2.medianBlur(image, size)
        return _image

    def Fill_Holes(self, image):
        _image = ndimage.binary_fill_holes(image).astype(int) * 255
        return _image

    def Opening(self, image):
        opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, self.kernel)
        return opening

    def Closing(self, image):
        closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, self.kernel)
        return closing

    def color_eraser(self, image, RGBarray):
        pass
    #======================================================================

    def resize_image(self, img, dsize, X, Y):
        re_image = cv2.resize(img, dsize, fx=X, fy=Y)
        return re_image

    #メイン
    def main(self, videofile_path):
        n = 0
        data = []
        cap = cv2.VideoCapture(videofile_path)

        if self.t_init is True:
            self.Trackbars_init()

        if cap.isOpened():
            print("INFO : The Video loaded successfully.")
        else:
            print("INFO : LOAD ERROR ***Chack video path or name.***")
            print("CAP : ", cap)
            exit()
        
        #最初に1フレームだけ表示して、STARTとGOALを選択する
        print("")
        print("[INFO] : This is the 1st frame.\n[INFO] : Choose start and goal positions and Click.")
        
        ret, frame0 = cap.read()
        cp_frame0 = frame0.copy()
        
        cv2.namedWindow(self.window_name0)
        mouse = Mouse(self.window_name0, cp_frame0, frame0)
        
        while True:
            if image_frag is not True:
                cp_frame0 = frame0
                print("change image")
                
            cv2.imshow(self.window_name0, cp_frame0)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        
        #メインのループ
        while(cap.isOpened()):
            ret, frame = cap.read()
            if frame is None:
                print("frame is None")
                break

            #画像の認識範囲を狭めるためのトリミング操作
            """
            h, w = frame.shape[:2]
            frame = frame[80:h, 0:w]
            """

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            #hsv決め打ちの時はココを編集
            if self.t_init is True:
                Lh, Ls, Lv = self.trackbars()[:3]
                Hh, Hs, Hv = self.trackbars()[3:]
            else:
                #青
                Lh, Ls, Lv = (40, 40, 109)
                Hh, Hs, Hv = (121, 255, 255)

            hsv_min = np.array([Lh,Ls,Lv])
            hsv_max = np.array([Hh,Hs,Hv])

            """
            print( "H:{0} - {1}\nS:{2} - {3}\nV:{4} - {5}\n-------------"
                .format(Lh, Hh, Ls, Hs, Lv, Hv))
            """

            mask = cv2.inRange(hsv, hsv_min, hsv_max)
            
            #ノイズ除去のためのアルゴリズム判定(True/False)
            if self.MB is True: mask = self.median_blar(mask, 3)
            try:
                if self.fill_holes is True: mask = self.Fill_Holes(mask)
            except: print("INFO : fill_holes is not working")
            if self.opening is True: mask = self.Opening(mask)
            if self.closing is True: mask = self.Closing(mask)
            if self.ColorErase is True: mask = self.color_eraser(mask, None)

            _, center, maxblob = self.analysis_blob(mask)
            #print("target num:",len(center))
            for i in center:
                cv2.circle(frame, (int(i[0]), int(i[1])), 10, (255, 0, 0),
                            thickness=-3, lineType=cv2.LINE_AA)

            center_x = int(maxblob["center"][0])
            center_y = int(maxblob["center"][1])

            print(center_x, center_y)

            cv2.circle(frame, (center_x, center_y), 30, (0, 200, 0),
                      thickness=3, lineType=cv2.LINE_AA)

            data.append([n, center_x, center_y])
            
            re_frame=self.resize_image(frame, None, .5, .5)
            cv2.imshow("Frame", re_frame)
            
            mask = self.resize_image(mask, None, .5, .5)
            cv2.imshow("mask image", mask)
            
            n += 1
            
            print("----------")

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        self.data_plot(data)

if __name__ == '__main__':

    video = "20191122/nihongi_f_l1.mp4"
    if len(sys.argv) < 2:
        videofile_path = video
    else:
        videofile_path = sys.argv[1]

    hsv_sup = HSV_supporter()
    hsv_sup.main(videofile_path)
    
