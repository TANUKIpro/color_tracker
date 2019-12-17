# -*- coding: utf-8 -*-

import sys
try:
    py_path = sys.path
    ros_CVpath = '/opt/ros/kinetic/lib/python2.7/dist-packages'
    if py_path[3] == ros_CVpath:
        print("[INFO] : ROS and OpenCV are competing")
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

global S2G
S2G = int(460 + 477) #STARTからGOALまでの距離(cm)

"""
class FromPoint2point:
    def __init__(self, x, x1, x2, y1, y2)
        pass
    def P2P_calc(x, x1, x2, y1, y2):
        fx = (((y2 - y1) / x2 - x1) * (x - x1)) + y1
        return fx 
"""
class Mouse:
    def __init__(self, window_name, cp_image, org_image, HumanHeight):
        self.cp_image     = cp_image
        self.org_image    = org_image
        self.event_call   = 0
        self.ClickedPoint = [None, None, None, None]
        self.Prediction   = [None, None, None, None]
        
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
            global image_frag
            
            #1回目のクリック
            if self.event_call == 1:
                image_frag = True
                self.ClickedPoint[0] = M_x
                self.ClickedPoint[1] = M_y
                print(self.ClickedPoint[0], self.ClickedPoint[1])
                
                cv2.circle(self.cp_image, (self.ClickedPoint[0], self.ClickedPoint[1]), 5, (0, 0, 255), -1)
                cv2.putText(self.cp_image, "START", (self.ClickedPoint[0] - 40,
                                                     self.ClickedPoint[1] - 10),
                                                     cv2.FONT_HERSHEY_SIMPLEX, 1., (0, 0, 255), 2, cv2.LINE_AA)
            
            #2回目のクリック
            elif self.event_call == 2:
                self.ClickedPoint[2] = M_x
                self.ClickedPoint[3] = M_y
                print(self.ClickedPoint[2], self.ClickedPoint[3])
                
                cv2.circle(self.cp_image, (self.ClickedPoint[2], self.ClickedPoint[3]), 5, (0, 255, 0), -1)
                cv2.putText(self.cp_image, "GOAL", (self.ClickedPoint[2] - 40,
                                                    self.ClickedPoint[3] - 10),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 1., (0, 255, 0), 2, cv2.LINE_AA)
                
                #START --> GOAL までの線を引っ張る
                cv2.line(self.cp_image, (self.ClickedPoint[0], self.ClickedPoint[1]),
                                        (self.ClickedPoint[2], self.ClickedPoint[3]),(255, 0, 0), 2)
                
                #TODO トラッキング範囲を推定。被験者の身長から割り出す <-- この処理本当に必要ぉ??
                S_x = self.ClickedPoint[0]
                S_y = self.ClickedPoint[1]
                G_x = self.ClickedPoint[2]
                G_y = self.ClickedPoint[3]

                #後の処理で同じことする。書いたやつアホ
                #単位ピクセルあたりの実際の距離を算出して、Y軸に減算
                px_d = G_x - S_x
                K    = S2G / px_d
                S2H  = HumanHeight / K
                #描画
                HS_y = int(S_y - S2H)
                HG_y = int(G_y - S2H)
                cv2.line(self.cp_image, (S_x, HS_y), (G_x, HG_y),(70, 220, 140), 2)
                self.Prediction = [S_x, HS_y, G_x, HG_y]
                print(self.Prediction)
            
            #それ以降のクリック
            elif self.event_call > 2:
                print("Update Image")
                image_frag = False
                
                #クリック回数、イメージに描いた線、マウス座標のリセット
                self.event_call = 0
                self.cp_image = self.org_image
                self.ClickedPoint = [None, None, None, None]
                self.Prediction   = [None, None, None, None]
                
    def getData(self):
        return self.mouseEvent

    def getEvent(self):
        return self.mouseEvent["event"]

    def getFlags(self):
        return self.mouseEvent["flags"]

    def posXY(self):
        return (self.mouseEvent["x"], self.mouseEvent["y"])
    
    def clicked_point(self):
        return self.ClickedPoint
        
    def Prediction(self):
        return self.Prediction

class HSV_supporter:
    def __init__(self):
        self.t_init     = False
        self.MB         = True
        self.fill_holes = False #Not working
        self.opening    = True
        self.closing    = True
        self.ColorErase = False #Not working
        
        self.kernel = np.ones((8,8),np.uint8)
        
        self.center_x = 0
        self.center_y = 0
        
        self.window_name0 = "Serection"
        self.window_name1 = "Frame"
        
        self.K            = 0                     # 単位ピクセルあたりの実際の距離(cm)
        self.frame_num    = 0                     # 全体のフレーム数
        self.frame_rate   = 30                    # 入力動画のfps
        self.frame_time   = 1 / self.frame_rate   # 単位フレームあたりの時間
        self.N_frame_time = 0                     # Nフレーム時の時間
        self.velocity     = 0                     # 瞬間の速度

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
    def data_plot(self, data, VideoName):
        data_np = np.array(data)
        if len(data_np) <= 0:
            print("too many indices for array")
            f = 0
            x = 0
            y = 0
            t = 0
        else:
            f = data_np[:,0]
            x = data_np[:,1]
            y = data_np[:,2]
            t = data_np[:,3]
            
        #ピクセル --> 距離(cm)に変換
        real_x = (self.K * x) * (1 / 100)
        real_y = (self.K * y) * (1 / 100)
        
        #フレームN --> フレームN+1 までの間に移動した微小距離dxの計算
        #x[1:]-x[:-1] で隣接する要素の引き算ができる
        diff_real_x = real_x[1:] - real_x[:-1]
        diff_real_y = real_y[1:] - real_y[:-1]
        
        #隣接するフレーム間の相対的な二次元の微小距離を算出
        dx = np.sqrt(diff_real_x**2 + diff_real_y**2)

        #瞬間の速度を計算
        Velocity = dx / self.frame_time
        
        #plt.rcParams["font.family"] = "Times New Roman"
        fig, (axL, axR) = plt.subplots(ncols = 2, sharex = "none", figsize = (10,4))
        fig.suptitle("VIDEO PATH : " + VideoName)
        
        #1つ目のグラフ描画
        axL.plot(f, x, "r-", linewidth=1.5, label = "x")
        axL.plot(f, y, "g-", linewidth=1.5, label = "y")
        axL.legend(fontsize=7,
                   frameon=True,
                   facecolor="lightgreen")
                   
        axL.set_title('< Frame - Pixel >')
        axL.set_xlabel('frame[mai]')
        axL.set_ylabel('Position[px]')
        axL.grid(True)

        #2つ目のグラフ描画
        axR.plot(t[1:], Velocity, "b-", linewidth=1.5, label = "Velocity")
        axR.legend(fontsize=7,
                   frameon=True,
                   facecolor="lightgreen")
                   
        axR.set_title('< Time - Velocity >')
        axR.set_xlabel('Time[sec]')
        axR.set_ylabel('Velocity[m/sec]')
        axR.set_ylim(-0.5,  4)
        axR.grid(True)
        
        
        fig.tight_layout()
        fig.subplots_adjust(top=0.85)
        plt.show()
        
        #データのセーブ
        sNAME = VideoName.strip('.mp4') + '.png'
        fig.savefig(sNAME)

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
        #TODO 特定色の消去する処理
        pass
    #======================================================================

    def resize_image(self, img, dsize, X, Y):
        re_image = cv2.resize(img, dsize, fx=X, fy=Y)
        return re_image
    
    def px2cm(self, S_x, S_y, G_x, G_y):
        px_d = np.sqrt((S_x - G_x)**2 + (S_y + G_y)**2)
        #単位ピクセルあたりの実際の距離
        K = S2G / px_d
        return K

    #メイン
    def main(self, videofile_path, HumanHeight):
        data = []
        cap = cv2.VideoCapture(videofile_path)

        if self.t_init is True:
            self.Trackbars_init()

        if cap.isOpened():
            print("[INFO] : The Video loaded successfully.")
        else:
            print("[INFO] : LOAD ERROR *** Chack video path or name ***")
            print("CAP : ", cap)
            exit()
        
        #最初の1フレームだけ表示して、STARTとGOALを選択する
        print("\n[INFO] : This is the 1st frame.\n[INFO] : Choose start and goal positions and Click.")
        print("[INFO] : Quit order 'q' Key")
        
        ret, frame0 = cap.read()
        cp_frame0 = frame0.copy()
        
        cv2.namedWindow(self.window_name0)
        mouse = Mouse(self.window_name0, cp_frame0, frame0, HumanHeight)
        
        #STARTとGOALの選択
        while(cap.isOpened()):
            if image_frag is not True:
                cp_frame0 = frame0
                #print("[INFO] : CLEAR")
                
            cv2.imshow(self.window_name0, cp_frame0)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[INFO] : Order 'q' key. Proceed to the next step!")
                time.sleep(.2)
                break

        cv2.destroyAllWindows()
        
        #マウス座標のアンパックとピクセルから距離の割り出し
        S_x, S_y, G_x, G_y = mouse.clicked_point()
        self.K = self.px2cm(S_x, S_y, G_x, G_y)
        #TODO 頭のトラッキング位置を推定して画像に描画した線を配列にしてデータをプロットする。
        #配列にするには、スタートからゴールまでのX座標を相対距離ピクセルで割れば分割できる。numpyで分割できるかもかも
        #Ps_x, Ps_y, Pg_x, Pg_y = mouse.Prediction()
        
        print("[INFO] : Distance from START to GOAL is {0} pixel".format(G_x - S_x))
        print("[INFO] : So {0}(cm) is calculated as {1} pixcel".format(S2G, G_x - S_x))
        print("[INFO] : K is ", self.K)
        time.sleep(.2)
        
        #メインのループ
        while(cap.isOpened()):
            ret, frame = cap.read()
            if frame is None:
                print("frame is None")
                break

            #最初の処理で指定したSTARTとGOALに合わせてトリミングする
            h, w = frame.shape[:2]
            frame = frame[:, S_x:G_x]

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
            if self.fill_holes is True: mask = self.Fill_Holes(mask)
            if self.opening is True: mask = self.Opening(mask)
            if self.closing is True: mask = self.Closing(mask)
            if self.ColorErase is True: mask = self.color_eraser(mask, None)

            _, center, maxblob = self.analysis_blob(mask)
            #print("target num:",len(center))
            
            #見つけた領域にとりあえず円を描画してみる
            #ちょい重いから、マシンスペックに応じてコメントアウトしとくといいかも
            for i in center:
                cv2.circle(frame, (int(i[0]), int(i[1])), 10, (255, 0, 0),
                            thickness=-3, lineType=cv2.LINE_AA)
            
            self.center_x = int(maxblob["center"][0])
            self.center_y = int(maxblob["center"][1])
            
            print(self.center_x, self.center_y)
            
            #ラベル付けされたブロブに円を描画
            cv2.circle(frame, (self.center_x, self.center_y), 30, (0, 200, 0),
                      thickness=3, lineType=cv2.LINE_AA)
            
            #表示する用のデータのトリミング
            if S_x <= self.center_x <= G_x:
                data.append([self.frame_num, self.center_x, self.center_y, self.N_frame_time])
                
                self.N_frame_time = self.frame_time * self.frame_num
                self.frame_num += 1
            else:
                continue
            
            re_frame=self.resize_image(frame, None, .8, .8)
            cv2.imshow(self.window_name1, re_frame)
            
            mask = self.resize_image(mask, None, .8, .8)
            cv2.imshow("mask image", mask)
            
            print("-----")

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        self.data_plot(data, videofile_path)

if __name__ == '__main__':
    if len(sys.argv) == 1:
        videofile_path = "20191122/nihongi_f_l1.mp4"
        HumanHeight     = 165.0
        
    elif len(sys.argv) == 2:
        videofile_path = sys.argv[1]
        HumanHeight     = 165.0
        
    elif len(sys.argv) == 3:
        videofile_path = sys.argv[1]
        HumanHeight     = sys.argv[2]

    hsv_sup = HSV_supporter()
    hsv_sup.main(videofile_path, HumanHeight)
    
