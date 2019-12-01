# ColorTracker
### /hsv_supporter.py
**[使い方]**  
★直接パスを指定する場合、以下を有効にして実行場所からの相対パスを指定
```bash
videofile_path = '../PATH/test.mp4'
```  
**[実行]**  
`$python3 hsv_supporter.py`  

  
★コンソール画面からパスを引数として受け取る場合、以下を有効にすること
```bash
videofile_path = sys.argv[1]
```
**[実行]**  
`$python3 hsv_supporter.py ~/PATH/test.mp4`  
  
## コードの解説(知見)
**[ROSとOpencvの競合]**  
```python
import sys
try:
    py_path = sys.path
    ros_CVpath = '/opt/ros/kinetic/lib/python2.7/dist-packages'
    if py_path[3] == ros_CVpath:
        print("INFO : ROS and OpenCV are competing")
        sys.path.remove(py_path[3])
except: pass
```  
__ROS__(Kinetic, Indigoで確認)のDesktopfullをインストールしてあるPCでopencvをインポートしようとすると、エラーを吐く。  
これを回避するために、~/.bashrcに記述されている`source /opt/ros/kinetic/setup.bash`でロードされるパスをpythonでopencvをインポートする前に追い出してあげればいい。

この回避方法は、[こちらのサイト](https://qiita.com/ReoNagai/items/112c3a8b6cd55c3e5380)が参考になった。
  
**[今回使ったノイズの除去アルゴリズム]**  
```python
#+-----[MedianBlur]-------+#
MB = True

#+-----[fill_holes]-------+#
fill_holes = False

#+-------[opening]--------+#
opening = True

#+-------[closing]--------+#
closing = True

```
★MB(MedianBlar) 中央値フィルタ)  
__cv2.medianBlur()__ 関数はカーネル内の全画素の中央値を計算します．ごま塩ノイズのようなノイズに対して効果的です．箱型フィルタとガウシアンフィルタの出力結果は原画像中には存在しない画素値を出力とするのに対して，中央値フィルタの出力は常に原画像中から選ばれています．そのためごま塩ノイズのような特異なノイズに対して効果的です．カーネルサイズは奇数でなければいけません．  
[ここから抜粋](http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_filtering/py_filtering.html)  

プログラム内では以下の関数を用意した。~~別に用意する必要はないがなんでか追加した~~
```python
def median_blar(image, size):
    _image = cv2.medianBlur(image, size)
    return _image
```

★opening(Opening) モルフォロジー変換(オープニング)  
オープニング処理は、収縮の後に膨張をする処理である．前述したようにノイズ除去に有効である．関数は __cv2.morphologyEx()__ を使う  
こんな感じでノイズ除去される  
![opening](/image/opening.png)  
プログラム内では以下の関数を用意した。
```python
def Opening(image):
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return opening
```

★closing(Closing) モルフォロジー変換(クロージング)  
クロージング処理はオープニング処理の逆の処理であり，膨張の後に収縮をする処理である．前景領域中の小さな(黒い)穴を埋めるのに役立る．オープニングと同様 __cv2.morphologyEx()__ 関数を使うが，第2引数のフラグに __cv2.MORPH_CLOSE__ を指定する点が異なる  
こんな感じでノイズ除去される  
![opening](/image/closing.png)  
プログラム内では以下の関数を用意した。
```python
def Closing(image):
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return closing
```
上2つは[ココ](http://lang.sist.chukyo-u.ac.jp/classes/OpenCV/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html)から抜粋した。ノイズに対するアルゴリズムはOpencvがほとんど関数を用意してくれている。ありがたい。
