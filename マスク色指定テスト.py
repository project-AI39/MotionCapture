import cv2
import numpy as np

# 初期の色範囲の制限
lower_color = np.array([0, 0, 0])
upper_color = np.array([255, 255, 255])

# 色範囲の制限を調整するスライダーのコールバック関数
def adjust_color_range(value):
    global lower_color, upper_color
    hue_range = cv2.getTrackbarPos('Hue Range', 'Frame')
    saturation_range = cv2.getTrackbarPos('Saturation Range', 'Frame')
    value_range = cv2.getTrackbarPos('Value Range', 'Frame')

    # lower_colorの値を更新
    lower_color[0] = max(0, hue_range - 10)
    lower_color[1] = max(0, saturation_range - 50)
    lower_color[2] = max(0, value_range - 50)

    # upper_colorの値を更新
    upper_color[0] = min(255, hue_range + 10)
    upper_color[1] = min(255, saturation_range + 50)
    upper_color[2] = min(255, value_range + 50)

    print("Lower Color:", lower_color)
    print("Upper Color:", upper_color)

# ウェブカメラのキャプチャを開始
cap = cv2.VideoCapture(0)

# ウィンドウを作成
cv2.namedWindow('Frame')

# スライダーを作成
cv2.createTrackbar('Hue Range', 'Frame', 0, 255, adjust_color_range)
cv2.createTrackbar('Saturation Range', 'Frame', 0, 255, adjust_color_range)
cv2.createTrackbar('Value Range', 'Frame', 0, 255, adjust_color_range)

while True:
    # カメラからフレームをキャプチャ
    ret, frame = cap.read()

    # フレームをHSVに変換
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 色範囲を適用してマスクを作成
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # マスクを使って元のフレームに色を適用
    res = cv2.bitwise_and(frame, frame, mask=mask)

    # フレームとマスクを表示
    cv2.imshow('Frame', np.hstack((frame, res)))

    # キー入力を待ち、'q'キーが押されたらループを抜ける
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# キャプチャを解放し、ウィンドウを閉じる
cap.release()
cv2.destroyAllWindows()
