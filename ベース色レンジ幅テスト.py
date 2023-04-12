import cv2
import numpy as np

# 初期の基準色と範囲の設定
h_base = 0
s_base = 0
v_base = 0
h_range = 0
s_range = 0
v_range = 0

# コールバック関数
def adjust_color_range(val):
    global h_base, s_base, v_base, h_range, s_range, v_range
    # スライダーの値を取得
    h_base = cv2.getTrackbarPos('H Base', 'Color Range')
    s_base = cv2.getTrackbarPos('S Base', 'Color Range')
    v_base = cv2.getTrackbarPos('V Base', 'Color Range')
    h_range = cv2.getTrackbarPos('H Range', 'Color Range')
    s_range = cv2.getTrackbarPos('S Range', 'Color Range')
    v_range = cv2.getTrackbarPos('V Range', 'Color Range')

# ウィンドウの作成
cv2.namedWindow('Color Range')

# トラックバーの作成
cv2.createTrackbar('H Base', 'Color Range', h_base, 179, adjust_color_range)
cv2.createTrackbar('S Base', 'Color Range', s_base, 255, adjust_color_range)
cv2.createTrackbar('V Base', 'Color Range', v_base, 255, adjust_color_range)
cv2.createTrackbar('H Range', 'Color Range', h_range, 179, adjust_color_range)
cv2.createTrackbar('S Range', 'Color Range', s_range, 255, adjust_color_range)
cv2.createTrackbar('V Range', 'Color Range', v_range, 255, adjust_color_range)

# Webカメラのキャプチャを開始
cap = cv2.VideoCapture(0)

while True:
    # フレームの取得
    ret, frame = cap.read()
    if not ret:
        break

    # フレームをHSVに変換
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 色範囲のマスクを作成
    lower_color = np.array([h_base - h_range, s_base - s_range, v_base - v_range])
    upper_color = np.array([h_base + h_range, s_base + s_range, v_base + v_range])
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # 元のフレームにマスクを適用
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # フレームとマスクを表示
    cv2.imshow('Original', frame)
    cv2.imshow('Mask', mask)
    cv2.imshow('Result', result)

    # 'q' キーが押されたらループを抜ける
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# キャプチャを解放し、ウィンドウを閉じる
cap.release()
cv2.destroyAllWindows()
