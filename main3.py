import cv2
import numpy as np

# カメラのキャプチャを開始
cap = cv2.VideoCapture(0)

# 色範囲の指定 (例: 青い色の範囲を指定)
lower_blue = np.array([17, 140, 140])
upper_blue = np.array([27, 255, 255])

while True:
    # カメラからフレームをキャプチャ
    ret, frame = cap.read()

    # フレームをHSVに変換
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # マスクを作成
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # 膨張処理
    kernel_dilate = np.ones((11,11), np.uint8)
    dilated = cv2.dilate(mask, kernel_dilate, iterations=1)
    
    # 輪郭を抽出 #method 引数,mode 引数 #https://pystyle.info/opencv-find-contours/#outline__4_1_1
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 各輪郭の中心座標を計算し、描画
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(frame, (cX, cY), 5, (0, 255, 0), -1)
            cv2.putText(frame, f'({cX}, {cY})', (cX - 50, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 輪郭を描画
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
    # 画像を表示
    cv2.imshow("Webカメラ", frame)
    cv2.imshow("mask", mask)
    cv2.imshow("dilated", dilated)

    # 'q'キーが押されたらループを終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# キャプチャをリリースし、ウィンドウを閉じる
cap.release()
cv2.destroyAllWindows()
