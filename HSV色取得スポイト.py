import cv2
import numpy as np

def select_color_range(event, x, y, flags, param):
    global lower_color, upper_color
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        pixel_value = hsv[y, x]
        lower_color = np.array([pixel_value[0] - 10, pixel_value[1] - 50, pixel_value[2] - 50])
        bace_color = np.array([pixel_value[0], pixel_value[1], pixel_value[2]])
        upper_color = np.array([pixel_value[0] + 10, pixel_value[1] + 50, pixel_value[2] + 50])
        print("Lower Color:", lower_color)
        print("Bace Color:", bace_color)
        print("Upper Color:", upper_color)

# ウェブカメラのキャプチャを開始
cap = cv2.VideoCapture(0)

# ウィンドウを作成し、マウスイベントを設定
cv2.namedWindow('Frame')
cv2.setMouseCallback('Frame', select_color_range)

while True:
    # カメラからフレームをキャプチャ
    ret, frame = cap.read()

    # フレームを表示
    cv2.imshow('Frame', frame)

    # キー入力を待ち、'q'キーが押されたらループを抜ける
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# キャプチャを解放し、ウィンドウを閉じる
cap.release()
cv2.destroyAllWindows()
