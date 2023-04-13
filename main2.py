import cv2
import numpy as np
import time

# Webカメラからの画像を取得する
cap = cv2.VideoCapture(0)

def frame_denoised(frame):
    # 非局所平均法処理時間計測開始
    denoised_start = time.perf_counter()
    # ノイズ除去のために非局所平均法を適用
    denoised = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
    # 非局所平均法処理時間計測終了
    denoised_end = time.perf_counter()
    # 非局所平均法処理時間を表示
    # print("Denoised Time:", denoised_end - denoised_start)
    return denoised

def mask_blurred(mask):
    # メディアンブラー処理時間計測開始
    blurred_start = time.perf_counter()
    # メディアンブラーを適用して画像の平滑化を行う
    blurred = cv2.medianBlur(mask, 15)
    # メディアンブラー処理時間計測終了
    blurred_end = time.perf_counter()
    # メディアンブラー処理時間を表示
    # print("Blurred Time:", blurred_end - blurred_start)
    return blurred

def mask_dilated(mask):
    # 膨張処理時間計測開始
    dilated_start = time.perf_counter()
    # 膨張処理を行う
    kernel_dilate = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(mask, kernel_dilate, iterations=1)
    # 膨張処理時間計測終了
    dilated_end = time.perf_counter()
    # 膨張処理時間を表示
    # print("Dilated Time:", dilated_end - dilated_start)
    return dilated

def mask_eroded(mask):
    # 収縮処理時間計測開始
    eroded_start = time.perf_counter()
    # 収縮処理を行う
    kernel_erode = np.ones((5, 5), np.uint8)
    eroded = cv2.erode(mask, kernel_erode, iterations=1)
    # 収縮処理時間計測終了
    eroded_end = time.perf_counter()
    # 収縮処理時間を表示
    # print("Eroded Time:", eroded_end - eroded_start)
    return eroded

while True:
    ret, frame = cap.read()

    # RGBからHSVに変換する
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 色でマスクを作成する
    lower_blue = np.array([17, 140, 140])
    upper_blue = np.array([27, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # ノイズ除去のために非局所平均法を適用  #かなり重い（0.9s）
    # denoised = frame_denoised(frame)
    # メディアンブラーを適用して画像の平滑化を行う #少し重い（0.03s）
    # blurred = mask_blurred(mask)
    # 膨張処理を行う #軽い（0.0003s）
    dilated = mask_dilated(mask)
    # 収縮処理を行う #軽い（0.0003s）
    eroded = mask_eroded(mask)

    # マスクを使って元のフレームに対してビット毎のAND演算を行い、ボールのみを抽出
    masked = cv2.bitwise_and(eroded, eroded, mask=mask)

    # マスクからボールの中心座標を取得
    moments = cv2.moments(mask)
    if moments["m00"] != 0:
        center_x = int(moments["m10"] / moments["m00"])
        center_y = int(moments["m01"] / moments["m00"])
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
        cv2.putText(frame, f"Center: ({center_x}, {center_y})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # フレームを表示
    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)
    cv2.imshow("eroded", eroded)
    cv2.imshow("masked", masked)

    # 'q'キーを押したらループを終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# キャプチャを解放してウィンドウを閉じる
cap.release()
cv2.destroyAllWindows()
