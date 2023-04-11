import cv2

# ウェブカメラの解像度とフレームレートを設定
width = 1920
height = 1080
fps = 30

# ウェブカメラをオープン
cap = cv2.VideoCapture(0)  # 0はデフォルトのカメラのインデックス

# ウェブカメラの解像度とフレームレートを設定
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv2.CAP_PROP_FPS, fps)

while True:
    # ウェブカメラからフレームをキャプチャ
    ret, frame = cap.read()

    # フレームをウィンドウに表示
    cv2.imshow('Web Camera', frame)

    # キャプチャしたフレームの解像度とフレームレートを取得
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_fps = int(cap.get(cv2.CAP_PROP_FPS))

    # 解像度とフレームレートをリアルタイムで表示
    cv2.putText(frame, f"Resolution: {frame_width}x{frame_height}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Frame Rate: {frame_fps} FPS", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # フレームをウィンドウに再表示
    cv2.imshow('Web Camera', frame)

    # 'q'キーを押したらループを抜ける
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ウェブカメラをリリースし、ウィンドウを閉じる
cap.release()
cv2.destroyAllWindows()
