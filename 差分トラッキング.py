import cv2

# カメラを開く
cap = cv2.VideoCapture(0)

# カメラのフレームサイズを取得
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# フレーム間の差分の閾値
threshold = 30

# モーション検出のための前のフレームの初期化
prev_frame = None

while True:
    # フレームをキャプチャ
    ret, frame = cap.read()

    # グレースケールに変換
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_frame is not None:
        # 前のフレームとの差分を計算
        frame_diff = cv2.absdiff(gray, prev_frame)

        # 差分が閾値を超えた領域を白くする
        ret, thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)

        # ノイズを削除
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # 輪郭を検出
        contours, hierarchy = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # モーション検出した領域を矩形で囲む
        for contour in contours:
            if cv2.contourArea(contour) < 500:  # 面積が小さいものは除外
                continue
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 現在のフレームを前のフレームとして保存
    prev_frame = gray

    # フレームを表示
    cv2.imshow('Motion Detection', frame)

    # 'q'キーでループを終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# キャプチャを解放し、ウィンドウを閉じる
cap.release()
cv2.destroyAllWindows()
