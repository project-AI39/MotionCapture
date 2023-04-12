import cv2

# カメラのキャプチャ
cap = cv2.VideoCapture(0)

# ボールの色の範囲を設定 (Hue, Saturation, Value)
lower_color = (100, 50, 50)
upper_color = (130, 255, 255)

while True:
    # フレームの読み込み
    ret, frame = cap.read()

    # フレームをHSVに変換
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 色の範囲内のピクセルを白に、それ以外を黒にするマスクを作成
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # マスクを使って元のフレームに対してビット毎のAND演算を行い、ボールのみを抽出
    ball = cv2.bitwise_and(frame, frame, mask=mask)

    # ボールの中心座標を取得
    moments = cv2.moments(mask)
    if moments["m00"] != 0:
        cX = int(moments["m10"] / moments["m00"])
        cY = int(moments["m01"] / moments["m00"])
        cv2.circle(frame, (cX, cY), 5, (0, 255, 0), -1)
        cv2.putText(frame, f"({cX}, {cY})", (cX - 50, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # フレームを表示
    cv2.imshow("Ball Tracking", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("ball", ball)

    # 'q'キーが押されたらループを抜ける
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# キャプチャを解放し、ウィンドウを閉じる
cap.release()
cv2.destroyAllWindows()
