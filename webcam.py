import cv2

# カメラデバイスのインデックス（通常は0）を指定してキャプチャオブジェクトを作成
cap = cv2.VideoCapture(0)

# 映像フォーマットの指定
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))  # MJPGフォーマットを指定

# 解像度の設定
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # 幅を1280に設定
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # 高さを720に設定

# フレームレートの設定
cap.set(cv2.CAP_PROP_FPS, 30)  # フレームレートを30fpsに設定

# カメラの設定が反映されたか確認するために、現在の設定値を取得して表示
fourcc = cap.get(cv2.CAP_PROP_FOURCC)
fourcc_bytes = int(fourcc).to_bytes(4, byteorder='little')
fourcc_str = fourcc_bytes.decode('ascii')
print('FourCC:', fourcc_str)

print('Width:', cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print('Height:', cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print('FPS:', cap.get(cv2.CAP_PROP_FPS))

while True:
    # フレームをキャプチャ
    ret, frame = cap.read()

    if ret:
        # 解像度の取得
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        resolution_text = f'Resolution: {width} x {height}'

        # フレームレートの取得
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps_text = f'FPS: {fps:.2f}'

        # 解像度とフレームレートをフレームに描画
        cv2.putText(frame, resolution_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, fps_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 解像度とフレームレートをコンソールに出力
        # print(resolution_text,fps_text)
        
        # フレームを表示
        cv2.imshow('Web Camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
