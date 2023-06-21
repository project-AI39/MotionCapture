import cv2

# カメラデバイスのインデックス（通常は0）を指定してキャプチャオブジェクトを作成
cap = cv2.VideoCapture(0)

# FourCCコードを直接指定（例: 'MJPG'）
fourcc_code = cv2.VideoWriter_fourcc(*'MJPG')

# FourCCコードを設定
cap.set(cv2.CAP_PROP_FOURCC, fourcc_code)

# 設定されたFourCCコードを取得して表示
fourcc = 0
fourcc = cap.get(cv2.CAP_PROP_FOURCC)
fourcc_bytes = int(fourcc).to_bytes(4, byteorder='little')
fourcc_str = fourcc_bytes.decode('ascii')
print('FourCC:', fourcc_str)
