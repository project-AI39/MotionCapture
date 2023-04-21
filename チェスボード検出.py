import cv2
import os

# チェスボードのサイズ
pattern_size = (7, 5)

# フォルダ内のすべての画像ファイルを取得
folder_path = './img0/cam0/'
image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')]

for image_file in image_files:
    # 画像を読み込み
    image_path = os.path.join(folder_path, image_file)
    image = cv2.imread(image_path)
    
    # チェスボードのコーナーを検出
    found, corners = cv2.findChessboardCorners(image, pattern_size)
    
    if found:
        # コーナーを描画
        cv2.drawChessboardCorners(image, pattern_size, corners, found)
        
        # 原点に点を描画
        origin = corners[0][0]
        cv2.circle(image, (int(origin[0]), int(origin[1])), 10, (0, 0, 255), -1)
        
        # 結果を表示
        cv2.imshow('Chessboard Corners', image)
        cv2.waitKey(0)
    else:
        print(f'チェスボードのコーナーが画像 {image_file} で検出できませんでした。')
