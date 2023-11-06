import cv2
import os
import numpy as np

# チェスボードのサイズ
pattern_size = (7, 5)

# チェスボードの3D座標
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

# フォルダ内のすべての画像ファイルを取得
folder_path = './img0/cam0/test/'
image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')]

# キャリブレーションのためのデータ
objpoints = []  # 3Dオブジェクトの点のリスト
imgpoints = []  # 2Dイメージ上の点のリスト

# すべての画像でチェスボードを検出
for image_file in image_files:
    # 画像を読み込み
    image_path = os.path.join(folder_path, image_file)
    image = cv2.imread(image_path)
    
    # チェスボードのコーナーを検出
    found, corners = cv2.findChessboardCorners(image, pattern_size)
    
    if found:
        objpoints.append(objp)
        imgpoints.append(corners)
        
        # コーナーを描画
        cv2.drawChessboardCorners(image, pattern_size, corners, found)
        
        # 原点に点を描画
        origin = corners[0][0]
        # cv2.circle(image, (int(origin[0]), int(origin[1]), 10, (0, 0, 255), -1))
        
        # 結果を表示
        # cv2.imshow('Chessboard Corners', image)
        # cv2.waitKey(0)
    else:
        print(f'チェスボードのコーナーが画像 {image_file} で検出できませんでした.')

# すべての画像でチェスボードが検出された場合のみキャリブレーションを実行
if len(objpoints) == len(image_files):
    # カメラキャリブレーションを実行
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image.shape[1::-1], None, None)

    # 内部パラメータと外部パラメータを出力
    print("内部パラメータ (カメラ行列):")
    print(camera_matrix)
    print("\n歪み係数:")
    print(dist_coeffs)
    print("\n外部パラメータ (回転ベクトル):")
    print(rvecs)
    print("\n外部パラメータ (並進ベクトル):")
    print(tvecs)
else:
    print("すべての画像でチェスボードが検出されなかったため、キャリブレーションを行えません。")
