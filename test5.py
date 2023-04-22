import numpy as np
import cv2
import glob
import pickle

# チェスボードの行数、列数、サイズを定義
rows = 5
cols = 7
square_size = 0.1

# キャリブレーション用のオブジェクトを定義
objp = np.zeros((rows * cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size

print("必要なものをインポートしました。")

# 画像が入っているフォルダのパスを定義
path_list = ['./img0/cam0/.png', './img0/cam1/.png', './img0/cam2/*.png']

for i, path in enumerate(path_list):
    print(f"カメラ{i}の画像を処理します。")
    
# フォルダから画像を読み込み、画像のリストを作成
images = []
image_points = []
for filename in glob.glob(path):
    img = cv2.imread(filename)
    ret, corners = cv2.findChessboardCorners(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (cols, rows), None)
    if ret == True:
        images.append(img)
        image_points.append(corners)
    else:
        print(f"Failed to detect chessboard corners in {filename}")

print(f"カメラ{i}の画像から、画像のリストを作成しました。")

# キャリブレーションを行い、内部パラメータと外部パラメータを求める
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([objp]*len(images), image_points, cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY).shape[::-1], None, None)

print(f"カメラ{i}のキャリブレーションを行い、内部パラメータと外部パラメータを求めました。")

# 内部パラメータと外部パラメータをprintする
print(f"カメラ{i}の内部パラメータ:")
print(mtx)
print(f"カメラ{i}の外部パラメータ:")
print(rvecs)
print(tvecs)
print("------------------------")
