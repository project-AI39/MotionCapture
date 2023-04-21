import numpy as np
import cv2
import glob

# チェスボードの行数、列数、サイズを定義
rows = 6
cols = 8
square_size = 0.1

# キャリブレーション用のオブジェクトを定義
objp = np.zeros((rows * cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size

print("初めに必要なものをインポートしました。")

# 画像が入っているフォルダのパスを定義
path = './img0/cam0/*.png'

print("画像が入っているフォルダのパスを定義しました。")

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

print("フォルダから画像を読み込み、画像のリストを作成しました。")

# キャリブレーションを行い、内部パラメータと外部パラメータを求める
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([objp]*len(images), image_points, cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY).shape[::-1], None, None)

print("キャリブレーションを行い、内部パラメータと外部パラメータを求めました。")

# 内部パラメータと外部パラメータをprintする
print("内部パラメータ:")
print(mtx)
print("外部パラメータ:")
print(rvecs)
print(tvecs)

print("内部パラメータと外部パラメータをprintしました。")

# 画像やカメラの位置が理解しやすいように３次元空間を作成して画像やカメラを描写
# 3D空間の作成

print("画像やカメラの位置が理解しやすいように３次元空間を作成して画像やカメラを描写します。")
