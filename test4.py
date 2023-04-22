import numpy as np
import cv2
import glob


# チェスボードの行数、列数、サイズを定義
rows = 5
cols = 7
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

# 内部パラメータの値を分割して表示
fx = mtx[0][0]
fy = mtx[1][1]
cx = mtx[0][2]
cy = mtx[1][2]
print("fx=", fx)
print("fy=", fy)
print("cx=", cx)
print("cy=", cy)

# 外部パラメータの値を分割して表示
rvecs_x = rvecs[0][0]
rvecs_y = rvecs[1][0]
rvecs_z = rvecs[2][0]
tvecs_x = tvecs[0][0]
tvecs_y = tvecs[1][0]
tvecs_z = tvecs[2][0]
rvecs_x = rvecs_x[0]
rvecs_y = rvecs_y[0]
rvecs_z = rvecs_z[0]
tvecs_x = tvecs_x[0]
tvecs_y = tvecs_y[0]
tvecs_z = tvecs_z[0]
print("rvecs_x=", rvecs_x)
print("rvecs_y=", rvecs_y)
print("rvecs_z=", rvecs_z)
print("tvecs_x=", tvecs_x)
print("tvecs_y=", tvecs_y)
print("tvecs_z=", tvecs_z)

# パラメータの値を保存
with open("calibration_params.txt", "w") as f:
    f.write("fx=" + str(fx) + "\n")
    f.write("fy=" + str(fy) + "\n")
    f.write("cx=" + str(cx) + "\n")
    f.write("cy=" + str(cy) + "\n")
    f.write("rvecs_x=" + str(rvecs_x) + "\n")
    f.write("rvecs_y=" + str(rvecs_y) + "\n")
    f.write("rvecs_z=" + str(rvecs_z) + "\n")
    f.write("tvecs_x=" + str(tvecs_x) + "\n")
    f.write("tvecs_y=" + str(tvecs_y) + "\n")
    f.write("tvecs_z=" + str(tvecs_z) + "\n")

print("パラメータの値を保存しました。")