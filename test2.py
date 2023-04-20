import numpy as np
import cv2

# カメラキャリブレーションに使用する画像数
num_imgs = 10

# 画像サイズ
img_width = 640
img_height = 480

# チェスボードのサイズ
board_width = 9
board_height = 6

# チェスボードの3D座標を計算する
objp = np.zeros((board_width * board_height, 3), np.float32)
objp[:, :2] = np.mgrid[0:board_width, 0:board_height].T.reshape(-1, 2)

# キャリブレーション用の画像を準備する
obj_points = []
img_points1 = []
img_points2 = []
images = []
for i in range(num_imgs):
    img1 = cv2.imread(f"left{i}.png")
    img2 = cv2.imread(f"right{i}.png")
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ret1, corners1 = cv2.findChessboardCorners(gray1, (board_width, board_height), None)
    ret2, corners2 = cv2.findChessboardCorners(gray2, (board_width, board_height), None)
    if ret1 and ret2:
        obj_points.append(objp)
        corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        img_points1.append(corners1)
        img_points2.append(corners2)
        images.append((img1, img2))

# カメラパラメータを計算する
ret1, K1, dist1, rvecs1, tvecs1 = cv2.calibrateCamera(obj_points, img_points1, (img_width, img_height), None, None)
ret2, K2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(obj_points, img_points2, (img_width, img_height), None, None)

# ステレオカメラキャリブレーションを行う
retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(obj_points, img_points1, img_points2, K1, dist1, K2, dist2, (img_width, img_height), criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

# ステレオカメラパラメータを計算する
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, (img_width, img_height), R, T)
map1x, map1y = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, (img_width, img_height), cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, (img_width, img_height), cv2.CV_32FC1)


# 画像を読み込む
img1, img2 = images[0]

# ステレオ画像のペアを取得する
rectified_img1 = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR)
rectified_img2 = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR)

# ステレオ対応を見つける
window_size = 3
min_disp = 0
num_disp = 112 - min_disp
stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
numDisparities=num_disp,
blockSize=window_size,
P1=8 * 3 * window_size ** 2,
P2=32 * 3 * window_size ** 2,
disp12MaxDiff=1,
uniquenessRatio=10,
speckleWindowSize=100,
speckleRange=32)

disparity = stereo.compute(rectified_img1, rectified_img2).astype(np.float32) / 16.0

# 視差マップから3D点を計算する
focal_length = cameraMatrix1[0, 0]
baseline = -T[0]
Q = np.float32([[1, 0, 0, -img_width/2.0],
[0,-1, 0, img_height/2.0],
[0, 0, 0, -focal_length],
[0, 0, 1, 0]])

points_3d = cv2.reprojectImageTo3D(disparity, Q)

# 画面上に選択された点を描画する
cv2.namedWindow("image")
cv2.imshow("image", img1)
cv2.waitKey(0)

# 画面上での座標を取得する
u, v = cv2.ginputs([img1], 1)

# 3D座標を計算する
point_3d = points_3d[v, u]
point_3d /= 1000.0 # convert to meters

# 結果を出力する
print("3D point: ", point_3d)

# ウィンドウを閉じる
cv2.destroyAllWindows()