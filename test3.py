import cv2
import glob
import numpy as np

# チェスボードの行数と列数
rows = 5
cols = 7

# 1マスのサイズ (10cm)
square_size = 0.1

# チェスボードのコーナーの3D座標
objp = np.zeros((rows * cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size

# 3D座標と2D座標の対応を保存するためのリスト
objpoints = []  # 3D座標
imgpoints = []  # 2D座標

# 画像の読み込み
images = []
for filename in sorted(glob.glob('./img0/cam0/*.png')):
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    images.append(gray)
# 画像の読み込みを終了
print('Image loading is finished.')
print (images)

# チェスボードのコーナーを検出
for i, gray in enumerate(images):
    ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
    if ret == True:
        # コーナーの位置を修正
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        # 3D座標と2D座標をリストに追加
        objpoints.append(objp)
        imgpoints.append(corners2)
        # チェスボードのコーナーを描画
        cv2.drawChessboardCorners(img, (cols, rows), corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)
    # 処理の進捗を表示
    print(f'Processing... {i + 1}/{len(images)}')
    # corners2の定義
    corners2 = None


cv2.destroyAllWindows()

# カメラキャリブレーションを実行
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, images[0].shape[::-1], None, None)

# 結果を表示
print('Camera Matrix:')
print(mtx)
print('Distortion Coefficients:')
print(dist)

# 画像の位置とカメラの位置を三次元で描写
img = cv2.imread('./img0/cam0/img1.png')
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# 歪みを補正
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# 描写
mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

# カメラ位置の描写
axis = np.float32([[0,0,0], [0,square_size*(rows-1),0], [square_size*(cols-1),square_size*(rows-1),0], [square_size*(cols-1),0,0],
                   [0,0,-square_size*(rows-1)], [0,square_size*(rows-1),-square_size*(rows-1)], [square_size*(cols-1),square_size*(rows-1),-square_size*(rows-1)], [square_size*(cols-1),0,-square_size*(rows-1)]])
axis = axis.reshape(-1,3)
 
# カメラ位置の回転ベクトルと移動ベクトルを計算
_, _, _, _, _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)

# 描写のための変換行列を計算
rotM = cv2.Rodrigues(rvecs)[0]
projM = np.hstack((rotM, tvecs))
cameraPos, _ = cv2.projectPoints(np.array([(0,0,0)]), rvecs, tvecs, mtx, dist)

# カメラ位置と姿勢を描写
imgpts, _ = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
img = cv2.drawFrameAxes(dst, mtx, dist, rvecs, tvecs, square_size, 2)

# 結果を描写
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

