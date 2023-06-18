import numpy as np
import cv2

# カメラ1のパラメータ
camera1_intrinsic = {"fx": 500, "fy": 500, "cx": 320, "cy": 240}
camera1_extrinsic = {"t": [0, 0, 0], "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}
camera1_resolution = (320, 240)

# カメラ2のパラメータ
camera2_intrinsic = {"fx": 550, "fy": 550, "cx": 640, "cy": 480}
camera2_extrinsic = {
    "t": [0.2, 0.3, 0.1],
    "R": [[0.96, -0.1, 0.2], [0.1, 0.99, 0], [-0.2, 0, 0.98]],
}
camera2_resolution = (640, 480)

# カメラ3のパラメータ
camera3_intrinsic = {"fx": 600, "fy": 600, "cx": 960, "cy": 720}
camera3_extrinsic = {
    "t": [0.1, -0.2, 0.3],
    "R": [[0.97, 0.1, 0], [-0.1, 0.97, -0.2], [0, 0.2, 0.97]],
}
camera3_resolution = (960, 720)

# ターゲット点の座標
target_point = (100, 200)

# カメラ行列の計算
camera1_matrix = np.array(
    [
        [camera1_intrinsic["fx"], 0, camera1_intrinsic["cx"]],
        [0, camera1_intrinsic["fy"], camera1_intrinsic["cy"]],
        [0, 0, 1],
    ]
)
camera2_matrix = np.array(
    [
        [camera2_intrinsic["fx"], 0, camera2_intrinsic["cx"]],
        [0, camera2_intrinsic["fy"], camera2_intrinsic["cy"]],
        [0, 0, 1],
    ]
)
camera3_matrix = np.array(
    [
        [camera3_intrinsic["fx"], 0, camera3_intrinsic["cx"]],
        [0, camera3_intrinsic["fy"], camera3_intrinsic["cy"]],
        [0, 0, 1],
    ]
)
# 回転行列と並進ベクトルの計算
R1 = np.array(camera1_extrinsic["R"])
t1 = np.array(camera1_extrinsic["t"])
R2 = np.array(camera2_extrinsic["R"])
t2 = np.array(camera2_extrinsic["t"])
R3 = np.array(camera3_extrinsic["R"])
t3 = np.array(camera3_extrinsic["t"])

# 2次元座標を行列形式に変換
pt1 = np.array([[target_point[0]], [target_point[1]], [1]])
pt2 = np.array([[target_point[0]], [target_point[1]], [1]])
pt3 = np.array([[target_point[0]], [target_point[1]], [1]])

# カメラ1での3次元座標の計算
P1 = np.hstack((camera1_matrix @ R1, camera1_matrix @ t1.reshape(3, 1)))
x1 = np.dot(np.linalg.inv(camera1_matrix), pt1)
x1 = x1 / np.linalg.norm(x1)
X1 = np.dot(-R1.T, t1)
X1 = np.dot(X1.reshape(1, 3), x1) * x1 + X1

# カメラ2での3次元座標の計算
P2 = np.hstack((camera2_matrix @ R2, camera2_matrix @ t2.reshape(3, 1)))
x2 = np.dot(np.linalg.inv(camera2_matrix), pt2)
x2 = x2 / np.linalg.norm(x2)
X2 = np.dot(-R2.T, t2)
X2 = np.dot(np.negative(X2.reshape(1, 3)), x2) * x2 + X2

# カメラ3での3次元座標の計算
P3 = np.hstack((camera3_matrix @ R3, camera3_matrix @ t3.reshape(3, 1)))
x3 = np.dot(np.linalg.inv(camera3_matrix), pt3)
x3 = x3 / np.linalg.norm(x3)
X3 = np.dot(-R3.T, t3)
X3 = np.dot(np.negative(X3.reshape(1, 3)), x3) * x3 + X3

# 3つの3次元座標を縦方向に結合
b = np.vstack((np.negative(X1.reshape(-1, 3)), np.negative(X2.reshape(-1, 3)), np.negative(X3.reshape(-1, 3))))



# 3つの3次元座標を用いて、三角測量によりターゲットの3次元座標を計算
A = np.vstack((P1[0:2, :], P2[0:2, :], P3[0:2, :]))
X = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, b)).reshape(3)



# 結果を表示
print(X)

