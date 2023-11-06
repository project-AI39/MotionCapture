# multiple view geometry
# 三角測量　triangulation
# 2 cam triangulation test code

#CV2
import cv2
#numpy
import numpy as np


# カメラ１設定
# カメラ1の内部パラメータ
K_1 = [[320, 0, 160], [0, 320, 120], [0, 0, 1]]
# カメラ1の外部パラメータ
# カメラ1の位置
X_1 = [0, 10, -10]
# カメラ1の回転行列
R_1 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
# カメラ1の投影座標
u_1 = 0
v_1 = 0


# カメラ2設定
# カメラ2の内部パラメータ
K_2 = [[320, 0, 160], [0, 320, 120], [0, 0, 1]]
# カメラ2の外部パラメータ
# カメラ2の位置
X_2 = [10, 10, 0]
# カメラ2の回転行列
R_2 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
# カメラ2の投影座標
u_2 = 0
v_2 = 0


# 三角測量の誤差
error = 0.1

# 三角測量
# カメラ1の投影行列
P_1 = np.dot(K_1, np.hstack((R_1, np.array(X_1).reshape(3,1))))
# カメラ2の投影行列
P_2 = np.dot(K_2, np.hstack((R_2, np.array(X_2).reshape(3,1))))
# 三角測量
X = cv2.triangulatePoints(P_1, P_2, (u_1, v_1), (u_2, v_2))
# # 三次元座標を正規化
# X = X/X[3]
# # 三次元座標の誤差
# error = np.linalg.norm(np.dot(P_1, X) - np.array([u_1, v_1, 1])) + np.linalg.norm(np.dot(P_2, X) - np.array([u_2, v_2, 1]))
print(X)
# print(error)
