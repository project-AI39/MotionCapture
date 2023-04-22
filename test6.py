import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# カメラ内部パラメータと外部パラメータを定義
fx=948.7147278896714
fy=1264.1658427870718
cx=963.4532191086563
cy=536.9201057274719
rvecs_x=-0.004131861300871871
rvecs_y=-0.34146591163488893
rvecs_z=0.00028835112493211186
tvecs_x=0.07995035544381054
tvecs_y=0.10694853761605323
tvecs_z=3.385298532837615

# 三次元空間を作成
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 描写範囲の指定
ax.set_xlim3d([-2, 2])
ax.set_ylim3d([-2, 2])
ax.set_zlim3d([-2, 2])

# 原点に座標軸を作成
ax.quiver(0, 0, 0, 1, 0, 0, length=1, normalize=True, color='r')# x軸 (縦)
ax.quiver(0, 0, 0, 0, 1, 0, length=1, normalize=True, color='g')# y軸 (横)
ax.quiver(0, 0, 0, 0, 0, 1, length=1, normalize=True, color='b')# z軸 (奥)

# カメラの座標に赤点を描写
camera_position = np.array([[tvecs_x], [tvecs_y], [tvecs_z]])
ax.scatter(camera_position[0], camera_position[1], camera_position[2], color='r')

# 画像の中心座標に青点を描写
ax.scatter(cx, cy, 0, color='b')

# カメラの向いている方向をベクトル線で表現し描写
rotation_matrix, _ = cv2.Rodrigues(np.array([[rvecs_x], [rvecs_y], [rvecs_z]], dtype=np.float32))
camera_direction = rotation_matrix @ np.array([[0], [0], [1]])
ax.quiver(camera_position[0], camera_position[1], camera_position[2],
          camera_direction[0], camera_direction[1], camera_direction[2],
          length=1, normalize=True, color='k')

# グラフを表示
plt.show()
