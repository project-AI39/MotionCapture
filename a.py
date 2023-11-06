import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def make(position, rotation_matrix=None):
    if rotation_matrix is None:
        ax.scatter(position[0], position[1], position[2], color='black')
    else:
        ax.scatter(position[0], position[1], position[2], color='fuchsia')
        
        # 座標から回転行列をもとに向いている方向に少し進んだ場所を計算
        direction = np.dot(rotation_matrix, np.array([1, 0, 0]))
        end_point = position + direction
        
        # 矢印を描画
        # ax.quiver(position[0], position[1], position[2], direction[0], direction[1], direction[2], color='orange', label='Direction')
        
        # 点を描画
        ax.scatter(end_point[0], end_point[1], end_point[2], color='red')
        
        # 座標から回転行列をもとに座標軸を示すベクトル線を計算
        x_axis = np.dot(rotation_matrix, np.array([1, 0, 0]))
        y_axis = np.dot(rotation_matrix, np.array([0, 1, 0]))
        z_axis = np.dot(rotation_matrix, np.array([0, 0, 1]))
        
        # 点線で座標軸を描画
        ax.plot([position[0], position[0] + 0.8 * x_axis[0]], [position[1], position[1] + 0.8 * x_axis[1]], [position[2], position[2] + 0.8 * x_axis[2]], 'r--', linewidth=1)
        ax.plot([position[0], position[0] + 0.8 * y_axis[0]], [position[1], position[1] + 0.8 * y_axis[1]], [position[2], position[2] + 0.8 * y_axis[2]], 'g--', linewidth=1)
        ax.plot([position[0], position[0] + 0.8 * z_axis[0]], [position[1], position[1] + 0.8 * z_axis[1]], [position[2], position[2] + 0.8 * z_axis[2]], 'b--', linewidth=1)
        pass

def calculate_rotation_matrix(x_rotation, y_rotation, z_rotation):
    # 各軸の回転行列を計算
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(np.radians(x_rotation)), -np.sin(np.radians(x_rotation))],
                   [0, np.sin(np.radians(x_rotation)), np.cos(np.radians(x_rotation))]])

    Ry = np.array([[np.cos(np.radians(y_rotation)), 0, np.sin(np.radians(y_rotation))],
                   [0, 1, 0],
                   [-np.sin(np.radians(y_rotation)), 0, np.cos(np.radians(y_rotation))]])

    Rz = np.array([[np.cos(np.radians(z_rotation)), -np.sin(np.radians(z_rotation)), 0],
                   [np.sin(np.radians(z_rotation)), np.cos(np.radians(z_rotation)), 0],
                   [0, 0, 1]])

    # 各軸の回転行列を積算して最終的な回転行列を計算
    R = np.dot(Rz, np.dot(Ry, Rx))

    return R

####################################################################################################
# 新しい3Dプロットを作成
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_zlim(-5, 5)

# 座標軸のラベルを設定
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 原点からの座標軸を示す向きベクトルを描写
ax.quiver(0, 0, 0, 1, 0, 0, color='r', label='X')
ax.quiver(0, 0, 0, 0, 1, 0, color='g', label='Y')
ax.quiver(0, 0, 0, 0, 0, 1, color='b', label='Z')
####################################################################################################

#カメラ１の位置
X_1 = [1, 0, 0]
#カメラ２の位置
X_2 = [0, 1, 0]


# カメラ１の回転角度 degrees
x_rotation = 0  # X軸回転
y_rotation = 0  # Y軸回転
z_rotation = 90  # Z軸回転
#カメラ１の回転行列
R_1 = calculate_rotation_matrix(x_rotation, y_rotation, z_rotation)

# カメラ２の回転角度 degrees
x_rotation = 0  # X軸回転
y_rotation = 0  # Y軸回転
z_rotation = 0  # Z軸回転
#カメラ２の回転行列
R_2 = calculate_rotation_matrix(x_rotation, y_rotation, z_rotation)

# カメラ１の焦点距離
f_1 = 50

# 主点１
principal_point_1 = (960, 540)

# 歪み係数１
skew_1 = 0

# 内部パラメータの計算
K_1 = np.array(
[
        [f_1, 0, principal_point_1[0]],
        [0, f_1, principal_point_1[1]],
        [0, skew_1, 1],
    ]
)

# カメラ２の焦点距離
f_2 = 50

# 主点２
principal_point_2 = (960, 540)

# 歪み係数２
skew_2 = 0

# 内部パラメータの計算
K_2 = np.array(
[
        [f_2, 0, principal_point_2[0]],
        [0, f_2, principal_point_2[1]],
        [0, skew_2, 1],
    ]
)
# カメラ1の投影座標
u_1 = 960
v_1 = 540
# カメラ2の投影座標
u_2 = 960
v_2 = 540

# 三角測量
# カメラ1の投影行列
P_1 = np.dot(K_1, np.hstack((R_1, np.array(X_1).reshape(3,1))))
# カメラ2の投影行列
P_2 = np.dot(K_2, np.hstack((R_2, np.array(X_2).reshape(3,1))))
# 三角測量
X = cv2.triangulatePoints(P_1, P_2, (u_1, v_1), (u_2, v_2))

print("X=",X)

# データをプロット
make(X_1, R_1)
make(X_2, R_2)
make(X)
# # 三次元座標を正規化
# X = X/X[3]
# make(X)

# 凡例を追加
ax.legend()
# プロット表示
plt.show()
