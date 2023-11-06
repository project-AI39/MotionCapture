# multiple view geometry
# 三角測量 triangulation
# 2 cam triangulation test code

#CV2
import cv2
#numpy
import numpy as np
#mathplotlib
import matplotlib.pyplot as plt


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

def prot():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)

    x = [0,X_1[0], X_2[0]]
    y = [0,X_1[1], X_2[1]]
    z = [0,X_1[2], X_2[2]]
    # 3Dでプロット
    ax.scatter(x, y, z, color='blue')
    # 矢印の表示
    ax.quiver(0, 0, 0, x, y, z, arrow_length_ratio=0.1)
    # カメラ１の向きを矢印で表示
    ax.quiver(X_1[0], X_1[1], X_1[2], R_1[0][0], R_1[1][0], R_1[2][0], arrow_length_ratio=0.1, color='red')
    ax.quiver(X_1[0], X_1[1], X_1[2], R_1[0][1], R_1[1][1], R_1[2][1], arrow_length_ratio=0.1, color='green')
    ax.quiver(X_1[0], X_1[1], X_1[2], R_1[0][2], R_1[1][2], R_1[2][2], arrow_length_ratio=0.1, color='blue')
    # カメラ２の向きを矢印で表示
    ax.quiver(X_2[0], X_2[1], X_2[2], R_2[0][0], R_2[1][0], R_2[2][0], arrow_length_ratio=0.1, color='red')
    ax.quiver(X_2[0], X_2[1], X_2[2], R_2[0][1], R_2[1][1], R_2[2][1], arrow_length_ratio=0.1, color='green')
    ax.quiver(X_2[0], X_2[1], X_2[2], R_2[0][2], R_2[1][2], R_2[2][2], arrow_length_ratio=0.1, color='blue')

    plt.show()
    pass


#カメラ１の位置
X_1 = [0, 0, 0]
#カメラ２の位置
X_2 = [1, 1, 0]


# カメラ１の回転角度 degrees
x_rotation = 0  # X軸回転
y_rotation = 0  # Y軸回転
z_rotation = 0  # Z軸回転
#カメラ１の回転行列
R_1 = calculate_rotation_matrix(x_rotation, y_rotation, z_rotation)

# カメラ２の回転角度 degrees
x_rotation = 0  # X軸回転
y_rotation = 0  # Y軸回転
z_rotation = -90  # Z軸回転
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

print("X_1:",X_1)
print("X_2:",X_2)
print("R_1:",R_1)
print("R_2:",R_2)
print("K_1:",K_1)
print("K_2:",K_2)
print("P_1:",P_1)
print("P_2:",P_2)
print(X)
prot()

