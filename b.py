import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
def calculate_camera_matrix(fx, fy, cx, cy, rotation_matrix, x_translation, y_translation, z_translation):
    # 内部パラメータ行列の設定
    intrinsic_matrix = np.array([[fx, 0, cx],
                                 [0, fy, cy],
                                 [0, 0, 1]])
    
    # 外部パラメータ行列の設定
    # カメラの位置
    position_matrix = np.array([[x_translation],
                                [y_translation],
                                [z_translation]])
    
    # 外部パラメータ行列の計算
    extrinsic_matrix = np.hstack((rotation_matrix, position_matrix))
    
    # カメラ行列の計算
    camera_matrix = np.dot(intrinsic_matrix, extrinsic_matrix)
    
    return intrinsic_matrix, position_matrix, rotation_matrix, extrinsic_matrix, camera_matrix
def triangulate(P1, P2, points1, points2):
    """
    三角測量を行う関数

    :param P1: カメラ1の3x4行列
    :param P2: カメラ2の3x4行列
    :param points1: カメラ1の画像上の対応点の座標（n x 2行列）
    :param points2: カメラ2の画像上の対応点の座標（n x 2行列）
    :return: 3D空間の点の座標（n x 3行列）
    """
    num_points = points1.shape[0]
    reconstructed_points = np.zeros((num_points, 3))

    A = np.zeros((4, 4))

    # カメラ1の画像座標
    x1, y1 = points1
    A[0] = x1 * P1[2] - P1[0]
    A[1] = y1 * P1[2] - P1[1]

    # カメラ2の画像座標
    x2, y2 = points2
    A[2] = x2 * P2[2] - P2[0]
    A[3] = y2 * P2[2] - P2[1]

    _, _, V = np.linalg.svd(A)
    reconstructed_points = V[-1, :-1] / V[-1, -1]

    return reconstructed_points
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
def rotation_vector_to_euler_angles(rotation_vector):
    # 正規化
    rotation_vector /= np.linalg.norm(rotation_vector)
    
    # 回転行列の計算
    x = np.arccos(rotation_vector[0, 0])  # x 軸の回転角度
    y = np.arccos(rotation_vector[1, 0])    # y 軸の回転角度
    z = np.arccos(rotation_vector[2, 0])    # z 軸の回転角度

    # 結果を度数法に変換
    x_deg = np.degrees(x)
    y_deg = np.degrees(y)
    z_deg = np.degrees(z)

    return x_deg, y_deg, z_deg
####################################################################################################
# 新しい3Dプロットを作成
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlim(-500, 500)
ax.set_ylim(-500, 500)
ax.set_zlim(-500, 500)

# 座標軸のラベルを設定
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 原点からの座標軸を示す向きベクトルを描写
ax.quiver(0, 0, 0, 1, 0, 0, color='r', label='X')
ax.quiver(0, 0, 0, 0, 1, 0, color='g', label='Y')
ax.quiver(0, 0, 0, 0, 0, 1, color='b', label='Z')
####################################################################################################

## 内部パラメータ行列の設定
fx1 = 2597.87121  # X方向の焦点距離
fy1 = 2597.55043  # Y方向の焦点距離
cx1 = 936.422977   # 画像の中心X座標
cy1 = 509.192332   # 画像の中心Y座標
## 外部パラメータ行列の設定
# カメラの位置
cam1_pos_x = 1.32459546 # カメラのX座標
cam1_pos_y = 1.81193277 # カメラのY座標
cam1_pos_z = 69.56969684 # カメラのZ座標
# カメラの姿勢
cam1_rot_x = 0.0 # カメラのX軸回転
cam1_rot_y = 0.0 # カメラのY軸回転
cam1_rot_z = 90.0 # カメラのZ軸回転
rotation_vector = np.array([[-0.01117237],
                            [-0.77528757],
                            [ 0.00485018]])
cam1_pos_x, cam1_pos_y, cam1_pos_z = rotation_vector_to_euler_angles(rotation_vector)
# 回転行列の計算
rotation_matrix_1 = calculate_rotation_matrix(cam1_rot_x, cam1_rot_y, cam1_rot_z)
# 画像上の対応点の座標
u_1 = 1157 # 画像上の対応点のX座標
v_1 = 749 # 画像上の対応点のY座標

## 内部パラメータ行列の設定
fx2 = 2597.87121  # X方向の焦点距離
fy2 = 2597.55043  # Y方向の焦点距離
cx2 = 936.422977   # 画像の中心X座標
cy2 = 509.192332   # 画像の中心Y座標
## 外部パラメータ行列の設定
# カメラの位置
cam2_pos_x = 1.4323922 # カメラのX座標
cam2_pos_y = 1.56932358 # カメラのY座標
cam2_pos_z = 48.70932853 # カメラのZ座標
# カメラの姿勢
cam2_rot_x = 0.0 # カメラのX軸回転
cam2_rot_y = 0.0 # カメラのY軸回転
cam2_rot_z = 0.0 # カメラのZ軸回転
rotation_vector = np.array([[-0.00895712],
                            [ 0.00713052],
                            [ 0.00014517]])
cam2_pos_x, cam2_pos_y, cam2_pos_z = rotation_vector_to_euler_angles(rotation_vector)
# 回転行列の計算
rotation_matrix_2 = calculate_rotation_matrix(cam1_rot_x, cam1_rot_y, cam1_rot_z)
# 画像上の対応点の座標
u_2 = 1358 # 画像上の対応点のX座標
v_2 = 858 # 画像上の対応点のY座標

# カメラ行列の計算
(
intrinsic_matrix_1,
position_matrix_1,
rotation_matrix_1,
extrinsic_matrix_1,
camera_matrix_1
) = calculate_camera_matrix(
fx1,
fy1,
cx1,
cy1,
rotation_matrix_1,
cam1_pos_x,
cam1_pos_y,
cam1_pos_z
)
(
intrinsic_matrix_2,
position_matrix_2,
rotation_matrix_2,
extrinsic_matrix_2,
camera_matrix_2
) = calculate_camera_matrix(
fx2,
fy2,
cx2,
cy2,
rotation_matrix_2,
cam2_pos_x,
cam2_pos_y,
cam2_pos_z
)


print("内部パラメータ行列:")
print(intrinsic_matrix_1)
print("カメラの位置:")
print(position_matrix_1)
print("回転行列:")
print(rotation_matrix_1)
print("外部パラメータ行列:")
print(extrinsic_matrix_1)
print("カメラ行列:")
print(camera_matrix_1)

print("内部パラメータ行列:")
print(intrinsic_matrix_2)
print("カメラの位置:")
print(position_matrix_2)
print("回転行列:")
print(rotation_matrix_2)
print("外部パラメータ行列:")
print(extrinsic_matrix_2)
print("カメラ行列:")
print(camera_matrix_2)

P1 = camera_matrix_1
P2 = camera_matrix_2
points1 = np.array([u_1, v_1])
points2 = np.array([u_2, v_2])

reconstructed_points = triangulate(P1, P2, points1, points2)

print("カメラ1行列:")
print(P1)
print("カメラ1の画像上の対応点:")
print(points1)

print("カメラ2行列:")
print(P2)
print("カメラ2の画像上の対応点:")
print(points2)

print("三次元座標:")
print(reconstructed_points)

# 三次元座標を描画
make(position_matrix_1, rotation_matrix_1)
make(position_matrix_2, rotation_matrix_2)
make(reconstructed_points)

# 凡例を追加
ax.legend()
# プロット表示
plt.show()