import sys
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def prepare_test_data(
    draw_test_data,
    draw_epipolar,
    surface_type,
    rot_euler_deg_0,
    rot_euler_deg_1,
    T_0_in_camera_coord,
    T_1_in_camera_coord,
    f,
    width,
    height,
):
    rot_mat_0 = euler_angle_to_rot_mat(
        rot_euler_deg_0[0], rot_euler_deg_0[1], rot_euler_deg_0[2]
    )
    trans_vec_0 = np.eye(3) * np.matrix(T_0_in_camera_coord).T
    rot_mat_1 = euler_angle_to_rot_mat(
        rot_euler_deg_1[0], rot_euler_deg_1[1], rot_euler_deg_1[2]
    )
    trans_vec_1 = np.eye(3) * np.matrix(T_1_in_camera_coord).T
    if surface_type == "CURVE":
        points_3d = create_curve_surface_points(5, 5, 0.2)
    elif surface_type == "PLANE":
        points_3d = create_curve_surface_points(5, 5, 0)
    elif surface_type == "CIRCLE":
        points_3d = create_circle_surface_points(5, 3)
    else:
        raise RuntimeError("Surface type is wrong")
    rodri_0, jac = cv2.Rodrigues(rot_mat_0)
    rodri_1, jac = cv2.Rodrigues(rot_mat_1)

    pp = (width / 2, height / 2)
    camera_matrix = np.matrix(
        np.array([[f, 0, pp[0]], [0, f, pp[1]], [0, 0, 1]], dtype="double")
    )
    dist_coeffs = np.zeros((5, 1))
    img_pnts_0, jac = cv2.projectPoints(
        points_3d, rodri_0, trans_vec_0, camera_matrix, dist_coeffs
    )
    img_pnts_0 = np.reshape(img_pnts_0, (img_pnts_0.shape[0], 2))
    img_pnts_1, jac = cv2.projectPoints(
        points_3d, rodri_1, trans_vec_1, camera_matrix, dist_coeffs
    )
    img_pnts_1 = np.reshape(img_pnts_1, (img_pnts_1.shape[0], 2))

    img_0 = np.full((height, width, 3), (255, 255, 255), np.uint8)
    img_1 = np.full((height, width, 3), (255, 255, 255), np.uint8)

    noise_scale = 0.2
    noised_img_pnts_0 = add_noise(img_pnts_0, noise_scale)
    noised_img_pnts_1 = add_noise(img_pnts_1, noise_scale)

    for pnt in noised_img_pnts_0:
        cv2.circle(img_0, (int(pnt[0]), int(pnt[1])), 3, (0, 0, 0), -1)
    for pnt in noised_img_pnts_1:
        cv2.circle(img_1, (int(pnt[0]), int(pnt[1])), 3, (255, 0, 0), -1)

    (
        F_true_1_to_2,
        rot_1_to_2,
        trans_1_to_2_in_camera_coord,
    ) = calculate_true_fundamental_matrix(
        rot_mat_0, rot_mat_1, T_0_in_camera_coord, T_1_in_camera_coord, camera_matrix
    )

    # if draw_epipolar:
    #     draw_epipolar_lines(img_0, noised_img_pnts_0, noised_img_pnts_1)

    # if draw_test_data:
    #     cv2.imshow("CAM0", cv2.resize(img_0, None, fx=0.5, fy=0.5))
    #     cv2.imshow("CAM1", cv2.resize(img_1, None, fx=0.5, fy=0.5))
    #     cv2.waitKey(0)

    return (
        img_pnts_0,
        img_pnts_1,
        noised_img_pnts_0,# 1
        noised_img_pnts_1,# 1
        F_true_1_to_2,
        rot_1_to_2,# 1
        trans_1_to_2_in_camera_coord,# 1
        points_3d,
    )
    
def create_circle_surface_points(radius, interval_deg):
    points = np.zeros((0, 3))
    for theta in range(360):
        if theta % interval_deg == 0:
            x = radius * np.cos(np.deg2rad(theta))
            y = radius * np.sin(np.deg2rad(theta))
            points = np.append(points, [[x, y, 0]], axis=0)

    return points

def create_curve_surface_points(row, col, z_scale):
    points = np.zeros((0, 3))
    for i in range(row + 1):
        for j in range(col + 1):
            x = i - row / 2
            y = j - col / 2
            z = x**2 * z_scale
            points = np.append(points, [[x, y, z]], axis=0)

    return points

def euler_angle_to_rot_mat(x_deg, y_deg, z_deg):
    x = x_deg / 180 * math.pi
    y = y_deg / 180 * math.pi
    z = z_deg / 180 * math.pi
    R_x = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    R_y = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    R_z = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])

    return np.dot(np.dot(R_x, R_y), R_z)

def add_noise(img_pnts, noise_scale):
    noise = np.random.normal(0, noise_scale, img_pnts.shape)
    noised_points = img_pnts + noise

    return noised_points

def calculate_true_fundamental_matrix(
    rot_mat_before,
    rot_mat_after,
    T_in_camera_coord_before,
    T_in_camera_coord_after,
    camera_matrix,
):
    rot_1_to_2 = np.dot(rot_mat_after, rot_mat_before.T)
    trans_1_to_2_in_camera_coord = (
        np.matrix(T_in_camera_coord_after).T
        - rot_1_to_2 * np.matrix(T_in_camera_coord_before).T
    )
    # trans_1_to_2_in_camera_coord_outer = create_outer_product(
    #     trans_1_to_2_in_camera_coord
    # )
    # A_inv = np.linalg.inv(camera_matrix)
    # F_true_1_to_2 = A_inv.T * trans_1_to_2_in_camera_coord_outer * rot_1_to_2 * A_inv

    # return normalize_F_matrix(F_true_1_to_2), rot_1_to_2, trans_1_to_2_in_camera_coord
    return None, rot_1_to_2, trans_1_to_2_in_camera_coord

def calculate_camera_matrix_from_RT(R, T, f):
    focal_arr = np.array([[f, 0, 0], [0, f, 0], [0, 0, 1]])
    I = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    P = np.dot(focal_arr, I)

    Rt = np.dot(R.T, T)
    R_T = R.T
    moved_arr = np.array(
        [
            [R_T[0, 0], R_T[0, 1], R_T[0, 2], Rt[0, 0]],
            [R_T[1, 0], R_T[1, 1], R_T[1, 2], Rt[1, 0]],
            [R_T[2, 0], R_T[2, 1], R_T[2, 2], Rt[2, 0]],
        ]
    )
    P_after = np.dot(focal_arr, moved_arr)

    return P, P_after

def simple_triangulation(P_0, P_1, f_0, points_0, points_1):
    x_0 = points_0[0]
    y_0 = points_0[1]
    x_1 = points_1[0]
    y_1 = points_1[1]
    T = np.array(
        [
            [
                f_0 * P_0[0, 0] - x_0 * P_0[2, 0],
                f_0 * P_0[0, 1] - x_0 * P_0[2, 1],
                f_0 * P_0[0, 2] - x_0 * P_0[2, 2],
            ],
            [
                f_0 * P_0[1, 0] - y_0 * P_0[2, 0],
                f_0 * P_0[1, 1] - y_0 * P_0[2, 1],
                f_0 * P_0[1, 2] - y_0 * P_0[2, 2],
            ],
            [
                f_0 * P_1[0, 0] - x_1 * P_1[2, 0],
                f_0 * P_1[0, 1] - x_1 * P_1[2, 1],
                f_0 * P_1[0, 2] - x_1 * P_1[2, 2],
            ],
            [
                f_0 * P_1[1, 0] - y_1 * P_1[2, 0],
                f_0 * P_1[1, 1] - y_1 * P_1[2, 1],
                f_0 * P_1[1, 2] - y_1 * P_1[2, 2],
            ],
        ]
    )
    p = np.array(
        [
            f_0 * P_0[0, 3] - x_0 * P_0[2, 3],
            f_0 * P_0[1, 3] - y_0 * P_0[2, 3],
            f_0 * P_1[0, 3] - x_1 * P_1[2, 3],
            f_0 * P_1[1, 3] - y_1 * P_1[2, 3],
        ]
    )
    ans = (-1) * np.dot(np.dot(np.linalg.inv(np.dot(T.T, T)), T.T), p)

    return np.array([ans[0], ans[1], ans[2]])

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


draw_test_data = False
draw_epipolar = False
rot_euler_deg_0 = [0, 0, 0] # X回転, Y回転, Z回転
rot_euler_deg_1 = [0, 0, 0] # X回転, Y回転, Z回転
T_0_in_camera_coord = [0, 0, 10] # X, Y, Z
T_1_in_camera_coord = [0, 0, 10] # X, Y, Z
f = 160
width = 1920
height = 1080

# 対応点
u_1 = 960
v_1 = 540
u_2 = 960
v_2 = 540
(
    img_pnts_0,
    img_pnts_1,
    noised_img_pnts_0,# 1
    noised_img_pnts_1,# 1
    F_true,
    rot_1_to_2,# 1
    trans_1_to_2_in_camera_coord,# 1
    points_3d,
) = prepare_test_data(
    draw_test_data,# 1
    draw_epipolar,# 1
    "CURVE",
    rot_euler_deg_0,# 1
    rot_euler_deg_1,# 1
    T_0_in_camera_coord,# 1
    T_1_in_camera_coord,# 1
    f,# 1
    width,# 1
    height,# 1
)
P_0, P_1 = calculate_camera_matrix_from_RT(
    rot_1_to_2, trans_1_to_2_in_camera_coord, f
)
i = 0
pos = simple_triangulation(
    P_0, P_1, width, noised_img_pnts_0[i], noised_img_pnts_1[i]
)


print(pos)

make(T_0_in_camera_coord, calculate_rotation_matrix(rot_euler_deg_0[0], rot_euler_deg_0[1], rot_euler_deg_0[2]))
make(T_1_in_camera_coord, calculate_rotation_matrix(rot_euler_deg_1[0], rot_euler_deg_1[1], rot_euler_deg_1[2]))
make(pos)

# 凡例を追加
ax.legend()
# プロット表示
plt.show()