# d1.py

import numpy as np
import math
import cv2
import dd


# calculate rotation matrix using Rodriguez's rotation formula
def rotation_vector_to_matrix(rotation_vector_x, rotation_vector_y, rotation_vector_z):
    theta = math.sqrt(
        rotation_vector_x**2 + rotation_vector_y**2 + rotation_vector_z**2
    )
    axis_x = rotation_vector_x / theta
    axis_y = rotation_vector_y / theta
    axis_z = rotation_vector_z / theta

    c = math.cos(theta)
    s = math.sin(theta)
    C = 1 - c

    x = axis_x
    y = axis_y
    z = axis_z

    # Rotation matrix
    rotation_matrix = np.array(
        [
            [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
            [x * y * C + z * s, y * y * C + c, y * z * C - x * s],
            [x * z * C - y * s, y * z * C + x * s, z * z * C + c],
        ]
    )

    return rotation_matrix


# calculate position matrix
def calculate_position_matrix(x, y, z):
    position_matrix = np.array(
        [
            [x],
            [y],
            [z],
        ]
    )
    return position_matrix


# calculate intrinsic matrix
def calculate_intrinsic_matrix(fx, fy, cx, cy, skew):
    intrinsic_matrix = np.array(
        [
            [fx, skew, cx],
            [0, fy, cy],
            [0, 0, 1],
        ]
    )
    return intrinsic_matrix


# calculate camera matrix
def calculate_camera_matrix(intrinsic_matrix, position_matrix, rotation_matrix):
    camera_matrix = np.dot(
        intrinsic_matrix, np.hstack((rotation_matrix, position_matrix))
    )
    return camera_matrix


# calculate 3D coordinates of corresponding points
def triangulate_points(P1, P2, p1, p2):
    A = np.zeros((4, 4))
    A[0] = p1[0] * P1[2] - P1[0]
    A[1] = p1[1] * P1[2] - P1[1]
    A[2] = p2[0] * P2[2] - P2[0]
    A[3] = p2[1] * P2[2] - P2[1]

    U, S, Vt = np.linalg.svd(A)
    X = Vt[-1, :] / Vt[-1, -1]

    return X[:3]


def main():
    # Camera 1 value
    cam1_position_x = 1.32459546
    cam1_position_y = 1.81193277
    cam1_position_z = 69.56969684
    cam1_position = np.array([cam1_position_x, cam1_position_y, cam1_position_z])
    cam1_rotation_vector_x = -0.01117237
    cam1_rotation_vector_y = -0.77528757
    cam1_rotation_vector_z = 0.00485018
    cam1_rotation = np.array(
        [cam1_rotation_vector_x, cam1_rotation_vector_y, cam1_rotation_vector_z]
    )
    cam1_fx = 2597.87121
    cam1_fy = 2597.55043
    cam1_cx = 936.422977
    cam1_cy = 509.192332
    cam1_skew = 0

    # Camera 2 value
    cam2_position_x = 1.4323922
    cam2_position_y = 1.56932358
    cam2_position_z = 48.70932853
    cam2_position = np.array([cam2_position_x, cam2_position_y, cam2_position_z])
    cam2_rotation_vector_x = -0.00895712
    cam2_rotation_vector_y = 0.00713052
    cam2_rotation_vector_z = 0.00014517
    cam2_rotation = np.array(
        [cam2_rotation_vector_x, cam2_rotation_vector_y, cam2_rotation_vector_z]
    )
    cam2_fx = 2597.87121
    cam2_fy = 2597.55043
    cam2_cx = 936.422977
    cam2_cy = 509.192332
    cam2_skew = 0

    # Camera 3 value
    cam3_position_x = 1.3195483
    cam3_position_y = 1.79905889
    cam3_position_z = 68.26501377
    cam3_position = np.array([cam3_position_x, cam3_position_y, cam3_position_z])
    cam3_rotation_vector_x = -0.0129551
    cam3_rotation_vector_y = 0.79861548
    cam3_rotation_vector_z = -0.00455293
    cam3_rotation = np.array(
        [cam3_rotation_vector_x, cam3_rotation_vector_y, cam3_rotation_vector_z]
    )
    cam3_fx = 2597.87121
    cam3_fy = 2597.55043
    cam3_cx = 936.422977
    cam3_cy = 509.192332
    cam3_skew = 0

    # Coordinates of corresponding points
    cam1_point = np.array(
        [
            [961, 541],
            [986, 540],
            [1012, 539],
            [1037, 540],
            [1062, 540],
            [1087, 540],
            [1111, 540],
            [1134, 540],
            [1157, 540],
            [959, 577],
            [986, 577],
            [1012, 577],
            [1037, 576],
            [1062, 576],
            [1087, 576],
            [1111, 576],
            [1134, 576],
            [1157, 576],
            [959, 616],
            [986, 615],
            [1157, 610],
            [959, 652],
            [986, 652],
            [1062, 649],
            [1087, 648],
            [1157, 645],
            [959, 691],
            [986, 690],
            [1063, 685],
            [1087, 684],
            [1157, 679],
            [959, 728],
            [986, 727],
            [1157, 715],
            [959, 766],
            [987, 764],
            [1011, 762],
            [1038, 759],
            [1062, 758],
            [1087, 755],
            [1110, 753],
            [1134, 752],
            [1156, 749],
        ]
    )
    cam2_point = np.array(
        [
            [960, 540],
            [1013, 540],
            [1067, 540],
            [1119, 540],
            [1173, 540],
            [1226, 540],
            [1280, 540],
            [1332, 540],
            [1387, 540],
            [960, 593],
            [1013, 594],
            [1067, 593],
            [1120, 593],
            [1172, 593],
            [1226, 593],
            [1280, 593],
            [1332, 593],
            [1386, 593],
            [960, 647],
            [1013, 647],
            [1387, 647],
            [959, 699],
            [1013, 700],
            [1172, 700],
            [1227, 701],
            [1387, 700],
            [959, 754],
            [1013, 753],
            [1173, 754],
            [1227, 753],
            [1386, 753],
            [959, 806],
            [1013, 807],
            [1387, 807],
            [959, 860],
            [1014, 860],
            [1066, 860],
            [1121, 860],
            [1172, 860],
            [1227, 860],
            [1279, 860],
            [1334, 860],
            [1386, 860],
        ]
    )
    cam3_point = np.array(
        [
            [960, 540],
            [986, 540],
            [1014, 540],
            [1042, 540],
            [1071, 540],
            [1100, 540],
            [1130, 540],
            [1160, 540],
            [1192, 540],
            [959, 577],
            [987, 578],
            [1014, 578],
            [1042, 579],
            [1071, 580],
            [1101, 580],
            [1130, 580],
            [1161, 581],
            [1192, 581],
            [960, 616],
            [987, 616],
            [1192, 621],
            [960, 653],
            [987, 654],
            [1071, 658],
            [1100, 659],
            [1191, 663],
            [960, 692],
            [987, 693],
            [1070, 697],
            [1100, 699],
            [1191, 703],
            [960, 728],
            [987, 731],
            [1192, 745],
            [960, 766],
            [988, 768],
            [1014, 771],
            [1043, 773],
            [1071, 775],
            [1101, 778],
            [1129, 780],
            [1162, 784],
            [1191, 785],
        ]
    )
    # cam1_point = np.array([986, 540])
    # cam2_point = np.array([1013, 540])

    # Calculate camera matrix for camera 1
    cam1_rotation_matrix = rotation_vector_to_matrix(
        cam1_rotation_vector_x, cam1_rotation_vector_y, cam1_rotation_vector_z
    )
    cam1_position_matrix = calculate_position_matrix(
        cam1_position_x, cam1_position_y, cam1_position_z
    )
    cam1_intrinsic_matrix = calculate_intrinsic_matrix(
        cam1_fx, cam1_fy, cam1_cx, cam1_cy, cam1_skew
    )
    cam1_camera_matrix = calculate_camera_matrix(
        cam1_intrinsic_matrix, cam1_position_matrix, cam1_rotation_matrix
    )

    # Calculate camera matrix for camera 2
    cam2_rotation_matrix = rotation_vector_to_matrix(
        cam2_rotation_vector_x, cam2_rotation_vector_y, cam2_rotation_vector_z
    )
    cam2_position_matrix = calculate_position_matrix(
        cam2_position_x, cam2_position_y, cam2_position_z
    )
    cam2_intrinsic_matrix = calculate_intrinsic_matrix(
        cam2_fx, cam2_fy, cam2_cx, cam2_cy, cam2_skew
    )
    cam2_camera_matrix = calculate_camera_matrix(
        cam2_intrinsic_matrix, cam2_position_matrix, cam2_rotation_matrix
    )

    # Calculate camera matrix for camera 3
    cam3_rotation_matrix = rotation_vector_to_matrix(
        cam3_rotation_vector_x, cam3_rotation_vector_y, cam3_rotation_vector_z
    )
    cam3_position_matrix = calculate_position_matrix(
        cam3_position_x, cam3_position_y, cam3_position_z
    )
    cam3_intrinsic_matrix = calculate_intrinsic_matrix(
        cam3_fx, cam3_fy, cam3_cx, cam3_cy, cam3_skew
    )
    cam3_camera_matrix = calculate_camera_matrix(
        cam3_intrinsic_matrix, cam3_position_matrix, cam3_rotation_matrix
    )

    # Calculate 3D coordinates of corresponding points
    # reconstructed_points = triangulate_points(
    #     cam1_camera_matrix, cam2_camera_matrix, cam1_point, cam2_point
    # )
    # print(reconstructed_points)

    # n_points = cam1_point.shape[0]
    # points = []
    # for i in range(n_points):
    #     p1 = cam1_point[i]
    #     p2 = cam2_point[i]

    #     point_3d = triangulate_points(cam1_camera_matrix, cam2_camera_matrix, p1, p2)
    #     points.append(point_3d)

    # points = np.asarray(points)
    # print(points)
    # dd.plot_3d_points_with_arrows(
    #     points,
    #     cam1_position,
    #     cam1_rotation,
    #     cam2_position,
    #     cam2_rotation,
    # )

    n_points = cam1_point.shape[0]
    points = []
    for i in range(n_points):
        p1 = cam1_point[i]
        p2 = cam3_point[i]

        point_3d = triangulate_points(cam1_camera_matrix, cam3_camera_matrix, p1, p2)
        points.append(point_3d)

    points = np.asarray(points)
    print(points)
    dd.plot_3d_points_with_arrows(
        points,
        cam1_position,
        cam1_rotation,
        cam3_position,
        cam3_rotation,
    )


if __name__ == "__main__":
    main()
