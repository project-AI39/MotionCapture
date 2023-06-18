import cv2
import numpy as np

# カメラの数を指定する
num_cameras = 3

# カメラの内部パラメータ行列と外部パラメータ行列をランダムに生成する
Ks = []
Rs = []
ts = []
for i in range(num_cameras):
    K = np.random.rand(3, 3)
    Ks.append(K)
    R = cv2.Rodrigues(np.random.rand(3))[0]
    Rs.append(R)
    t = np.random.rand(3, 1)
    ts.append(t)

# 2つのカメラからの対応点座標をランダムに生成する
num_points = 10
pts1 = []
pts2 = []
for i in range(num_points):
    x = np.random.rand(3, 1)
    pts1.append(x)
    pts2.append([])
    for j in range(1, num_cameras):
        x_ = R.dot(x) + t
        x_ = K.dot(x_[:2] / x_[2])
        x_ += np.random.randn(2, 1) * 5
        pts2[i].append(x_)

# cv::sfm::triangulatePoints関数を呼び出して、三角測量を行う
points_3d = np.zeros((num_points, 3))
for i in range(num_points):
    A = np.zeros((4 * num_cameras, 4))
    for j in range(num_cameras):
        K = Ks[j]
        R = Rs[j]
        t = ts[j]
        pt1 = np.append(pts1[i], 1)
        pt2 = np.append(pts2[i][j], 1)
        A[4*j:4*j+4, :] = np.kron(pt1, K.dot(R).dot(np.hstack((np.eye(3), -t)))).reshape(4, 4)
        b = np.linalg.svd(A)[2][-1, :]
        points_3d[i, :] = b[:3] / b[3]

# points_3dには、対応点座標から三角測量によって得られた3次元点座標が格納される
