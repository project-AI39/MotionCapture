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




# カメラ1のパラメータ
X_1 = np.array([1, 3, 2]) # カメラ座標
R_1 = np.eye(3) # 回転行列 
K_1 = np.array([[2597.87121, 0.00000000, 936.422977],
 [0.00000000, 2597.55043, 509.192332],
 [0.00000000, 0.00000000, 1.00000000]]) # 内部パラメータ
u_1 = 320 # 対応点のu座標
v_1 = 240 # 対応点のv座標

# カメラ2のパラメータ 
X_2 = np.array([3, 1, 2])
R_2 = np.array([[ 0.5, -0.5,  0.        ],
               [ 0.5,  0.5,  0.        ],
               [ 0.         , 0.         ,  1.        ]])
K_2 = np.array([[2597.87121, 0.00000000, 936.422977],
 [0.00000000, 2597.55043, 509.192332],
 [0.00000000, 0.00000000, 1.00000000]])
u_2 = 320  
v_2 = 240

# カメラ行列を求める
P_1 = K_1 @ np.hstack([R_1, X_1.reshape(3,1)])
P_2 = K_2 @ np.hstack([R_2, X_2.reshape(3,1)])

# 三角測量
X = cv2.triangulatePoints(P_1, P_2, (u_1, v_1), (u_2, v_2))
X = X/X[3]
print(X)

# データをプロット
make(X_1, R_1)
make(X_2, R_2)
make(X)

# 凡例を追加
ax.legend()
# プロット表示
plt.show()