import numpy as np

# カメラパラメータ
X_1 = np.array([0, 0, 0])
R_1 = np.array([[1,0,0],[0,1,0],[0,0,1]]) 
K_1 = np.array([[1, 0, 960], [0, 1, 540], [0, 0, 1]])

X_2 = np.array([10, 0, 0])  
R_2 = np.array([[ 0.70710678, -0.70710678,  0.        ],
               [ 0.70710678,  0.70710678,  0.        ],
               [ 0.         , 0.         ,  1.        ]])
K_2 = np.array([[1, 0, 960], [0, 1, 540], [0, 0, 1]])

# 投影行列計算
P_1 = K_1 @ np.hstack((R_1, X_1.reshape(3,1)))
P_2 = K_2 @ np.hstack((R_2, X_2.reshape(3,1)))

# 対応点
u_1 = 960
v_1 = 540
u_2 = 960
v_2 = 540

# 三角測量
A = np.zeros((4,4))
A[0] = u_1 * P_1[2] - P_1[0]
A[1] = v_1 * P_1[2] - P_1[1] 
A[2] = u_2 * P_2[2] - P_2[0]
A[3] = v_2 * P_2[2] - P_2[1]

U, S, Vh = np.linalg.svd(A)
X = Vh[-1,0:-1]/Vh[-1,-1]

print("X=",X)