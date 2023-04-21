import numpy as np
import cv2

# チェスボードのサイズ(縦, 横)
board_size = (8, 6)

# チェスボードの正方形の1辺の長さ（メートル単位）
square_size = 0.1

# チェスボードの座標を計算する
objp = np.zeros((board_size[0]*board_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2) * square_size

# チェスボードの画像を生成する
img = np.zeros((board_size[1]*int(square_size*1000)+2*int(square_size*1000), board_size[0]*int(square_size*1000)+2*int(square_size*1000), 3), dtype=np.uint8)
img.fill(255)

# 偶数行と奇数行で色を交互に変える
for i in range(board_size[1]):
    for j in range(board_size[0]):
        if i % 2 == 0 and j % 2 == 0:
            img[(i+1)*int(square_size*1000):(i+2)*int(square_size*1000), (j+1)*int(square_size*1000):(j+2)*int(square_size*1000)] = (0, 0, 0)
        elif i % 2 == 1 and j % 2 == 1:
            img[(i+1)*int(square_size*1000):(i+2)*int(square_size*1000), (j+1)*int(square_size*1000):(j+2)*int(square_size*1000)] = (0, 0, 0)

# チェスボードの外側に白枠を描画する
img[0:int(square_size*1000), :] = 255
img[-int(square_size*1000):, :] = 255
img[:, 0:int(square_size*1000)] = 255
img[:, -int(square_size*1000):] = 255

# 画像を保存する
cv2.imwrite('chessboard.png', img)
