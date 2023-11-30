import dd

points = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [5, 5, 5],
]
cam1_position = [0, 0, 0]
cam1_rotation = [1, 1, 1]
cam2_position = [1, 0, 0]
cam2_rotation = [1, 0, 1]

dd.plot_3d_points_with_arrows(
    points,
    cam1_position,
    cam1_rotation,
    cam2_position,
    cam2_rotation,
)
