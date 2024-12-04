# Module transforming RGB images into YCbCr
import numpy as np
import numpy.linalg as alg

mat = np.array(
    [[65.481, 128.553, 24.966], [-37.797, -74.203, 112.0], [112.0, -93.786, -18.214]]
)

col = np.array([[16, 128, 128]])


def rgb_to_ycbcr(rgb: tuple) -> tuple:
    a = np.asarray(rgb)
    b = mat.dot(a)
    return tuple(b + col)


def ycbcr_to_rgb(t: tuple) -> tuple:
    a = np.asarray(t)
    b = alg.inv(mat)
    c = a - col
    d = b.dot(c[0])
    return tuple(d)
