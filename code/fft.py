# fast-fourier transforms

import complex as cpx
from numpy import log2
from cmath import pi, exp, cos
from scipy.fftpack import dct, idct


def FFT(vector:list) -> list:
    """calculate the fast fourier tranform of a vector
    
    parameters
    ----------
        -vector : list of Complex object
    
    return
    ------
        - 1-D fast fourier transform of the vector"""
    n = len(vector) 
    assert log2(n).is_integer(), "make sure that the length of the arguement is a power of 2"
    if n == 1:
        return vector
    poly_even, poly_odd = vector[::2] , vector[1::2]
    res_even, res_odd = FFT(poly_even), FFT(poly_odd)
    res = [cpx.Complex(0)] * n 
    for j in range(n//2):
        w_j = cpx.exp_to_literal(-2*pi*j/n)
        product = w_j * res_odd[j]
        res[j] = res_even[j] + product
        res[j + n//2] = res_even[j] - product
    return res

def IFFT_aux(vector:list) -> list:
    """auxiliary function that makes the recursive steps of the IFFT algorithm
    parameters
    ----------
        -vector : list of Complex object
    
    return
    ------
        - partial inverse of the 1-D fast fourier transform of the vector (lack the division by n)"""
    n = len(vector) 
    assert log2(n).is_integer(), "make sure that the length of the arguement is a power of 2"
    if n == 1:
        return vector
    poly_even, poly_odd = vector[::2] , vector[1::2]
    res_even, res_odd = IFFT_aux(poly_even), IFFT_aux(poly_odd)
    res = [cpx.Complex(0)] * n 
    for j in range(n//2):
        w_j = cpx.exp_to_literal((2 * pi * j) / n)
        product = w_j * res_odd[j]
        res[j] = res_even[j] + product
        res[j + n//2] = res_even[j] - product
    return res

def IFFT(vector:list) -> list:
    """caclulate the inverse of the fast fourier tranform of a vector (in order to have ifft(fft(poly)) == poly)
    
    parameters
    ----------
        -vector : list of Complex object
    
    return
    ------
        - inverse of the 1-D fast fourier transform of the vector"""
    n = len(vector)
    res = IFFT_aux(vector)
    for i in range(n):
        res[i] = res[i] / cpx.Complex(n)
    return res

def DCT(vector:list, orthogonalize:bool =False, norm="forward"):
    """calculate the one-dimensional type-II discrete cosine tranform of a matrix (MAKHOUL) (using the FFT function previously defined)
    
    parameters
    ----------
        - vector: list of Numerical Object
        
    return
    ------
        - discrete cosine tranform of the input"""
    N = len(vector)
    temp = vector[ : : 2] + vector[-1 - N % 2 : : -2] 
    temp = FFT(temp)
    factor = - pi / (N * 2)
    result = [2 * (val * (cpx.exp_to_literal(i * factor))).re for (i, val) in enumerate(temp)]
    if orthogonalize:
        result[0] *= 2 ** (-1 / 2)
    if norm == "ortho":
        result[0] *= (N) **(-1 / 2)
        result[1::] = [(2 / N) ** (1 / 2) * result[i] for i in range(1, len(result))]
    return result

def IDCT(vector:list):
    """calculate the one-dimensional "inverse" type-III discrete cosine tranform of a matrix (MAKHOUL) (using the FFT function previously defined)
    
    parameters
    --------
        - vector: list of Numerical Object
        
    return
    ------
        - type-III discrete cosine tranform of the input"""
    N = len(vector)
    factor = - pi / (N * 2)
    temp = [(cpx.Complex(val) if i > 0 else (cpx.Complex(val) / cpx.Complex(2))) * cpx.exp_to_literal(i * factor) for (i, val) in enumerate(vector)]
    temp = FFT(temp)
    temp = [val.re for val in temp]
    result = [None] * N
    result[ : : 2] = temp[ : (N + 1) // 2]
    result[-1 - N % 2 : : -2] = temp[(N + 1) // 2 : ]
    return result

if __name__ == "__main__":
    vectorCpx= [cpx.Complex(5), cpx.Complex(2), cpx.Complex(4), cpx.Complex(8)]
    vector = [5, 2, 4, 8]
    print("DCT : ", DCT(vectorCpx))
    print("inverse + DCT : ", IDCT((DCT(vectorCpx))))
    print("scipy dct :", dct(vector))
    print("scipy + inverse dct: ", dct(idct(vector)))
    print("scipy dct (ortho) : ", dct(vector, norm = "ortho"))
    print("scipy inverse + dct (ortho) : ", idct(dct(vector, norm="ortho"), norm="ortho"))