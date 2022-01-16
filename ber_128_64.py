# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 09:28:56 2022

@author: yinri
"""
import numpy as np
from numpy.linalg import matrix_power

from decoder import decoder_gen

def hex2bin(s):
    x = np.array([hex2bin_single(c) for c in s])
    return x.ravel()

def hex2bin_single(s):
    """
    Parameters
    ----------
    s : string of a single character, one of 'a-fA-F0-9'    

    Returns
    -------
    a list of binary representation of s
    e.g. hex2bin_single('e') returns [1, 1, 1, 0]

    """
    a = np.array([[0,0,0,0], [0,0,0,1], [0,0,1,0], [0,0,1,1],
                  [0,1,0,0], [0,1,0,1], [0,1,1,0], [0,1,1,1],
                  [1,0,0,0], [1,0,0,1], [1,0,1,0], [1,0,1,1],
                  [1,1,0,0], [1,1,0,1], [1,1,1,0], [1,1,1,1]])
    
    return a[int(s, 16)]

phi = np.roll(np.eye(16), 1, axis=1)
h = np.block([
    [np.eye(16)+matrix_power(phi, 7), matrix_power(phi, 2), 
     matrix_power(phi, 14), matrix_power(phi, 6),
     np.zeros([16,16]), np.eye(16),
     matrix_power(phi, 13), np.eye(16)],
    [matrix_power(phi, 6), np.eye(16)+matrix_power(phi, 15),
     np.eye(16), phi,
     np.eye(16), np.zeros([16, 16]),
     np.eye(16), matrix_power(phi, 7)],
    [matrix_power(phi, 4), phi,
     np.eye(16)+matrix_power(phi, 15), matrix_power(phi, 14),
     matrix_power(phi, 11), np.eye(16),
     np.zeros([16, 16]), matrix_power(phi, 3)],
    [np.eye(16), phi,
     matrix_power(phi, 9), np.eye(16)+matrix_power(phi, 13),
     matrix_power(phi, 14), phi,
     np.eye(16), np.zeros([16, 16])]])
h = h%2

w_str = list(['0E69166BEF4C0BC2', '7766137EBB248418', 'C480FEB9CD53A713', '4EAA22FA465EEA11'])
w = np.zeros([64, 64])
w[0, :] = hex2bin(w_str[0])
w[16, :] = hex2bin(w_str[1])
w[32, :] = hex2bin(w_str[2])
w[48, :] = hex2bin(w_str[3])

for n in range(4):
    for k in range(1, 16):
        for m in range(4):
            w[n*16+k, m*16:(m+1)*16] = np.roll(w[n*16, m*16:(m+1)*16], k)

g = np.block([np.eye(64), w])

# to verify h and g, run
# np.sum(np.matmul(h, g.T)%2)
# if the result is 0, then h and g are correct

SNR_MIN = 3.5
SNR_MAX = 5
SNR_STEP = 0.5
Eb_N0 = np.arange(SNR_MIN, SNR_MAX+1, SNR_STEP)
SNR = 10**(Eb_N0/10.0)
BER = np.zeros(SNR.shape)

rng = np.random.default_rng(1)
loop = 0
for snr in SNR:
    N0 = 1/snr
    err_num = 0
    for k in range(5000):
        x = rng.integers(2, size=64)
        c = np.matmul(x, g) % 2
        tx = 1-2*c
        noise = rng.normal(0, np.sqrt(N0), 128)
        llr_coded = tx + noise
        dec_data = decoder_gen(llr_coded, h, 20)
        err_num += np.where(tx!=dec_data)[0].size
        
    BER[loop] = err_num/128/5000
    loop += 1