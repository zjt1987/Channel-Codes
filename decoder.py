# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 10:43:23 2022
A general decoder for LDPC codes
Min-Sum algorithm
@author: yinri
"""
import numpy as np

def decoder_gen(llr, H, num_iter):
    n_row = H.shape[0]
    n_col = H.shape[1]
    y = np.zeros(n_col)
    
    msg = np.zeros(H.shape)
    for n in range(n_row):
        idx_row = np.where(H[n, :] == 1)
        msg[n, idx_row] = llr[idx_row]
    
    for n in range(num_iter):
        for m in range(n_row):
            idx_row = np.where(H[m,:] == 1)
            msg[m, idx_row], c = cnp(msg[m, idx_row].ravel())
            
        for m in range(n_col):
            idx_col = np.where(H[:,m] == 1)
            msg[idx_col, m], y[m] = vnp(msg[idx_col, m], llr[m])
            
        #w = (1-y) / 2
        
    return y


def vnp(msg, llr):
    s = np.sum(msg) + llr
    a = np.sign(s)
    return s-msg, a

def cnp(msg):
    s = np.sign(msg)
    a = np.abs(msg)
    pos = np.argpartition(a, 2)[:2]
    #print(a)
    g_min = a[pos]
    c = np.prod(s)
    x = np.zeros(msg.shape)
    
    for n in range(x.size):
        s_y = c / s[n]
        if n == pos[0]:
            x[n] = s_y * g_min[1]
        else:
            x[n] = s_y * g_min[0]
            
    return x, c
    
    