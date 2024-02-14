#!/usr/bin/env python

#-------------------------------------------------------
# BandaLoRE package
# Approximate a given LD matrix using the sum of 
# a banded martrix and a low-rank matrix

# Version: 0.0.1
#-------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import time, sys, logging
from functools import reduce
import argparse
import scipy.linalg
import torch
import scipy.sparse.linalg
from sklearn.utils.extmath import randomized_svd
import scipy.sparse
from pandas_plink import read_plink1_bin, read_grm, read_rel, read_plink

__version__ = '0.0.1'

borderline = "<><><<>><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>"
short_borderline = "<><><<>><><><><><><><><><><><><><><><><><><><><><><><><><><><><>\n"
header ="\n"
header += borderline +"\n"
header += "<>\n"
header += "<> BandaLoRE: Banded and Low-Rank approximation of the empirical LD matrix\n"
header += "<> Version: {}\n".format(str(__version__))
header += "<> (C) 2024 Hui Li, Xiang Meng, Rahul Mazumder and Xihong Lin\n"
header += "<> Harvard University Department of Biostatistics\n"
header += "<> MIT Sloan School of Management, Operations Research Center and Center for Statistics and Data Science\n"
header += "<> MIT License Copyright (c) 2024 Hui Li \n"
header += borderline + "\n"
header += "<> Note:  It is recommended to run your own QC on the input before using this program. \n"
header += "<> Software-related correspondence: hui_li@g.harvard.edu \n"
header += borderline +"\n"
header += "\n\n"

pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 800)
pd.set_option('display.precision', 12)
pd.set_option('max_colwidth', 800)
pd.set_option('colheader_justify', 'left')

np.set_printoptions(linewidth=800)
np.set_printoptions(precision=3)

def sec_to_str(t):
    '''Convert seconds to days:hours:minutes:seconds'''
    [d, h, m, s, n] = reduce(lambda ll, b : divmod(ll[0], b) + ll[1:], [(t, 1), 60, 60, 24])
    f = ''
    if d > 0:
        f += '{D}d:'.format(D=d)
    if h > 0:
        f += '{H}h:'.format(H=h)
    if m > 0:
        f += '{M}m:'.format(M=m)

    f += '{S}s'.format(S=s)
    return f

#====================
# Lanzcos algorithm
#====================
def Lanczos(A, v, r=100, rediag=True):
    # A: specified as a matrix pxp
    # reference from: https://en.wikipedia.org/wiki/Lanczos_algorithm
    p = len(v)
    if r > p: r = p;
    
    # initialize variables
    V = np.zeros((p,r))
    T = np.zeros((r,r))
    beta = np.zeros(r) # the first entry will not be used
    alpha = np.zeros(r)
    vo = np.zeros(p)

    # step 1. 
    V[:,0] = v / np.linalg.norm(v)

    # step 2. 
    w = np.dot(A,V[:,0])
    alpha[0] = np.dot(w,V[:,0])
    w = w - alpha[0] * V[:,0]

    # step 3.
    for j in range(1,r):
        beta[j] = np.linalg.norm(w)
        
        if beta[j] != 0:
            V[:,j] = w / beta[j]

        if rediag:
            # logging.info("Rediagonalization for all vectors up to j")
            for i in range(j-1):
                V[:,j] = V[:,j] - np.dot(np.conj(V[:,j]), V[:,i])*V[:,i]
            # normalize the new v
            V[:,j] = V[:,j] / np.linalg.norm(V[:,j])

        w = np.dot(A,V[:,j])
        alpha[j] = np.dot(w,V[:,j])
        w = w - alpha[j]*V[:,j] - beta[j]*V[:,j-1]

    return alpha, beta[1:], V

def Lanczos_vec(X_sp, v, r=100, rediag=True):
    p = len(v)
    if r > p: r = p;
    # initialize variables
    V = np.zeros((p,r))
    T = np.zeros((r,r))
    beta = np.zeros(r) # the first entry will not be used
    alpha = np.zeros(r)
    vo = np.zeros(p)
    # step 1. 
    V[:,0] = v / np.linalg.norm(v)
    # step 2. 
    w = X_sp.T.dot(X_sp.dot(V[:,0]))
    # w = np.dot(A,V[:,0])
    alpha[0] = np.dot(w, V[:,0])
    w = w - alpha[0] * V[:,0]
    # step 3.
    for j in range(1,r):
        beta[j] = np.linalg.norm(w)
        if beta[j] != 0:
            V[:,j] = w / beta[j]
        if rediag:
            # logging.info("Rediagonalization for all vectors up to j")
            for i in range(j-1):
                V[:,j] = V[:,j] - np.dot(V[:,j], V[:,i])*V[:,i] #np.conj(V[:,j])
            # normalize the new v
            V[:,j] = V[:,j] / np.linalg.norm(V[:,j])
        w = X_sp.T.dot(X_sp.dot(V[:,j]))
        # w = np.dot(A,V[:,j])
        alpha[j] = np.dot(w, V[:,j])
        w = w - alpha[j]*V[:,j] - beta[j]*V[:,j-1]
    return alpha, beta[1:], V

#====================
# BCD joint algs
#====================
def blockgrad_UL_tune(H, Uinit, Linit, bandlen, lowr, stepsize_L, stepsize_U, max_iter, stop_ratio = 0, ifprint = True):
    
    # H: data matrix
    # Uinit/Linit: initial solutions
    # bandlen: band width
    # lowr: rank
    # stepsize_L/stepsize_U: initial value of s_L and s_U
    # max_iter: number of iterations
    # stop_ratio: terminate alg if \|H - LL^T - UU^T \|_F^2 <= stop_ratio * \|H\|_F^2
    # ifprint: whether print results

    
    st = time.time()
    U = np.copy(Uinit)
    L = np.copy(Linit)
    loss_H = np.zeros(max_iter)
    time_L = np.zeros(max_iter)
    time_U = np.zeros(max_iter)
    p = H.shape[0]
    grad_eta = 1e-2
    max_searchtimeU = 20
    max_searchtimeL = 20
    num_band = (2*bandlen-1)*p - bandlen * (bandlen-1)
    num_LL = int(bandlen*p - bandlen * (bandlen-1) / 2)
    band_ind = np.ones((p,p)) * np.tri(p, k=bandlen-1) * (np.tri(p, k=bandlen-1).T)
    
    U = torch.from_numpy(U)
    L = torch.from_numpy(L)
    H = torch.from_numpy(H)
    
    
    H_fro = torch.sum(H**2)
    Bsp1 = np.zeros((p+1,), dtype='int')
    Bsp2 = np.zeros((num_band,), dtype='int')
    for i4 in range(p):
        imin = np.maximum(0,i4-bandlen+1)
        imax = np.minimum(p,i4+bandlen)
        Bsp1[i4+1] = Bsp1[i4] + (imax-imin)
        Bsp2[Bsp1[i4]:Bsp1[i4+1]] = np.arange(imin,imax)
        
    Lsp1 = np.zeros((p+1,), dtype='int')
    Lsp2 = np.zeros((num_LL,), dtype='int')
    Lsp3 = np.zeros((num_LL,))
    for i4 in range(p):
        imin = np.maximum(0,i4-bandlen+1)
        imax = np.minimum(p,i4+1)
        Lsp1[i4+1] = Lsp1[i4] + (imax-imin)
        Lsp2[Lsp1[i4]:Lsp1[i4+1]] = np.arange(imin,imax)
        Lsp3[Lsp1[i4]:Lsp1[i4+1]] = L[i4,imin:imax].numpy()
        
    Lsp1T = np.zeros((p+1,), dtype='int')
    Lsp2T = np.zeros((num_LL,), dtype='int')
    Lsp3T = np.zeros((num_LL,))
    for i4 in range(p):
        imin = np.maximum(0,i4)
        imax = np.minimum(p,i4+bandlen)
        Lsp1T[i4+1] = Lsp1T[i4] + (imax-imin)
        Lsp2T[Lsp1T[i4]:Lsp1T[i4+1]] = np.arange(imin,imax)
        Lsp3T[Lsp1T[i4]:Lsp1T[i4+1]] = L[imin:imax,i4].numpy()
        
    Lsp = torch.sparse_csr_tensor(Lsp1,Lsp2,Lsp3, dtype=torch.float64)
    LspT = torch.sparse_csr_tensor(Lsp1T,Lsp2T,Lsp3T, dtype=torch.float64)
    grad_Lsp3 = np.zeros((num_LL,))
    grad_Lsp3T = np.zeros((num_LL,))
    
    B = (Lsp@LspT).to_dense()
    UUT =  U@(U.T)
    res = H - B - UUT
    time_init = time.time() - st
    del Uinit, Linit, L
    
    U_stepinit = stepsize_U
    L_stepinit = stepsize_L
    
    for i in range(max_iter):
        
        st = time.time()
        grad_U = -4*res@U 
        grad_Unorm = np.linalg.norm(grad_U,'fro')**2
        linestep_U = U_stepinit
        search_time = 0
        ori_loss = np.linalg.norm(res,'fro')**2
        HB = H - B
        while search_time < max_searchtimeU:
            U_test = U - linestep_U*grad_U
            UUT_test = U_test@(U_test.T)    
            res_test = HB - UUT_test    
            cur_loss = np.linalg.norm(res_test,'fro')**2
            des_ratio = (ori_loss-cur_loss) / grad_Unorm / linestep_U

            if ifprint:
                print("Iter {}, search time (U) is {}, descent ratio (U) is {}".format(i,search_time, des_ratio))
            if des_ratio > grad_eta:
                U = U_test
                UUT = UUT_test
                res = res_test
                break
            linestep_U /= 2
            search_time += 1
        
        del HB, UUT_test, res_test
        time_U[i] = time.time() - st
        U_stepinit = linestep_U * 2
        st = time.time()
        
        ressp = (res*band_ind).to_sparse_csr()
        grad_L = (ressp@Lsp).to_dense()
        grad_Lnorm = 0
        for i4 in range(p):
            imin = np.maximum(0,i4-bandlen+1)
            imax = np.minimum(p,i4+1)
            grad_Lsp3[Lsp1[i4]:Lsp1[i4+1]] = grad_L[i4,imin:imax].numpy()
            grad_Lnorm += np.linalg.norm(grad_L[i4,imin:imax])**2
            
        for i4 in range(p):
            imin = np.maximum(0,i4)
            imax = np.minimum(p,i4+bandlen)
            grad_Lsp3T[Lsp1T[i4]:Lsp1T[i4+1]] = grad_L[imin:imax,i4].numpy()
        
        linestep_L = L_stepinit
        search_time = 0
        ori_loss = np.linalg.norm(res,'fro')**2
        HU = H-UUT 
        loss_tmp = ori_loss / H_fro
        
        while search_time < max_searchtimeL:
            Lsp3test = Lsp3 - (-4)*linestep_L*grad_Lsp3
            Lsp3Ttest = Lsp3T - (-4)*linestep_L*grad_Lsp3T
            
            Lsptest = torch.sparse_csr_tensor(Lsp1,Lsp2,Lsp3test, dtype=torch.float64)
            LspTtest = torch.sparse_csr_tensor(Lsp1T,Lsp2T,Lsp3Ttest, dtype=torch.float64)
            B_test = (Lsptest@LspTtest).to_dense()
            res_test = HU - B_test   
            cur_loss = np.linalg.norm(res_test,'fro')**2
            des_ratio = (ori_loss-cur_loss) / grad_Lnorm / linestep_L

            if ifprint:
                print("Iter {}, search time (L) is {}, descent ratio (L) is {}".format(i,search_time, des_ratio))
            if des_ratio > grad_eta:
                Lsp3 = Lsp3test
                Lsp3T = Lsp3Ttest
                Lsp = Lsptest
                LspT = LspTtest
                B = B_test
                res = res_test
                break
            linestep_L /= 2
            search_time += 1 
        
        loss_H[i] = np.linalg.norm(res,'fro')**2 
        time_L[i] = time.time() - st
        L_stepinit = linestep_L * 2
        
        del HU, B_test, res_test
        
        if ifprint:
            print("Iter {}, loss (U) is {}, loss (L) is {}".format(i,loss_tmp, loss_H[i]))
        
        print(loss_H[i] , stop_ratio)
        if loss_H[i] < stop_ratio * H_fro:
            break
        
    
    return (Lsp@LspT).to_dense().numpy(), U.numpy(), loss_H, (time_init,time_L,time_U)


#====================
# GD sequential algs
#====================
def grad_Lband(H, Linit, bandlen, stepsize_L, max_iter, ifprint = True):
    
    # H: data matrix
    # Linit: initial solutions
    # bandlen: band width
    # stepsize_L: initial value of s_L 
    # max_iter: number of iterations
    # ifprint: whether print results

    
    st = time.time()
    L = np.copy(Linit)
    loss_H = np.zeros(max_iter)
    time_L = np.zeros(max_iter)
    p = H.shape[0]
    grad_eta = 1e-2
    max_searchtimeL = 20
    num_band = (2*bandlen-1)*p - bandlen * (bandlen-1)
    num_LL = int(bandlen*p - bandlen * (bandlen-1) / 2)
    band_ind = np.ones((p,p)) * np.tri(p, k=bandlen-1) * (np.tri(p, k=bandlen-1).T)
    
    L = torch.from_numpy(L)
    H = torch.from_numpy(H)
    H_fro = torch.sum(H**2)
    Bsp1 = np.zeros((p+1,), dtype='int')
    Bsp2 = np.zeros((num_band,), dtype='int')
    for i4 in range(p):
        imin = np.maximum(0,i4-bandlen+1)
        imax = np.minimum(p,i4+bandlen)
        Bsp1[i4+1] = Bsp1[i4] + (imax-imin)
        Bsp2[Bsp1[i4]:Bsp1[i4+1]] = np.arange(imin,imax)
        
    Lsp1 = np.zeros((p+1,), dtype='int')
    Lsp2 = np.zeros((num_LL,), dtype='int')
    Lsp3 = np.zeros((num_LL,))
    for i4 in range(p):
        imin = np.maximum(0,i4-bandlen+1)
        imax = np.minimum(p,i4+1)
        Lsp1[i4+1] = Lsp1[i4] + (imax-imin)
        Lsp2[Lsp1[i4]:Lsp1[i4+1]] = np.arange(imin,imax)
        Lsp3[Lsp1[i4]:Lsp1[i4+1]] = L[i4,imin:imax].numpy()
        
    Lsp1T = np.zeros((p+1,), dtype='int')
    Lsp2T = np.zeros((num_LL,), dtype='int')
    Lsp3T = np.zeros((num_LL,))
    for i4 in range(p):
        imin = np.maximum(0,i4)
        imax = np.minimum(p,i4+bandlen)
        Lsp1T[i4+1] = Lsp1T[i4] + (imax-imin)
        Lsp2T[Lsp1T[i4]:Lsp1T[i4+1]] = np.arange(imin,imax)
        Lsp3T[Lsp1T[i4]:Lsp1T[i4+1]] = L[imin:imax,i4].numpy()
        
    Lsp = torch.sparse_csr_tensor(Lsp1,Lsp2,Lsp3, dtype=torch.float64)
    LspT = torch.sparse_csr_tensor(Lsp1T,Lsp2T,Lsp3T, dtype=torch.float64)
    grad_Lsp3 = np.zeros((num_LL,))
    grad_Lsp3T = np.zeros((num_LL,))
    
    time_init = time.time() - st
    del L
    
    for i in range(max_iter):
        
        st = time.time()
        res = H - (Lsp@LspT).to_dense()
        ressp = (res*band_ind).to_sparse_csr()
        grad_L = (ressp@Lsp).to_dense()
        grad_Lnorm = 0
        for i4 in range(p):
            imin = np.maximum(0,i4-bandlen+1)
            imax = np.minimum(p,i4+1)
            grad_Lsp3[Lsp1[i4]:Lsp1[i4+1]] = grad_L[i4,imin:imax].numpy()
            grad_Lnorm += np.linalg.norm(grad_L[i4,imin:imax])**2
            
        for i4 in range(p):
            imin = np.maximum(0,i4)
            imax = np.minimum(p,i4+bandlen)
            grad_Lsp3T[Lsp1T[i4]:Lsp1T[i4+1]] = grad_L[imin:imax,i4].numpy()
        
        linestep_L = stepsize_L
        search_time = 0
        ori_loss = np.linalg.norm(res,'fro')**2
        loss_tmp = ori_loss
        
        while search_time < max_searchtimeL:
            Lsp3test = Lsp3 - (-4)*linestep_L*grad_Lsp3
            Lsp3Ttest = Lsp3T - (-4)*linestep_L*grad_Lsp3T
            
            Lsptest = torch.sparse_csr_tensor(Lsp1,Lsp2,Lsp3test, dtype=torch.float64)
            LspTtest = torch.sparse_csr_tensor(Lsp1T,Lsp2T,Lsp3Ttest, dtype=torch.float64)
            cur_loss = np.linalg.norm(H - (Lsptest@LspTtest).to_dense() ,'fro')**2
            des_ratio = (ori_loss-cur_loss) / grad_Lnorm / linestep_L

            if ifprint:
                print("Iter {}, search time (L) is {}, descent ratio (L) is {}".format(i,search_time, des_ratio))
            if des_ratio > grad_eta:
                Lsp3 = Lsp3test
                Lsp3T = Lsp3Ttest
                Lsp = Lsptest
                LspT = LspTtest
                break
            linestep_L /= 3
            search_time += 1 
        
        loss_H[i] = np.linalg.norm(res,'fro')**2 / H_fro 
        time_L[i] = time.time() - st
        
        
        if ifprint:
            print("Iter {}, loss (L) is {}".format(i, loss_H[i]))
    
    return (Lsp@LspT).to_dense().numpy(), loss_H, (time_init,time_L)

def grad_Ulowr(H, Uinit, lowr, stepsize_U, max_iter, ifprint = True):
    
    # H: data matrix
    # Uinit: initial solutions
    # lowr: rank
    # stepsize_U: initial value of s_U
    # max_iter: number of iterations
    # ifprint: whether print results

    
    st = time.time()
    U = np.copy(Uinit)
    loss_H = np.zeros(max_iter)
    time_U = np.zeros(max_iter)
    p = H.shape[0]
    grad_eta = 1e-2
    max_searchtimeU = 5
    
    U = torch.from_numpy(U)
    H = torch.from_numpy(H)
    H_fro = torch.sum(H**2)
    
    time_init = time.time() - st
    
    for i in range(max_iter):
        
        st = time.time()
        res = H - U@(U.T)
        grad_U = -4*res@U 
        grad_Unorm = np.linalg.norm(grad_U,'fro')**2
        linestep_U = stepsize_U
        search_time = 0
        ori_loss = np.linalg.norm(res,'fro')**2

        while search_time < max_searchtimeU:
            U_test = U - linestep_U*grad_U    
            cur_loss = np.linalg.norm(H - U_test@(U_test.T)  ,'fro')**2
            des_ratio = (ori_loss-cur_loss) / grad_Unorm / linestep_U

            if ifprint:
                print("Iter {}, search time (U) is {}, descent ratio (U) is {}".format(i,search_time, des_ratio))
                
            if des_ratio > grad_eta:
                U = U_test
                break
            linestep_U /= 2
            search_time += 1
        
        
        time_U[i] = time.time() - st
        loss_H[i] = np.linalg.norm(res,'fro')**2 / H_fro
        st = time.time()

        if ifprint:
            print("Iter {}, loss (U) is {}".format(i, loss_H[i]))
    
    return U.numpy(), loss_H, (time_init,time_U)

#=======================
# other core functions
#=======================
def LBFGS_acc(H, Uinit, Linit, bandlen, lowr, step_H, maxsearch_H, grad_eta, limit_m, max_iter, stop_ratio = 0, ifprint = True):
    
    # H: data matrix
    # Uinit/Linit: initial solutions
    # bandlen: band width
    # lowr: rank
    # stepsize_L/stepsize_U: initial value of s_L and s_U
    # max_iter: number of iterations
    # stop_ratio: terminate alg if \|H - LL^T - UU^T \|_F^2 <= stop_ratio * \|H\|_F^2
    # ifprint: whether print results
    
    st = time.time()
    
    loss_H = np.zeros(max_iter)
    time_H = np.zeros(max_iter)

    p = H.shape[0]
    num_band = (2*bandlen-1)*p - bandlen * (bandlen-1)
    num_LL = int(bandlen*p - bandlen * (bandlen-1) / 2)
    band_ind = np.ones((p,p)) * np.tri(p, k=bandlen-1) * (np.tri(p, k=bandlen-1).T)
    
    U = torch.from_numpy(Uinit)
    L = torch.from_numpy(Linit)
    H = torch.from_numpy(H)
    H_fro = torch.sum(H**2)
    
    Lsp1 = np.zeros((p+1,), dtype='int')
    Lsp2 = np.zeros((num_LL,), dtype='int')
    Lsp3 = np.zeros((num_LL,))
    for i4 in range(p):
        imin = np.maximum(0,i4-bandlen+1)
        imax = np.minimum(p,i4+1)
        Lsp1[i4+1] = Lsp1[i4] + (imax-imin)
        Lsp2[Lsp1[i4]:Lsp1[i4+1]] = np.arange(imin,imax)
        Lsp3[Lsp1[i4]:Lsp1[i4+1]] = L[i4,imin:imax].numpy()
        
    Lsp1T = np.zeros((p+1,), dtype='int')
    Lsp2T = np.zeros((num_LL,), dtype='int')
    Lsp3T = np.zeros((num_LL,))
    for i4 in range(p):
        imin = np.maximum(0,i4)
        imax = np.minimum(p,i4+bandlen)
        Lsp1T[i4+1] = Lsp1T[i4] + (imax-imin)
        Lsp2T[Lsp1T[i4]:Lsp1T[i4+1]] = np.arange(imin,imax)
        Lsp3T[Lsp1T[i4]:Lsp1T[i4+1]] = L[imin:imax,i4].numpy()
        
    Lsp = torch.sparse_csr_tensor(Lsp1,Lsp2,Lsp3, dtype=torch.float64)
    LspT = torch.sparse_csr_tensor(Lsp1T,Lsp2T,Lsp3T, dtype=torch.float64)
    
    res = H - (Lsp@LspT).to_dense() - U@(U.T)
    time_init = time.time() - st
    del L
    
    U_hist = np.zeros((p*lowr, limit_m))
    Ugrad_hist = np.zeros((p*lowr, limit_m))
    L_hist = np.zeros((num_LL, limit_m))
    Lgrad_hist = np.zeros((num_LL, limit_m))
    LT_hist = np.zeros((num_LL, limit_m))
    LTgrad_hist = np.zeros((num_LL, limit_m))
    rho_hist = np.zeros((limit_m,))
    step_prev = step_H
    
    for i in range(max_iter):
        
        st = time.time()
        
        
        ori_loss = torch.sum(res**2)
        
        grad_U = ((-4*res@U).reshape(-1)).numpy()
        ressp = (res*band_ind).to_sparse_csr()
        grad_L = (ressp@Lsp).to_dense()
        grad_Lsp3 = np.zeros((num_LL,))
        grad_Lsp3T = np.zeros((num_LL,))
        for i4 in range(p):
            imin = np.maximum(0,i4-bandlen+1)
            imax = np.minimum(p,i4+1)
            grad_Lsp3[Lsp1[i4]:Lsp1[i4+1]] = (-4)*grad_L[i4,imin:imax].numpy()
            
        for i4 in range(p):
            imin = np.maximum(0,i4)
            imax = np.minimum(p,i4+bandlen)
            grad_Lsp3T[Lsp1T[i4]:Lsp1T[i4+1]] = (-4)*grad_L[imin:imax,i4].numpy()
            
        
            
        if i >= 1:    
            U_hist[:,i%limit_m] = (U - U_prev).numpy().reshape(-1)
            Ugrad_hist[:,i%limit_m] = (grad_U - Ugrad_prev).reshape(-1)
            L_hist[:,i%limit_m] = (Lsp3 - L_prev).reshape(-1)
            Lgrad_hist[:,i%limit_m] = (grad_Lsp3 - Lgrad_prev).reshape(-1)
            LT_hist[:,i%limit_m] = (Lsp3T - LT_prev).reshape(-1)
            LTgrad_hist[:,i%limit_m] = (grad_Lsp3T - LTgrad_prev).reshape(-1)
            
            rho_hist[i%limit_m] = 1 / (U_hist[:,i%limit_m] @ Ugrad_hist[:,i%limit_m] + 
                                       L_hist[:,i%limit_m] @ Lgrad_hist[:,i%limit_m] ) 
        
        U_prev = torch.clone(U)
        Ugrad_prev = np.copy(grad_U)
        L_prev = np.copy(Lsp3)
        Lgrad_prev = np.copy(grad_Lsp3)
        LT_prev = np.copy(Lsp3T)
        LTgrad_prev = np.copy(grad_Lsp3T)
        
        q_Ugrad = np.copy(grad_U)
        q_Lgrad = np.copy(grad_Lsp3)
        q_LTgrad = np.copy(grad_Lsp3T)
        alpha = np.zeros((limit_m,))
        for bi in range(np.minimum(i,limit_m)):
            
            alpha[(i-bi)%limit_m] = rho_hist[(i-bi)%limit_m] *  (U_hist[:,(i-bi)%limit_m] @ q_Ugrad + 
                                                                 L_hist[:,(i-bi)%limit_m] @ q_Lgrad ) 
            q_Ugrad -= alpha[(i-bi)%limit_m] * Ugrad_hist[:,(i-bi)%limit_m]
            q_Lgrad -= alpha[(i-bi)%limit_m] * Lgrad_hist[:,(i-bi)%limit_m]
            q_LTgrad -= alpha[(i-bi)%limit_m] * LTgrad_hist[:,(i-bi)%limit_m]
            
            #print(q_Ugrad[:10])
            #print(q_Lgrad[:10])
            #print(q_LTgrad[:10])
        
        if i >= 1:
            gamma = 1 / rho_hist[i%limit_m] / (Ugrad_hist[:,i%limit_m] @ Ugrad_hist[:,i%limit_m] + 
                                           Lgrad_hist[:,i%limit_m] @ Lgrad_hist[:,i%limit_m] ) 
        else:
            gamma = 1
        z_Ugrad = gamma * q_Ugrad
        z_Lgrad = gamma * q_Lgrad
        z_LTgrad = gamma * q_LTgrad
        #print("gamma is ",gamma)
        #print(z_Ugrad[:10])
        #print(z_Lgrad[:10])
        #print(z_LTgrad[:10])
        
        for bi in range(np.minimum(i,limit_m)-1,-1,-1):
            beta = rho_hist[(i-bi)%limit_m] *  (Ugrad_hist[:,(i-bi)%limit_m] @ z_Ugrad + 
                                                                 Lgrad_hist[:,(i-bi)%limit_m] @ z_Lgrad )
            z_Ugrad += U_hist[:,(i-bi)%limit_m] * (alpha[(i-bi)%limit_m] - beta)
            z_Lgrad += L_hist[:,(i-bi)%limit_m] * (alpha[(i-bi)%limit_m] - beta)
            z_LTgrad += LT_hist[:,(i-bi)%limit_m] * (alpha[(i-bi)%limit_m] - beta)
            
        
        linestep = step_prev * 2
        search_time = 0
        
        #print(z_Ugrad[:10])
        #print(z_Lgrad[:10])
        #print(z_LTgrad[:10])
        
        while search_time < maxsearch_H:
            
            U_test = U - linestep * torch.from_numpy(z_Ugrad.reshape(p,lowr))
            Lsp3test = Lsp3 - linestep * z_Lgrad
            Lsp3Ttest = Lsp3T - linestep * z_LTgrad
            
            Lsptest = torch.sparse_csr_tensor(Lsp1,Lsp2,Lsp3test, dtype=torch.float64)
            LspTtest = torch.sparse_csr_tensor(Lsp1T,Lsp2T,Lsp3Ttest, dtype=torch.float64)
            B_test = (Lsptest@LspTtest).to_dense()
            res_test = H - U_test@U_test.T - B_test   
            cur_loss = np.linalg.norm(res_test,'fro')**2
            
            prod_z = np.maximum(1e-6, z_Ugrad @ grad_U + z_Lgrad @ grad_Lsp3)
            des_ratio = (ori_loss - cur_loss) / prod_z / linestep

            
            if ifprint:
                print("Iter {}, search time (L) is {}, descent ratio (L) is {}".format(i,search_time, des_ratio))
                print(ori_loss, cur_loss, prod_z * linestep)
                
            
            if des_ratio > grad_eta:
                Lsp3 = Lsp3test
                Lsp3T = Lsp3Ttest
                Lsp = Lsptest
                LspT = LspTtest
                U = U_test
                B = B_test
                res = res_test
                step_prev = linestep
                break
            linestep /= 2
            search_time += 1 
        
        loss_H[i] = cur_loss
        time_H[i] = time.time() - st
        
        
        if ifprint:
            print("Iter {}, loss is {}, ".format(i, loss_H[i]))
        
 
            
        if loss_H[i] < stop_ratio:
            break
        
    
    return (Lsp@LspT).to_dense().numpy(), U.numpy(), loss_H, (time_init,time_H)

# =================
# DEFINE ARGS 
# =================
## Argument parsers
parser = argparse.ArgumentParser(description="\n BandaLoRE approximation algorithm.")

## Input and output file paths
IOfile = parser.add_argument_group(title="Input and output options")
IOfile.add_argument('--input_fp', default=None, type=str, help="File path to the LD matrix for decomposition. Default assumes it's saved in .npz format.")
IOfile.add_argument('--plink_bfile_fp', default=None, type=str, help="File path to the raw genotype file. Default assumes it's saved in plink bfile format.")
IOfile.add_argument('--input_type', default='LD', type=str, help="Type of the input, either the LD matrix or the genotype matrix X. If LD is provided, LD decomposition will be performed directly; if X is provided, standardization will be performed on columns of X.")
IOfile.add_argument('--output_fp', default=None, type=str, help="File path to save the approximation of the LD matrix. A log file will also be saved.")

## LD approximation flags
LD_approx = parser.add_argument_group(title="Flags related to LD approximation")
LD_approx.add_argument('--LD_approx_B', default=100, type=int, 
    help='Central bandwidth of the banded component. For hyperparameter selection, this flag specifies the starting value of the bandwidth.')
LD_approx.add_argument('--LD_approx_R', default=100, type=int, 
    help='Number of low-rank factors for the off-centralband component. For hyperparameter selection, this flag specifies the starting value of the number of low rank factors.')
LD_approx.add_argument('--LD_approx_method', default="block_UL_PSD_band_lr", type=str, help="The LD approxiamtion method to use. Available options include: Lanczos, block_UL_joint, block_new_UL_joint, LBFGS_acc, seq_band_lr, PSD_band_lr, joint. Default is the PSD_band_lr with some optimization.")

## Operators
if __name__ == '__main__':
    args = parser.parse_args()

    ## Instantiate log file and masthead
    logging.basicConfig(format='%(asctime)s %(message)s', filename=args.output_fp + '.log', filemode='w', level=logging.INFO, datefmt='%Y/%m/%d %I:%M:%S %p')
    logging.getLogger().addHandler(logging.StreamHandler())

    header_sub = header
    header_sub += "Calling ./BandaLoRE.py \\\n"
    defaults = vars(parser.parse_args(''))
    opts = vars(args)
    non_defaults = [x for x in opts.keys() if opts[x] != defaults[x]]
    options = ['--'+x.replace('_','-')+' '+str(opts[x])+' \\' for x in non_defaults]
    header_sub += '\n'.join(options).replace('True','').replace('False','')
    header_sub = header_sub[0:-1] + '\n'

    logging.info(header_sub)
    logging.info("Beginning BandaLoRE approximation...")

    try:
        # band width b and rank r
        bandlen = int(args.LD_approx_B)
        lowr = int(args.LD_approx_R)

        # load the LD data
        logging.info('Reading in the LD matrix directly')
        H = np.load(args.input_fp)['LD']
        p = H.shape[0]

        #=============================
        # Run LD decomposition
        #=============================
        start_time = time.time()
        logging.info('Starting the LD approximation at {T}'.format(T=time.ctime()))
        logging.info("Approximation method: {}".format(args.LD_approx_method))
        logging.info("Size of the LD matrix is: {}".format(p))
        logging.info("Approximating using (B,R) = ({}, {})".format(bandlen, lowr))

        if args.LD_approx_method == "block_UL_PSD_band_lr":
            # Linit = np.linalg.cholesky(H)
            H_tensor = torch.tensor(H)
            Linit_torch = torch.linalg.cholesky(H_tensor)
            Linit = Linit_torch.numpy()

            # Linit = np.random.rand(p, p)
            L_ind = np.ones((p,p)) * np.tri(p, k=0) * (np.tri(p, k=bandlen-1).T)
            Linit = Linit * L_ind

            B, loss_H1, time_H1 = grad_Lband(H, Linit, bandlen, stepsize_L=1e-5, max_iter=30, ifprint = True)
            H_res = H - B
            
            Uinit, Sdiag, _  = randomized_svd(H_res, lowr) 
            Uinit = Uinit * Sdiag**(1/2)

            U, loss_H2, time_H2 = grad_Ulowr(H_res, Uinit, lowr, stepsize_U=1e-5, max_iter=10, ifprint = True)

        elif args.LD_approx_method == "block_UL_PSD_lr_band":
            Uinit, Sdiag, _  = randomized_svd(H, lowr) 
            Uinit = Uinit * Sdiag**(1/2)

            U, loss_H1, time_H1 = grad_Ulowr(H, Uinit, lowr, stepsize_U=1e-5, max_iter=10, ifprint = True)
            H_res = H - U.dot(U.T)

            L_ind = np.ones((p,p)) * np.tri(p, k=0) * (np.tri(p, k=bandlen-1).T)
            Linit = H_res * L_ind

            B, loss_H2, time_H2 = grad_Lband(H_res, Linit, bandlen, stepsize_L=1e-5, max_iter=30, ifprint = True)

        elif args.LD_approx_method == "block_UL_joint":
            # Linit = np.linalg.cholesky(H)
            H_tensor = torch.tensor(H)
            Linit_torch = torch.linalg.cholesky(H_tensor)
            Linit = Linit_torch.numpy()
            
            # Linit = np.random.rand(p, p)
            L_ind = np.ones((p,p)) * np.tri(p, k=0) * (np.tri(p, k=bandlen-1).T)
            Linit = Linit * L_ind

            Linit2 = torch.from_numpy(Linit).to_sparse_csr()
            Linit2T = torch.from_numpy(Linit.T).to_sparse_csr()
            Uinit, Sdiag, _  = randomized_svd(H - (Linit2@Linit2T).to_dense().numpy(), lowr)
            Uinit = Uinit * Sdiag**(1/2)

            rsvd_time = round(time.time() - start_time, 2)
            logging.info('tictok: SVD -- in seconds: {T}'.format(T=rsvd_time))

            B, U, loss_H, time_H = blockgrad_UL_tune(H, Uinit, Linit, bandlen, lowr, stepsize_L=2e-2, stepsize_U=2e-2, max_iter=200, stop_ratio = 0, ifprint = True)

        elif args.LD_approx_method == "block_UL_seq_band_lr":
            B_ind =  np.ones((p,p)) * np.tri(p, k=bandlen-1) * (np.tri(p, k=bandlen-1).T)
            B = H * B_ind
            H_res = H - B
            Uinit, Sdiag, _  = randomized_svd(H_res, lowr) 
            Uinit = Uinit * Sdiag**(1/2)

            U, loss_H, time_H = grad_Ulowr(H_res, Uinit, lowr, stepsize_U=2e-2, max_iter=20, ifprint = True)

        logging.info("Dimension of B: {}".format(B.shape))
        logging.info("Dimension of R: {}".format(U.shape))

        # B = LL^T is banded matrix, UU^T is low rank matrix
        # loss_H: loss \|H-LL^T-UU^T\|_F^2 of each iteration
        # time_H: time consumption of rach iteration

        np.savez(args.output_fp + "LDdecomp", B=B, U=U)

        #=============================
        # Output approx performance
        #=============================
        logging.info("Evaluating LD approx performance")
        LD_norm = np.linalg.norm(H)
        H_res = H - B - U.dot(U.T)
        res_norm = np.linalg.norm(H_res)
        perc_approx = 1 - (res_norm/LD_norm)**2
        logging.info("% Approximated: {}".format(perc_approx*100))


        #=============================
        # Working in progress funcs
        #=============================
        if args.LD_approx_method == "Lanczos":
            # initialize the random unit random vector
            v0 = np.random.rand(p)
            v0 = v0 / np.linalg.norm(v0)

            # compute V and T (in tri-diagonal form)
            alpha, beta, V = Lanczos(H, v0, r = lowr)
            lanzcos_time = round(time.time() - start_time, 2)
            logging.info('tictok: Lanzcos runtime -- in seconds: {T}'.format(T=lanzcos_time))

            # eigen-decompose the tri-diagonal matrix
            w, v = scipy.linalg.eigh_tridiagonal(alpha, beta)
            eigval = w[::-1]
            eigvec = v[:, ::-1]
            eig_tridiag_time = round(time.time() - start_time, 2)
            logging.info('tictok: Eigendecomposition for tridiagonal matrix runtime -- in seconds: {T}'.format(T=eig_tridiag_time))

            # convert eigvec(T) back to eigvec(H)
            eigvec = V.dot(eigvec)

            # # verify the correctness of eigen calculation
            # w_H, v_H = scipy.linalg.eigh(H)
            # logging.info(eigvec[:4, :4])
            # logging.info(v_H[:,::-1][:4, :4])
            # logging.info(eigval[:4])
            # logging.info(w_H[::-1][:4])

            # reconstruct LD based on the selected eigvec
            LD_lr = eigvec.dot(np.diag(eigval).dot(np.transpose(eigvec)))
            F_norm = np.linalg.norm(H - LD_lr)
            LD_norm = np.linalg.norm(H)
            logging.info("Approximated proportion of the LD: {}".format(1 - F_norm / LD_norm))
            quit()

        elif args.LD_approx_method == "LBFGS_acc":
            # generate Uinit and Linit
            Linit = np.linalg.cholesky(H)
            L_ind = np.ones((p,p)) * np.tri(p, k=0) * (np.tri(p, k=bandlen-1).T)
            Linit = Linit * L_ind

            Linit2 = torch.from_numpy(Linit).to_sparse_csr()
            Linit2T = torch.from_numpy(Linit.T).to_sparse_csr()
            Uinit, Sdiag, _  = randomized_svd(H - (Linit2@Linit2T).to_dense().numpy(), lowr)
            Uinit = Uinit * Sdiag**(1/2)

            rsvd_time = round(time.time() - start_time, 2)
            logging.info('tictok: SVD -- in seconds: {T}'.format(T=rsvd_time))

            B, U, loss_H, time_H = LBFGS_acc(H, Uinit, Linit, bandlen, lowr, step_H = 1e-2, maxsearch_H = 20, grad_eta = 1e-3, limit_m = 10, max_iter = 100, stop_ratio = 0, ifprint = True)

        time_elapsed = round(time.time() - start_time, 2)
        logging.info('Time elapsed in seconds: {T}'.format(T=time_elapsed))


    except Exception as e:
        logging.error(e,exc_info=True)
        logging.info('Analysis terminated from error at {T}'.format(T=time.ctime()))
        time_elapsed = round(time.time() - start_time, 2)
        logging.info('Total time elapsed: {T}'.format(T=sec_to_str(time_elapsed)))

    logging.info('Total time elapsed: {}'.format(sec_to_str(time.time()-start_time)))
