# -*- coding: utf-8 -*-
"""
Created on Thu May 19 10:35:28 2016

@author: Balthazar
"""

import pymaxflow
import numpy as np
from time import time

## Alpha-expansion related functions

def solve_binary_problem(indices, D0, D1, L):
    '''
    solve the labeling problem     
        min_x    \sum_i  D_i(x_i) + \sum_{ij}  V_ij(x_i,x_j)     
        with x_i \in {0,1}
    '''
    sz = D0.size
    sh = D0.shape
    e  = pymaxflow.PyEnergy(sz, sz*4)

    # variables
    first_node_idx = e.add_variable(sz)
    
    # add unary terms
    e.add_term1_vectorized(indices.ravel(), D0.ravel(), D1.ravel())
    
    for k in range(len(L) // 6):
        e.add_term2_vectorized(L[6*k].ravel(),L[6*k+1].ravel(),
                               L[6*k+2].ravel(),L[6*k+3].ravel(),
                               L[6*k+4].ravel(),L[6*k+5].ravel())
    
    Emin = e.minimize()       
    out  = e.get_var_vectorized()
    return Emin, out.reshape(sh)
 
## Image displacements : circular shifts, warps, calculation of displacement maps
   
def shift_image(im, dx, dy):
    ''' circular shift of dx, dy'''
    cim=np.copy(im)
    return np.roll(np.roll(cim,dy,axis=0),dx,axis=1)
    
def warp(im,mx,my):
    '''''' 
    sh = im.shape
    
    px, py = np.meshgrid(range(sh[1]), range(sh[0]))
    px = px + mx 
    py = py + my
    
    #truncated 
    px = np.mod(px,sh[1])
    py = np.mod(py,sh[0])#    
    Out = im[py[:].astype(np.int),px[:].astype(np.int),] 
    return Out
    
def compute_displacement_map(m, shifts, mask=None):
    M = np.copy(mask)
    nlabels = len(shifts)

    # no displacement map
    mx, my = m * 0, m * 0
    
    for t in range(nlabels):
        mx = mx + shifts[t][0] * (m==t)
        my = my + shifts[t][1] * (m==t)
        
    #apply mask
    if type(mask)!=type(None):
        mx = mx * (M>0)
        my = my * (M>0)
    return mx, my

## Distance

def diff_image(a, b):
    if len(a.shape) > 2:
        return np.sum(np.abs(a - b)**2, axis=2) ** (1 / float(2))
    else:
        return np.abs(a - b)

def difflabel(mx, my, mx1, my1):
    eps = 0.01
    k2 = 0.01
    K2 = k2 * np.ones(mx.shape)
    N = np.sqrt((mx - mx1) ** 2 + (my - my1) ** 2)
    return (eps * mint(N, K2)).astype(np.float32)

## Utilitaries
        
def intensity(im):
    return(1/3*np.sum(im,axis=2))
    
def mint(A,B):
    return 1 * (A * (A <= B) + B * (B < A))

def frontiere(mask, neighborhood):
    M = 1 - mask
    K = M * 0
    n = len(neighborhood)
    for i, j in zip(neighborhood[:,0], neighborhood[:,1]):
       K += shift_image(M, i, j)  
    F = K > 0
    return F * mask
   
def penaltydata(mx,my,mask):
    sh = mask.shape
    px, py = np.meshgrid(range(sh[1]), range(sh[0]))
    px = px + mx 
    py = py + my
    C=1500000
    K1=C*(px>(sh[1]-1))
    K2=C*(py>(sh[0]-1))
    K3=C*(px<0)
    K4=C*(py<0)
    px = np.mod(px,sh[1])
    py = np.mod(py,sh[0])
    I=C*mask[py.astype(np.int),px.astype(np.int)]
    return (I+K1+K2+K3+K4)*mask

def square_neighborhood(n):
    dx, dy = np.meshgrid(np.arange(-n // 2 + 1, n // 2 + 1), np.arange(-n//2 + 1, n // 2 + 1 ))
    dx, dy = dx.ravel(), dy.ravel()
    return np.array([dx, dy]).T
    
def round_neighborhood(r):
    r = int(r)
    result = []
    for k in range(-r-1, r + 2):
        for i in range(-r-1, r + 2):
            if (k ** 2 + i ** 2) <= r**2:
                result.append(np.array((k,i)))
    return np.array(result)
    
def rayon(mask):
    m = mask.copy()
    n = 0
    while 1 in m:
        n += 1 
        m = m - frontiere(m, np.array([[1,0],[-1,0],[0,1],[0,-1]]))
    return n

w = lambda x : np.exp(-0.25 * (x**2))   

def dataterm(im, mask, shifts, data_neighborhood):
    #bob = time()
    width, height, depth = im.shape
    im = im * (np.dstack((1 - mask, ) * depth))
    card_offsets = len(shifts)
    card_neighbors = len(data_neighborhood)
    result = np.zeros((width, height, card_offsets))
    neighims = np.zeros((width, height, depth, card_neighbors))
    w = lambda x : np.exp(-0.25*(x**2))
    
    for index_neighbor in range(card_neighbors):
        dx, dy = data_neighborhood[index_neighbor]
        neighims[:, :, :, index_neighbor] = shift_image(im, dx, dy)
    
    print 'Calcul du dataterm. ', card_offsets, 'offsets. 0% ',
    p = 0.0
    for index_offset in range(card_offsets):
        if float(index_offset) / card_offsets > p:
            print '#',
            p += 0.05
        mx, my = shifts[index_offset]
        Mx = mx * np.ones((width, height)) * mask
        My = my * np.ones((width, height)) * mask
        partial_result = np.zeros((width, height))
        normalisation = np.zeros((width, height))
        
        for index_neighbor in range(card_neighbors):
            dx, dy = data_neighborhood[index_neighbor]
            neighim = neighims[:, :, :, index_neighbor].copy()
            offsetim = warp(neighim, Mx, My)
            neigh_length = (dx ** 2 + dy ** 2) ** 0.5
            partial_result += w(neigh_length) * (diff_image(neighim, offsetim) ** 2) * (neighim[:,:,0]>0) * (offsetim[:,:,0]>0)
            normalisation += w(neigh_length) * (neighim[:,:,0]>0) * (offsetim[:,:,0]>0)
        
        normalisation += normalisation == 0
        partial_result /= normalisation
        result[:, :, index_offset] = partial_result + penaltydata(Mx, My, mask)
    print '100%'
    #print 'Temps de calcul = ', time()-bob
    return result