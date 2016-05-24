# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 14:08:41 2016

@author: Rémi
"""

import numpy as np
from scipy.misc import imread
import pylab as plt
#from skimage import io
from time import time
from inpaintools import *
  
w=lambda x : np.exp(-0.25*(x**2))    
    
def Data(im, mx, my, mask, F, neighborhood):
    #F=frontiere(mask,neighborhood)
    M = np.copy(mask)
    I = im * (np.dstack((M==0, M==0, M==0)))
    e = []
    E = []
    D = []
    S = np.zeros(mask.shape)
    Data = np.zeros(mask.shape)
    for i, j in zip(neighborhood[:,0], neighborhood[:,1]):
        u = shift_image(I, i, j)
        v = warp(u, mx, my)
        e.append(u)
        E.append(v)
        x = (i ** 2 + j ** 2) ** 0.5
        D.append(w(x) * ((diff_image(u,v) ** 2) * ((u[:,:,0]>0) * (v[:,:,0]>0))))
        S = S + w(x) * ((v[:,:,0]>0) * (u[:,:,0]>0))
    S2 = S + ((S * F)==0)
    for Dataterm in D:
        Data = Data + (1 / S2) * Dataterm
    P = penaltydata(mx, my, mask)
    return ((Data + 5000000 * ((S * F)==0)) * (F > 0)) + P
    
def Datat(im, mask, shifts, neighborhood = np.array([[0,1],[0,-1],[0,2],[0,-2],[1,0],[-1,0],[2,0],[-2,0]])):
    F = frontiere(mask,neighborhood)
    n, k, v = im.shape
    l = shifts.shape[0]
    D = np.zeros((n,k,l))
    for m in range(l):
        mx = shifts[m,0]
        my = shifts[m,1]
        Mx = mx * np.ones((n,k)) * mask
        My = my * np.ones((n,k)) * mask
        Dm = Data(im, Mx, My, mask, F, neighborhood)
        D[:, :, m] = Dm
    return D    
    
def dataterm(im, mask, shifts, data_neighborhood):
    im = im * (np.dstack((1 - mask, 1 - mask, 1 - mask)))
    width, height, depth = im.shape
    card_offsets = len(shifts)
    card_neighbors = len(data_neighborhood)
    result = np.zeros((width, height, card_offsets))
    neighims = np.zeros((width, height, depth, card_neighbors))
    
    for index_neighbor in range(card_neighbors):
        dx, dy = data_neighborhood[index_neighbor]
        neighims[:, :, :, index_neighbor] = shift_image(im, dx, dy)
    
    for index_offset in range(card_offsets):
        mx, my = shifts[index_offset]
        Mx = mx * np.ones((width, height)) * mask
        My = my * np.ones((width, height)) * mask
        partial_result = np.zeros((width, height))
        normalisation = np.zeros((width, height))
        
        for index_neighbor in range(card_neighbors):
            neighim = neighims[:, :, :, index_neighbor].copy()
            offsetim = warp(neighim, Mx, My)
            neigh_length = (dx ** 2 + dy ** 2) ** 0.5
            partial_result += w(neigh_length) * (diff_image(neighim, offsetim) ** 2) * (neighim[:,:,0]>0) * (offsetim[:,:,0]>0)
            normalisation += w(neigh_length) * (neighim[:,:,0]>0) * (offsetim[:,:,0]>0)
        
        normalisation += normalisation == 0
        partial_result /= normalisation
        result[:, :, index_offset] = partial_result + penaltydata(Mx, My, mask)
    
    return result
            
def shiftmaps(im, mask, shifts, D, rounds = 1):
    neighborhood = np.array([[0,1],[0,-1],[1,0],[-1,0]])
    F = frontiere(mask, neighborhood)
    M = np.copy(mask * (F==0))
    F2 = frontiere(mask, np.array([[1,0],[0,1]]))
    M = M + F2
    nlabels = len(shifts)
    sz = np.prod(mask.shape[0:2])
    sh = mask.shape[0:2]
    
    indices = np.arange(sz).reshape(sh).astype(np.int32)

    # generate a random initial labeling
    
    labelmap = np.random.randint(nlabels,size=sz).reshape(sh) * (mask>0) *0
    mx, my = compute_displacement_map(labelmap, shifts, mask)
    #D0=Data(im,mx,my,mask,F).astype(np.float32)
    E = 2000000 + np.sum(D[:, :, 0])
     
        
    # alpha expansion loop
    for t in range(rounds*len(shifts)):
        currlab = np.mod(t,len(shifts))
        ai,aj = shifts[currlab]
        # compute warp for the current map
        mx, my = compute_displacement_map(labelmap, shifts, mask)
                
        # compute warped image for the current map 
        outM = warp(im, mx, my) 
        # compute warped image for the candidate shift 
        outAlpha = warp(im, (mx*0+ai)*mask, (my*0+aj)*mask)    # can be precomputed

          
        # construct the baseline regularity term 4-connected           
        for i,j in ((1,0),(0,1)):
            tmpx = shift_image(mx,-i,-j) 
            tmpy = shift_image(my,-i,-j)
            outQ = warp(im, tmpx, tmpy)
            
            #p+m(q)   
            p1 = indices                      # current pixel
            p2 = shift_image(indices,-i,-j)   # always the same idx
            
            if(i==1):
                E00 = (diff_image(outM, outQ) + difflabel(mx, my, tmpx, tmpy)) * M
                E01 = (diff_image(outM, outAlpha) + difflabel(mx, my, mx * 0 + ai, my * 0 + aj)) * M
                E10 = (diff_image(outAlpha, outQ) + difflabel(tmpx, tmpy, mx * 0 + ai, my * 0 + aj)) * M
                E11 = 0 * E00   # 0
                e1, e2 = p1, p2
               
                    
            else:
                _E00 = (diff_image(outM, outQ) + difflabel(mx, my, tmpx, tmpy))*M
                _E01 = (diff_image(outM, outAlpha) + difflabel(mx, my, mx * 0 + ai, my * 0 + aj)) * M
                _E10 = (diff_image(outAlpha, outQ) + difflabel(tmpx, tmpy, mx * 0 + ai, my * 0 + aj)) * M
                _E11 = 0 * E00                  
                _e1,_e2 = p1, p2

        ix, iy = np.meshgrid(np.arange(sh[1]), np.arange(sh[0]))
#        
        D0 = D[iy, ix, labelmap].astype(np.float32)
        D1 = D[:, :, currlab].astype(np.float32)
        
        # trim the last row/column of the edge matrices
        energy, blab = solve_binary_problem(indices, D0, D1, 
                        e1[:,:-1], e2[:,:-1], E00[:,:-1], E01[:,:-1], E10[:,:-1], E11[:,:-1],
                        _e1[:-1,:], _e2[:-1,:], _E00[:-1,:], _E01[:-1,:], _E10[:-1,:], _E11[:-1,:])
        
        # update solution  
        if energy<E:
            E=energy
            print "it:%02d \t l:%d \t (%d,%d) \t changed:%d \t e:%f"%(t, currlab, ai, aj, np.sum(blab[:]), energy)
            plt.figure(2+t)
            mx, my = compute_displacement_map(labelmap, shifts, mask)
            outM = warp(im, mx, my)
            plt.imshow(outM/255.)
            labelmap = labelmap * (1 - blab) + currlab * blab
            
    mx, my = compute_displacement_map(labelmap, shifts, mask)
    outM = warp(im, mx, my)
    
    return outM, labelmap    
    

maxshift = 200
numshifts = 20

shifts = np.random.randint(maxshift,size=numshifts*2).reshape((numshifts,2)) - maxshift/2 

im = imread('elephant2_300x225_rgb.jpg').squeeze().astype(np.float32)
mask = imread('elephant2_300x225_msk.jpg').squeeze().astype(np.float32)

#im = imread('foule.jpg').squeeze().astype(np.float32)
#
#mask = imread('foulem.jpg').squeeze().astype(np.float32)
#mask=mask[:,:,0]
#im=np.dstack((im,im,im))

mask = mask > 10
  
T0 = time()

D = dataterm(im, mask, shifts, square_neighborhood(5))

out, labelmap = shiftmaps(im,mask,shifts,D,rounds=2)
T1=time() 
print(T1-T0)
plt.figure(1)
plt.imshow(out/np.max(out))
plt.figure(2)
plt.imshow(labelmap)