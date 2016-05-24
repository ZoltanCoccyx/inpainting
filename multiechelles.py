# -*- coding: utf-8 -*-
"""
Created on Wed May 18 13:03:07 2016

@author: D
"""

import numpy as np
from scipy.misc import imread
from numpy.lib.stride_tricks import as_strided as ast
from inpaintools import *
from shiftmap import *
from matplotlib import pyplot as plt

maxshift = 200
numshifts = 80
shifts = np.random.randint(maxshift,size=numshifts*2).reshape((numshifts,2)) - maxshift/2
im = imread('elephant2_300x225_rgb.jpg').squeeze().astype(np.float32)
mask = imread('elephant2_300x225_msk.jpg').squeeze().astype(np.float32)
sz = np.prod(mask.shape[0:2])
sh = mask.shape[0:2]
labelmap = np.random.randint(numshifts,size=sz).reshape(sh) * (mask>0)
mx, my = compute_displacement_map(labelmap, shifts, mask)
offsetmap=np.array([mx,my])

def changescale(offsetmap, mask, im):
    '''Pour l'instant, ne gère que les images 2^nx2^n'''
    scale = int(mask.shape[0] / offsetmap.shape[1]) //2
    scaledmask = (np.sum(block_view(mask, (scale, scale)), axis = (2, 3)) > 0)* 1
    newmap = np.zeros((2 * offsetmap.shape[1], 2 * offsetmap.shape[2]))
    newmap[::2,::2] = offsetmap
    newmap[1::2,::2] = offsetmap
    newmap[1::2,1::2] = offsetmap
    newmap[::2,1::2] = offsetmap
    scaledim = (np.sum(block_view(im, (scale, scale)), axis = (2,3)) / scale ** 2) * scaledmask
    return newmap * scaledmask, scaledmask, scaledim

def block_view(A, block= (2, 2)):
    """Provide a 2D block view to 2D array. No error checking made.
    Therefore meaningful (as implemented) only for blocks strictly
    compatible with the shape of A."""
    # simple shape and strides computations may seem at first strange
    # unless one is able to recognize the 'tuple additions' involved ;-)
    shape= (A.shape[0]/ block[0], A.shape[1]/ block[1])+ block
    strides= (block[0]* A.strides[0], block[1]* A.strides[1])+ A.strides
    return ast(A, shape= shape, strides= strides)

def grad(im,L):
    intensity = np.sum(im,axis=2)/3
    grad_x = shift_image(intensity,0,-1)-intensity
    grad_y = shift_image(intensity,-1,0)-intensity
    gx = np.sum(block_view(grad_x,(2**L,2**L)),axis=(2,3)) / (2**L)
    gy = np.sum(block_view(grad_y,(2**L,2**L)),axis=(2,3)) / (2**L)
    Gx = np.sum(block_view(np.abs(grad_x),(2**L,2**L)),axis=(2,3)) / (2**L)
    Gy = np.sum(block_view(np.abs(grad_x),(2**L,2**L)),axis=(2,3)) / (2**L)
    return gx, gy, Gx, Gy

def multiscale(im, mask, L = 2):
    
    sz = np.prod(mask.shape[0:2])
    sh = mask.shape[0:2]
    width, heigth = mask.shape
    
    # Initialisation des variables à la plus basse échelle
    scaledmask = (np.sum(block_view(mask, (2**L,2**L)), axis = (2, 3)) > 0) * 1
    scaledim = (np.sum(block_view(im, (2**L,2**L)), axis = (2,3)) / scale ** 2) * scaledmask
    labelmap=np.zeros((sh[0]*(2**(-L)),sh[0]*(2**(-L))))
    shifts = square_neighborhood(2 * rayon(mask)) #TODO : crude
    data_neighborhood = square_neighborhood(7)
    dataterm = datat(im, mask, shifts, data_neighborhood)
    
    # recupération de la première carte d'offsets
    out, labelmap = shiftmaps(scaledim, scaledmask, shifts, dataterm, 2)
    
    # système de perturbations, peut être pris plus grand, mais en aucun cas plus petit.
    perturbations = np.array([[0, 1], [0, -1], [1, 0], [-1, 0], [0, 0]])
    
    # remontée en échelle
    for k in range(L):
        offsetmap, scaledmask, scaledim = changescale(offsetmap, mask, im)
        
        # Construction du dataterm à cette échelle
        D = np.array((width, heigth, len(perturbations)))
        for dxdy in perturbations:
            M = np.copy(scaledmask)
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
    
    