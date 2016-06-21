# -*- coding: utf-8 -*-
"""
Created on Wed May 18 13:03:07 2016

@author: Balthazar
"""

import numpy as np
import pylab as plt
import patchmatch as pm
#from pritch import *
#from inpaitools import *

#im = imread('elephant2_300x225_rgb.jpg').squeeze()
#mask = imread('elephant2_300x225_msk.jpg').squeeze()
#mask = mask > 10

#im = imread('kom07.png').squeeze()
#mask = imread('kom07_msk.png', 'png').squeeze()
#mask = mask>0

def offset_system(dx, dy, m):
    xedges = np.arange(np.min(dx), np.max(dx) + 2)
    yedges = np.arange(np.min(dy), np.max(dy) + 2)
    # xsize = np.max(dx) + 1 - np.min(dx)
    ysize = np.max(dy) + 1 - np.min(dy)
    h = np.histogram2d(dx.ravel(), dy.ravel(), bins = (xedges, yedges))[0]
    hh = h.copy()
    result = []
    for k in range(m):
        maximum = np.max(h)
        index = np.argmax(h)
        result.append([index // ysize + np.min(dx), index % ysize + np.min(dy)])
        h[index // ysize, index % ysize] = 0
    result = np.array(result)
    return result, hh
    
def he_sun(im, mask, m,data_neighborhood, smoothness_neighborhood, rounds):
    im = im.astype(np.float32)
    sh = im.shape
    mask_broadcast = np.zeros((sh[0],sh[1],1))
    mask_broadcast[:,:,0] = mask
    mask_broadcast = mask_broadcast.astype(np.float32)
    D = np.zeros((sh[0],sh[1],2))
    D = D.astype(np.float32)
    cost = np.zeros((sh[0],sh[1]))
    cost = cost.astype(np.float32)
    r = int(rayon(mask))
    pm.pm(im*(1-mask_broadcast), im*(1-mask_broadcast), D, cost, 5, 20, 3*r)
    dx, dy = D[:,:,0], D[:,:,1] #partie qui manque à cause de patch match
    shifts, hh = offset_system(dx, dy, m)
    out, labelmap = pritch(im, mask, shifts, data_neighborhood, smoothness_neighborhood, rounds)
    return out, shifts, labelmap

data_neighborhood = square_neighborhood(3)
smoothness_neighborhood = square_neighborhood(3)
out, shifts, labelmap = he_sun(im, mask, 99,data_neighborhood, smoothness_neighborhood, 2)

def petite_fonction(n):
    return 'om' + str(n//10) + str(n%10)

for i in range(1,13):
    print(i)
    r=petite_fonction(i)
    im = imread('k' + r +'.png').squeeze()
    mask = imread('k' + r +'_msk.png', 'png').squeeze()
    mask = mask>0
    out, shifts, labelmap = he_sun(im, mask, 99, data_neighborhood, smoothness_neighborhood, 2)
    m = np.max(out)
    out = out/m
    plt.imsave('C:\Users\D\Desktop\inpainting\image\k' + r + '_hesun.png',out,vmin=0,vmax=255,format='png')
    plt.imsave('C:\Users\D\Desktop\inpainting\image\k' + r + '_hesunlabel.png',labelmap,vmin=0,vmax=98,format='png')