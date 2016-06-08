# -*- coding: utf-8 -*-
"""
Created on Wed May 18 13:03:07 2016

@author: Balthazar
"""

import numpy as np

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
    
def he_sun(im, mask, m):
    dx, dy = 1, 1 #partie qui manque à cause de patch match
    shifts = offset_system(dx, dy, m)
    data_energy = dataterm(im, mask, shifts, square_neighborhood(3))
    out, labelmap = shiftmaps(im, mask, shifts, data_energy)
    return out, shifts