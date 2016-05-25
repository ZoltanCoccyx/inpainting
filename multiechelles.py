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

def low_resolution_coordinates(im, scaledmask, L):
    scale = im.shape[0] // scaledmask.shape[0]
    intensity = np.sum(im,axis=2)/3
    grad_x = shift_image(intensity,0,-1)-intensity
    grad_y = shift_image(intensity,-1,0)-intensity
    gx = np.sum(block_view(grad_x,(scale, scale)),axis=(2,3)) / (scale ** 2)
    gy = np.sum(block_view(grad_y,(2**L,2**L)),axis=(2,3)) / (scale ** 2)
    Gx = np.sum(block_view(np.abs(grad_x),(2**L,2**L)),axis=(2,3)) / (scale ** 2)
    Gy = np.sum(block_view(np.abs(grad_x),(2**L,2**L)),axis=(2,3)) / (scale ** 2)
    scaledmask = (np.sum(block_view(mask, (2**L,2**L)), axis = (2, 3)) > 0) * 1
    scaledim = (np.sum(block_view(im, (2**L,2**L)), axis = (2,3)) / scale ** 2) * scaledmask
    R, G, B = scaledim[:, :, 0], scaledim[:, :, 1], scaledim[:, :, 2]
    return R, G, B, gx, gy, Gx, Gy

def multiscale(im, mask, L = 2):
    
    sz = np.prod(mask.shape[0:2])
    sh = mask.shape[0:2]
    width, heigth = mask.shape
    
    # Initialisation des variables à la plus basse échelle
    scaledmask = (np.sum(block_view(mask, (2**L,2**L)), axis = (2, 3)) > 0) * 1
    scaledim = low_resolution_coordinates(im, scaledmask) ### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    labelmap=np.zeros((sh[0]*(2**(-L)),sh[0]*(2**(-L))))
    shifts = square_neighborhood(2 * rayon(mask)) #TODO : crude
    data_neighborhood = square_neighborhood(7)
    data_energy = dataterm(scaledim, mask, shifts, data_neighborhood)
    
    # recupération de la première carte d'offsets
    out, labelmap = shiftmaps(scaledim, scaledmask, shifts, data_energy, 2)
    
    # système de perturbations, peut être pris plus grand, mais en aucun cas plus petit.
    perturbations = np.array([[0, 1], [0, -1], [1, 0], [-1, 0], [0, 0]])
    
    # remontée en échelle
    for k in range(L):
        offsetmap, scaledmask, scaledim = changescale(offsetmap, mask, im)
        shifts = 2 * shifts
        
        # Construction du dataterm à cette échelle
        im = im * (np.dstack((1 - mask, 1 - mask, 1 - mask)))
        width, height, depth = im.shape
        card_perturbations = len(perturbations)
        card_neighbors = len(data_neighborhood)
        data_energy = np.zeros((width, height, card_offsets))
        neighims = np.zeros((width, height, depth, card_neighbors))
        
        for index_neighbor in range(card_neighbor):
            dx, dy = data_neighborhood[index_neighbor]
            neighims[:, :, :, index_neighbor] = shift_image(im, dx, dy)
        
        for index_perturbation in range(card_perturbations):
            print(index_offset)
            tempshifts = shifts + perturbations[index_pertubation]
            Mx, My = compute_displacement_map(offsetmap, tempshifts, scaledmask) # Principale différence avec le dataterm normal
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
            data_energy[:, :, index_offset] = partial_result + penaltydata(Mx, My, mask)
        
            # Traitement de cette échelle par la méthode de Pritch
            #def shiftmaps(im, mask, shifts, D, smoothness_neighborhood = np.array([[0,1],[0,-1],[1,0],[-1,0]]),  rounds = 1):
            smoothness_neighborhood = np.array([[0,1],[0,-1],[1,0],[-1,0]])            
            F = frontiere(mask, smoothness_neighborhood)
            M = np.copy(mask - F)
            sz = np.prod(scaledmask.shape[0:2])
            sh = scaledmask.shape[0:2]
            
            indices = np.arange(sz).reshape(sh).astype(np.int32)
        
            # generate a trivial initial labeling
            mx, my = np.zeros(sh), np.zeros(sh) # a priori pas de pb, à ce stade (0,0) est bien dans le voisinage
            #D0=Data(im,mx,my,mask,F).astype(np.float32)
            E = 2000000 + np.sum(D[:, :, card_perturbations // 2]) # TODO : cherche le zero, ne plus supposer qu'il est au milieu
             
            half_neighborhood = [np.array([0,1])]
            for k in smoothness_neighborhood:
                if k[0] > 0:
                    half_neighborhood.append(k)
            half_neighborhood = np.array(half_neighborhood)
            
            # alpha expansion loop
            for t in range(rounds * card_perturbations):
                
                currlab = np.mod(t, card_perturbations)
                ai, aj = perturbations[currlab]
                # compute warp for the current map
                tempshifts = shifts + perturbations[currlab]
                mx, my = compute_displacement_map(labelmap, tempshifts, mask)
                
                # compute warped image for the current map 
                outM = warp(im, mx, my) 
                # compute warped image for the candidate shift 
                outAlpha = warp(im, ai*mask, aj*mask)    # can be precomputed
                
                L = []
                
                for i,j in half_neighborhood:
                    tmpx = shift_image(mx,-i,-j) 
                    tmpy = shift_image(my,-i,-j)
                    outQ = warp(im, tmpx, tmpy)
                    Mnew = M + frontiere(mask, np.array([[i,j]]))
                    
                    #p+m(q)   
                    p1 = indices                      # current pixel
                    p2 = shift_image(indices,-i,-j)   # always the same idx
                    nullarray = (0 * mx).astype('float32')
                    
                    L.append(p1[:-j if j else None,:-i if i else None])
                    L.append(p2[:-j if j else None,:-i if i else None])
                    L.append(((diff_image(outM, outQ) ** 2 + difflabel(mx, my, tmpx, tmpy)) * Mnew)[:-j if j else None,:-i if i else None]) #E00
                    L.append(((diff_image(outM, outAlpha) ** 2 + difflabel(mx, my, nullarray + ai, nullarray + aj)) * Mnew)[:-j if j else None,:-i if i else None]) #E01
                    L.append(((diff_image(outAlpha, outQ) ** 2 + difflabel(tmpx, tmpy, nullarray + ai, nullarray + aj)) * Mnew)[:-j if j else None,:-i if i else None]) #E10
                    L.append(nullarray[:-j if j else None,:-i if i else None]) #E11
        
                ix, iy = np.meshgrid(np.arange(sh[1]), np.arange(sh[0]))
                D0 = D[iy, ix, labelmap].astype(np.float32)
                D1 = D[:, :, currlab].astype(np.float32)
                
                energy, blab = solve_binary_problem(indices, D0, D1, L)
                
                # update solution
                if energy<E:
                    E=energy
                    print "it:%02d \t l:%d \t (%d,%d) \t changed:%d \t e:%f"%(t, currlab, ai, aj, np.sum(blab[:]), energy)
                    labelmap = labelmap * (1 - blab) + currlab * blab
            
        
        
    