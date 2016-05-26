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
from time import time

def changescale(offsetmap, mask, im):
    '''Pour l'instant, ne gère que les images 2^nx2^n'''
    scale = int((mask.shape[0] + 1) / offsetmap.shape[0]) // 2 # le +1 vient d'une situation du style 225 -> 113 ou l'échelle détectée serait 0
    scaledmask = (np.sum(block_view(mask, (scale, scale)), axis = (2, 3)) > 0)* 1
    newmap = np.zeros((2 * offsetmap.shape[0], 2 * offsetmap.shape[1]))
    newmap[::2,::2] = offsetmap
    newmap[1::2,::2] = offsetmap
    newmap[1::2,1::2] = offsetmap
    newmap[::2,1::2] = offsetmap
    scaledR = np.sum(block_view(im[:, :, 0], (scale, scale)), axis = (2,3)) / scale ** 2 * scaledmask
    scaledG = np.sum(block_view(im[:, :, 1], (scale, scale)), axis = (2,3)) / scale ** 2 * scaledmask
    scaledB = np.sum(block_view(im[:, :, 2], (scale, scale)), axis = (2,3)) / scale ** 2 * scaledmask
    scaledim = np.dstack((scaledR, scaledG, scaledB))
    return newmap[: - (scaledmask.shape[0] % 2) if (scaledmask.shape[0] % 2) else None , 
                  : - (scaledmask.shape[1] % 2) if (scaledmask.shape[1] % 2) else None] * scaledmask, scaledmask, scaledim 
                  # pour ne pas forcer une image impaire dans une image paire

def block_view(A, block= (2, 2)): # TODO: gérer les effets de bors : important si trou au bord
    """Provide a 2D block view to 2D array. No error checking made.
    Therefore meaningful (as implemented) only for blocks strictly
    compatible with the shape of A."""
    # simple shape and strides computations may seem at first strange
    # unless one is able to recognize the 'tuple additions' involved ;-)
    shape= (A.shape[0]/ block[0], A.shape[1]/ block[1])+ block
    strides= (block[0]* A.strides[0], block[1]* A.strides[1])+ A.strides
    return ast(A, shape= shape, strides= strides)

def low_resolution_coordinates(im, scaledmask):
    scale = im.shape[0] // scaledmask.shape[0] # pas le même problème qu'au dessus car block_view coupe les méchants bords, ce qui n'est pas bien
    intensity = np.sum(im,axis=2)/3
    grad_x = shift_image(intensity,0,-1)-intensity
    grad_y = shift_image(intensity,-1,0)-intensity
    gx = np.sum(block_view(grad_x,(scale, scale)),axis=(2,3)) / (scale ** 2)
    gy = np.sum(block_view(grad_y,(scale, scale)),axis=(2,3)) / (scale ** 2)
    Gx = np.sum(block_view(np.abs(grad_x),(scale, scale)),axis=(2,3)) / (scale ** 2)
    Gy = np.sum(block_view(np.abs(grad_x),(scale, scale)),axis=(2,3)) / (scale ** 2)
    scaledR = np.sum(block_view(im[:, :, 0], (scale, scale)), axis = (2,3)) / scale ** 2 * scaledmask
    scaledG = np.sum(block_view(im[:, :, 1], (scale, scale)), axis = (2,3)) / scale ** 2 * scaledmask
    scaledB = np.sum(block_view(im[:, :, 2], (scale, scale)), axis = (2,3)) / scale ** 2 * scaledmask
    newim = np.array([scaledR, scaledG, scaledB, gx, gy, Gx, Gy]).transpose((1, 2, 0))
    return newim

def multiscale(im, mask, L = 2):
    
    sz = np.prod(mask.shape[0:2])
    sh = mask.shape[0:2]
    width, heigth, depth = im.shape
    
    # Initialisation des variables à la plus basse échelle
    scaledmask = (np.sum(block_view(mask, (2**L,2**L)), axis = (2, 3)) > 0) * 1
    scaledim = low_resolution_coordinates(im, scaledmask)
    labelmap = np.zeros((sh[0]*(2**(-L)),sh[0]*(2**(-L)))) # TODO : trouver 0 ?
    shifts = round_neighborhood(rayon(mask) / 2 ** L)
    data_neighborhood = square_neighborhood(3)
    data_energy = dataterm(scaledim, scaledmask, shifts, data_neighborhood)

    print 'Dataterm initial calculé'    
    
    # recupération de la première carte d'offsets
    out, labelmap = shiftmaps(scaledim, scaledmask, shifts, data_energy, rounds = 2)
    
    print 'Labelmap initial calculé'
    
    # remontée en échelle
    for k in range(L):
        print ' ! Echelle ', L - k, '!'
        labelmap, scaledmask, scaledim = changescale(labelmap, mask, im)
        shifts = 2 * shifts # adaptation of the offsets to the new scale
        
        # Construction du dataterm à cette échelle
        width, height, depth = scaledim.shape
        scaledim = scaledim * (np.dstack((1 - scaledmask, ) * depth))
        card_perturbations = len(perturbations)
        card_neighbors = len(data_neighborhood)
        data_energy = np.zeros((width, height, card_perturbations))
        neighims = np.zeros((width, height, depth, card_neighbors))
        
        for index_neighbor in range(card_neighbors):
            dx, dy = data_neighborhood[index_neighbor]
            neighims[:, :, :, index_neighbor] = shift_image(scaledim, dx, dy)
        
        print 'Neighims calculées'
        
        print 'Calcul du dataterm à cette échelle. ', card_perturbations, 'perturbations.'
        for index_perturbation in range(card_perturbations):
            tempshifts = shifts + perturbations[index_perturbation]
            Mx, My = compute_displacement_map(labelmap, tempshifts, scaledmask) # Principale différence avec le dataterm normal
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
            data_energy[:, :, index_perturbation] = partial_result + penaltydata(Mx, My, scaledmask)
            
        print 'Done' 
        
        # Traitement de cette échelle par alpha-expansion des perturbations
        
        # système de perturbations, peut être pris plus grand, mais en aucun cas plus petit.
        perturbations = np.array([[0, 1], [0, -1], [1, 0], [-1, 0], [0, 0]])
        
        smoothness_neighborhood = np.array([[0,1],[0,-1],[1,0],[-1,0]])            
        F = frontiere(scaledmask, smoothness_neighborhood)
        M = np.copy(scaledmask - F)
        sz = np.prod(scaledmask.shape[0:2])
        sh = scaledmask.shape[0:2]
        
        indices = np.arange(sz).reshape(sh).astype(np.int32)
    
        # generate a trivial initial perturbation labeling for this scale
        perturbationmap = np.zeros(sh).astype(np.int32)
        mx, my = np.zeros(sh), np.zeros(sh) # a priori pas de pb, à ce stade (0,0) est bien dans le voisinage
        #D0=Data(im,mx,my,mask,F).astype(np.float32)
        E = 2000000 + np.sum(D[:, :, card_perturbations // 2]) # TODO : cherche le zero, ne plus supposer qu'il est au milieu
         
        half_neighborhood = [np.array([0,1])]
        for k in smoothness_neighborhood:
            if k[0] > 0:
                half_neighborhood.append(k)
        half_neighborhood = np.array(half_neighborhood)
        
        rounds = 2            
        ######################################################################
        # alpha expansion loop
        print "Calcul de l'alpha-expansion à cette échelle. ", card_perturbations, 'perturbations.'
        for t in range(rounds * card_perturbations):
            
            currlab = np.mod(t, card_perturbations)
            ai, aj = perturbations[currlab]
            # compute warp for the current map
            tempshifts = shifts + 
            dx, dy = compute_displacement_map(perturbationmap, perturbations[currlab], scaledmask)
            mx, my = label
            
            # compute warped image for the current map 
            outM = warp(scaledim, mx, my) 
            # compute warped image for the candidate shift 
            outAlpha = warp(scaledim, ai*scaledmask, aj*scaledmask)    # can be precomputed
            
            L = []
            
            for i,j in half_neighborhood:
                tmpx = shift_image(mx,-i,-j) 
                tmpy = shift_image(my,-i,-j)
                outQ = warp(scaledim, tmpx, tmpy)
                Mnew = M + frontiere(scaledmask, np.array([[i,j]]))
                
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
            labelmap = labelmap.astype(np.int32)
            D0 = D[iy, ix, labelmap].astype(np.float32)
            D1 = D[:, :, currlab].astype(np.float32)
            
            energy, blab = solve_binary_problem(indices, D0, D1, L)
            
            # update solution
            if energy<E:
                E=energy
                print "it:%02d \t l:%d \t (%d,%d) \t changed:%d \t e:%f"%(t, currlab, ai, aj, np.sum(blab[:]), energy)
                labelmap = labelmap * (1 - blab) + currlab * blab
        
        print 'Done'