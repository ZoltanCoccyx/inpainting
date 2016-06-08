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

im = imread('elephant2_300x225_rgb.jpg').squeeze().astype(np.float32)
im=im[1:,:-4]
mask = imread('elephant2_300x225_msk.jpg').squeeze().astype(np.float32)
mask=mask[1:,:-4]
im = im[::-1,,:,:]
mask = mask[::-1,,:]
mask = mask > 10

def changescale(mx, my, mask, im):
    '''Pour l'instant, ne gère que les images 2^nx2^n'''
    scale = int((mask.shape[0] + 1) / mx.shape[0]) // 2 # le +1 vient d'une situation du style 225 -> 113 ou l'échelle détectée serait 0
    scaledmask = (np.sum(block_view(mask, (scale, scale)), axis = (2, 3)) > 0)* 1
    newmx = np.zeros((2 * mx.shape[0], 2 * mx.shape[1]))
    newmx[::2,::2] = mx
    newmx[1::2,::2] = mx
    newmx[1::2,1::2] = mx
    newmx[::2,1::2] = mx
    newmy = np.zeros((2 * my.shape[0], 2 * my.shape[1]))
    newmy[::2,::2] = my
    newmy[1::2,::2] = my
    newmy[1::2,1::2] = my
    newmy[::2,1::2] = my
    scaledR = np.sum(block_view(im[:, :, 0], (scale, scale)), axis = (2,3)) / scale ** 2
    scaledG = np.sum(block_view(im[:, :, 1], (scale, scale)), axis = (2,3)) / scale ** 2
    scaledB = np.sum(block_view(im[:, :, 2], (scale, scale)), axis = (2,3)) / scale ** 2
    scaledim = np.dstack((scaledR, scaledG, scaledB))
    print(newmx.shape)
    return newmx * scaledmask, newmy * scaledmask,  scaledmask, scaledim 
                  # pour ne pas forcer une image impaire dans une image paire PAS BEAU !

def block_view(A, block= (2, 2)): # TODO: gérer les effets de bords : important si trou au bord
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
    scaledR = np.sum(block_view(im[:, :, 0], (scale, scale)), axis = (2,3)) / scale ** 2
    scaledG = np.sum(block_view(im[:, :, 1], (scale, scale)), axis = (2,3)) / scale ** 2
    scaledB = np.sum(block_view(im[:, :, 2], (scale, scale)), axis = (2,3)) / scale ** 2
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
    shifts = round_neighborhood(2*rayon(mask) / 2 ** L)
    data_neighborhood = square_neighborhood(3)
    data_energy = dataterm(scaledim, scaledmask, shifts, data_neighborhood)

    print 'Dataterm initial calculé'    
    
    # recupération de la première carte d'offsets
    out, labelmap = shiftmaps(scaledim, scaledmask, shifts, data_energy, rounds = 2)
    print(np.max(out[:,:,:3]))
    plt.figure(2)
    plt.imshow(out[:,:,:3]/np.max(out[:,:,:3]))
    cumulx, cumuly = compute_displacement_map(labelmap, shifts)
    print 'Offsets initiaux calculés'
    
    # Etant donné que labelmap + shifts est plus tenable, l'idée est d'entretenir
    # les tableaux Mx, et My qui contiennent la somme des résultats trouvés à 
    # chaque échelle : offsets "principaux" + perturabtion1 + perturbation2 + ...    
    
    # système de perturbations, peut être pris plus grand, mais en aucun cas plus petit.
    perturbations = square_neighborhood(5) #np.array([[0, 1], [0, -1], [1, 0], [-1, 0], [0, 0]])
    
    # remontée en échelle
    for k in range(L):
        print ' ! Echelle ', L - k, '!'
        cumulx, cumuly, scaledmask, scaledim = changescale(cumulx, cumuly, mask, im)
        cumulx, cumuly = 2 * cumulx, 2 * cumuly # adaptation of the offsets to the new scale
        plt.figure(55*k)
        plt.imshow(scaledim/255)
        plt.figure(8759**k)
        plt.imshow(warp(scaledim, cumulx, cumuly)/255)
        
        # Construction du dataterm à cette échelle
        width, height, depth = scaledim.shape
        maskedscaledim = scaledim * (np.dstack((1 - scaledmask, ) * depth))
        card_perturbations = len(perturbations)
        card_neighbors = len(data_neighborhood)
        data_energy = np.zeros((width, height, card_perturbations))
        neighims = np.zeros((width, height, depth, card_neighbors))
        
        for index_neighbor in range(card_neighbors):
            dx, dy = data_neighborhood[index_neighbor]
            neighims[:, :, :, index_neighbor] = warp(maskedscaledim, cumulx + dx, cumuly + dy)
        
        print 'Neighims calculées'
        
        print 'Calcul du dataterm à cette échelle. ', card_perturbations, 'perturbations.'
        for index_perturbation in range(card_perturbations):
            mx, my = perturbations[index_perturbation]
            cumulpertx, cumulperty = cumulx - mx * scaledmask, cumuly - my * scaledmask
            partial_result = np.zeros((width, height))
            normalisation = np.zeros((width, height))
            
            for index_neighbor in range(card_neighbors):
                dx, dy = data_neighborhood[index_neighbor]
                neighim = neighims[:, :, :, index_neighbor].copy()
                offsetim = warp(neighim, cumulpertx, cumulperty)
                neigh_length = (dx ** 2 + dy ** 2) ** 0.5
                partial_result += w(neigh_length) * (diff_image(neighim, offsetim) ** 2) * (neighim[:,:,0]>0) * (offsetim[:,:,0]>0)
                normalisation += w(neigh_length) * (neighim[:,:,0]>0) * (offsetim[:,:,0]>0)
            
            normalisation += normalisation == 0
            partial_result /= normalisation
            data_energy[:, :, index_perturbation] = partial_result + penaltydata(cumulpertx, cumulperty, scaledmask) # sûr ? avec le mx, my ?
            
        print 'Done' 
        
        # Traitement de cette échelle par alpha-expansion des perturbations
        
        smoothness_neighborhood = np.array([[0,1],[0,-1],[1,0],[-1,0]])     # TODO : paresseux, doit pouvoir être réglé
        F = frontiere(scaledmask, smoothness_neighborhood)
        M = np.copy(scaledmask - F)
        sz = np.prod(scaledmask.shape[0:2])
        sh = scaledmask.shape[0:2]
        
        indices = np.arange(sz).reshape(sh).astype(np.int32)
    
        # generate a trivial initial perturbation labeling for this scale
        perturbationmap = np.zeros(sh).astype(np.int32)
        
        E = 2000000 + np.sum(data_energy[:, :, card_perturbations // 2]) # TODO : chercher le zero, ne plus supposer qu'il est au milieu
         
        half_neighborhood = [np.array([0,1])]
        for bob in smoothness_neighborhood:
            if bob[0] > 0:
                half_neighborhood.append(bob)
        half_neighborhood = np.array(half_neighborhood)
        
        rounds = 2            
        ######################################################################
        # alpha expansion loop
        print "Calcul de l'alpha-expansion à cette échelle. ", card_perturbations, 'perturbations.'
        for t in range(rounds * card_perturbations):
            
            currlab = np.mod(t, card_perturbations)
            ai, aj = perturbations[currlab]
            # compute warp for the current map
            pertx, perty = compute_displacement_map(perturbationmap, perturbations, scaledmask)
            cumulpertx, cumulperty = cumulx - pertx, cumuly - perty
            
            # compute warped image for the current map 
            outM = warp(scaledim, cumulpertx, cumulperty) 
            # compute warped image for the candidate shift 
            outAlpha = warp(scaledim, cumulpertx - ai * scaledmask, cumulperty - aj * scaledmask)
            
            energy_list = []
            
            for i,j in half_neighborhood:
                neighMx = shift_image(cumulpertx,-i,-j) 
                neighMy = shift_image(cumulperty,-i,-j)
                outQ = warp(scaledim, neighMx, neighMy)
                Mnew = M + frontiere(scaledmask, np.array([[i,j]]))
                
                #p+m(q)   
                p1 = indices                      # current pixel
                p2 = shift_image(indices,-i,-j)   # always the same idx
                energy_list.append(p1[:-j if j else None,:-i if i else None])
                energy_list.append(p2[:-j if j else None,:-i if i else None])
                
                nullarray = (0 * cumulx).astype('float32')                
                E00 = (((diff_image(outM, outQ) ** 2 + difflabel(cumulpertx, cumulperty, neighMx, neighMy)) * Mnew)[:-j if j else None,:-i if i else None]).astype(np.float32) #E00
                E01 = (((diff_image(outM, outAlpha) ** 2 + difflabel(cumulpertx, cumulperty, cumulx + ai, cumuly + aj)) * Mnew)[:-j if j else None,:-i if i else None]).astype(np.float32) #E01
                E10 = (((diff_image(outAlpha, outQ) ** 2 + difflabel(neighMx, neighMy, cumulx + ai, cumuly + aj)) * Mnew)[:-j if j else None,:-i if i else None]).astype(np.float32) #E10
                E11 = (nullarray[:-j if j else None,:-i if i else None]).astype(np.float32) #E11
                            
                energy_list.append(E00) #E00
                energy_list.append(E01) #E01
                energy_list.append(E10) #E10
                energy_list.append(E11) #E11
                
            ix, iy = np.meshgrid(np.arange(sh[1]), np.arange(sh[0]))
            perturbationmap = perturbationmap.astype(np.int32)
            D0 = data_energy[iy, ix, perturbationmap].astype(np.float32)
            D1 = data_energy[:, :, currlab].astype(np.float32)
            
            energy, blab = solve_binary_problem(indices, D0, D1, energy_list)
            
            # update solution
            if energy<E:
                E=energy
                print "it:%02d \t l:%d \t (%d,%d) \t changed:%d \t e:%f"%(t, currlab, ai, aj, np.sum(blab[:]), energy)
                perturbationmap = perturbationmap * (1 - blab) + currlab * blab
        
        pertx, perty = compute_displacement_map(perturbationmap, perturbations, scaledmask)
        cumulx, cumuly = cumulx + pertx, cumuly + perty
        print 'Done'
        print(k, 'wldkghdfckwsugdvckqjs<wsdwvckljwsgvbdcfkjgsdf')
        plt.figure(716+k)
        plt.imshow(cumulx)
        plt.figure(1716+k)
        plt.imshow(cumuly)
    
    return warp(im, cumulx, cumuly), cumulx, cumuly