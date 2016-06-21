# -*- coding: utf-8 -*-
"""
Created on Wed Jun 08 14:44:02 2016

@author: D
"""

import numpy as np
from scipy.misc import imread, imsave
import pylab as plt
from inpaintools import *
  
w = lambda x : np.exp(-0.25 * (x**2))    

im = imread('elephant2_300x225_rgb.jpg').squeeze()
mask = imread('elephant2_300x225_msk.jpg').squeeze()
mask = mask > 10

#im=np.random.randint(255,size=(5,5,1))
#mask=np.zeros((5,5))
#mask[1:4,1:4]=1

errors = 0.
errorsSUM = 0.

numshifts = 99
#numshifts = 50
#numshifts = 25
shifts = np.zeros((numshifts, 2))

shifts[50:,:] = square_neighborhood(7) * np.array([30, 5])
shifts[0:25,:] = square_neighborhood(5) * 5
shifts[25:50,:] = square_neighborhood(5) * 2


#shifts = np.random.randint((numshifts, 2),)

#shifts = np.array([(i,j) for i in range(-1,2) for j in range(-1,1)])

data_neighborhood = square_neighborhood(3)
smoothness_neighborhood = square_neighborhood(3)


def alpha_expansion_step(im, mask, mx, my, alphax, alphay, D0, D1, smoothness_neighborhood):
    sh = mask.shape
    im = im.astype(np.float32)
    indices = np.arange(0, np.prod(sh)).reshape(sh).astype(np.int32)
    outM = warp(im, mx, my)
    outalpha = warp(im, alphax, alphay)

    half_neighborhood = smoothness_neighborhood
    
    L = []    
    for i, j in half_neighborhood:
        a, b, c, d = -j if j<0 else None,-j if j>0 else None,-i if i<0 else None,-i if i>0 else None
        p1 = indices[a:b,c:d]
        p2 = (shift_image(indices, -i, -j))[a:b,c:d]

        L.append(p1.astype(np.int32))
        L.append(p2.astype(np.int32))        
        
        Mnew = mask * shift_image(mask, -i,-j)#* shift_image(mask, i,j)


        tempx = shift_image(mx, -i, -j)
        tempy = shift_image(my, -i, -j) 
        outQ = warp(im, tempx, tempy) 
        E00 = ((diff_image(outM, outQ) + 0*difflabel(mx, my, tempx, tempy)) * Mnew).astype(np.float32)
        E10 = ((diff_image(outalpha, outQ) + 0*difflabel(alphax, alphay, tempx, tempy)) * Mnew).astype(np.float32)
        E01 = ((diff_image(outM, outalpha) + 0*difflabel(mx, my, alphax, alphay)) * Mnew).astype(np.float32)
        E11 = mx * 0

        L.append(E00[a:b,c:d].astype(np.float32))
        L.append(E10[a:b,c:d].astype(np.float32))
        L.append(E01[a:b,c:d].astype(np.float32))
        L.append(E11[a:b,c:d].astype(np.float32))
        
        if np.sum((E01[a:b,c:d]-E00[a:b,c:d]+E10[a:b,c:d]-E11[a:b,c:d])<0) > 0:

                global errors
                global errorsSUM
                errors = errors + 1
                X = (E01[a:b,c:d]-E00[a:b,c:d]+E10[a:b,c:d]-E11[a:b,c:d])
                errorsSUM = errorsSUM + np.sum(((X<0)*X)[:])
                
                if  np.abs(np.sum(((X<0)*X)[:])) > 0.0001:
                    plt.figure()
                    plt.imshow(mask[a:b,c:d] + 2 * (X<0), interpolation= 'nearest')
                    plt.show()
    #                plt.pause(10**(-3))               
               
        energy, alphamap = solve_binary_problem(indices, D0.astype(np.float32), D1.astype(np.float32), L)
    
    return energy, alphamap


def datatermX(im, mask, Mx, My, data_neighborhood):
    width, height, depth = im.shape
    im = im * (np.dstack((1 - mask, ) * depth))
    card_neighbors = len(data_neighborhood)
    result = np.zeros((width, height), dtype=np.float32)
    neighims = np.zeros((width, height, depth, card_neighbors), dtype=np.float32)
    w = lambda x : np.exp(-0.25*(x**2))
    
    for index_neighbor in range(card_neighbors):
        dx, dy = data_neighborhood[index_neighbor]
        neighims[:, :, :, index_neighbor] = shift_image(im, dx, dy)
    
    if 1:
        Mx = Mx * mask
        My = My * mask
        partial_result = np.zeros((width, height), dtype=np.float32)
        normalisation = np.zeros((width, height), dtype=np.float32)
        
        for index_neighbor in range(card_neighbors):
            dx, dy = data_neighborhood[index_neighbor]
            neighim = neighims[:, :, :, index_neighbor].copy()
            offsetim = warp(neighim, Mx, My)
            neigh_length = (dx ** 2 + dy ** 2) ** 0.5
            partial_result += w(neigh_length) * (diff_image(neighim, offsetim) ** 1) * (neighim[:,:,0]>0) * (offsetim[:,:,0]>0)
            normalisation  += w(neigh_length) * (neighim[:,:,0]>0) * (offsetim[:,:,0]>0)
        
        normalisation  += normalisation == 0
        partial_result /= normalisation
        result[:, :] = partial_result + penaltydata(Mx, My, mask)
    return result


def alpha_expansion(im, mask, shifts, data_energy, smoothness_neighborhood, data_neighborhood, rounds):
    sh = mask.shape
    #labelmap = np.zeros(sh).astype(np.int32)
    mx = np.zeros(sh).astype(np.int32)
    my = np.zeros(sh).astype(np.int32)
    energy = float('inf')

    D0 = datatermX(im,mask,mx,my,data_neighborhood)
    
    for t in range(rounds * len(shifts)):
        #print t
        currlab = t % len(shifts)
        alpha = shifts[currlab]
#        alpha = np.random.randint(-300,300, size=2)
#        ix, iy = np.meshgrid(np.arange(sh[1]), np.arange(sh[0]))
#        D0 = data_energy[iy, ix, labelmap]
#        D1 = data_energy[:,:,currlab]
        D1 = datatermX(im,mask,mask*0+alpha[0],mask*0+alpha[1],data_neighborhood)
        #mx, my = compute_displacement_map(labelmap, shifts, mask)
        mx, my = mx * mask, my * mask

        new_energy, alphamap = alpha_expansion_step(im, mask, mx, my, mask * alpha[0], mask * alpha[1], D0, D1, smoothness_neighborhood)
        
        if new_energy < energy :
            energy = new_energy
            #labelmap = currlab * alphamap + labelmap * (1 - alphamap)
            mx = alpha[0] * alphamap + mx * (1 - alphamap)
            my = alpha[1] * alphamap + my * (1 - alphamap)
            D0 =       D1 * alphamap + D0 * (1 - alphamap)

            print "it:%02d \t l:%d \t (%d,%d) \t changed:%d \t e:%f"%(t, currlab, alpha[0], alpha[1], np.sum(alphamap[:]), energy)
#            plt.figure(1); plt.imshow(warp(im, mx, my));  plt.pause(0.0001)
    
    return mx,my



def alpha_expansion_minibatch(im, mask, mx,my,shifts, data_energy, smoothness_neighborhood, data_neighborhood, rounds):
    D0 = datatermX(im,mask,mx,my,data_neighborhood)
    
    energy = float('inf')
    for t in range(rounds * len(shifts)):
        currlab = t % len(shifts)
        alpha = shifts[currlab]
        D1 = data_energy[:,:,currlab]
#        D1 = datatermX(im,mask,mask*0+alpha[0],mask*0+alpha[1],data_neighborhood)

        new_energy, alphamap = alpha_expansion_step(im, mask, mx, my, mask * alpha[0], mask * alpha[1], D0, D1, smoothness_neighborhood)
        
        #plt.figure(2); plt.imshow(alphamap);  plt.pause(0.0001)
        if new_energy < energy :
            energy = new_energy
            mx = alpha[0] * alphamap + mx * (1 - alphamap)
            my = alpha[1] * alphamap + my * (1 - alphamap)
            D0 =       D1 * alphamap + D0 * (1 - alphamap)

            print "it:%02d \t l:%d \t (%d,%d) \t changed:%d \t e:%f"%(t, currlab, alpha[0], alpha[1], np.sum(alphamap[:]), energy)
        #    plt.figure(1); plt.imshow(warp(im, mx, my));  plt.pause(0.0001)
    
    return mx,my
    
    
def pritch(im, mask, shifts, data_neighborhood, smoothness_neighborhood, rounds):


## single batch
#    data_energy = 0 #dataterm(im, mask, shifts, data_neighborhood)
#    mx, my = alpha_expansion(im, mask, shifts, data_energy, smoothness_neighborhood, data_neighborhood, rounds)
##    mx, my = compute_displacement_map(labelmap, shifts)
#    mx, my = mx * mask, my * mask



###  minibatch
    mx = np.zeros(mask.shape).astype(np.int32)
    my = np.zeros(mask.shape).astype(np.int32)
    batchsize = 15 #len(shifts)
    for i in range(1,len(shifts),batchsize):
       ss  = shifts[i:i+batchsize]
       data_energy = dataterm(im, mask, ss, data_neighborhood)
       mx, my = alpha_expansion_minibatch(im, mask, mx,my,ss, data_energy, smoothness_neighborhood, data_neighborhood, rounds=2)
#       plt.figure(1); plt.imshow(warp(im, mx, my));  plt.pause(0.0001)


### refine loop
#    energy = float('inf')
#    D0 = datatermX(im,mask,mx,my,data_neighborhood)
#    for t in range(5):
#       for i,j in ((1,0),(0,-1),(0,1),(-1,0)):#,(1,-1),(-1,1),(1,1),(-1,-1)):
#          amx = mx + i*1
#          amy = my + j*1
#          D1 = datatermX(im,mask,amx,amy,data_neighborhood)
#          new_energy, alphamap = alpha_expansion_step(im, mask, mx,my,amx,amy, D0, D1, smoothness_neighborhood)
#   
#          if new_energy < energy :
#              energy = new_energy
#              #labelmap = currlab * alphamap + labelmap * (1 - alphamap)
#              mx = amx * alphamap + mx * (1 - alphamap)
#              my = amy * alphamap + my * (1 - alphamap)
#              D0 =  D1 * alphamap + D0 * (1 - alphamap)
#              print "%d (%d,%d) \t changed:%d \t e:%f"%(t, i*t, j*t, np.sum(alphamap[:]), energy)
#              plt.figure(1); plt.imshow(warp(im, mx, my));  plt.pause(0.0001)



### all refine
#    energy = float('inf')
#    mx = np.zeros(mask.shape).astype(np.int32)
#    my = np.zeros(mask.shape).astype(np.int32)
#    D0 = datatermX(im,mask,mx,my,data_neighborhood) 
#    for i in range(len(shifts)):
#       ss = shifts[i]
#       #ss = np.random.randint(-300,300, size=2)
#       amx = mx*1 + ss[0]
#       amy = my*1 + ss[1]
##       D0 = datatermX(im,mask,mx,my,data_neighborhood)
#       D1 = datatermX(im,mask,amx,amy,data_neighborhood) 
#       new_energy, alphamap = alpha_expansion_step(im, mask, mx, my, amx, amy, D0, D1, smoothness_neighborhood)
#
#       if new_energy < energy :
#           energy = new_energy
#           #labelmap = currlab * alphamap + labelmap * (1 - alphamap)
#           mx = amx * alphamap + mx * (1 - alphamap)
#           my = amy * alphamap + my * (1 - alphamap)
#           D0 =  D1 * alphamap + D0 * (1 - alphamap)
#           print "%d (%d,%d) \t changed:%d \t e:%f"%(i, ss[0], ss[1], np.sum(alphamap[:]), energy)
#           #plt.figure(1); plt.imshow(warp(im, mx, my));  plt.pause(0.0001)

    return warp(im, mx, my), mx, my

if __name__ == "__main__":
   import sys
   if len(sys.argv)>1:
      im = imread(sys.argv[1]).squeeze()
      mask = imread(sys.argv[2]).squeeze()
   mask = mask > 10
   sub=1
   im   = im[::sub,::sub,:];
   mask = mask[::sub,::sub];
   np.set_printoptions(threshold=np.nan)
   smoothness_neighborhood = np.array([[0,1], [1,0], [1,1], [1,-1]])
   smoothness_neighborhood = np.array([[-1,0]])
   smoothness_neighborhood = np.array([[0,1], [1,0]])
   
   out, mx,my  = pritch(im, mask, shifts, data_neighborhood, smoothness_neighborhood, 1)
   if len(sys.argv)>3:
      imsave(sys.argv[3], np.uint8(out))
   else:
      plt.figure(1); plt.imshow(warp(im, mx, my));  plt.show()
   print errors, errorsSUM

 plt.imsave('C:\Users\D\Desktop\inpainting\image\' + r + '_pritchlabel.png',labelmap,vmin=0,vmax=255,format='png')