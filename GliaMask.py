#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 10:32:13 2017

@author: jaclynbeck
"""

import scipy as sp
import matplotlib.pyplot as plt
import cv2 # installed with 'conda install -c conda-forge opencv'


def create_Mask(img):
    thresh = sp.median(img)
    bw = sp.zeros_like(img)
    bw[img > thresh] = 1
    
    kernel = sp.ones((3,3),sp.uint8)
    mask = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    
    mask[mask > 0] = 1

    return mask


if __name__ == '__main__':
    img_fname = '/Users/jaclynbeck/Desktop/BaramLab/Substack (8).tif' #'/Users/jaclynbeck/Desktop/BaramLab/C2-8-29-17_CRH-tdTomato+CX3CR1-GFP P8 PVN CES_Female 1 L PVN_a_.i...CX3CR1-GFP P8 PVN CES_Female 1 L PVN_a_GREEN_t1.tif'
    output_img = '/Users/jaclynbeck/Desktop/BaramLab/gliaMask.tif'
    
    img = cv2.imread(img_fname, flags=(cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE))
    mask = create_Mask(img)
    plt.imshow(mask)
    plt.show()
    
    sp.misc.imsave('/Users/jaclynbeck/Desktop/BaramLab/pyGliaMask.tif', mask)
    