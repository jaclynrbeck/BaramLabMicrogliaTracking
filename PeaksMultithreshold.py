#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 12:00:58 2017

@author: jaclynbeck
"""

import scipy as sp
import matplotlib.pyplot as plt
import detect_peaks as dp 


def peaks_multithreshold(img, peak_threshold):    
    L = img.max()+1;
    
    counts, binEdges = sp.histogram(img, L)
    thresholds = dp.detect_peaks(counts, threshold=peak_threshold)
    
    return thresholds


if __name__ == '__main__':
    img_fname = '/Users/jaclynbeck/Desktop/BaramLab/Substack (1).tif' #C2-8-29-17_CRH-tdTomato+CX3CR1-GFP P8 PVN CES_Female 1 L PVN_a_.i...CX3CR1-GFP P8 PVN CES_Female 1 L PVN_a_GREEN_t1.tif'
    peak_threshold = 100; 
    
    img = sp.misc.imread(img_fname)
    thresholds = peaks_multithreshold(img, peak_threshold)
    print(thresholds)

    L = img.max() + 1; 
    
    if L == 256:
        img = img - img.min()
        img = sp.round_(img/img.max()*255)
        
    k = thresholds
    bw = sp.zeros_like(img, dtype='uint8')
        
    for i in range(k.size):
        bw[img >= k[i]] = 20 + 5*(i+1)

    plt.imshow(bw / bw.max())
    plt.show()
    
    sp.misc.imsave('/Users/jaclynbeck/Desktop/BaramLab/pyGliaMask.tif', bw*255/bw.max())