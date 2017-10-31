#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 15:19:03 2017

@author: jaclynbeck
"""

import cv2
import scipy as sp
import matplotlib.pyplot as plt
#from AmbrosioTortorelliMinimizer import *


"""
Contrast Limited Adaptive Histogram equalization to increase contrast
"""
def equalize_img(img):
    if img.dtype != 'uint8':
        img = (img * 255.0 / img.max()).astype('uint8')

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(img)

    return cl1 


"""
Denoise using a bilateral filter to preserve edges
"""
def denoise_img(img):
    img = sp.array(img, dtype=sp.float32)   
    return cv2.bilateralFilter(img, 5, 75, 75)


"""
Equalizes img histogram, denoises the image, and does a preliminary
adaptive thresholding to remove low-value pixels
"""
def preprocess_img(img):
    img = equalize_img(img)
    img = denoise_img(img).astype('uint8')
    
    #bw = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    #                           cv2.THRESH_BINARY, 65, 0)
    
    #img[bw==0] = 0
    
    return img
    

def bbox_has_overlap(bbox1, bbox2):
    if bbox1[2] < bbox2[0] or bbox1[3] < bbox2[1]:
        return False
        
    if bbox1[0] > bbox2[2] or bbox1[1] > bbox2[3]:
        return False
        
    return True


def bbox_overlap_area(bbox1, bbox2):
    if not bbox_has_overlap(bbox1, bbox2):
        return 0
    
    xmin = max(bbox1[0], bbox2[0])
    ymin = max(bbox1[1], bbox2[1])
    xmax = min(bbox1[2], bbox2[2])
    ymax = min(bbox1[3], bbox2[3])
    
    return (xmax-xmin)*(ymax-ymin)


def bbox_area(bbox):
    return (bbox[2]-bbox[0])*(bbox[3]-bbox[1])


def bbox_significant_overlap(bbox1, bbox2, overlap_percent):
    overlap = bbox_overlap_area(bbox1, bbox2)
        
    if overlap == 0:
        return False
    
    # Overlap needs to be significant
    if (overlap / bbox_area(bbox1) > overlap_percent) \
        or (overlap / bbox_area(bbox2) > overlap_percent):
            return True
        
    return False
    

    
def plot_levels(levels, img_size, display=False):
    bw = sp.zeros(img_size, dtype='uint16')
    
    for i in range(len(levels)):
        L = levels[len(levels)-i-1]
        
        if L.isBackgroundLevel():
            color = 0
        else:
            color = L.threshold+1
        
        #bw[L.rows(),L.cols()] = color
        for region in L.regions:
            bw[region.rows(), region.cols()] = color 
    
    if display:
        plt.imshow(bw*255.0/bw.max())
        plt.show()
    
    return bw


def plot_somas(somas, img_size, display=False):
    s_img = sp.zeros(img_size, dtype='uint8')
    color = 0
    for soma in somas:
        s_img[soma.rows(), soma.cols()] = 20 + 5*color
        color += 1
        
    s_img = sp.array(s_img * (255.0 / s_img.max()), dtype='uint8')
    
    if display:
        plt.imshow(s_img)
        plt.show()
    
    return s_img


def plot_seed_regions(seedRegions, img_size, display=False):
    s_img = sp.zeros(img_size, dtype='uint8')
    
    for seed in seedRegions:
        s_img[seed.rows(), seed.cols()] = 128
        for soma in seed.somas:
            s_img[soma.rows(),soma.cols()] = 255
        
    s_img = sp.array(s_img * (255.0 / s_img.max()), dtype='uint8')
    
    if display:
        plt.imshow(s_img)
        plt.show()
    
    return s_img
    


def plot_seed_points(seedRegions, img_size, display=False):
    bw = sp.zeros(img_size, dtype='uint8')
    
    for seed in seedRegions:
        for pts in seed.seedPoints:
            bw[pts[:,0],pts[:,1]] = 255
    
    if display:
        plt.imshow(bw)
        plt.show()
        
    return bw


def plot_cell_regions(regions, somas, img_size, display=False):
    bw = sp.zeros(img_size, dtype='uint8')
    
    for region in regions:
        bw[region.rows(), region.cols()] = 128
    
    for soma in somas:
        bw[soma.rows(), soma.cols()] = 255
    
    if display:
        plt.imshow(bw)
        plt.show()
        
    return bw


def plot_skeleton(regions, somas, img_size, display=False):
    bw = sp.zeros(img_size, dtype='uint8')
    
    for region in regions:
        bw[region.skeletonRows(), region.skeletonCols()] = 255
        
    for soma in somas:
        bw[soma.rows(), soma.cols()] = 255
    
    if display:
        plt.imshow(bw)
        plt.show()
        
    return bw