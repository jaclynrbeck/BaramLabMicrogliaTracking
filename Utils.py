#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 15:19:03 2017

@author: jaclynbeck
"""

import scipy as sp
import matplotlib.pyplot as plt

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
    

    
def plot_levels(levels, width, height, display=False):
    bw = sp.zeros((width, height), dtype='uint16')
    
    for i in range(len(levels)):
        L = levels[len(levels)-i-1]
        
        if L.isBackgroundLevel():
            color = 0
        else:
            color = L.threshold+1
        
        #bw[L.rows(),L.cols()] = color
        for r in range(L.regionCount):
            bw[L.regionRows(r), L.regionCols(r)] = color 
    
    if display:
        plt.imshow(bw*255.0/bw.max())
        plt.show()
    
    return bw


def plot_somas(somas, display=False):
    s_img = sp.zeros((1024,1024), dtype='uint8')
    color = 0
    for soma in somas:
        s_img[soma.rows(), soma.cols()] = 20 + 5*color
        color += 1
        
    s_img = sp.array(s_img * (255.0 / s_img.max()), dtype='uint8')
    
    if display:
        plt.imshow(s_img)
        plt.show()
    
    return s_img


def plot_seed_regions(seedRegions, display=False):
    s_img = sp.zeros((1024,1024), dtype='uint8')
    color = 0
    for seed in seedRegions:
        s_img[seed.rows(), seed.cols()] = 20 + 5*color
        for soma in seed.somas:
            s_img[soma.rows(),soma.cols()] = 255
            
        color += 1
        
    s_img = sp.array(s_img * (255.0 / s_img.max()), dtype='uint8')
    
    if display:
        plt.imshow(s_img)
        plt.show()
    
    return s_img
    


def plot_seed_points(points, display=False):
    bw = sp.zeros((1024,1024), dtype='uint8')
    
    for pts in points:
        bw[pts[:,0],pts[:,1]] = 255
    
    if display:
        plt.imshow(bw)
        plt.show()
        
    return bw