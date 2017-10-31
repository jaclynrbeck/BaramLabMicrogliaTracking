#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 20:54:49 2017

@author: jaclynbeck
"""

import scipy as sp
import timeit
import Utils
import skimage.morphology as skm
from libtiff import TIFF
import cv2
import FindSomas as fs


MIN_OBJECT_SIZE = 10*10


def skeletonize_level(img, somas, threshold):
    bw = sp.zeros_like(img, dtype='uint8')
    bw[img > threshold] = 1
    
    number, labels, stats, centoids = cv2.connectedComponentsWithStats(bw, connectivity=4)
    
    valid = sp.where(stats[:,4] > MIN_OBJECT_SIZE)[0][1:]
    regions = sp.zeros_like(bw)
    
    for v in valid:         
        regions[labels == v] = 1

    #for soma in somas:
    #    regions[soma.rows(), soma.cols()] = 0
        
    skeleton = skm.skeletonize(regions)
    return skeleton


def get_thresholds(img):
    # We want coarse thresholding from 50-85% at intervals of 5, then fine
    # thresholding from 90-99% at intervals of 1
    percentiles = list(sp.arange(70, 86, 5)) + list(sp.arange(90, 100))
    thresholds = sp.percentile(img, percentiles)
    
    thresholds = sp.array(list(set(thresholds)))
    thresholds.sort()
    return thresholds


def parse_regions(skeleton):
    bw = sp.zeros_like(skeleton, dtype='uint8')
    bw[skeleton > 0] = 1
    
    number, labels, stats, centoids = cv2.connectedComponentsWithStats(bw, connectivity=8)
    
    valid = sp.where(stats[:,4] > 10)[0][1:]
    region_mask = sp.zeros_like(bw)
    
    for v in valid:         
        region_mask[labels == v] = 1
    
    skeleton[region_mask == 0] = 0
    return skeleton


def delete_template_matches(bw, skeleton, template):
    res = cv2.matchTemplate(bw, template, cv2.TM_SQDIFF)
    pts = sp.vstack(sp.where(res == 0)).T # Exact matches only
    
    size = template.shape
    
    for p in range(pts.shape[0]):
        rows = sp.arange(pts[p,0], pts[p,0]+size[0])
        cols = sp.arange(pts[p,1], pts[p,1]+size[1])
        coords_r, coords_c = sp.meshgrid(rows, cols, indexing='ij')
    
        skeleton[coords_r, coords_c] = 0
        
    return skeleton
    
    
def remove_artifacts(skeleton):
    bw = sp.zeros_like(skeleton, dtype='uint8')
    bw[skeleton > 0] = 1

    new_skeleton = skeleton.copy()
    
    template1 = sp.eye(5, dtype='uint8')
    template2 = template1[::-1]
    template3 = sp.array([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0]], dtype='uint8')
    template4 = template3.T
    
    for T in (template1, template2, template3, template4):
        new_skeleton = delete_template_matches(bw, new_skeleton, T)
            
    return new_skeleton
        

def trace_image(img, index, kernel, output_dir):
    img = Utils.preprocess_img(img)
    
    thresholds = get_thresholds(img)
    [soma_threshold, somas] = fs.find_somas_single_image(img)
    
    skeletons = sp.zeros_like(img, dtype='uint8')

    for t in thresholds: 
        skeletons += skeletonize_level(img, somas, t)     
        
    skeletons[skeletons == 1] = 0
    
    #skeletons = remove_artifacts(skeletons)
    skeletons = parse_regions(skeletons)
    
    new_skeleton = cv2.morphologyEx(sp.minimum(skeletons, 1), cv2.MORPH_CLOSE, kernel)
    skeleton_final = skm.skeletonize(new_skeleton)

    return skeleton_final
    

"""
Main function for debugging
"""
if __name__ == '__main__':
    img_fname  = '/Users/jaclynbeck/Desktop/BaramLab/full_zstack_video_green.tif' #'/Users/jaclynbeck/Desktop/BaramLab/C2-8-29-17_CRH-tdTomato+CX3CR1-GFP P8 PVN CES_Female 1 L PVN_a_.i...CX3CR1-GFP P8 PVN CES_Female 1 L PVN_a_GREEN_t1.tif'
    output_dir = '/Users/jaclynbeck/Desktop/BaramLab/'

    tif = TIFF.open(img_fname, mode='r')
    out_tif = TIFF.open(output_dir + 'tst2.tif', mode='w')
    
    start_time = timeit.default_timer()
    index = 0
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    
    for img in tif.iter_images(): 
        img.byteswap(True)
        skeleton = trace_image(img, index, kernel, output_dir)
        index += 1
        out_tif.write_image(skeleton.astype('uint8')*255)
        
    elapsed = timeit.default_timer() - start_time
    print(elapsed)
    
    tif.close()
    out_tif.close()


