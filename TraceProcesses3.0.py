#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 20:54:49 2017

@author: jaclynbeck
"""

import scipy as sp
import timeit
import skimage.morphology as skm
from libtiff import TIFF
import cv2
import FindSomas as fs


MIN_OBJECT_SIZE = 10*10


def skeletonize_level(img, threshold):
    bw = sp.zeros_like(img, dtype='uint8')
    bw[img > threshold] = 1
    
    number, labels, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=4)
    
    valid = sp.where(stats[:,4] > MIN_OBJECT_SIZE)[0][1:]
    regions = sp.zeros_like(bw)
    
    for v in valid:      
        regions[labels == v] = 1

    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    #regions = cv2.morphologyEx(regions, cv2.MORPH_CLOSE, kernel)
    #regions = cv2.morphologyEx(regions, cv2.MORPH_OPEN, kernel)
    
    skeleton = skm.skeletonize(regions).astype('uint8')

    #tst = 2*skeleton + regions
    #sp.misc.imsave("/Users/jaclynbeck/Desktop/BaramLab/skel_" + str(int(threshold)) + ".tif", tst*128)
    return skeleton


def get_thresholds(img):
    # We want to threshold the top 80-99% of pixels in intervals of 1 percentile
    percentiles = list(sp.arange(80, 100))
    thresholds = sp.percentile(img, percentiles)
    
    thresholds = sp.array(list(set(thresholds)))
    thresholds.sort()
    return thresholds


def parse_regions(skeleton):
    bw = sp.zeros_like(skeleton, dtype='uint8')
    bw[skeleton > 0] = 1
    
    number, labels, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=8)
    
    valid = sp.where(stats[:,4] > 10)[0][1:]
    region_mask = sp.zeros_like(bw)
    
    for v in valid:         
        region_mask[labels == v] = 1
    
    skeleton[region_mask == 0] = 0
    return skeleton
        

def trace_image(img, index, kernel, output_dir):
    sobelx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    sobelxy = sp.sqrt(sobelx**2 + sobely**2)
    sobelxy = (sobelxy * (255.0/sobelxy.max())).astype('uint8')
    
    #sp.misc.imsave("/Users/jaclynbeck/Desktop/BaramLab/sobel_"+str(z)+".tif", sobelxy)
    
    thresh = sp.percentile(sobelxy, 80)
    bw = sp.zeros_like(sobelxy)
    bw[sobelxy > thresh] = 255
    
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    #bw_closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k)
    #bw_closed = cv2.morphologyEx(bw_closed, cv2.MORPH_ERODE, k)
    
    im2, contours, hierarchy = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    bw2 = cv2.drawContours(sp.zeros_like(bw), contours, -1, 1, thickness=-1)
    
    bw2 = cv2.morphologyEx(bw2, cv2.MORPH_ERODE, k)

    #cv2.imshow("contours", bw2)
    #cv2.imshow("sobel", sobelxy)
    #n_components, labels = cv2.connectedComponents(edges, connectivity=8)
    #cv2.imshow("edges", edges)
    #cv2.waitKey()
    
    skeleton_final = skm.skeletonize(bw2).astype('uint8')*128
    
    #new_skeleton = cv2.morphologyEx(sp.minimum(skeletons, 1), cv2.MORPH_CLOSE, kernel)
    #skeleton_final = skm.skeletonize(new_skeleton).astype('uint8')*128
    
    [soma_threshold, somas] = fs.find_somas_single_image(img)
    for soma in somas:
        skeleton_final[soma.rows(), soma.cols()] = 0
        
    for soma in somas:
        skeleton_final[soma.rows(), soma.cols()] = 0
        skeleton_final[soma.centroid[0], soma.centroid[1]] = 255
        skeleton_final[soma.contourRows(), soma.contourCols()] = 200

    #add_value = 128-skeletons.max()
    #skeleton_final[skeleton_final == 128] = skeletons[skeleton_final == 128] + add_value

    return skeleton_final
    

"""
Main function for debugging
"""
if __name__ == '__main__':
    img_fname  = '/Users/jaclynbeck/Desktop/BaramLab/videos/A_LPVN_T1_08202017/processed_max_projection.tif' 
    output_dir = '/Users/jaclynbeck/Desktop/BaramLab/videos/A_LPVN_T1_08202017/'

    tif = TIFF.open(img_fname, mode='r')
    out_tif = TIFF.open(output_dir + 'processed_skeleton_max_projection.tif', mode='w')
    
    start_time = timeit.default_timer()
    index = 0
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    
    window = []
    
    for img in tif.iter_images(): 
        skeleton = trace_image(img, index, kernel, output_dir)
            
        index += 1
        out_tif.write_image(skeleton)
        
    elapsed = timeit.default_timer() - start_time
    print(elapsed)
    
    tif.close()
    out_tif.close()


