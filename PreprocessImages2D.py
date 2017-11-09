#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 14:07:23 2017

@author: jaclynbeck
"""

from libtiff import TIFF
import scipy as sp
import cv2
import timeit
from skimage.measure import ransac
from skimage.transform import AffineTransform
#from matplotlib import pyplot as plt


def warp_image_affine(img, drift):
    img2 = cv2.warpAffine(img, drift, img.shape)
    
    return img2


"""
This method doesn't work well, but leaving code in for future reference. 
"""
def register_with_SIFT(img1, img2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    orb = cv2.ORB_create()
    
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)
    
    matches = bf.match(des2,des1) # des2 = query, des1 = training
    matches = [(M.trainIdx, M.queryIdx) for M in matches]
    
    #diffs = []
    src = []
    dst = []
    
    for m in matches:
        #diffs.append(sp.array(kp2[m[1]].pt) - sp.array(kp1[m[0]].pt))
        src.append(sp.array(kp1[m[0]].pt))
        dst.append(sp.array(kp2[m[1]].pt))

    model_robust, inliers = ransac((sp.array(src), sp.array(dst)), 
                                   AffineTransform, min_samples=3,
                                   residual_threshold=2, max_trials=100)
    
    drift = cv2.estimateRigidTransform(sp.array(dst)[inliers], sp.array(src)[inliers], True)
    print(sp.round_(drift, 2))
    #drift = sp.round_(sp.mean(sp.array(diffs)[inliers], axis=0)).astype('int16')
    return drift #sp.array([[1, 0, -drift[0]], [0, 1, -drift[1]]])


def register_affine(img1, img2):
    (cc, warp_matrix) = cv2.findTransformECC(img2, img1, None, cv2.MOTION_AFFINE)
    return warp_matrix


def preprocess_images_2D(img_fname, output_fname, window_size):
    img_tif  = TIFF.open(img_fname, mode='r')
    averaged_tif = TIFF.open(output_fname, mode='w')
    
    window = []
    
    img0 = img_tif.read_image()
    img0.byteswap(True)
    img0 = (img0 * (255.0/img0.max())).astype('uint8')
    
    for img in img_tif.iter_images():
        img.byteswap(True)
        
        # OpenCV registration function requires a uint8 image
        img_uint8 = (img * (255.0/img.max())).astype('uint8')
        drift = register_affine(img0, img_uint8)
        
        # Warp the original uint16 image
        new_img = warp_image_affine(img, drift)
        window.append(new_img)
        
        # Keep the uint8 version of the warped image for the next round of 
        # registration
        img0 = (new_img * (255.0/new_img.max())).astype('uint8')
        
        if len(window) < window_size:
            continue
        
        # Sum all images in the window. Don't bother dividing since it doesn't
        # really matter for this image set (no potential for overflow of the
        # uint16 limit). 
        averaged = sum(window)
        averaged_tif.write_image(averaged.astype('uint16'))
        
        window.clear() #.remove(window[0])

    img_tif.close()
    averaged_tif.close()



if __name__=='__main__':
    img_fname    = '/Users/jaclynbeck/Desktop/BaramLab/8-29-17_CRH-tdTomato+CX3CR1-GFP P8 PVN CES_Female 1 L PVN_b_4D_1.ims - C=1.tif' #'/Users/jaclynbeck/Desktop/BaramLab/C2-8-29-17_CRH-tdTomato+CX3CR1-GFP P8 PVN CES_Female 1 L PVN_a_.i...CX3CR1-GFP P8 PVN CES_Female 1 L PVN_a_GREEN_t1.tif'
    output_fname = '/Users/jaclynbeck/Desktop/BaramLab/averaged_max_projection.tif'
    
    start_time = timeit.default_timer()

    preprocess_images_2D(img_fname, output_fname, 3)
            
    elapsed = timeit.default_timer() - start_time
    print(elapsed)

    
    