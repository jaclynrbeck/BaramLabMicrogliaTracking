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
import Utils
import javabridge # required by bioformats
import bioformats # installed with pip install python-bioformats
from skimage import restoration


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
    
    src = []
    dst = []
    
    for m in matches:
        src.append(sp.array(kp1[m[0]].pt))
        dst.append(sp.array(kp2[m[1]].pt))

    model_robust, inliers = ransac((sp.array(src), sp.array(dst)), 
                                   AffineTransform, min_samples=3,
                                   residual_threshold=2, max_trials=100)
    
    drift = cv2.estimateRigidTransform(sp.array(dst)[inliers], sp.array(src)[inliers], True)
    print(sp.round_(drift, 2))

    return drift 


def register_affine(img1, img2):
    (cc, warp_matrix) = cv2.findTransformECC(img2, img1, None, cv2.MOTION_AFFINE)
    return warp_matrix


def preprocess_images_2D(img_fname, output_fname, window_size):
    javabridge.start_vm(class_path=bioformats.JARS)
    averaged_tif = TIFF.open(output_fname, mode='w')
    psf = cv2.imread("/Users/jaclynbeck/Desktop/BaramLab/PSF.png", cv2.IMREAD_GRAYSCALE)
    psf = psf / sp.sum(psf)

    xml = bioformats.get_omexml_metadata(path=img_fname)
    ome_xml = bioformats.OMEXML(xml)
    sizeX = ome_xml.image().Pixels.SizeX
    sizeY = ome_xml.image().Pixels.SizeY
    numImages = ome_xml.image().Pixels.SizeT
    physX = ome_xml.image().Pixels.PhysicalSizeX
    physY = ome_xml.image().Pixels.PhysicalSizeY
    
    print("File contains " + str(numImages) + " images. Size: " + str(sizeX) \
          + " x " + str(sizeY) + " px (" + str(physX*sizeX) + " x " + \
          str(physY*sizeY) +  " um)")
    
    truncate = sp.ceil(30/0.633) # number of frames that equals 30 mins
    if numImages > truncate:
        numImages = int(truncate)
        print("Truncating to " + str(numImages) + " images (30 minutes)")
    
    window = []
    img0 = None
    
    with bioformats.ImageReader(img_fname) as reader:
        for t in range(numImages):
            img = reader.read(c=1, z=0, t=t, series=None, index=None, rescale=True, wants_max_intensity=False, channel_names=None, XYWH=None)
            deconvolved = restoration.richardson_lucy(img, psf, iterations=50)
            deconvolved = (deconvolved*65535).astype('float32')
            #img = (img*65535).astype('uint16')
            #img = Utils.preprocess_img(img)
            
            if img0 is not None:
                drift = register_affine(img0, img)
    
                # Warp the image
                warped_img_deconvolved = warp_image_affine(deconvolved, drift)
                warped_img_orig = warp_image_affine(img, drift)
            else:
                warped_img_deconvolved = deconvolved
                warped_img_orig = img
                
            window.append(warped_img_deconvolved)
            
            # Keep the warped image for the next round of registration
            img0 = warped_img_orig
            
            if len(window) < window_size:
                continue
            
            # Sum all images in the window. Don't bother dividing since it doesn't
            # really matter for this image set (no potential for overflow of the
            # uint16 limit). 
            averaged = sum(window)
            averaged_tif.write_image(averaged) #averaged.astype('uint8'))
        
            window.clear() 
            
        reader.close()

    averaged_tif.close()
    
    #javabridge.kill_vm()



if __name__=='__main__':
    img_fname    = '/Users/jaclynbeck/Desktop/BaramLab/videos/A_LPVN_T1_08202017/8-20-17_crh-tdtomato+cx3cr1-gfp p8 pvn ces_male 1 l pvn t1_b_4d.ims' 
    output_fname = '/Users/jaclynbeck/Desktop/BaramLab/videos/A_LPVN_T1_08202017/processed_max_projection_deconvolved_50iter.tif'
    
    start_time = timeit.default_timer()

    preprocess_images_2D(img_fname, output_fname, 1)
            
    elapsed = timeit.default_timer() - start_time
    print(elapsed)

    
    