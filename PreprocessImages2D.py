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
import javabridge # required by bioformats
import bioformats # installed with pip install python-bioformats
from skimage import restoration


def warp_image_affine(img, drift):
    img2 = cv2.warpAffine(img, drift, img.shape)
    
    return img2


def register_affine(img1, img2):
    (cc, warp_matrix) = cv2.findTransformECC(img2, img1, None, cv2.MOTION_AFFINE)
    return warp_matrix


def preprocess_images_2D(img_fname, output_fname, psf_fname, window_size, deconvolution_iters):
    javabridge.start_vm(class_path=bioformats.JARS)
    averaged_tif = TIFF.open(output_fname, mode='w')
    psf = cv2.imread(psf_fname, cv2.IMREAD_GRAYSCALE)
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
            print("Processing image " + str(t) + "...")
            img = reader.read(c=1, z=0, t=t, series=None, index=None, rescale=True, wants_max_intensity=False, channel_names=None, XYWH=None)
            deconvolved = restoration.richardson_lucy(img, psf, iterations=deconvolution_iters)
            deconvolved = (deconvolved*65535).astype('float32')
            
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
            
            averaged = averaged * (65536.0 / averaged.max())
                
            averaged_tif.write_image(averaged.astype('uint16')) 
        
            window.clear() 
            
        reader.close()

    averaged_tif.close()
    
    #javabridge.kill_vm()



if __name__=='__main__':
    img_fname    = '/Users/jaclynbeck/Desktop/BaramLab/videos/B_LPVN_T1_10052017/10-05-17_crh-tdtomato+cx3cr1-gfp p8 pvn ctl_male 2 l pvn t1_b_4d_male 2 l pvn t1.ims' 
    output_fname = '/Users/jaclynbeck/Desktop/BaramLab/videos/B_LPVN_T1_10052017/processed_max_projection_deconvolved_5iter.tif'
    psf_fname    = '/Users/jaclynbeck/Desktop/BaramLab/PSF.png'
    start_time = timeit.default_timer()

    preprocess_images_2D(img_fname, output_fname, psf_fname, 1, 5)
            
    elapsed = timeit.default_timer() - start_time
    print(elapsed)

    
    