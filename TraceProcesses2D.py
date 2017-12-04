#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Traces the skeletons of the microglia in every image in an image stack. Also
locates somas and tracks them across frames. 

Created on Thu Oct 5 20:54:49 2017

@author: jaclynbeck
"""

import scipy as sp
import timeit
import skimage.morphology as skm
from libtiff import TIFF
import cv2
import FindSomas as fs
import pickle
import os


# A default threshold to use if none is specified
THRESHOLD = 85


"""
Takes a single image frame, thresholds it to create a binary file, and 
skeletonizes the binary image to get the structure of the microglia. Regions
of the binary image that are too small (i.e. have less than 20 pixels in them)
are removed prior to skeletonization. 

The skeleton is then altered so that each white pixel in the skeleton takes on
the pixel value of the corresponding pixel in the original image. The minimum
spanning tree algorithm uses the difference in pixel values in the skeleton
to decide what to connect first. 

Input: 
    img - (MxN ndarray) uint16 image to skeletonize
    threshold - (int, between 0 and 100) the percent of pixels to threshold
                to black. 80-90 works well as a value on deconvolved images. 
    
Output:
    skeleton - (MxN ndarray) uint8 skeleton image
    bw - (MxN ndarray) uint8 thresholded image
"""
def trace_image(img, threshold):
    # Threshold the image
    thresh = sp.percentile(img, threshold)
    bw = sp.zeros_like(img, dtype='uint8')
    bw[img > thresh] = 1

    # Get a list of all connected regions in the thresholded image
    number, labels, stats, centroids = cv2.connectedComponentsWithStats(bw, 
                                                                connectivity=4)
    
    # Here we process whichever list is shorter to help with code speed: 
    # valid regions or invalid (too small) regions
    invalid = sp.where(stats[:,4] < 20)[0]
    valid   = sp.where(stats[:,4] >= 20)[0]
    
    # Processing the valid list: make a black image and only mark valid regions
    # as white
    if len(valid) < len(invalid):
        bw = sp.zeros_like(img, dtype='uint8')
        
        for v in valid[1:]: # Skip valid[0], which is background
            bw[labels == v] = 1
    
    # Processing the invalid list: use the thresholded images and zero out
    # invalid regions
    else:
        for v in invalid:      
            bw[labels == v] = 0
    
    # Skeletonize the binary image and turn it from True/False values to 
    # a 'uint16' type. Then, instead of using the default black/white skeleton,
    # substitute the image pixel values in for all of the skeleton pixels so
    # that the minimum spanning tree algorithm can use those values. 
    skeleton = skm.skeletonize(bw).astype(img.dtype)
    skeleton[skeleton > 0] = img[skeleton > 0]
    
    # This rebalances the distribution of pixels a little bit. The distribution
    # is really skewed toward the left such that when converting to a uint8,
    # the majority of pixels would be indistinguishable due to being < 250. 
    # Instead we take the top 20% of pixel values and make them the 20% 
    # threshold value, so that the total range of pixel values is a little 
    # smaller and we can distinguish between lower-valued pixels better. 
    thresh = sp.percentile(skeleton[skeleton > 0], 80)
    skeleton[skeleton > thresh] = thresh
    
    # Convert to a uint8, setting the max value of skeleton pixels as 250.
    skeleton = sp.ceil((skeleton * (250.0/skeleton.max()))).astype('uint8')

    return skeleton, bw


"""
Skeletonizes all images in an image stack, and finds all somas in the image
stack. Somas are correlated across frames to provide consistency. All output
is put in the same directory as the input .tif file. 

Input:
    img_fname    - Full file path to the .tif file to skeletonize
    output_fname - Local file name to save the skeleton to. 
                    * Must end in .tif
    soma_fname   - Local file name to save the soma information to. 
                    * Must end in .p
    threshold    - int, the percentage of pixels to threshold to zero in the
                   .tif image. 80-90 typically works well. 
    
Output:
    (no output), but several files are created through this process:
        skeletonized image stack
        soma pickle file holding the soma spatial information
"""
def trace_all_images(img_fname, output_fname, soma_fname, threshold=THRESHOLD):
    # Make the local filenames full file paths
    path = os.path.dirname(img_fname)
    output_fname = path + "/" + output_fname
    soma_fname = path + "/" + soma_fname
    
    # Open the input and output image files
    tif = TIFF.open(img_fname, mode='r')
    out_tif = TIFF.open(output_fname, mode='w')

    videoSomas = []
    skeletons = []
    threshold_images = []
    
    frame = 0
    
    # Skeletonize each image and add the skeletons to a list
    for img in tif.iter_images(): 
        # Checks for big endian, change to little endian. For some reason
        # some images got written as big endian and the bytes need to be 
        # swapped. 
        if img.mean() > 20000: 
            img = img.byteswap()
            
        skeleton, thresh_img = trace_image(img, threshold) 
        skeletons.append(skeleton)
        threshold_images.append(thresh_img)
        
        # Find the somas for this image and add them to the list
        [soma_threshold, frame_somas] = fs.find_somas_single_image(img, frame)
        videoSomas = fs.combine_somas(videoSomas, frame_somas)
        
        frame += 1
    
    # Track somas across frames. Frames missing somas that appear in adjacent
    # frames will have somas interpolated/filled in. 
    videoSomas = fs.interpolate_somas(videoSomas, threshold_images)
        
    # Write each skeleton frame to the output file. Before writing, draw the
    # somas onto the image. Soma pixels have value 254, soma centroids have
    # value 255. 
    frame = 0
    for skeleton in skeletons:
        for soma in videoSomas:
            frameSoma = soma.somaAtFrame(frame)

            if frameSoma is not None:
                skeleton[frameSoma.rows(), frameSoma.cols()] = 254
                skeleton[frameSoma.centroid[0], frameSoma.centroid[1]] = 255
            
        out_tif.write_image(skeleton)
        frame += 1
        
    tif.close()
    out_tif.close()

    # Pickle the soma information
    with open(soma_fname, 'wb') as f:
        pickle.dump(videoSomas, f)
    

"""
Main function, used for debugging or for running a single .tif file by itself.
Preferably the skeletonize function should be used with "AnalyzeAllVideos"
instead for a little more error checking of file names/paths. 
"""
if __name__ == '__main__':
    img_fname  = '/Volumes/Baram Lab/2-photon Imaging/2-01-17_PVN CX3CR1-GFP+CRH-tdTomato P8 CES/video_processing/2-01-17_PVN CX3CR1-GFP+CRH-tdTomato P8 CES_Male 3 R PVN T1_b_4D_Male 3 R PVN T1/preprocessed_max_projection_10iter2.tif' 
    output_fname = 'skeleton.tif'
    soma_fname = 'somas.p'

    start_time = timeit.default_timer()
    
    trace_all_images(img_fname, output_fname, soma_fname, 89)
        
    elapsed = timeit.default_timer() - start_time
    print(elapsed)


