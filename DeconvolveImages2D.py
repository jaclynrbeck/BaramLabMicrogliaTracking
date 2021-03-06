#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deconvolves every image in a fluorescence image stack and registers the images
to correct for drift.

Created on Wed Oct 18 14:07:23 2017

@author: jaclynbeck
"""

from libtiff import TIFF
import scipy as sp
import numpy as np
import cv2
import timeit
import javabridge # required by bioformats
import bioformats # installed with pip install python-bioformats
from skimage import restoration
import os
import pickle
from Objects import ImageMetadata


"""
Given an image and an affine transformation matrix, apply the affine transform
to the image.

Input: 
    img   - (MxN ndarray), An image to transform
    drift - (2x3 ndarray), The affine transformation matrix
    
Output:
    img2 - The transformed image
"""
def warp_image_affine(img, drift):
    img2 = cv2.warpAffine(img, drift, img.shape)
    
    return img2


"""
Registers two images to each other using OpenCV's ECC function.

Input:
    img1 - (MxN ndarray), The first image
    img2 - (MxN ndarray), The second image, to be registered to the first image
    
Output:
    warp_matrix - (2x3 ndarray), The affine transformation matrix
"""
def register_affine(img1, img2):
    (cc, warp_matrix) = cv2.findTransformECC(img2, img1, None, 
                                             cv2.MOTION_AFFINE)
    return warp_matrix


"""
Deconvolves and registers all images in the image stack. All output will be
put in the same directory as the .ims file under the subdirectory
"video_processing/<img_fname without .ims>/"

Input:
    img_fname       - Full file path to the .ims file
    output_fname    - Local file name to save the deconvolved image stack to.
                       * Must end in .tif
    psf_fname       - Full file path to the PSF file that describes the point
                      spread function of the microscope. Must be a .png. 
    metadata_fname  - Local file name to save the image metadata to. 
                        * Must end in .p
    window_size     - If averaging several deconvolved frames together, this is
                      how many frames to average together. Set to 1 if no 
                      averaging is desired. 1 or 3 is recommended. 
    deconvolution_iters - How many iterations of Richardson-Lucy deconvolution
                          to perform. Start with 10, and raise it to 20 or 50
                          if the resulting image is still too noisy. 
    
Output:
    (no output), however several files are created through this process:
        deconvolved image stack
        metadata pickle file holding the image metadata
"""
def deconvolve_images_2D(img_fname, output_fname, psf_fname, metadata_fname, 
                         window_size, deconvolution_iters):
    
    # Create the subdirectory "video_processing/<img_fname>"
    path = os.path.dirname(img_fname)
    path = os.path.join(path, "video_processing", os.path.basename(img_fname)[0:-4])
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Output and metadata will go in this subdirectory
    output_fname = os.path.join(path, output_fname)
    metadata_fname = os.path.join(path, metadata_fname)
    
    # This is needed for the bioformats library
    javabridge.start_vm(class_path=bioformats.JARS)
    
    # Open the output file and the PSF file
    averaged_tif = TIFF.open(output_fname, mode='w')
    psf = cv2.imread(psf_fname, cv2.IMREAD_GRAYSCALE)
    psf = psf / np.sum(psf) # Normalize so the sum of all pixels is 1

    # Process the metadata
    xml = bioformats.get_omexml_metadata(path=img_fname)
    ome_xml = bioformats.OMEXML(xml)
    
    meta = ImageMetadata(ome_xml)
    
    # Print some general information about the file
    print(img_fname)        
    print("File contains " + str(meta.numImages) + " images. Size: " + \
          str(meta.sizeX) + " x " + str(meta.sizeY) + " px (" + \
          str(meta.physX*meta.sizeX) + " x " + \
          str(meta.physY*meta.sizeY) +  " um)")
    
    minutes = np.mean(sp.array(meta.timeDeltas)) / 60.0
    truncate = np.ceil(30/minutes) # number of frames that equals 30 mins
    if meta.numImages > truncate:
        meta.numImages = int(truncate)
        print("Truncating to " + str(meta.numImages) + " images (30 minutes)")
    
    window = []
    img0 = None

    numImages = meta.numImages
    
    # Read each image, deconvolve it, and register it with the previous image
    with bioformats.ImageReader(img_fname) as reader:
        for t in range(numImages):
            print("Processing image " + str(t) + "...")
            
            # Read channel 1 (green), z 0, frame t
            img = reader.read(c=1, z=0, t=t, series=None, index=None, 
                              rescale=True, wants_max_intensity=False, 
                              channel_names=None, XYWH=None)
            
            # Deconvolve and set to the range of a uint16, but keep it as a
            # float for now
            deconvolved = restoration.richardson_lucy(img, psf, 
                                                iterations=deconvolution_iters)
            deconvolved = deconvolved.astype('float32')
            
            # Register the image with the previous image, but we use the 
            # original (non-deconvolved) images to find the registration matrix
            # because it produces more reliable registration. 
            if img0 is not None:
                try:
                    drift = register_affine(img0, img)
                except cv2.error as e:
                    print("Error registering frame " + str(t))
                    break
    
                # Warp the images (both deconvolved and original)
                warped_img_deconvolved = warp_image_affine(deconvolved, drift)
                warped_img_orig = warp_image_affine(img, drift)
            else:
                warped_img_deconvolved = deconvolved
                warped_img_orig = img
            
            # Save the warped deconvolved image for output
            window.append(warped_img_deconvolved)
            
            # Keep the warped original image for the next round of registration
            img0 = warped_img_orig
            
            # If averaging, wait until we have <window_size> frames before 
            # writing the averaged output. If not averaging, the code will 
            # go directly to writing output. 
            if len(window) < window_size:
                continue
            
            # Average all images in the window. (If not averaging, this step
            # does nothing.)
            averaged = sum(window) / window_size
            
            # Re-adjust the range of the pixels to the range of a uint16 and
            # center the median at the same place in each frame
            averaged = np.minimum(averaged * (400.0 / np.median(averaged)), 65535.0)
            
            # Write this as a little endian uint16 image
            averaged_tif.write_image(averaged.astype('<u2')) 
        
            window.clear() 
            
        reader.close()

    averaged_tif.close()
    
    # Pickle the image metadata to a file for later use
    with open(metadata_fname, 'wb') as f:
        pickle.dump(meta, f)
    
    # Technically this should be uncommented but it interferes with the ability
    # to run multiple calls of this deconvolve function in a row so it is
    # commented out. 
    #javabridge.kill_vm() 


"""
Main function, used for debugging or for running a single .ims file by itself.
Preferably the deconvolve function should be used with "AnalyzeAllVideos"
instead for a little more error checking of file names/paths. 
"""
if __name__=='__main__':
    img_fname    = '/Users/jaclynbeck/Desktop/BaramLab/videos/A_LPVN_T1_08202017/8-20-17_crh-tdtomato+cx3cr1-gfp p8 pvn ces_male 1 l pvn t1_b_4d.ims' 
    output_fname = 'preprocessed_max_projection_10iter.tif'
    metadata_fname = 'img_metadata.p'
    psf_fname    = '/Users/jaclynbeck/Desktop/BaramLab/PSF_GL_squared_920nm_single.png'
    start_time = timeit.default_timer()

    deconvolve_images_2D(img_fname, output_fname, psf_fname, metadata_fname, 1, 10)
            
    elapsed = timeit.default_timer() - start_time
    print(elapsed)

    
    