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
import cv2
import timeit
import javabridge # required by bioformats
import bioformats # installed with pip install python-bioformats
from skimage import restoration
import os
import datetime
import pickle


"""
This class stores metadata from the image in a more human-readable format.
Metadata comes from the OME XML data in the .ims file. This is intended for
use with max projection images only and is not for z-stacks. 

Data stored in this class:
    sizeX:      Width of the images in pixels
    sizeY:      Height of the images in pixels
    numImages:  Number of frames in the stack
    physX:      Physical width of each pixel in microns
    physY:      Physical height of each pixel in microns
    imgTimes:   List of datetime timestamps corresponding to each frame
    timeDeltas: List of differences in time between frames. timeDeltas[0] is
                the difference between frame 0 and 1, timeDeltas[1] is between
                frame 1 and 2, etc. 
"""
class ImageMetadata(object):
    __slots__ = 'sizeX', 'sizeY', 'numImages', 'physX', 'physY', \
                'imgTimes', 'timeDeltas'
    
    """
    Initialization. The bioformats library uses the schema from 2013 but the
    .ims files use the schema from 2016 so there are several fixes in this 
    function to account for the difference. 
    
    Input: 
        ome_xml: a bioformats.OMEXML object created from the .ims file metadata
    """           
    def __init__(self, ome_xml):
        # This is kind of dirty but it fixes an issue getting annotations since
        # this library is behind by a few years.
        bioformats.omexml.NS_ORIGINAL_METADATA = \
                            "http://www.openmicroscopy.org/Schemas/OME/2016-06"
        ome_xml.ns['sa'] = ome_xml.ns['ome'] 
    
        # Extract the basic pixel information
        self.sizeX = ome_xml.image().Pixels.SizeX
        self.sizeY = ome_xml.image().Pixels.SizeY
        self.numImages = ome_xml.image().Pixels.SizeT
        self.physX = ome_xml.image().Pixels.PhysicalSizeX
        self.physY = ome_xml.image().Pixels.PhysicalSizeY
        
        # Use the annotations to get the time stamps / time deltas
        annotations = ome_xml.structured_annotations
        annotations.ns['sa'] = annotations.ns['ome'] # Same issue fix
    
        self.timeDeltas = []
        self.imgTimes = {}
        last_dt = None
        for i in range(self.numImages):
            # There are fields called "TimePoint<#>", i.e. "TimePoint10" for 
            # frame 10. Get those fields. 
            tp = "TimePoint"+str(i)
            if annotations.has_original_metadata(tp):
                time = annotations.get_original_metadata_value(tp)
                
                # Turn the text into a datetime object. It does not properly
                # ingest microseconds so there is an extra step to add the
                # microseconds to the datetime object. 
                dt = datetime.datetime.strptime(time[:-4], '%Y-%m-%d %H:%M:%S')
                dt = dt.replace(microsecond=int(time[-3:])*1000)
                self.imgTimes[i] = dt
                
                # Calculate the time delta between this frame and the last one
                if last_dt is not None:
                    diff = dt - last_dt
                    self.timeDeltas.append(diff.total_seconds())
                
                last_dt = dt
            else:
                last_dt = None


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
    path += "/video_processing/" + os.path.basename(img_fname)[0:-4]
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Output and metadata will go in this subdirectory
    output_fname = path + "/" + output_fname
    metadata_fname = path + "/" + metadata_fname
    
    # This is needed for the bioformats library
    javabridge.start_vm(class_path=bioformats.JARS)
    
    # Open the output file and the PSF file
    averaged_tif = TIFF.open(output_fname, mode='w')
    psf = cv2.imread(psf_fname, cv2.IMREAD_GRAYSCALE)
    psf = psf / sp.sum(psf) # Normalize so the sum of all pixels is 1

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
    
    minutes = sp.mean(sp.array(meta.timeDeltas)) / 60.0
    truncate = sp.ceil(30/minutes) # number of frames that equals 30 mins
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
            
            # Re-adjust the range of the pixels to the range of a uint16
            averaged = averaged * (65535.0 / averaged.max())
            
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
    #commented out. 
    #javabridge.kill_vm() 


"""
Main function, used for debugging or for running a single .ims file by itself.
Preferably the deconvolve function should be used with "AnalyzeAllVideos"
instead for a little more error checking of file names/paths. 
"""
if __name__=='__main__':
    img_fname    = '/Volumes/Baram Lab/2-photon Imaging/7-30-17_CRH-tdTomato+CX3CR1-GFP P8 PVN Ctrl/7-30-17_CRH-tdTomato+CX3CR1-GFP P8 PVN Ctrl_Male 1 R PVN T1_b_4D.ims' 
    output_fname = 'preprocessed_max_projection_10iter.tif'
    metadata_fname = 'img_metadata.p'
    psf_fname    = '/Volumes/Baram Lab/2-photon Imaging/PSF_GL_squared_920nm_single.png'
    start_time = timeit.default_timer()

    deconvolve_images_2D(img_fname, output_fname, psf_fname, metadata_fname, 1, 10)
            
    elapsed = timeit.default_timer() - start_time
    print(elapsed)

    
    