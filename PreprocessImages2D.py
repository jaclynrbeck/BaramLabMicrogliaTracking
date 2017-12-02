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
import os
import datetime
import pickle

class ImageMetadata(object):
    __slots__ = 'sizeX', 'sizeY', 'numImages', 'physX', 'physY', \
                'imgTimes', 'timeDeltas'
                
    def __init__(self, ome_xml):
        # This is kind of dirty but it fixes an issue getting annotations since
        # this library is behind by a few years.
        bioformats.omexml.NS_ORIGINAL_METADATA = "http://www.openmicroscopy.org/Schemas/OME/2016-06"
        ome_xml.ns['sa'] = ome_xml.ns['ome'] 
    
        self.sizeX = ome_xml.image().Pixels.SizeX
        self.sizeY = ome_xml.image().Pixels.SizeY
        self.numImages = ome_xml.image().Pixels.SizeT
        self.physX = ome_xml.image().Pixels.PhysicalSizeX
        self.physY = ome_xml.image().Pixels.PhysicalSizeY
        
        annotations = ome_xml.structured_annotations
        annotations.ns['sa'] = annotations.ns['ome'] # Same issue fix
    
        self.timeDeltas = []
        self.imgTimes = {}
        last_dt = None
        for i in range(self.numImages):
            if annotations.has_original_metadata("TimePoint"+str(i)):
                time = annotations.get_original_metadata_value("TimePoint"+str(i))
                dt = datetime.datetime.strptime(time[:-4], '%Y-%m-%d %H:%M:%S')
                dt = dt.replace(microsecond=int(time[-3:])*1000)
                self.imgTimes[i] = dt
                
                if last_dt is not None:
                    diff = dt - last_dt
                    self.timeDeltas.append(diff.total_seconds())
                
                last_dt = dt
            else:
                last_dt = None


def warp_image_affine(img, drift):
    img2 = cv2.warpAffine(img, drift, img.shape)
    
    return img2


def register_affine(img1, img2):
    (cc, warp_matrix) = cv2.findTransformECC(img2, img1, None, cv2.MOTION_AFFINE)
    return warp_matrix


def preprocess_images_2D(img_fname, output_fname, psf_fname, metadata_fname, 
                         window_size, deconvolution_iters):
    path = os.path.dirname(img_fname)
    path += "/video_processing/" + os.path.basename(img_fname)[0:-4] 
    if not os.path.exists(path):
        os.makedirs(path)
    
    output_fname = path + "/" + output_fname
    metadata_fname = path + "/" + metadata_fname
    
    javabridge.start_vm(class_path=bioformats.JARS)
    averaged_tif = TIFF.open(output_fname, mode='w')
    psf = cv2.imread(psf_fname, cv2.IMREAD_GRAYSCALE)
    psf = psf / sp.sum(psf)

    xml = bioformats.get_omexml_metadata(path=img_fname)
    ome_xml = bioformats.OMEXML(xml)
    
    meta = ImageMetadata(ome_xml)
    
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
    
    with bioformats.ImageReader(img_fname) as reader:
        for t in range(numImages):
            print("Processing image " + str(t) + "...")
            img = reader.read(c=1, z=0, t=t, series=None, index=None, rescale=True, wants_max_intensity=False, channel_names=None, XYWH=None)
            deconvolved = restoration.richardson_lucy(img, psf, iterations=deconvolution_iters)
            deconvolved = (deconvolved*65535).astype('float32')
            
            if img0 is not None:
                try:
                    drift = register_affine(img0, img)
                except cv2.error as e:
                    print("Error registering frame " + str(t))
                    break
    
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
            
            averaged = averaged * (65535.0 / averaged.max())
                
            averaged_tif.write_image(averaged.astype('<u2')) 
        
            window.clear() 
            
        reader.close()

    averaged_tif.close()
    
    with open(metadata_fname, 'wb') as f:
        pickle.dump(meta, f)
    
    #javabridge.kill_vm()



if __name__=='__main__':
    img_fname    = '/Volumes/Baram Lab/2-photon Imaging/2-01-17_PVN CX3CR1-GFP+CRH-tdTomato P8 CES/2-01-17_PVN CX3CR1-GFP+CRH-tdTomato P8 CES_Male 3 R PVN T1_b_4D_Male 3 R PVN T1.ims' 
    output_fname = 'preprocessed_max_projection_10iter2.tif'
    metadata_fname = 'img_metadata2.p'
    psf_fname    = '/Volumes/Baram Lab/2-photon Imaging/PSF_GL_squared_890nm_single.png'
    start_time = timeit.default_timer()

    preprocess_images_2D(img_fname, output_fname, psf_fname, metadata_fname, 1, 10)
            
    elapsed = timeit.default_timer() - start_time
    print(elapsed)

    
    