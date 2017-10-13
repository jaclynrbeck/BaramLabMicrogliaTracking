#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 13:12:56 2017

@author: jaclynbeck
"""

import scipy as sp
import skimage.measure as skm
import skimage.filters as skf
from libtiff import TIFFfile
import timeit
import Utils

"""
This class represents a soma in a single frame
"""
class FrameSoma:
    """
    Global variables for this class
    """
    MIN_SOMA_RADIUS = 10.0# Somas must have at least this radius to be valid
    MIN_SOMA_SIZE = MIN_SOMA_RADIUS*MIN_SOMA_RADIUS # Somas must have at least
                                                    # this many pixels to be
                                                    # valid
    
    """
    Initializes the object.
        frameNum - (int) The frame number this soma appears in (TODO)
        coords - (Nx2 ndarray) All pixel coordinates for this soma
        bbox - (1x4 ndarray, list, or tuple) Bounding box of the soma 
               containing the fields [xmin, ymin, xmax, ymax]
    """
    def __init__(self, frameNum, coords, bbox):
        self.frameNum = frameNum
        self.coordinates = coords
        self.bbox = sp.array(bbox)
        
        # The centroid is the mean of rows and mean of columns, turned into
        # a 1x2 ndarray
        self.centroid = sp.round_( sp.array( (sp.mean(self.coordinates[:,0]), 
                                              sp.mean(self.coordinates[:,1])), 
                                              ndmin=2 ) )
    
    """
    Shortcut for accessing the image row coordinates
    """
    def rows(self):
        return self.coordinates[:,0]
    
    """
    Shortcut for accessing the image column coordinates 
    """
    def cols(self):
        return self.coordinates[:,1]
    
    """
    This is what will get printed out when using print(frame) 
    """
    def __repr__(self):
        s = "Frame:\t" + str(self.frameNum) + "\n" + \
            "Centroid: [" + str(int(self.centroid[0,0])) + ", " + \
                            str(int(self.centroid[0,1])) + "]\n" + \
            "Box:\t" + str(self.bbox)
            
        return s


"""
This class represents a soma as a 3D object across multiple frames  
"""
class Soma3D:
    def __init__(self, idNum, frame):    
        self.id = idNum
        self.frames = [frame]
        self.frameNums = [frame.frameNum]
        self.coordinates = frame.coordinates
        

    def addFrame(self,frame):
        self.frames.append(frame)
        self.frameNums.append(frame.frameNum)
        self.coordinates = sp.concatenate((self.coordinates,frame.coordinates))
    
    def getId(self):
        return self.id
    
    def rows(self):
        return self.coordinates[:,0]
    
    def cols(self):
        return self.coordinates[:,1]
    
    
    """ 
    Tests to see if soma objects in different frames are actually the same.
    Assumes that between Z-levels, the microglia centroid does not shift more
    than 20 pixels in any direction. 
    """
    def isMatch(self, frame):
        # If they're in the same frame they can't be the same soma
        if frame.frameNum in self.frameNums:
            return False
        
        # If they're centered on the same spot they are the same soma
        for f in self.frames:
            if sp.all(f.centroid == frame.centroid):
                return True
            
            diff = f.centroid - frame.centroid
            if sp.sqrt(diff[0]**2 + diff[1]**2) < FrameSoma.MIN_SOMA_RADIUS:
                return True

        return False
 
    
    """
    This is what will get printed out when using print(soma) or during 
    debugging. 
    """
    def __repr__(self):
        s = "ID:\t" + str(self.id) + "\n" + \
            "Frames:\t" + str(self.frameNums) + "\n" + \
            "Centroid: " + str(self.frames[0].centroid)
            
        return s
    
    
# TODO not used yet. For 3D
def match_somas(somas):
    somas_final = []
    
    idNum = 0
    
    for soma in somas:
        match = False; 
        for f in somas_final:
            if f.isMatch(soma):
                match = True
                f.addFrame(soma)
                break
                
        if match == False:
            somas_final.append(Soma3D(idNum, soma))
            idNum += 1
            
    return somas_final
        

"""
Finds all interconnected pixels in a frame and labels them as an object. If the
object is large enough, it is marked as a soma and added to the list.

Input:
    bw - (MxN ndarray) Thresholded image with values of 0 or 255
    frame - (int) The frame number (for soma labelling purposes)
    
Output:
    somas - (list) List of FrameSoma objects
"""
def label_objects(bw, frame):
    # Find all interconnected pixel regions
    labels = skm.label(bw, background=False, connectivity=2) # TODO can opencv do this faster?
    props = skm.regionprops(labels)
    counts, edges = sp.histogram(labels, labels.max()+1)
    
    somas = []
    
    # Find all the labelled regions with a large enough number of pixels
    valid = sp.where(counts > FrameSoma.MIN_SOMA_SIZE)
    
    # Create a soma object for each object that is larger than the minimum size
    for v in valid[0]:
        if v == 0: # Ignore the 'background' label, which will always be 0
            continue

        bbox = props[v-1].bbox
        
        # Ignore all somas within 10 px of the edges
        if (min(bbox) <= 10) or (max(bbox) >= bw.shape[0]-10):
            continue
        
        # Create the soma object and add it to the list
        somas.append(FrameSoma(frame, props[v-1].coords, bbox))
    
    return somas


"""
Finds all the somas in a single max projection image. 

The image is thresholded using Otsu's method, and all pixels above the 
threshold are likely to be somas. Interconnected pixels are labelled as 
objects, and those of adequate size are returned as valid somas. 

Input:
    img - (MxN ndarray) The image to search
    
Output:
    list containing:
        threshold - (double) The threshold at which all pixels above it are 
                             labelled as soma pixels
        somas - (list) List of FrameSoma objects 
"""
def find_somas_single_image(img):
    threshold = skf.threshold_otsu(img[img > 0], img.max()+1)
    
    bw = sp.zeros_like(img, dtype='uint8')
    bw[img > threshold] = 255

    return [threshold, label_objects(bw, 0)]
    

"""
Finds all the somas in a z-stack of images and correlates their positions in
three dimensions. 

A single threshold, calculated by Otsu's method on a histogram containing
counts from all of the images, is used for all the images. 

Input:
    images - (ZxMxN ndarray) Z-stack of images. "Z" is the number of images, 
             "M" is the height (rows), and "N" is the width (columns).
    
Output:
    list containing:
        threshold - (double) The threshold at which all pixels above it are 
                             labelled as soma pixels
        somas_final - (list) List of Soma3D objects 
"""
def find_somas_3D(images):
    threshold = skf.threshold_otsu(images[images > 0], images.max()+1)
    
    somas = []
    
    # For each image, threshold it and label its somas
    for z in range(images.shape[0]):
        img = images[z]
    
        bw = sp.zeros_like(img, dtype='uint8')
        bw[img > threshold] = 255

        somas += label_objects(bw, z)
        
    # Match the somas from each image to find the 3D structure
    somas_final = match_somas(somas)        
    return [threshold, somas_final]
    


"""
Main method for debugging.
"""
if __name__ == '__main__':
    img_fname = '/Users/jaclynbeck/Desktop/BaramLab/Substack (1).tif' #'/Users/jaclynbeck/Desktop/BaramLab/C2-8-29-17_CRH-tdTomato+CX3CR1-GFP P8 PVN CES_Female 1 L PVN_a_.i...CX3CR1-GFP P8 PVN CES_Female 1 L PVN_a_GREEN_DENOISED_t1.tif'
    output_img = '/Users/jaclynbeck/Desktop/BaramLab/somas.tif'
    
    tiff = TIFFfile(img_fname)
    samples, sample_names = tiff.get_samples()

    start_time = timeit.default_timer()
    
    [threshold, somas] = find_somas_single_image(samples[0])
    
    elapsed = timeit.default_timer() - start_time
    print(elapsed)
    
    # For debugging and display
    s_img = Utils.plot_somas(somas, display=True)
    sp.misc.imsave(output_img, s_img)
    