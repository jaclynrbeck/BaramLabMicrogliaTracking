#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 13:12:56 2017

@author: jaclynbeck
"""

import scipy as sp
import skimage.measure as skm
from libtiff import TIFFfile
import timeit
import Utils
import cv2

"""
This class represents a soma in a single frame
"""
class FrameSoma(object):
    # Defining all the variables ahead of time with __slots__ helps with
    # memory management and makes access quicker
    __slots__ = 'frameNum', 'coordinates', 'bbox', 'centroid'
    
    """
    Global variables for this class
    """
    MIN_SOMA_SIZE = 10*10 # Somas must have at least this many pixels to be
                          # valid -- 100 px = 24 micrometers
    
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
class Soma3D(object):
    # Defining all the variables ahead of time with __slots__ helps with
    # memory management and makes access quicker
    __slots__ = 'id', 'frames', 'frameNums', 'coordinates'
    
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
    labels = skm.label(bw, background=False, connectivity=1)
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
Given a set of coordinates and their bounding box, this method finds the 
contour around those points and returns the coordinates of the contour points.

Input:
    coordinates - (list of 2 1xM ndarrays) The coordinates of all pixels in 
                    this region. 0 is rows, 1 is columns
    bbox - (1x4 ndarray or list) Bounding box with [xmin, ymin, xmax, ymax]
    
Output:
    rc - (Nx2 ndarray) The coordinates of the pixels on the region contour
"""
def get_contour(img):  
    # contours is a tuple, contours[1] contains the coordinates needed
    contours = cv2.findContours(img.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours[1] if c.shape[0] > 1]

    # contours is a list of M contours, where each element is an Nx2 array of
    # coordinates belonging to each contour. This code flattens the array
    # into (M*N)x2 and removes duplicate points. 
    contours = [sp.reshape(c,(c.shape[0]*c.shape[1], c.shape[2])) for c in contours]
    contours = sp.concatenate(contours)
    contours = sp.array(list(set(tuple(p) for p in contours))) # Remove duplicate points
    
    # contours are in terms of x,y instead of row,col, so the coordinates need
    # to be reversed. This also undoes the coordinate adjustment done at the
    # beginning of this function to account for using only the bounding box
    rc = sp.column_stack((contours[:,1]+bbox[0],contours[:,0]+bbox[1]))
    
    return rc


"""
Finds all the somas in a single max projection image. 

The image is thresholded by finding the top 1% of pixels and using those
pixels as potential somas. Interconnected pixels are labelled as objects, and 
those of adequate size are returned as valid somas. 

Input:
    img - (MxN ndarray) The image to search
    
Output:
    list containing:
        threshold - (double) The threshold at which all pixels above it are 
                             labelled as soma pixels
        somas - (list) List of FrameSoma objects 
"""
def find_somas_single_image(img):
    threshold = sp.percentile(img, 99)
    
    bw = sp.zeros_like(img, dtype='uint8')
    bw[img > threshold] = 255
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_DILATE, kernel)

    somas = label_objects(bw, 0)
    return [threshold, somas]
    

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
    threshold = sp.percentile(images, 99)
    
    somas = []
    
    # For each image, threshold it and label its somas
    for z in range(images.shape[0]):
        img = images[z]
        
        contours = get_contour(img)
    
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
    img_fname = '/Users/jaclynbeck/Desktop/BaramLab/C2-8-29-17_CRH-tdTomato+CX3CR1-GFP P8 PVN CES_Female 1 L PVN_a_.i...CX3CR1-GFP P8 PVN CES_Female 1 L PVN_a_GREEN_t1.tif' #'/Users/jaclynbeck/Desktop/BaramLab/C2-8-29-17_CRH-tdTomato+CX3CR1-GFP P8 PVN CES_Female 1 L PVN_a_.i...CX3CR1-GFP P8 PVN CES_Female 1 L PVN_a_GREEN_DENOISED_t1.tif'
    output_img = '/Users/jaclynbeck/Desktop/BaramLab/somas.tif'
    
    tiff = TIFFfile(img_fname)
    samples, sample_names = tiff.get_samples()

    start_time = timeit.default_timer()
    
    for i in range(samples[0].shape[0]):
        samples[0][i] = Utils.preprocess_img(samples[0][i])
    
    #[threshold, somas] = find_somas_single_image(samples[0])
    [threshold, somas] = find_somas_3D(samples[0])
    
    elapsed = timeit.default_timer() - start_time
    print(elapsed)
    
    # For debugging and display
    s_img = Utils.plot_somas(somas, display=True)
    sp.misc.imsave(output_img, s_img)
    