#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 20:54:49 2017

@author: jaclynbeck
"""

import scipy as sp
from libtiff import TIFFfile
import FindSomas as fs
import timeit
import Utils
import skimage.measure as skmeas
import skimage.morphology as skmorph
import matplotlib.pyplot as plt


"""
This object represents a region of connected pixels in an image
"""
class RegionObject(object):
    # Defining all the variables ahead of time with __slots__ helps with
    # memory management and makes access quicker
    __slots__ = 'coordinates', 'skeleton', 'count', 'bbox'
    
    """
    Global variables for this object
    """
    MIN_OBJ_SIZE = 10*10 # Objects must be this large to be added as a region
    
    """
    Initialization
        threshold - (int) The threshold at which this region was found
        priority - (int) How high the threshold is. 0 = highest. This will 
                         correspond to block size in the seed point sampling.
        coordinates - (Nx2 ndarray) The pixel coordinates of this region
    """
    def __init__(self, coordinates):
        self.coordinates = coordinates
        self.count = self.coordinates.shape[0]
        
        rows = self.rows()
        cols = self.cols()
        self.bbox = [rows.min(),cols.min(),rows.max(),cols.max()]
        
        self.skeleton = self.skeletonize()
    
    """
    Shortcut for accessing the image row coordinates
    Outputs Nx1 array of row coordinates
    """  
    def rows(self): 
        return self.coordinates[:,0]
    
    """
    Shortcut for accessing the image column coordinates 
    Outputs Nx1 array of column coordinates
    """
    def cols(self):
        return self.coordinates[:,1]
    
    """
    Shortcut for calculating the centroid of the region 
    Outputs 1x2 array of coordinates
    """
    def centroid(self):
        return sp.round_(sp.reshape(sp.mean(self.coordinates, axis=0), (1,2)))
    
    """
    Shortcut for accessing the skeleton row coordinates
    Outputs Nx1 array of row coordinates
    """
    def skeletonRows(self):
        return self.skeleton[:,0]
    
    """
    Shortcut for accessing the skeleton column coordinates 
    Outputs Nx1 array of column coordinates
    """
    def skeletonCols(self):
        return self.skeleton[:,1]
    
    """
    Skeletonizes this region so the midline of each process becomes a line.
    """
    def skeletonize(self):
        size = [self.bbox[2]-self.bbox[0]+1, self.bbox[3]-self.bbox[1]+1]
        bw = sp.zeros(size, dtype='uint8')
        
        bw[self.rows()-self.bbox[0], self.cols()-self.bbox[1]] = 1
        skeleton = skmorph.skeletonize(bw)
        
        rows, cols = sp.where(skeleton > 0)
        return sp.vstack((rows+self.bbox[0], cols+self.bbox[1])).T
    
    
    """
    This is what will get printed out when using print(region)
    """
    def __repr__(self):
        s = "{Centroid: " + str(self.centroid()) + ", Count: " + \
            str(self.count) + "}"
        return s
    

def find_cell_regions(img, somas):     
    # Threshold the image
    bw = sp.zeros_like(img, dtype='uint8')
    bw[img > 0] = 255
    
    # Remove the somas so that only processes are found
    for soma in somas:
        bw[soma.rows(), soma.cols()] = 0
    
    # Find objects that are large enough
    labels = skmeas.label(bw, background=False, connectivity=1)
    counts, edges = sp.histogram(labels, labels.max()+1)
    
    valid = sp.where((counts > RegionObject.MIN_OBJ_SIZE))
    
    # If only the background was labelled, stop here
    if len(valid[0]) == 0:
        return None

    props = skmeas.regionprops(labels)
    
    regions = []
    
    # Record the valid labelled regions in the level object.
    for v in valid[0]:
        if v == 0: # Background would be labeled as 0, ignore it
            continue
        
        # TODO Exclude anything that comes too close to the edges ?
        #bbox = props[v-1].bbox
        #if (min(bbox) < 10) or (bbox[2] > (img.shape[0]-10)) \
        #    or (bbox[3] > (img.shape[1]-10)):
        #    continue
        
        regions.append(RegionObject(props[v-1].coords))
       
    return regions



def trace_skeleton(regions, somas):
    bw = sp.zeros((1024,1024))
    
    for region in regions:
        bw[region.skeleton[:,0], region.skeleton[:,1]] = 1
        
    #bw = Utils.plot_cell_regions(regions, images.shape, False)
    
    #skeleton = skmorph.skeletonize(bw)
    
    #rows, cols = sp.where(skeleton > 0)
    #coordinates = sp.vstack((rows, cols)).T
    return bw


"""
Main function for debugging
"""
if __name__ == '__main__':
    img_fname = '/Users/jaclynbeck/Desktop/BaramLab/Substack (8).tif' #'/Users/jaclynbeck/Desktop/BaramLab/C2-8-29-17_CRH-tdTomato+CX3CR1-GFP P8 PVN CES_Female 1 L PVN_a_.i...CX3CR1-GFP P8 PVN CES_Female 1 L PVN_a_GREEN_t1.tif'
    output_dir = '/Users/jaclynbeck/Desktop/BaramLab/'
    peak_threshold = 100
    is_stack = False
    slice_of_interest = 8 # For debugging. This slice's objects will be shown as a b/w image
    
    if is_stack == False:
        slice_of_interest = 0
    
    tiff = TIFFfile(img_fname)
    samples, sample_names = tiff.get_samples()
    tiff.close()
    
    if is_stack:
        images = samples[0]
    else:
        images = samples[0][0]

    start_time = timeit.default_timer()
    
    images = Utils.preprocess_img(images) 
    
    #image_stack = sp.reshape(images, (1,images.shape[0], images.shape[1]))
    [soma_threshold, somas] = fs.find_somas_single_image(images)
    
    regions = find_cell_regions(images, somas)
    #skeleton = trace_skeleton(regions, somas)
    
    elapsed = timeit.default_timer() - start_time
    print(elapsed)
    
    # For debugging and display
    
    s_img = Utils.plot_somas(somas, (1024,1024), True)
    sp.misc.imsave(output_dir + 'somas.tif', s_img)
    
    bw = Utils.plot_cell_regions(regions, somas, (1024, 1024), True)
    sp.misc.imsave(output_dir + 'regions.tif', bw)
    
    skeleton = Utils.plot_skeleton(regions, somas, (1024,1024), True)
    sp.misc.imsave(output_dir + 'skeleton.tif', skeleton)
