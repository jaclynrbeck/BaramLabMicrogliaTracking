#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 14:54:31 2017

@author: jaclynbeck
"""

import scipy as sp
import skimage.measure as skm


"""
This object represents a region of connected pixels in an image
"""
class RegionObject:
    """
    Initialization
        threshold - (int) The threshold at which this region was found
        coordinates - (Nx2 ndarray) The pixel coordinates of this region
    """
    def __init__(self, threshold, coordinates):
        self.threshold = threshold
        self.coordinates = coordinates
        self.count = self.coordinates.shape[0]
        
        rows = self.rows()
        cols = self.cols()
        self.bbox = [rows.min(),cols.min(),rows.max(),cols.max()]
    
    """
    Shortcut for accessing the image row coordinates
    """  
    def rows(self): # TODO make sure this is actually row/col and not x/y
        return self.coordinates[:,0]
    
    """
    Shortcut for accessing the image column coordinates 
    """
    def cols(self):
        return self.coordinates[:,1]
    
    """
    'Less than' function for sorting by threshold
    """
    def __lt__(self, other):
         return self.threshold < other.threshold
    
    """
    This is what will get printed out when using print(region)
    """
    def __repr__(self):
        s = "Threshold: " + str(self.threshold) + ", Count: " + self.count 
        return s
    

"""
This object is a container for all pixel regions (RegionObjects) at a given
threshold
"""
class LevelObject:
    
    """
    Global variables for this object
    """
    MIN_OBJ_SIZE = 10*10 # Objects must be this large to be added as a region
    MAX_OBJ_SIZE = 1024*1024*0.1 # A microglia should not fill > 10% of the image
    
    """
    Initialization
        threshold - (int) The threshold of this level
    """
    def __init__(self, threshold):
        self.threshold = threshold
        self.regions = []
        self.backgroundLevel = False
        self.count = 0
        self.regionCount = 0    
    
    """
    Creates a RegionObject for the set of coordinates and saves it in the list
        coords - (Nx2 ndarray) Coordinates of the region to be added
    """
    def addLabelledRegion(self, coords):    
        self.regions.append(RegionObject(self.threshold, coords))
        self.count += coords.shape[0]
        self.regionCount += 1
    
    """
    Shortcut for accessing the image row coordinates of a given region
    Outputs Nx1 ndarray of row coordinates
    """
    def regionRows(self, region):
        return self.regions[region].rows()
    
    """
    Shortcut for accessing the image column coordinates of a given region
    Outputs Nx1 ndarray of column coordinates
    """
    def regionCols(self, region):
        return self.regions[region].cols()
    
    """
    Shortcut for accessing a specific RegionObject in the list
    Outputs a RegionObject
    """
    def regionObject(self, region):
        return self.regions[region]
    
    """
    Labels this object as "background" so it will be ignored in processing
    """
    def setBackgroundLevel(self):
        self.backgroundLevel = True
    
    """
    Returns True if this object has been labelled as "background"
    """    
    def isBackgroundLevel(self):
        return self.backgroundLevel
    
    """
    Shortcut for getting the bounding box of a given region
    Outputs a 1x4 ndarray containing [xmin, ymin, xmax, ymax]
    """
    def getRegionBBox(self, region):
        return self.regions[region].bbox
        
    
    """
    'Less than' function for sorting by threshold
    """
    def __lt__(self, other):
         return self.threshold < other.threshold
        
        
    """
    This is what will get printed out when using print(level) 
    """
    def __repr__(self):
        s = "Threshold:\t\t" + str(int(self.threshold))
        if self.backgroundLevel:
            s += " (Background)"
            
        s += "\n"
        s += "Pixels:\t\t" + str(self.count) + "\n" + \
             "Regions:\t" + str(self.regionCount) + "\n\n"
            
        return s


def prune_background(levels):
    for i in sp.arange(1,len(levels)-1):
        if (levels[i].regionCount() < levels[i-1].regionCount()) and \
            (levels[i].regionCount() >= levels[i+1].regionCount()):
                levels[i].setBackgroundLevel()
                
    if levels[-1].regionCount() < levels[-2].regionCount():
        levels[-1].setBackgroundLevel()
        

        
def find_objects_one_level(bw, img, threshold): 
    if sp.where(img > threshold)[0].size < LevelObject.MIN_OBJ_SIZE:
        return None
    
    bw[img > threshold] = 255
    
    labels = skm.label(bw, background=False, connectivity=1)
    counts, edges = sp.histogram(labels, labels.max()+1)
    
    valid = sp.where((counts > LevelObject.MIN_OBJ_SIZE) \
                     & (counts < LevelObject.MAX_OBJ_SIZE))
    
    # If only the background was labelled, stop here
    if len(valid[0]) == 0:
        return None
    
    level = LevelObject(threshold)
    props = skm.regionprops(labels)
    
    # Record the valid labelled regions in the level object.
    for v in valid[0]:
        if v == 0: # Background would be labeled as 0, ignore it
            continue
        
        # TODO Exclude anything that comes too close to the edges 
        #bbox = props[v-1].bbox
        #if (min(bbox) < 10) or (bbox[2] > img.shape[0]) or (bbox[3] > img.shape[1]):
        #    continue
        
        level.addLabelledRegion(props[v-1].coords)
       
    return level


def find_objects_in_levels(img, thresholds):
    bw = sp.zeros_like(img, dtype='uint8')
    
    levels = []
    
    for k in thresholds:
        level = find_objects_one_level(bw, img, k)
        if level is not None:
            levels.append(level)

    return levels

        

def get_levels(img, thresholds, somas):    
    [height, width] = img.shape
    
    if thresholds.size == 1:
        thresholds = sp.array(thresholds, ndmin=1)
    
    thresholds[::-1].sort() # Start at highest threshold and go to lowest
    
    levels = find_objects_in_levels(img, thresholds)
    
    #prune_background(levels)
    
    # TODO I think I want to do this section eventually?
    
    #bw = Utils.plot_levels(levels, height, width, False)
    
    # Subtract somas from the level information
    #for s in somas:
    #    bw[s.rows(),s.cols()] = 0
        
    #for L, i in zip(levels,range(len(levels))):
    #    coords = sp.where(bw == L.threshold+1)
    #    L.setCoordinates(sp.column_stack((coords[0],coords[1])))
    
    levels = [L for L in levels if L.count > 0]
    
    return levels