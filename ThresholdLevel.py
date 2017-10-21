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
class RegionObject(object):
    # Defining all the variables ahead of time with __slots__ helps with
    # memory management and makes access quicker
    __slots__ = 'threshold', 'priority', 'coordinates', 'count', 'bbox'
    
    """
    Initialization
        threshold - (int) The threshold at which this region was found
        priority - (int) How high the threshold is. 0 = highest. This will 
                         correspond to block size in the seed point sampling.
        coordinates - (Nx2 ndarray) The pixel coordinates of this region
    """
    def __init__(self, threshold, priority, coordinates):
        self.threshold = threshold
        self.priority  = priority
        self.coordinates = coordinates
        self.count = self.coordinates.shape[0]
        
        rows = self.rows()
        cols = self.cols()
        self.bbox = [rows.min(),cols.min(),rows.max(),cols.max()]
    
    """
    Shortcut for accessing the image row coordinates
    Outputs Nx1 array of row coordinates
    """  
    def rows(self): # TODO make sure this is actually row/col and not x/y
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
    'Less than' function for sorting by priority
    """
    def __lt__(self, other):
         return self.priority < other.priority
    
    """
    This is what will get printed out when using print(region)
    """
    def __repr__(self):
        s = "{Threshold: " + str(self.threshold) + ", Count: " + \
            str(self.count) + "}"
        return s
    

"""
This object is a container for all pixel regions (RegionObjects) at a given
threshold
"""
class LevelObject(object):
    # Defining all the variables ahead of time with __slots__ helps with
    # memory management and makes access quicker
    __slots__ = 'threshold', 'priority', 'regions', 'backgroundLevel', \
                'count', 'regionCount'
    
    """
    Global variables for this object
    """
    MIN_OBJ_SIZE = 10*10 # Objects must be this large to be added as a region
    
    """
    Initialization
        threshold - (int) The threshold of this level
        priority - (int) How high the threshold is. 0 = highest. This will 
                         correspond to block size in the seed point sampling.
    """
    def __init__(self, threshold, priority):
        self.threshold = threshold
        self.priority  = priority
        self.regions = []
        self.backgroundLevel = False
        self.count = 0
        self.regionCount = 0 # TODO this can be deleted if no background pruning
    
    """
    Creates a RegionObject for the set of coordinates and saves it in the list
        coords - (Nx2 ndarray) Coordinates of the region to be added
    """
    def addLabelledRegion(self, coords):    
        self.regions.append(RegionObject(self.threshold, self.priority, coords))
        self.count += coords.shape[0]
        self.regionCount += 1
    
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


# TODO do I need this function anymore?
def prune_background(levels):
    for i in sp.arange(1,len(levels)-1):
        if (levels[i].regionCount <= levels[i-1].regionCount) and \
            (levels[i].regionCount >= levels[i+1].regionCount):
                levels[i].setBackgroundLevel()
                
    if levels[-1].regionCount <= levels[-2].regionCount:
        levels[-1].setBackgroundLevel()
        

"""
Finds objects present at one level's threshold value and above. If a found
object is large enough, it is added to that level's list.

TODO do I want to go straight to getting contours here? Are the region 
coordinates actually necessary?

Input: 
    img - (MxN ndarray) Original image
    threshold - (int) The threshold level. All pixels above this threshold are
                used in the object finding process. 
    priority - (int) How high the threshold is. 0 = highest in the list. 
    
Output:
    level - LevelObject if objects were found, "None" otherwise
"""     
def find_objects_one_level(img, threshold, priority): 
    # Don't even process if there aren't enough pixels in this layer
    if sp.where(img > threshold)[0].size < LevelObject.MIN_OBJ_SIZE:
        return None
    
    # Threshold the image
    bw = sp.zeros_like(img, dtype='uint8')
    bw[img > threshold] = 255
    
    # Find objects that are large enough
    labels = skm.label(bw, background=False, connectivity=1)
    counts, edges = sp.histogram(labels, labels.max()+1)
    
    valid = sp.where((counts > LevelObject.MIN_OBJ_SIZE)) #\
                     #& (counts < LevelObject.MAX_OBJ_SIZE))
    
    # If only the background was labelled, stop here
    if len(valid[0]) == 0:
        return None
    
    level = LevelObject(threshold, priority)
    props = skm.regionprops(labels)
    
    # Record the valid labelled regions in the level object.
    for v in valid[0]:
        if v == 0: # Background would be labeled as 0, ignore it
            continue
        
        # TODO Exclude anything that comes too close to the edges ?
        #bbox = props[v-1].bbox
        #if (min(bbox) < 10) or (bbox[2] > img.shape[0]) or (bbox[3] > img.shape[1]):
        #    continue
        
        level.addLabelledRegion(props[v-1].coords)
       
    return level


"""
For every threshold, find the objects that exist at that threshold to locate
the cell bodies. 

Input: 
    img - (NxM ndarray) Image to process
    thresholds - (1xK ndarray) Thresholds to consider in the process
    
Output: 
    levels - (list) List of LevelObjects
"""
def find_objects_in_levels(img, thresholds):
    levels = []
    
    for k,p in zip(thresholds,range(len(thresholds))):
        level = find_objects_one_level(img, k, p)
        if level is not None:
            levels.append(level)

    return levels

        
"""
Given a set of thresholds, quantize the image into levels. Each level will have
a list of regions of connected pixels in that level. The levels list will be
in sorted order from highest threshold to lowest. 

Input:
    img - (MxN ndarray) The image to process
    thresholds - (1xK ndarray) The thresholds to use
    somas - (list) List of FrameSoma objects (TODO unused currently)
    
Output:
    levels - (list) List of LevelObjects
"""
def get_levels(img, thresholds, somas):    
    # If there's only one threshold, turn it into an array
    if thresholds.size == 1:
        thresholds = sp.array(thresholds, ndmin=1)
    
    # Start at highest threshold and go to lowest
    thresholds[::-1].sort() 
    
    levels = find_objects_in_levels(img, thresholds)
    
    #prune_background(levels)
    
    # TODO I think I want to do this section eventually?
    
    #bw = Utils.plot_levels(levels, height, width, False)
    
    # Subtract somas from the level information
    #for s in somas:
    #    bw[s.rows(),s.cols()] = 0
    
    # Remove any levels that didn't have any valid objects in them 
    levels = [L for L in levels if L.count > 0]
    
    return levels