#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 13:12:56 2017

@author: jaclynbeck
"""

import scipy as sp
from skimage import filters
import cv2


"""
This class represents a bounding box. This is here for readability so that the
indexing in the bounding box is unambiguous. 
"""
class BoundingBox(object):
    __slots__ = 'row_min', 'row_max', 'col_min', 'col_max'
    
    def __init__(self, row_min, col_min, row_max, col_max):
        self.row_min = row_min
        self.col_min = col_min
        self.row_max = row_max
        self.col_max = col_max
        
    def asArray(self):
        return sp.array([self.row_min, self.col_min, self.row_max, self.col_max])
    
    def width(self):
        return self.col_max - self.col_min + 1
    
    def height(self):
        return self.row_max - self.row_min + 1
    
    def __repr__(self):
        return str(self.asArray())
    
"""
This class represents a soma in a single frame
"""
class FrameSoma(object):
    # Defining all the variables ahead of time with __slots__ helps with
    # memory management and makes access quicker
    __slots__ = 'frameNum', 'coordinates', 'bbox', 'centroid', 'contour'
    
    """
    Global variables for this class
    """
    MIN_SOMA_SIZE = 20*20 # Somas must have at least this many pixels to be
                          # valid -- 100 px = 24 micrometers
    
    """
    Initializes the object.
        frameNum - (int) The frame number this soma appears in (TODO)
        coords - (Nx2 ndarray) All pixel coordinates for this soma
        bbox - (1x4 ndarray, list, or tuple) Bounding box of the soma 
               containing the fields [xmin, ymin, xmax, ymax]
    """
    def __init__(self, frameNum, coords, bbox=None, contour=None):
        self.frameNum = frameNum
        self.coordinates = coords
        
        if bbox is None:
            self.calculateBBox()
        else:    
            self.bbox = bbox
            
        if contour is None:
            self.calculateContour()
        else:
            self.contour = contour
        
        # The centroid is the mean of rows and mean of columns, turned into
        # a (2,) ndarray
        self.centroid = sp.round_( sp.array( (sp.mean(self.coordinates[:,0]), 
                                              sp.mean(self.coordinates[:,1])) 
                                            )).astype('int16')
    
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
    Shortcut for accessing the contour row coordinates
    """
    def contourRows(self):
        return self.contour[:,0]
    
    """
    Shortcut for accessing the contour column coordinates
    """
    def contourCols(self):
        return self.contour[:,1]
    
    """
    Calculates this soma's bounding box
    """
    def calculateBBox(self):
        self.bbox = BoundingBox(min(self.coordinates[:,0]), 
                                min(self.coordinates[:,1]), 
                                max(self.coordinates[:,0]), 
                                max(self.coordinates[:,1]))
    
    """
    Calculates the contour of the soma
    """    
    def calculateContour(self):
        bw = sp.zeros((self.bbox.height(), self.bbox.width()), dtype='uint8')
        bw[self.rows()-self.bbox.row_min, self.cols()-self.bbox.col_min] = 1
        
        contours = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
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
        self.contour = sp.column_stack((contours[:,1]+self.bbox.row_min,
                                        contours[:,0]+self.bbox.col_min))
        
    
    def __copy__(self):
        newSoma = FrameSoma(self.frameNum, self.coordinates.copy(), 
                            contour=self.contour.copy())
        return newSoma
    
    
    """
    This is what will get printed out when using print(frame) 
    """
    def __repr__(self):
        s = "Frame:\t" + str(self.frameNum) + "\n" + \
            "Centroid: [" + str(int(self.centroid[0])) + ", " + \
                            str(int(self.centroid[1])) + "]\n" + \
            "Box:\t" + str(self.bbox)
            
        return s
    

"""
This class represents a single soma across multiple frames in a video, i.e.
a collection of FrameSomas that represent the same object
"""
class VideoSoma(object):
    __slots__ = 'frameSomas', 'frames', 'coordinateSpread'
    
    def __init__(self, frameSoma):
        self.frameSomas = {frameSoma.frameNum: frameSoma}
        self.frames     = [frameSoma.frameNum]
        self.coordinateSpread = set([tuple(C) for C in frameSoma.coordinates])
    
    def addSoma(self, frameSoma):
        self.frameSomas[frameSoma.frameNum] = frameSoma
        self.frames.append(frameSoma.frameNum)
        spread = set([tuple(C) for C in frameSoma.coordinates])
        self.coordinateSpread = set.union(spread, self.coordinateSpread)
        
    def somaAtFrame(self, frame):
        if frame in self.frameSomas:
            return self.frameSomas[frame]
        
        return None

        
    """ 
    Tests to see if soma objects in different frames are actually the same.
    Assumes that between frames, the microglia centroid does not shift more
    than 20 pixels in any direction. 
    """
    def isMatch(self, frameSoma):
        # If they're in the same frame they can't be the same soma
        if frameSoma.frameNum in self.frames:
            return (False, -1)
        
        # If they're centered on the same spot they are the same soma
        dist = self.distanceTo(frameSoma)
        if dist < 20:
            return (True, dist)

        return (False, -1)
    
    
    def distanceTo(self, frameSoma):
        lastFrame = sorted(self.frames)[-1]
        lastSoma = self.frameSomas[lastFrame]
    
        diff = lastSoma.centroid - frameSoma.centroid
        dist = sp.sqrt(diff[0]**2 + diff[1]**2)
        
        return dist
    
    
    def overlaps(self, other):
        intersect = set.intersection(self.coordinateSpread, other.coordinateSpread)
        if len(intersect) == 0:
            return []
        
        overlap_frames = []
        for f in sorted(self.frames):
            if f not in other.frames:
                continue
            
            s1 = self.frameSomas[f]
            s2 = other.frameSomas[f]
            
            coords1 = set([tuple(C) for C in s1.coordinates])
            coords2 = set([tuple(C) for C in s2.coordinates])
            
            intersect = set.intersection(coords1, coords2)
            if len(intersect) > 0:
                overlap_frames.append(f)
    
        return overlap_frames
    
    
    def mergeWith(self, other):
        #self.coordinateSpread = set()
        
        for f in set.union(set(self.frames), set(other.frames)):
            # Only this soma appears in this frame. No action needs to be taken
            if f not in other.frames:
                pass
                #s1 = self.frameSomas[f]
                #self.addSoma(s1)
            
            # Only the other's soma appears in this frame. Use that one
            elif f not in self.frames:
                s2 = other.frameSomas[f]
                self.addSoma(s2)

            # Both somas appear and need to have their coordinates merged
            else:
                s1 = self.frameSomas[f]
                s2 = other.frameSomas[f]
                
                coords1 = set([tuple(C) for C in s1.coordinates])
                coords2 = set([tuple(C) for C in s2.coordinates])
                
                new_coords = sp.array(list(set.union(coords1, coords2)))
                new_soma = FrameSoma(f, new_coords)
    
                #self.addSoma(new_soma)
                self.frameSomas[f] = new_soma
                del s1, s2
        
        # Correct the frames list, which will now have duplicates due to the 
        # "addSoma" function
        #self.frames = list(set(self.frames))
    
    
    def nearestFrame(self, frame):
        if frame in self.frames:
            return self.frameSomas[frame]
        
        dist = abs(sp.array(self.frames) - frame)
        index = sp.argsort(dist)
        
        return self.frameSomas[self.frames[index[0]]]
   
    def __lt__(self, other):
        return len(self.frames) < len(other.frames)
    
    def __repr__(self):
        s = "Frames: " + str(sorted(self.frameSomas.keys())) + "\n" + \
            "Centroid: " + str(self.frameSomas[self.frames[0]].centroid) + "\n"
            
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
    number, labels, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=4)
    
    valid = sp.where(stats[:,4] >= FrameSoma.MIN_SOMA_SIZE)[0]
    
    #labels = skm.label(bw, background=False, connectivity=1)
    #props = skm.regionprops(labels)
    #counts, edges = sp.histogram(labels, labels.max()+1)
    
    somas = []
    
    # Find all the labelled regions with a large enough number of pixels
    #valid = sp.where(counts > FrameSoma.MIN_SOMA_SIZE)
    
    # Create a soma object for each object that is larger than the minimum size
    for v in valid:
        if v == 0: # Ignore the 'background' label, which will always be 0
            continue

        coords = sp.vstack(sp.where(labels == v)).T
        bbox = (min(coords[:,0]), min(coords[:,1]), \
                max(coords[:,0]), max(coords[:,1]))
        #bbox = props[v-1].bbox
        
        # Ignore all somas within 10 px of the edges
        #if (min(bbox) <= 10) or (max(bbox) >= bw.shape[0]-10):
        #    continue
        
        bbox_obj = BoundingBox(bbox[0], bbox[1], bbox[2], bbox[3])
        
        # Create the soma object and add it to the list
        somas.append(FrameSoma(frame, coords, bbox_obj))
    
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

The image is thresholded by using pixels above the Otsu threshold. 
Interconnected pixels are labelled as objects, and those of adequate size are 
returned as valid somas. 

Input:
    img - (MxN ndarray) The image to search
    
Output:
    list containing:
        threshold - (double) The threshold at which all pixels above it are 
                             labelled as soma pixels
        somas - (list) List of FrameSoma objects 
"""
def find_somas_single_image(img, frame):
    threshold = filters.threshold_otsu(img) 
    bw = sp.zeros_like(img, dtype='uint8')  
    bw[img > threshold] = 255
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_DILATE, kernel)

    somas = label_objects(bw, frame)
    return [threshold, somas]


def combine_somas(videoSomas, frameSomas):
    for f in frameSomas:
        matches = []
        for v in videoSomas:
            isMatch, dist = v.isMatch(f)
            if isMatch:
                matches.append((dist, v))
                
        if len(matches) == 0:
            videoSomas.append(VideoSoma(f))
            
        elif len(matches) == 1:
            matches[0][1].addSoma(f)
            
        else:
            matches.sort()
            matches[0][1].addSoma(f)
            
    return videoSomas


def interpolate_somas(videoSomas, threshold_images):
    for v in videoSomas:
        toAdd = []
        for frame in range(len(threshold_images)):
            if frame not in v.frames:
                frameSoma = v.nearestFrame(frame)
                bw = threshold_images[frame]
                
                ind = sp.where(bw[frameSoma.rows(), frameSoma.cols()] > 0)[0]
                coords = sp.vstack((frameSoma.rows()[ind], frameSoma.cols()[ind])).T
                
                if len(coords) > 10: # Filter out noise
                    newSoma = FrameSoma(frame, coords)
                    toAdd.append(newSoma)
        
        for soma in toAdd:
            v.addSoma(soma)
        
    toRemove = []
    for i in range(len(videoSomas)):
        v1 = videoSomas[i]
        for j in sp.arange(i+1, len(videoSomas)):
            v2 = videoSomas[j]
            
            overlap_frames = v1.overlaps(v2)
            if len(overlap_frames) > 0:
                v2.mergeWith(v1)
                toRemove.append(v1)
                break
        
    for v in toRemove:
        videoSomas.remove(v)
        
    return videoSomas
    

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
    
    