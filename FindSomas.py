#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains several object definitions and functions related to finding and 
tracking microglia somas across an image series. 

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
        return sp.array([self.row_min, self.col_min, self.row_max, 
                         self.col_max])
    
    def width(self):
        return self.col_max - self.col_min + 1
    
    def height(self):
        return self.row_max - self.row_min + 1
    
    def __repr__(self):
        return str(self.asArray())
    
    
"""
This class represents a soma in a single frame. All coordinates are with 
respect to (row, col), not (x,y).

Fields:
    frameNum    - Which frame this soma appears in
    coordinates - (Nx2 ndarray) list of coordinates belonging to the soma body
    bbox        - BoundingBox object describing the bounding box around the
                  soma body
    centroid    - (2, ndarray) the soma centroid coordinates
    contour     - (Mx2 ndarray) list of coordinates belonging to the soma 
                  contour
"""
class FrameSoma(object):
    # Defining all the variables ahead of time with __slots__ helps with
    # memory management and makes access quicker
    __slots__ = 'frameNum', 'coordinates', 'bbox', 'centroid', 'contour'
    
    """
    Global variables for this class
    """
    MIN_SOMA_SIZE = 281 # Somas must have at least this many pixels to 
                        # be valid: 281 px^2 ~= 16.7 microns^2
    
    """
    Initializes the object.
        frameNum - (int) The frame number this soma appears in (TODO)
        coords   - (Nx2 ndarray) All pixel coordinates for this soma
        bbox     - (1x4 ndarray, list, or tuple) Bounding box of the soma 
                   containing the fields [xmin, ymin, xmax, ymax].
                   Will be calculated if not provided. 
        contour  - (Mx2 ndarray) list of coordinates for the soma contour.
                   Will be calculated if not provided. 
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
        self.centroid = sp.round_( 
                            sp.mean(self.coordinates, axis=0) ).astype('int16')
    
    
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
    Calculates the contour of the soma. OpenCV's contour-finding algorithm
    returns a list of lists, so most of this function involves converting that
    to a usable Nx2 ndarray. 
    """    
    def calculateContour(self):
        bw = sp.zeros((self.bbox.height(), self.bbox.width()), dtype='uint8')
        bw[self.rows()-self.bbox.row_min, self.cols()-self.bbox.col_min] = 1
        
        contours = cv2.findContours(bw, cv2.RETR_EXTERNAL, 
                                    cv2.CHAIN_APPROX_NONE)
        contours = [c for c in contours[1] if c.shape[0] > 1]
    
        # contours is a list of M contours, where each element is an Nx2 array 
        # of coordinates belonging to each contour. This code flattens the 
        # array into (M*N)x2 and removes duplicate points. 
        contours = [sp.reshape(c,(c.shape[0]*c.shape[1], c.shape[2])) \
                    for c in contours]
        contours = sp.concatenate(contours)
        contours = sp.array(list(set(tuple(p) for p in contours))) 
        
        # contours are in terms of x,y instead of row,col, so the coordinates 
        # need to be reversed. This also undoes the coordinate adjustment done 
        # at the beginning of this function to account for using only the 
        # bounding box
        self.contour = sp.column_stack((contours[:,1]+self.bbox.row_min,
                                        contours[:,0]+self.bbox.col_min))
        
    
    """
    This is used when calling soma.copy(). It creates a new soma object
    with the same frame, coordinates, and contour. 
    """
    def __copy__(self):
        newSoma = FrameSoma(self.frameNum, self.coordinates.copy(), 
                            contour=self.contour.copy())
        return newSoma
    
    
    """
    This is what will get printed out when using print(soma) 
    """
    def __repr__(self):
        s = "Frame:\t" + str(self.frameNum) + "\n" + \
            "Centroid: [" + str(int(self.centroid[0])) + ", " + \
                            str(int(self.centroid[1])) + "]\n" + \
            "Box:\t" + str(self.bbox)
            
        return s
    

"""
This class represents a single soma across multiple frames in a video, i.e.
a collection of FrameSomas that represent the same object.

Fields:
    frameSomas - Dictionary of FrameSoma objects where the keys are the frame
                 numbers: {frameNum: <FrameSoma for that frame>}
    frames     - List of all frames that this VideoSoma appears in
    coordinateSpread - set of (row,col) tuples corresponding to all coordinates
                       of all soma bodies across all frames this VideoSoma
                       appears in
"""
class VideoSoma(object):
    __slots__ = 'frameSomas', 'frames', 'coordinateSpread'
    
    """
    Initialization
    
    Input: 
        frameSoma - the initial FrameSoma object for the first frame this
                    object appears in
    """
    def __init__(self, frameSoma):
        self.frameSomas = {frameSoma.frameNum: frameSoma}
        self.frames     = [frameSoma.frameNum]
        self.coordinateSpread = set([tuple(C) for C in frameSoma.coordinates])
    
    """
    Adds a FrameSoma to the list. The frameSomas dictionary, frames list, 
    and coordinate spread are all updated. 
    
    Input:
        frameSoma - A FrameSoma object that belongs to this VideoSoma
    """
    def addSoma(self, frameSoma):
        self.frameSomas[frameSoma.frameNum] = frameSoma
        self.frames.append(frameSoma.frameNum)
        spread = set([tuple(C) for C in frameSoma.coordinates])
        self.coordinateSpread = set.union(spread, self.coordinateSpread)
    

    """
    Gets the FrameSoma at the specified frame.
    
    Input:
        frame - integer. The frame number
        
    Output:
        FrameSoma object, or None
    """    
    def somaAtFrame(self, frame):
        if frame in self.frameSomas:
            return self.frameSomas[frame]
        
        return None

        
    """ 
    Tests to see if soma objects in different frames are actually the same.
    Assumes that between frames, the microglia centroid does not shift more
    than 20 pixels in any direction. 
    
    Input: 
        frameSoma - the FrameSoma object to compare to all this VideoSoma's
                    frameSomas. 
                    
    Output:
        tuple(True/False, distance) - True if there is a match, False if not.
            Distance is the distance between soma centroids. It will be -1
            if the match field is False. 
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
    
    
    """
    Calculates the distance between the input frameSoma's centroid and the 
    centroid of the soma in the closest frame. 
    
    Input: 
        frameSoma - The FrameSoma object to compare to
        
    Output:
        dist - distance between centroids
    """
    def distanceTo(self, frameSoma):
        # This will be the closest frame to the current frame
        lastFrame = sorted(self.frames)[-1] 
        lastSoma = self.frameSomas[lastFrame]
    
        diff = lastSoma.centroid - frameSoma.centroid
        dist = sp.sqrt(diff[0]**2 + diff[1]**2)
        
        return dist
    
    
    """
    Checks to see if two VideoSoma objects overlap in space at any point in the
    time series, which may happen when parts of a single soma fall below the 
    detection threshold and generate two VideoSoma objects before it joins 
    back into one soma. This function is called on post-interpolated VideoSoma
    data. 
    
    Input:
        other - The VideoSoma object to compare to
        
    Output:
        overlap_frames - a list of frames in which these two somas overlap.
                         Can be empty. 
    """
    def overlaps(self, other):
        # Fastest test: do their point spreads overlap at all? If not, stop
        # searching. 
        intersect = set.intersection(self.coordinateSpread, 
                                     other.coordinateSpread)
        if len(intersect) == 0:
            return []
        
        # If they do overlap, search frame by frame for specific soma object
        # overlaps
        overlap_frames = []
        for f in sorted(self.frames):
            if f not in other.frames:
                continue
            
            s1 = self.frameSomas[f]
            s2 = other.frameSomas[f]
            
            coords1 = set([tuple(C) for C in s1.coordinates])
            coords2 = set([tuple(C) for C in s2.coordinates])
            
            # Do the two somas overlap? If so, add this frame to the overlap
            # frames list
            intersect = set.intersection(coords1, coords2)
            if len(intersect) > 0:
                overlap_frames.append(f)
    
        return overlap_frames
    
    
    """
    Merges one VideoSoma's data with another, when they have been determined 
    to represent the same soma. The frameSomas dict, frames list, and
    coordinate spreads are merged together. 
    
    Input: 
        other - the VideoSoma object to merge into this one
    """
    def mergeWith(self, other):
        for f in set.union(set(self.frames), set(other.frames)):
            # Only this soma appears in this frame. No action needs to be taken
            if f not in other.frames:
                pass
            
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
                
                # Merge the coordinates, ensuring there are no duplicates by
                # making them a set before converting them to an array. 
                new_coords = sp.array(list(set.union(coords1, coords2)))
                new_soma = FrameSoma(f, new_coords)
    
                # This new soma replaces the old one in the array. Delete
                # both old soma objects. 
                self.frameSomas[f]  = new_soma
                other.frameSomas[f] = None
                del s1, s2
    
    
    """
    Given a frame number, return the soma at the nearest frame number to that
    one. 
    
    Input:
        frame - a frame number
        
    Output:
        FrameSoma object
    """
    def nearestFrame(self, frame):
        # No searching required if this VideoSoma is present at that frame
        if frame in self.frames:
            return self.frameSomas[frame]
        
        # Otherwise find the nearest frame number by distance
        dist = abs(sp.array(self.frames) - frame)
        index = sp.argsort(dist)
        
        # Return the soma at the closest frame number
        return self.frameSomas[self.frames[index[0]]]
   
    
    """
    For less-than comparison during a sort function. One VideoSoma is "less 
    than" another VideoSoma if it has fewer frames. 
    """
    def __lt__(self, other):
        return len(self.frames) < len(other.frames)
    
    
    """
    For debugging. This gets printed out with print(soma)
    """
    def __repr__(self):
        s = "Frames: " + str(sorted(self.frameSomas.keys())) + "\n" + \
            "Centroid: " + str(self.frameSomas[self.frames[0]].centroid) + "\n"
            
        return s
        

"""
Finds all interconnected pixels in a frame and labels them as an object. If the
object is large enough, it is marked as a soma and added to the list.

Input:
    bw    - (MxN ndarray) Thresholded image with values of 0 or 255
    frame - (int) The frame number (for soma labelling purposes)
    
Output:
    somas - (list) List of FrameSoma objects
"""
def label_objects(bw, frame):
    # Find all interconnected pixel regions
    number, labels, stats, centroids = cv2.connectedComponentsWithStats(bw, 
                                                                connectivity=4)
    
    valid = sp.where(stats[:,4] >= FrameSoma.MIN_SOMA_SIZE)[0]
    
    somas = []
    
    # Create a soma object for each object that is larger than the minimum size
    for v in valid:
        if v == 0: # Ignore the 'background' label, which will always be 0
            continue

        coords = sp.vstack(sp.where(labels == v)).T
        bbox = (min(coords[:,0]), min(coords[:,1]), \
                max(coords[:,0]), max(coords[:,1]))
        
        bbox_obj = BoundingBox(bbox[0], bbox[1], bbox[2], bbox[3])
        
        # Create the soma object and add it to the list
        somas.append(FrameSoma(frame, coords, bbox_obj))
    
    return somas


"""
Finds all the somas in a single max projection image. 

The image is thresholded by using pixels above the Otsu threshold. 
Interconnected pixels are labelled as objects, and those of adequate size are 
returned as valid somas. 

Input:
    img - (MxN ndarray) The image to search
    
Output:
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


"""
Tracks somas across the time series by combining individual FrameSomas to 
matching FrameSomas in other frames. The resulting collection is a "VideoSoma"
object. 

Input:
    videoSomas - A list of all VideoSoma objects that have been collected up
                 to the current frame
    frameSomas - A list of the FrameSoma objects in the current frame
    
Output: 
    videoSomas - The updated list of VideoSomas, with the current FrameSomas
                 added. 
"""
def combine_somas(videoSomas, frameSomas):
    for f in frameSomas:
        matches = []
        
        # Find all VideoSoma objects that could match with this FrameSoma
        # (by proximity in other frames)
        for v in videoSomas:
            isMatch, dist = v.isMatch(f)
            if isMatch:
                matches.append((dist, v))
        
        # If there are no matches, this FrameSoma becomes the first one in a
        # new VideoSoma        
        if len(matches) == 0:
            videoSomas.append(VideoSoma(f))
        
        # If there's one match, add this FrameSoma to the matching VideoSoma
        elif len(matches) == 1:
            matches[0][1].addSoma(f)
        
        # If there is more than one match, choose the VideoSoma that is 
        # closest in proximity
        else:
            matches.sort()
            matches[0][1].addSoma(f)
            
    return videoSomas


"""
Interpolates soma locations in frames where a soma may have been missed. 
For each VideoSoma, any missing frames are filled in by using the coordinates
of the soma in the nearest frame. Any of those coordinates that are white in 
the thresholded image become the interpolated soma for that frame. 

Once all VideoSoma objects have their somas interpolated, this function then
finds VideoSoma objects that may represent the same soma, and combines them.
This may happen when a soma falls below threshold for a few frames before
reappearing, generating two VideoSoma objects for the same soma, or when parts
of the soma fall below threshold and make it appear split before combining
again in a later frame. 

Input:
    videoSomas - list of VideoSoma objects
    threshold_images - list of thresholded images from the deconvolved image
                       stack. 
    
Output: 
    videoSomas - Updated list of VideoSoma objects
"""
def interpolate_somas(videoSomas, threshold_images):
    for v in videoSomas:
        toAdd = []
        for frame in range(len(threshold_images)):
            # If this frame is missing, interpolate the soma
            if frame not in v.frames:
                frameSoma = v.nearestFrame(frame)
                bw = threshold_images[frame]
                
                # Find all coordinates of the soma body in the nearest frame
                # that are also white in this frame's thresholded image
                ind = sp.where(bw[frameSoma.rows(), frameSoma.cols()] > 0)[0]
                coords = sp.vstack((frameSoma.rows()[ind], 
                                    frameSoma.cols()[ind])).T
                
                if len(coords) > 10: # Filter out noise
                    newSoma = FrameSoma(frame, coords)
                    toAdd.append(newSoma)
        
        # We cannot alter the videoSoma's frames while we are trying to iterate
        # through them, so alter them once the loop is done
        for soma in toAdd:
            v.addSoma(soma)
    
    # Merge VideoSomas that actually represent the same soma across all frames    
    toRemove = []
    for i in range(len(videoSomas)):
        v1 = videoSomas[i]
        for j in sp.arange(i+1, len(videoSomas)):
            v2 = videoSomas[j]
            
            # Find all frames where these two VideoSomas overlap spatially
            # (which would be due to interpolation of somas in missing frames)
            overlap_frames = v1.overlaps(v2)
            if len(overlap_frames) > 0:
                # Keep v2, mark v1 for deletion
                v2.mergeWith(v1)
                toRemove.append(v1)
                break
    
    # We cannot alter the VideoSomas list while iterating through it so remove
    # the dead VideoSomas when the loop is done    
    for v in toRemove:
        videoSomas.remove(v)
        
    return videoSomas
    
    
    