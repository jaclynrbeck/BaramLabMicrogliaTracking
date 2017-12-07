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
from Objects import FrameSoma, VideoSoma, BoundingBox

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
def find_somas_single_image(img, frame, threshold_percentile):
    diff = 100-threshold_percentile
    threshold_percentile = 100-diff/2
    threshold = sp.percentile(img, threshold_percentile)
    
    bw = sp.zeros_like(img, dtype='uint8')  
    bw[img > threshold] = 255
    
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    
    bw = cv2.morphologyEx(bw, cv2.MORPH_ERODE, k1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_DILATE, k2)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k1)

    somas = label_objects(bw, frame)
    return somas


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
    
    
    