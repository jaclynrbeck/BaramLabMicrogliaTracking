#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 13:14:38 2017

@author: jaclynbeck
"""

import bioformats
import datetime
import scipy as sp
import cv2
import Utils


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
        for i in range(0,self.numImages):
            # There are fields called "TimePoint<#>", i.e. "TimePoint10" for 
            # frame 10. Get those fields. 
            tp = "TimePoint"+str(i+1)
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
        frameNum - (int) The frame number this soma appears in
        coords   - (Nx2 ndarray) All pixel coordinates for this soma
        bbox     - (1x4 ndarray, list, or tuple) Bounding box of the soma 
                   containing the fields [rowmin, colmin, rowmax, colmax].
                   Will be calculated if not provided. 
        contour  - (Mx2 ndarray) list of coordinates for the soma contour.
                   Will be calculated if not provided. 
    """
    def __init__(self, frameNum, coords, bbox=None, contour=None):
        self.frameNum = int(frameNum)
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
This class represents a node in a directed graph.

Class variables:
    value       - Integer ID (index into the tree returned by mst)
    coordinates - (3,) ndarray of image coordinates (row, col, px val)
    parent      - DirectedNode that is the parent of this node
    children    - List of DirectedNodes that are children of this node
    length      - Physical distance (geodesic) between this node and the 
                  center node of the tree. 
"""   
class DirectedNode(object):
    __slots__ = 'value', 'coordinates', 'parent', 'children', 'length'
    
    """
    Initialization
    
    Input: 
        value - Integer ID (index into the tree returned by mst)
        coordinates - Real image coordinates of this node
    """
    def __init__(self, value, coordinates):
        self.value = value
        self.coordinates = coordinates[0:2].astype('int16')
        self.parent = None
        self.children = []
        self.length = 0
    
    """
    Sets this node's parent node
    """    
    def addParent(self, node):
        self.parent = node
        self.length = self.parent.length + sp.sqrt(sp.sum((self.parent.coordinates-self.coordinates)**2))
        
    
    """
    Checks to see if the given branch in the tree should be pruned. A branch
    is pruned if the length (in pixels) from leaf to root of branch is
    less than 10. If so, the root of the branch is returned so it can be 
    deleted. 
    
    This is a recursive function.
    
    Input:
        leafNode - the DirectedNode that is the leaf (endpoint) of the branch
        previousNode - the DirectedNode that was the last node checked. This is
                       always passed along in the recursion because if the
                       current node has more than one child, we need to know
                       which child branch we are examining. 
        
    Output:
        previousNode - a DirectedNode representing the base of the branch to
                       be pruned. 
    """    
    def checkForPruning(self, leafNode, previousNode):
        # If this node has more than one child, the branch might be prunable
        if len(self.children) > 1:
            # If the length of the branch is too small, return the previous
            # node in the recursive loop, which is the root of the branch
            if leafNode.length - self.length < 10:
                return previousNode
            
            # If the branch is long enough, don't prune
            else:
                return None
        
        # This node has only one child so this is part of a branch. Go up to
        # its parent to see if it can prune
        if self.parent is not None:
            return self.parent.checkForPruning(leafNode, self)
        
        return None
        
    
    """
    Method for debugging. Prints this node's value and children, and has the
    children print themselves too. 
    """    
    def printAsTree(self, tabLevel):
        s = str(self.value) + " (" + str(self.length) + ")\n"
        for c in self.children:
            s += "  "*tabLevel + " -> " + c.printAsTree(tabLevel+1)
            
        return s
    
    """
    This is what will be displayed when calling print(DirectedNode). This 
    method is not recursive. 
    """
    def __repr__(self):
        s = "Value: " + str(self.value) + ", Length: " + str(self.length) 
        if self.parent is None:
            s += ", Parent: None"
        else:
            s += ", Parent: " + str(self.parent.value)
            
        s += ", Children: [ " 
        for c in self.children:
            s += str(c.value) + "  "
        s += "]\n"
        
        return s
    

"""
This class represents a directed tree graph with no cycles. 

Class Variables:
    nodes      - List of all DirectedNodes in the tree
    leaves     - List of DirectedNodes that have no child nodes
    centerNode - DirectedNode that is the root (center) of the tree
"""
class DirectedTree(object):
    __slots__ = 'nodes', 'leaves', 'centerNode'
    
    
    """
    Initialization. The tree is built elsewhere, but all this class needs to
    know at this point is the center node. 
    
    Input: 
        centerNode - DirectedNode that will be the root of the tree
    """
    def __init__(self, centerNode):
        self.nodes = []
        self.leaves = []
        self.centerNode = centerNode
        
    
    """
    Removes short branches (< 10 pixels long) from the tree, which are likely 
    to be noise or artifacts from skeletonization. 
    """
    def prune(self):
        # We can't delete nodes from a list we are iterating through, so this
        # saves which nodes we want to remove 
        leaves_to_remove = [] 
        
        # For each leaf, checkForPruning will return the base of the leaf's 
        # branch if that branch should be pruned.
        for node in self.leaves:
            removed = node.checkForPruning(node, node)
            if removed is not None: 
                leaves_to_remove.append(node)
                self.deleteNode(removed)
        
        for node in leaves_to_remove:
            self.leaves.remove(node)
            
    
    """
    Deletes a node and all of its child branches from the tree. This is a 
    recursive function. 
    
    Input: 
        node - The node to delete
    """            
    def deleteNode(self, node):
        # Delete the child nodes first
        for c in node.children:
            self.deleteNode(c)
        
        # Remove this node from the parent node's list of children
        if node.parent is not None:
            node.parent.children.remove(node)
         
        # Remove this node from the master list of nodes
        self.nodes.remove(node)
        
        # Delete the object to free up memory
        del node
        
    
    """
    This method is called during 'pickle'. It translates all of the class's 
    data into a dictionary so the object can be written to a file. 
    Since pickle doesn't handle the amount of recursion contained in this 
    class's list of DirectedNodes, this method turns every node's child and 
    parent lists into lists of ID numbers instead of lists of DirectedNodes.
    The proper lists are restored during unpickle based on the ID number. 
    """        
    def __getstate__(self):
        stateDict = {'nodes': self.nodes,
                     'leaves': self.leaves, 
                     'centerNode': self.centerNode}
        
        # For each node, node.parent becomes the parent's index into the
        # node list and node.children becomes a list of the children's indices. 
        for node in stateDict['nodes']:
            if node.parent is not None:
                node.parent = stateDict['nodes'].index(node.parent)
                
            if len(node.children) > 0:
                new_children = []
                for c in node.children:
                    new_children.append(stateDict['nodes'].index(c))
                node.children = new_children

        return stateDict
    
    
    """
    This method is called during 'pickle load'. It restores each DirectedNode's
    parent and child lists to DirectedNodes instead of ID numbers, opposite of
    what is done in __getstate__. 
    """
    def __setstate__(self, stateDict):
        self.nodes = stateDict['nodes']
        self.leaves = stateDict['leaves']
        self.centerNode = stateDict['centerNode']
        
        # For each node, node.parent becomes the DirectedNode at the index
        # into the node list and node.children becomes a list of DirectedNodes
        # at the provided indices. 
        for node in self.nodes:
            if node.parent is not None:
                node.parent = self.nodes[node.parent]
            
            if len(node.children) > 0:
                new_children = []
                for c in node.children:
                    new_children.append(self.nodes[c])
                    
                node.children = new_children
                
 
"""
This class represents a process tip across multiple frames in the video. This 
is used to calculate statistics about a process. 

Class Variables: 
    tips           - Dictionary of DirectedNode objects that are leaves of the
                     tree. Dictionary is {frame: leaf}
    tipIDs         - List of tip IDs of the tip in each frame. A tip ID is a
                     tuple of (frame, DirectedNode)
    velocityX      - List of the x-coordinate velocity of the tip between
                     frames, indexed by frame. In microns per minute. 
    velocityY      - List of the y-coordinate velocity of the tip between
                     frames, indexed by frame. In microns per minute. 
    velocityMagnitude - List of the velocity of the tip between frames, indexed
                     by frame. In microns per minute.
    length         - List of the length of the process from tip to soma in each
                     frame, indexed by frame. In microns. 
    lengthVelocity - List of the change in tip length between frames, indexed
                     by frame. In microns per minute. 
    location       - List of the pixel (x,y) coordinates of the process
                     tip in each frame, indexed by frame number.
"""               
class ProcessTip(object):
    __slots__ = 'tips', 'tipIDs', 'velocityX', 'velocityY', \
                'velocityMagnitude', 'length', 'lengthVelocity', 'location'
    
    """
    Initialization. Sets up the tip dictionary with two inital matching tips 
    from different frames. 
    
    Input:
        tipID1 - tuple ID of leaf/tip 1: (frame1, leaf1)
        tipID2 - tuple ID of leaf/tip 2: (frame2, leaf2)
        frame1 - frame number that leaf1 appears in
        frame2 - frame number that leaf2 appears in
        leaf1  - DirectedNode representing a process tip in one frame
        leaf2  - DirectedNode representing a matching process tip in a 
                 different frame. 
    """
    def __init__(self, tipID1, tipID2, frame1, frame2, leaf1, leaf2):
        self.tipIDs = [tipID1, tipID2]
        self.tips = {frame1: leaf1, frame2: leaf2}

    """
    Adds a matching tip from a new frame to the existing list of tips/frames. 
    
    Input: 
        tipID - tuple ID of the leaf: (frame, leaf)
        frame - the frame this tip is in
        leaf  - the DirectedNode object representing the tip
    """
    def addFrame(self, tipID, frame, leaf):
        self.tipIDs.append(tipID)
        self.tips[frame] = leaf
    
    """
    Gets a list of frames that this tip exists in
    """    
    def getFrames(self):
        return [f for f in self.tips.keys()]
    
    """
    Calculates statistics about this process tip and stores them in class 
    variables.
    
    Current calculations:
        VelocityX - list, microns per minute moved in the X-direction
        VelocityY - list, microns per minute moved in the Y-direction
        VelocityMagnitude - list, microns per minute moved (magnitude)
        Length - list, microns. The length from soma to tip, following the path
                 of the branch. This is not Euclidean distance. 
        LengthVelocity - list, microns per minute. How much the length is 
                 changing between frames. 
        Location - list of (X,Y) pixel locations of the tip in each frame
    
    Input: 
        metadata  - ImageMetadata object
        numFrames - number of frames in the video
    """    
    def calculateData(self, metadata, numFrames):
        self.velocityX = sp.full((numFrames,), None)
        self.velocityY = sp.full((numFrames,), None)
        self.velocityMagnitude  = sp.full((numFrames,), None)
        self.length = sp.full((numFrames,), None)
        self.lengthVelocity = sp.full((numFrames,), None)
        self.location = sp.full((numFrames,), None)
    
        frames = [k for k in self.tips.keys()]
        self.length[frames[0]] = self.tips[frames[0]].length * metadata.physX
        
        coords = self.tips[frames[0]].coordinates
        self.location[frames[0]] = "(" + str(coords[1]) + " " + str(coords[0]) + ")"
        
        for i in range(len(frames)-1):
            f1 = frames[i]
            f2 = frames[i+1]

            tip1 = self.tips[f1]
            tip2 = self.tips[f2]
            
            # Assume physical size X = physical size Y
            self.length[f2] = tip2.length * metadata.physX
            coords = tip2.coordinates
            self.location[frames[i+1]] = "(" + str(coords[1]) + " " + str(coords[0]) + ")"
            
            delta = metadata.imgTimes[f2] - metadata.imgTimes[f1]
            velocity = (tip2.coordinates - tip1.coordinates) * metadata.physX / (delta.total_seconds()/60.0)
            self.velocityX[f2] = velocity[1]
            self.velocityY[f2] = velocity[0]
            self.velocityMagnitude[f2]  = sp.sqrt(sp.sum(velocity**2))   
            self.lengthVelocity[f2] = (self.length[f2]-self.length[f1]) / (delta.total_seconds()/60.0)
            

"""
This class represents a microglia across all frames in the video. It holds
information on all its soma locations, tree structures, and statistics.

Class Variables:
    trees           - dictionary of DirectedTree objects, one tree per frame,
                      indexed by frame. 
    somas           - VideoSoma object representing all the somas in the video
                      that belong to this microglia
    processTips     - list of ProcessTip objects representing all the tips in
                      all frames that belong to this microglia
    somaVelocityX   - list of soma velocities in the x-direction for each 
                      frame. In microns per minute, indexed by frame. 
    somaVelocityY   - list of soma velocities in the y-direction for each 
                      frame. In microns per minute, indexed by frame. 
    somaVelocityMagnitude - list of magnitudes of soma velocity for each frame.
                      in microns per minute, indexed by frame. 
    somaArea        - list of soma areas for each frame. In microns^2, indexed
                      by frame. 
    domainArea      - list of domain areas (convex hull of microglia soma/tips)
                      for each frame. In microns^2, indexed by frame. 
    numberOfProcesses - list of the number of processes in each frame, indexed
                      by frame. 
    numberOfMainBranches - list of the number of branches directly off the 
                      soma in each frame. Indexed by frame. 
    somaConvexity   - list of measures of soma convexity for each frame, 
                      indexed by frame. 
    somaCentroidX   - list of the x-coordinates of the somas in each frame, 
                      indexed by frame. 
    somaCentroidY   - list of the y-coordinates of the somas in each frame, 
                      indexed by frame. 
"""
class Microglia(object):
    __slots__ = 'trees', 'somas', 'processTips', 'somaVelocityX', \
                'somaVelocityY', 'somaVelocityMagnitude', 'somaArea', \
                'domainArea', 'numberOfProcesses', \
                'numberOfMainBranches', 'somaConvexity', 'somaCentroidX', \
                'somaCentroidY'
    
    """
    Initialization. The only information needed at this point is the list of 
    somas. 
    
    Input:
        somas - a VideoSoma object
    """
    def __init__(self, somas):
        self.somas = somas
        self.processTips = []
        self.trees = {}
        
    
    """
    Adds a new DirectedTree object to the trees dictionary.
    
    Input:
        tree  - the DirectedTree to add
        frame - the frame number this tree is in
    """        
    def addTreeAtFrame(self, tree, frame):
        self.trees[frame] = tree
        
    
    """
    Calculates statistics about the soma throughout the video and stores them
    in class variables.
    
    Current calculations:
        Soma Centroid X - list of the centroid X-coordinates in each frame
        Soma Centroid Y - list of the centroid Y-coordinates in each frame
        Soma Velocity X - list of the X-direction velocity (centroid movement)
                          in each frame
        Soma Velocity Y - list of the Y-direction velocity (centroid movement)
                          in each frame
        Soma Velocity Magnitude - list of the magnitude of velocity (centroid 
                          movement) in each frame. 
        Soma Area       - list of soma areas in each frame
        Soma Convexity  - list of soma convexities in each frame
        
    Input: 
        metadata  - ImageMetadata object
        numFrames - number of frames in the video
    """
    def calculateSomaMovement(self, metadata, numFrames):
        self.somaCentroidX = sp.full((numFrames,), None)
        self.somaCentroidY = sp.full((numFrames,), None)
        self.somaVelocityX = sp.full((numFrames,), None)
        self.somaVelocityY = sp.full((numFrames,), None)
        self.somaVelocityMagnitude = sp.full((numFrames,), None)
        self.somaArea = sp.full((numFrames,), None)
        self.somaConvexity = sp.full((numFrames,), None)
        
        frames = [k for k in self.trees.keys()]
        f1 = frames[0]
        s1 = self.somas.frameSomas[f1]
        c1 = s1.centroid
        
        self.somaCentroidX[f1] = c1[1]
        self.somaCentroidY[f1] = c1[0]
        
        # Here we are assuming that physical size X = physical size Y
        microns2 = metadata.physX * metadata.physY # microns^2
        self.somaArea[f1] = len(s1.coordinates) * microns2
        hull = cv2.convexHull(s1.coordinates.astype('int32'))
        self.somaConvexity[f1] = min(self.somaArea[f1] / (round(cv2.contourArea(hull))*microns2), 1.0)
        
        for i in range(1, len(frames)):
            f2 = frames[i]
            s2 = self.somas.frameSomas[f2]
            c2 = s2.centroid
            
            self.somaCentroidX[f2] = c2[1]
            self.somaCentroidY[f2] = c2[0]
        
            delta = metadata.imgTimes[f2] - metadata.imgTimes[f1]
            velocity = (c2 - c1)*metadata.physX / (delta.total_seconds()/60.0)
            self.somaVelocityX[f2] = velocity[1]
            self.somaVelocityY[f2] = velocity[0]
            self.somaVelocityMagnitude[f2] = sp.sqrt(sp.sum(velocity**2))
            
            self.somaArea[f2] = len(s2.coordinates) * microns2
            
            hull = cv2.convexHull(s2.coordinates.astype('int32'))
            self.somaConvexity[f2] = min(self.somaArea[f2] / (round(cv2.contourArea(hull))*microns2), 1.0)
            
            f1 = f2
            s1 = s2
            c1 = c2
            
    
    """
    Matches process tips between frames. For each set of trees (tree1, tree2), 
    leaves in tree1 are "matched" to leaves in tree2 by closest distance. Only
    leaves that match in both directions (i.e. leaf1 is the closest leaf to 
    leaf2, and leaf2 is the closest leaf to leaf1) are considered "matched". 
    
    Matched leaves are added to ProcessTip objects that will keep track of 
    leaves that match through the video. 
    """
    def matchLeaves(self):
        frames = sorted(list(self.trees.keys()))
        f1 = frames[0]
        leaves1 = self.trees[f1].leaves
        coords1 = [L.coordinates.astype('int32') for L in leaves1]
        
        matchDict = {}
    
        for i in range(1, len(frames)):
            f2 = frames[i]
            leaves2 = self.trees[f2].leaves
            coords2 = [L.coordinates.astype('int32') for L in leaves2]
            
            # Match leaves in frame-1 and frame
            matches = Utils.find_matches(coords1, coords2)
            
            # Any matches that are > 50 pixels apart are probably not really
            # matches. Discard those and keep the rest. 
            for m in matches:
                dist = m[2]
                if dist <= 50: 
                    leaf1 = (f1, leaves1[m[0]])
                    leaf2 = (f2, leaves2[m[1]])
                
                    matchDict[leaf1] = leaf2
                   
            f1 = f2
            leaves1 = leaves2
            coords1 = coords2
        
        # Go through the list of matches and create a ProcessTip that holds
        # the chain of matching leaves throughout the video. 
        while len(matchDict) > 0:
            leaf1 = list(matchDict.keys())[0]
            leaf2 = matchDict.pop(leaf1)
            
            # Seed with leaf1 and leaf2 as the first match
            pTip = ProcessTip(leaf1, leaf2, leaf1[0], leaf2[0], leaf1[1], leaf2[1])
            
            # Go through the matching dict and get the rest of the leaves in 
            # the chain: what matches to leaf2, what matches to that one, etc.
            while leaf2 in matchDict.keys():
                leaf2 = matchDict.pop(leaf2)
                pTip.addFrame(leaf2, leaf2[0], leaf2[1])
                
            self.processTips.append(pTip)
            
    
    """
    Calculates statistics about the microglia and its tips throughout the video
    and stores them in class variables. This is separate from the soma
    calculations. 
    
    Current calculations:
        Domain Area - list of the area of the convex hull of the microglia in
                      each frame
        Number Of Processes - list of the number of process tips in each frame
        Number Of Main Branches - list of the number of branches directly off
                      the soma in each frame. 
        Process Tip Statistics - See the ProcessTip object
        
    Input: 
        metadata  - ImageMetadata object
        numFrames - number of frames in the video
    """
    def calculateLeafData(self, metadata, numFrames):
        self.domainArea = sp.full((numFrames,), None)
        self.numberOfProcesses = sp.zeros((numFrames,))
        self.numberOfMainBranches = sp.full((numFrames,), None)
        
        for p in self.processTips:
            p.calculateData(metadata, numFrames)
            frames = p.getFrames()
            self.numberOfProcesses[frames] += 1
        
        self.numberOfProcesses[self.numberOfProcesses == 0] = None
        
        # Here we are assuming physical size X = physical size Y
        microns2 = metadata.physX * metadata.physY # microns^2
        
        for frame, tree in self.trees.items():
            coords = sp.array([N.coordinates for N in tree.nodes])
            hull = cv2.convexHull(coords.astype('int32'))
            self.domainArea[frame] = cv2.contourArea(hull) * microns2
            
            branches = [N for N in tree.centerNode.children if len(N.children) > 0]
            self.numberOfMainBranches[frame] = len(branches)
    
    
    """
    Given a set of data, calculates the mean of the data over certain time
    points.
    
    Input:
        data - a list of data, one data point per frame
        window_size - how many data points to average at once. For example,
            if there are 30 frames in the video and window_size = 10, this
            function will calculate the average of data points 0-9, 10-19, and
            20-29. 
        
    Output:
        averages - a list of length (numFrames / window_size), where the first
                   value is the average of window 1, the second value is the
                   average of window 2, etc. 
    """
    def means(self, data, window_size):
        averages = []
        for i in sp.arange(0, len(data), window_size):
            end = min(i+window_size, len(data))
            valid = [d for d in data[i:end] if d is not None]
            if len(valid) > 0:
                averages.append(sp.mean(valid))
            else:
                averages.append(None)
        
        averages.append(sp.mean([d for d in data if d is not None]))
        
        return averages
    
    
    """
    Calculates the mean value of a set of mean values. This is used to filter
    down process data. Realistically it should be taking the mean over all raw
    values, not means, but there is so much process data and enough means that
    there should not be much of a difference.
    
    Input:
        mID  - the microglia ID number
        data - a list of averaged data to take the mean of. Each element in
               this list is another list of averages over different time 
               windows for a specific process. 
        
    Output:
        m_mean - a list of length (window_size / numFrames) (see function 
                 "means"), where the first element is the mID, the second
                 element is the average of time point 1 across all the data, 
                 the third element is the average of time point 2 across all
                 the data, etc. 
    """
    def meanOfMeans(self, mID, data):
        arr = sp.array(data)
        m_mean = [mID]
        
        if len(arr) < 1:
            return []
        
        for i in range(2,arr.shape[1]):
            values = [x for x in arr[:,i] if x is not None]
            if len(values) > 0:
                m_mean += [sp.mean(values)]
            else:
                m_mean += [None]
            
        return m_mean
    
    
    """
    Converts class variables to a dictionary that will be used to write the csv
    files for this microglia. 
    
    Input: 
        mID - the microglia ID number
        window_size - the number of data points to average together when 
                      calculating averaged statistics. For example, if the 
                      video has 30 frames and window_size = 10, data points
                      0-9, 10-19, and 20-29 will be averaged to produce 3 
                      values. 
                      
    Output:
        idData      - list that starts with the microglia ID and then contains
                      the (x,y) coordinates for the soma in each frame, as 
                      strings
        somaData    - Dictionary containing lists for "VelocityX", "VelocityY",
                      "VelocityMagnitude", "Area", and "Convexity"
        processData - Dictionary containing lists for "VelocityX", "VelocityY",
                      "VelocityMagnitude", "Length", "LengthVelocity", 
                      "DomainArea", "NumberOfProcesses", 
                      "NumberOfMainBranches", and "Identity" (for processes)
        averageData - Dictionary containing lists of averaged data for all of
                      the above information, plus process length, process 
                      length velocity, and process velocity averaged by 
                      microglia in addition to by process
    """
    def dataToDict(self, mID, window_size):
        idData = [mID]+["(" + str(x) + " " + str(y) + ")" for x,y in \
                         zip(self.somaCentroidX, self.somaCentroidY)]
        
        somaData = {"VelocityX": [mID] + list(self.somaVelocityX), 
                    "VelocityY": [mID] + list(self.somaVelocityY), 
                    "VelocityMagnitude": [mID] + list(self.somaVelocityMagnitude),
                    "Area": [mID] + list(self.somaArea),
                    "Convexity": [mID] + list(self.somaConvexity)}
        
        averageData = {"SomaVelocityX": [mID] + self.means(self.somaVelocityX, window_size),
                       "SomaVelocityY": [mID] + self.means(self.somaVelocityY, window_size),
                       "SomaVelocityMagnitude": [mID] + self.means(self.somaVelocityMagnitude, window_size),
                       "SomaArea": [mID] + self.means(self.somaArea, window_size),
                       "SomaConvexity": [mID] + self.means(self.somaConvexity, window_size),
                       "ProcessVelocityX": [], "ProcessVelocityY": [], 
                       "ProcessVelocityMagnitude": [], "ProcessLength": [],
                       "ProcessLengthVelocity": [], 
                       "DomainArea": [mID] + self.means(self.domainArea, window_size),
                       "NumberOfProcesses": [mID] + self.means(self.numberOfProcesses, window_size), 
                       "NumberOfMainBranches": [mID] + self.means(self.numberOfMainBranches, window_size),
                       "ProcessLengthByMicroglia": [],
                       "ProcessLengthVelocityByMicroglia": [],
                       "ProcessVelocityMagnitudeByMicroglia": []}
        
        processData = {"VelocityX": [], "VelocityY": [], 
                       "VelocityMagnitude": [], "Length": [], 
                       "LengthVelocity": [], "DomainArea": [],
                       "NumberOfProcesses": [], "NumberOfMainBranches": [],
                       "Identity": []}
        
        for p in self.processTips:
            valid = [v for v in p.velocityX if v is not None]
            if len(valid) < 5: # Exclude tips that aren't in 5 or more frames
                continue
            
            tID = [mID, self.processTips.index(p)]
            processData["VelocityX"].append(tID + list(p.velocityX))
            processData["VelocityY"].append(tID + list(p.velocityY))
            processData["VelocityMagnitude"].append(tID + list(p.velocityMagnitude))
            processData["Length"].append(tID + list(p.length))
            processData["LengthVelocity"].append(tID + list(p.lengthVelocity))
            processData["Identity"].append(tID+list(p.location))
            
            averageData["ProcessVelocityX"].append(tID + self.means(p.velocityX, window_size))
            averageData["ProcessVelocityY"].append(tID + self.means(p.velocityY, window_size))
            averageData["ProcessVelocityMagnitude"].append(tID + self.means(p.velocityMagnitude, window_size))
            averageData["ProcessLength"].append(tID + self.means(p.length, window_size))
            averageData["ProcessLengthVelocity"].append(tID + self.means(p.lengthVelocity, window_size))
        
        averageData["ProcessLengthByMicroglia"] = self.meanOfMeans(mID, averageData["ProcessLength"])
        averageData["ProcessLengthVelocityByMicroglia"] = self.meanOfMeans(mID, averageData["ProcessLengthVelocity"]) 
        averageData["ProcessVelocityMagnitudeByMicroglia"] = self.meanOfMeans(mID, averageData["ProcessVelocityMagnitude"])
        
        processData["DomainArea"] = [mID] + list(self.domainArea)
        processData["NumberOfProcesses"] = [mID] + list(self.numberOfProcesses)
        processData["NumberOfMainBranches"] = [mID] + list(self.numberOfMainBranches)
            
        return idData, somaData, processData, averageData
