#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 20:54:49 2017

@author: jaclynbeck
"""

import scipy as sp
from libtiff import TIFFfile
import PeaksMultithreshold as pm
import FindSomas as fs
import timeit
import OtsuMultithreshold as omt
import GliaMask as gm
import ThresholdLevel as tl
import Utils
import cv2 # installed with 'conda install -c conda-forge opencv'
import MST
import skimage.morphology as skm

"""
This object represents a self-contained region of objects that overlap at each
threshold. It will contain one or more somas and the list of RegionObjects
that overlap this area. 
""" 
class SeedRegion(object):
    # Defining all the variables ahead of time with __slots__ helps with
    # memory management and makes access quicker
    __slots__ = 'bbox', 'somas', 'regions', 'regionCount', 'seedPoints'
    
    """
    Initialization
        bbox - (1x4 ndarray or list) Bounding box with [xmin, ymin, xmax, ymax]
    """
    def __init__(self, bbox):
        self.bbox   = bbox
        self.somas  = []
        self.regions = []
        self.regionCount = 0
        self.seedPoints = []
    
    """
    Adds a FrameSoma object to the list of somas
    """    
    def addSoma(self, soma):
        self.somas.append(soma)
    
    """
    Returns true if this region has been assigned at least one soma
    """
    def hasSomas(self):
        return len(self.somas) > 0
    
    """
    Adds a RegionObject that was identified as belonging to this seed region
    """
    def addRegion(self, regionObject):
        self.regions.append(regionObject)
        self.regionCount += 1
    
    """
    Adds a set of seed points that have been sampled from this region
    """    
    def addSeedPoints(self, pts):
        self.seedPoints.append(pts)
    
    """
    Shortcut for getting the row coordinates of the largest RegionObject in 
    the list, which will always be last due to sorting. 
    """    
    def rows(self):
        return self.regions[-1].rows()
    
    """
    Shortcut for getting the column coordinates of the largest RegionObject in 
    the list, which will always be last due to sorting. 
    """ 
    def cols(self):
        return self.regions[-1].cols()
    
    """
    Returns true if this region's bounding box and the input bounding box
    overlap by at least 90%, indicating that they are the same region.
    """    
    def overlaps(self, bbox):
        return Utils.bbox_significant_overlap(self.bbox, bbox, 0.9)
    
    """
    This is what gets printed out with print(SeedRegion)
    """
    def __repr__(self):
        s = "{Somas: " + str(len(self.somas)) + ", Regions: " + \
            str(self.regionCount) + "}"
        return s
            
    
    
"""
Given a contour and a block size, sample seed points around the contour. 
The image is divided into (block_size)x(block_size) windows. In each window, 
if at least one contour point is there, the point with the smallest pixel
intensity is used as the sample. 

Input:
    img - (MxN ndarray) The original image with pixel intensities
    edge_points - (Kx2 ndarray) Array of contour pixel coordinates
    bbox - (1x4 ndarray or list) Bounding box with [xmin, ymin, xmax, ymax]
    block_size - (int) The width of the block to sample from
    
Output:
    points - (Jx2 ndarray) Array of sampled points from the contour
"""
def sample_seed_points(img, edge_points, bbox, block_size):
    # Dimensions of the bounding box
    height = bbox[2]-bbox[0]+1
    width  = bbox[3]-bbox[1]+1
    
    # Make an image that is the same size as the bounding box. Adjust
    # coordinates so that (0,0) is the upper left corner of the bounding box.
    # Add real pixel intensities to the new image
    bw = sp.zeros((height,width))
    bw[edge_points[:,0]-bbox[0],edge_points[:,1]-bbox[1]] = \
                                        img[edge_points[:,0],edge_points[:,1]]
    
    points = []
    
    # The two for loops create a (block_size)x(block_size) sampling window
    for row in sp.arange(0, height, block_size):
        for col in sp.arange(0, width, block_size):
            # The block coordinates
            block = bw[row:row+block_size,col:col+block_size]
            
            # If theres a contour point in the block, choose the lowest 
            # intensity point as the sample
            if sp.any(block > 0):
                (x,y) = sp.where(block > 0)
                smallest = sp.argmin(bw[row+x,col+y])
                
                # Undo the coordinate adjustment above so the coordinates are
                # with respect to the full image again
                point_r = row + x[smallest] + bbox[0]
                point_c = col + y[smallest] + bbox[1]

                points.append(sp.array([point_r,point_c]))
                
    return sp.array(points)


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
def get_contour(coordinates, bbox):  
    # Dimensions of the bounding box
    height = bbox[2]-bbox[0]+1
    width  = bbox[3]-bbox[1]+1
    
    # Make an image that is the same size as the bounding box. Adjust
    # coordinates so that (0,0) is the upper left corner of the bounding box.
    # Add contour points to the image
    bw = sp.zeros((height, width))
    bw[coordinates[0]-bbox[0],coordinates[1]-bbox[1]] = 255

    # contours is a tuple, contours[1] contains the coordinates needed
    contours = cv2.findContours(bw.astype(sp.uint8).copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = contours[1]

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


def find_midpoints(coordinates, bbox):
    # Dimensions of the bounding box
    height = bbox[2]-bbox[0]+1
    width  = bbox[3]-bbox[1]+1
    
    # Make an image that is the same size as the bounding box. Adjust
    # coordinates so that (0,0) is the upper left corner of the bounding box.
    # Add region points to the image
    bw = sp.zeros((height, width))
    bw[coordinates[0]-bbox[0],coordinates[1]-bbox[1]] = 255
    
    midpoints = []
    
    for row in range(height):
        col_start = None
        col_end = None
        for col in range(width):
            if bw[row,col] == 255:
                if col_start is None:
                    col_start = col
            
            elif col_start is not None:
                col_end = col-1
                midpoint = round((col_end+col_start)/2)
                
                midpoints.append([row, midpoint])
                col_start = None
                col_end = None
            
        if col_end is None and col_start is not None:
            col_end = width
            midpoint = round((width+col_start)/2)
            midpoints.append([row, midpoint])
    
    midpoints = sp.array(midpoints)


"""
Seed points are sampled points around the contour of a region at a given 
threshold. This method samples every X points along the contour of each 
region in a seed region, where X is determined by how high the threshold is.

The highest threshold is sampled in 2x2 blocks, the next highest in 3x3 blocks,
etc. 

Input:
    img - (MxN ndarray) The original image
    seedRegions - (list) List of SeedRegions to sample
    
Output:
    seedRegions - (list) The list of SeedRegions
"""
def get_all_seed_points(img, seedRegions):
    for seed in seedRegions:        
        # Sample all the pixels in the rest of the regions
        for region in seed.regions: #[1:]:
            #find_midpoints([region.rows(), region.cols()], region.bbox)
            # Calculate the block size. Priority = 0 for highest threshold
            block_size = region.priority+2
            
            # Regions with the highest threshold are either somas or tips of
            # processes. Only sample the contour of these areas since the 
            #inside is not part of the tree
            if region.priority == 0:
                edge_pts = get_contour([region.rows(),region.cols()], region.bbox)
                seed_pts = sample_seed_points(img, edge_pts, region.bbox, block_size)
                
                seed_pts = sp.concatenate((region.centroid(), seed_pts))
            
            else:
                # Find the contour and sample the points along the contour
                edge_pts = get_contour([region.rows(),region.cols()], region.bbox)
                seed_pts = sample_seed_points(img, edge_pts, region.bbox, block_size)
            
            seed.addSeedPoints(seed_pts.astype('int32'))
        
    return seedRegions


def get_all_seed_points2(img, levels):
    seedRegions = []
    
    for L in levels:
        sr = SeedRegion(None)
        
        for region in L.regions:
            #find_midpoints([region.rows(), region.cols()], region.bbox)
            # Calculate the block size. Priority = 0 for highest threshold
            block_size = region.priority+2
            
            # Regions with the highest threshold are either somas or tips of
            # processes. Only sample the contour of these areas since the 
            #inside is not part of the tree
            if region.priority == 0:
                edge_pts = get_contour([region.rows(),region.cols()], region.bbox)
                seed_pts = sample_seed_points(img, edge_pts, region.bbox, block_size)
                
                seed_pts = sp.concatenate((region.centroid(), seed_pts))
            
            else:
                # Find the contour and sample the points along the contour
                edge_pts = get_contour([region.rows(),region.cols()], region.bbox)
                seed_pts = sample_seed_points(img, edge_pts, region.bbox, block_size)
            
            sr.addRegion(region)
            sr.addSeedPoints(seed_pts.astype('int32'))
        
        seedRegions.append(sr)
    return seedRegions


"""
Goes through all the levels and finds which regions in each level overlap with
regions from other levels. A single stack of overlapping regions becomes a 
"seed region" -- which defines a region where a tree will be built.

Input: 
    levels - (list) List of LevelObjects
    somas - (list) List of FrameSomas
    
Output:
    seedRegions - (list) List of SeedRegion objects
"""
def combine_regions(levels, somas):
    # The lowest level will have the largest regions. Start by adding somas
    # to these regions
    lowestL = levels[-1]
    seedRegions = []
    
    for region in lowestL.regions:        
        # Each large region is the start of a seed region
        seedRegion = SeedRegion(region.bbox)

        # See if this region contains one or more somas. More than one soma 
        # can fit inside a region
        for soma in somas:
            if seedRegion.overlaps(soma.bbox):
                seedRegion.addSoma(soma)
        
        # If this has at least one soma, it's valid. Add it to the list
        #if seedRegion.hasSomas():
        seedRegion.addRegion(region)
        seedRegions.append(seedRegion)
        
                
    # Now find the rest of the overlapping regions in each seed region. 
    # Go from second-lowest level to the top level where the somas are
    for L in levels[-2::-1]:
        for region in L.regions:
            # If it matches a current known seed region, add it and go to 
            # the next level region
            for seed in seedRegions:
                if seed.overlaps(region.bbox):
                    seed.addRegion(region)
                    break # There can only be one region this one overlaps, 
                          # by design 
    
    #seedRegions = [S for S in seedRegions if (S.regionCount > 1)]
    
    # Ensure that the list of regions is sorted from highest threshold to lowest              
    for seed in seedRegions:
        seed.regions.sort()
        
    return seedRegions
    

"""
Main function for debugging
"""
if __name__ == '__main__':
    img_fname = '/Users/jaclynbeck/Desktop/BaramLab/Substack (1).tif' #'/Users/jaclynbeck/Desktop/BaramLab/C2-8-29-17_CRH-tdTomato+CX3CR1-GFP P8 PVN CES_Female 1 L PVN_a_.i...CX3CR1-GFP P8 PVN CES_Female 1 L PVN_a_GREEN_t1.tif'
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
    
    #mask = gm.create_Mask(images)
    
    #images[mask == 0] = 0
    
    images = Utils.preprocess_img(images) 
    
    #image_stack = sp.reshape(images, (1,images.shape[0], images.shape[1]))
    [soma_threshold, somas] = fs.find_somas_single_image(images)
    
    #thresholds = pm.peaks_multithreshold(images, peak_threshold)
    #print(thresholds)
    [maxSig, thresholds] = omt.otsu_multithreshold(images, 10, soma_threshold)
    print(thresholds)
    
    thresholds = thresholds[thresholds < soma_threshold] 
    thresholds = sp.concatenate(([0], thresholds, [soma_threshold]))
    
    levels = tl.get_levels(images, thresholds, somas)
    levels = [L for L in levels if not L.isBackgroundLevel() and L.regionCount < 1000]
    
    seedRegions = combine_regions(levels, somas)
    seedRegions = get_all_seed_points(images, seedRegions)
    #seedRegions = get_all_seed_points2(images, levels)
    
    seeds = seedRegions[10].seedPoints
    seed_pts = []
    
    for s in seeds:
        for p in s:
            seed_pts.append(tuple(p))
    
    # Removes duplicates
    seed_pts = sp.array(list(set(seed_pts)))
        
    distance = MST.build_distance_matrix(seed_pts)
    
    bw = Utils.plot_levels(levels, (1024,1024), False)

    #tree = MST.init_tree(seed_pts)
    #tree = MST.build_tree(tree, seed_pts)
    
    #edges = MST.do_mst(seed_pts, distance)
    #plt.plot(edges[0], edges[1], '-k')
    
    elapsed = timeit.default_timer() - start_time
    print(elapsed)
    
    # For debugging and display
    #levels.remove(levels[-1])
    print(levels)
    
    bw = Utils.plot_levels(levels, (1024,1024), True)    
    sp.misc.imsave(output_dir + 'objects_' + str(thresholds.size) + '.tif', bw*255.0/bw.max())
    
    bw[bw > 0] = 1
    skeleton = skm.skeletonize(bw)
    plt.imshow(skeleton)
    plt.show()
    sp.misc.imsave(output_dir + 'skeleton.tif', skeleton)
    
    s_img = Utils.plot_somas(somas, (1024,1024), True)
    sp.misc.imsave(output_dir + 'somas.tif', s_img)
    
    r_img = Utils.plot_seed_regions(seedRegions, (1024,1024), True)
    sp.misc.imsave(output_dir + 'regions.tif', r_img)
    
    cont_img = Utils.plot_seed_points(seedRegions, (1024,1024), True)
    sp.misc.imsave(output_dir + 'contours.tif',cont_img)
    
    #plt.plot(edges[0], edges[1], '-k')