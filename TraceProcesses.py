#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 20:54:49 2017

@author: jaclynbeck
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 13:12:56 2017

@author: jaclynbeck
"""

import scipy as sp
from libtiff import TIFFfile
import matplotlib.pyplot as plt
import PeaksMultithreshold as pm
import FindSomas as fs
import timeit
import OtsuMultithreshold as omt
import GliaMask as gm
from mst_clustering import MSTClustering # installed with 'pip install mst_clustering'
import ThresholdLevel as tl
import Utils
import cv2


"""
This class represents a microglia in a single frame
"""
class Microglia:
    def __init__(self, soma):
        self.soma = soma
        self.centroid = soma.centroid
        self.bbox = soma.bbox
        self.seedPoints = []
        self.regions = []
        self.regionCount = 0
        
    def addSeedPoints(self, coords):
        self.seedPoints.append(coords)
        
    def getBBox(self):
        return self.bbox
    
    def addRegion(self, coords):
        self.regions.append(coords)
        self.regionCount += 1
        
    def regionRows(self, region):
        return self.regions[region][0]
    
    def regionCols(self, region):
        return self.regions[region][1]
    
    def overlaps(self, bbox):
        return Utils.bbox_significant_overlap(self.bbox, bbox, 0.9)
    

class SeedRegion:
    def __init__(self, bbox):
        self.bbox   = bbox
        self.somas  = []
        self.levels = []
        self.levelCount = 0
        self.seedPoints = []
        
    def addSoma(self, soma):
        self.somas.append(soma)
        
    def hasSoma(self, soma):
        return (soma in self.somas)
    
    def hasSomas(self):
        return len(self.somas) > 0
    
    def addLevel(self, regionObject):
        self.levels.append(regionObject)
        self.levelCount += 1
        
    def addSeedPoints(self, pts):
        self.seedPoints.append(pts)
        
    def rows(self):
        return self.levels[0].rows()
    
    def cols(self):
        return self.levels[0].cols()
    
    def levelRows(self, level):
        return self.levels[self.levelCount-level-1].rows()
    
    def levelCols(self, level):
        return self.levels[self.levelCount-level-1].cols()
        
    def overlaps(self, bbox):
        return Utils.bbox_significant_overlap(self.bbox, bbox, 0.9)
    
    

def sample_seed_points(img, edge_points, bbox, block_size):
    height = bbox[2]-bbox[0]+1
    width  = bbox[3]-bbox[1]+1
    
    bw = sp.zeros((height,width))
    bw[edge_points[:,0]-bbox[0],edge_points[:,1]-bbox[1]] = \
                                        img[edge_points[:,0],edge_points[:,1]]
    
    points = []
    
    for row in sp.arange(0, height, block_size):
        for col in sp.arange(0, width, block_size):
            block = bw[row:row+block_size,col:col+block_size]
            if sp.any(block > 0):
                (x,y) = sp.where(block > 0)
                smallest = sp.argmin(bw[row+x,col+y])
                
                point_r = row + x[smallest] + bbox[0]
                point_c = col + y[smallest] + bbox[1]

                points.append(sp.array([point_r,point_c]))
                
    return sp.array(points)


def get_contour(coordinates, bbox):    
    height = bbox[2]-bbox[0]+1
    width  = bbox[3]-bbox[1]+1
    
    bw = sp.zeros((height, width))
    bw[coordinates[0]-bbox[0],coordinates[1]-bbox[1]] = 255

    contours = cv2.findContours(bw.astype(sp.uint8).copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = contours[1]

    contours = [sp.reshape(c,(c.shape[0]*c.shape[1], c.shape[2])) for c in contours]
    contours = sp.concatenate(contours)
    contours = sp.array(list(set(tuple(p) for p in contours))) # Remove duplicate points
    
    # contours are in terms of x,y instead of row,col, so they need to 
    # be reversed. We also need to add the row/col start index to account
    # for using only the bounding box
    rc = sp.column_stack((contours[:,1]+bbox[0],contours[:,0]+bbox[1]))
    
    return rc


def get_all_seed_points(img, seedRegions):    
    for seed in seedRegions:
        for level in range(seed.levelCount):
            block_size = level+2
            
            edge_pts = get_contour([seed.levelRows(level),seed.levelCols(level)], seed.bbox)
            
            seed_pts = sample_seed_points(img, edge_pts, seed.bbox, block_size)
            
            seed.addSeedPoints(seed_pts.astype('int32'))
        
    return seedRegions


def combine_regions(levels, somas):
    # The lowest level will have the largest regions. Start by adding somas
    lowestL = levels[-1]
    seedRegions = []
    
    for region in range(lowestL.regionCount):
        r_bbox = lowestL.getRegionBBox(region)
        
        seedRegion = SeedRegion(r_bbox)

        # More than one soma can fit inside a region
        for soma in somas:
            if seedRegion.overlaps(soma.bbox):
                seedRegion.addSoma(soma)
                
        if seedRegion.hasSomas():
            seedRegion.addLevel(lowestL.regionObject(region))
            seedRegions.append(seedRegion)
        
                
    # Go from second-lowest level to the top level
    for L in levels[-2::-1]:
        for region in range(L.regionCount):
            r_bbox = L.getRegionBBox(region)
            
            # If it matches a current known seed region, add it and go to 
            # the next level region
            for seed in seedRegions:
                if seed.overlaps(r_bbox):
                    seed.addLevel(L.regionObject(region))
                    break # There can only be one region this one overlaps by design
      
        
    return seedRegions
    

if __name__ == '__main__':
    img_fname = '/Users/jaclynbeck/Desktop/BaramLab/Substack (1).tif' #'/Users/jaclynbeck/Desktop/BaramLab/C2-8-29-17_CRH-tdTomato+CX3CR1-GFP P8 PVN CES_Female 1 L PVN_a_.i...CX3CR1-GFP P8 PVN CES_Female 1 L PVN_a_GREEN_t1.tif'
    output_img = '/Users/jaclynbeck/Desktop/BaramLab/somas.tif'
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
    
    mask = gm.create_Mask(images)
    
    images[mask == 0] = 0
    
    image_stack = sp.reshape(images, (1,images.shape[0], images.shape[1]))
    [soma_threshold, somas] = fs.find_somas(image_stack)
    
    #thresholds = pm.peaks_multithreshold(images, peak_threshold)
    #print(thresholds)
    [maxSig, thresholds] = omt.otsu_multithreshold(images, 20)
    print(thresholds)
    
    thresholds = thresholds[thresholds < soma_threshold] 
    
    thresholds = sp.concatenate((thresholds, [soma_threshold]))
    
    levels = tl.get_levels(images, thresholds, somas)
    
    seedRegions = combine_regions(levels, somas)
    Utils.plot_seed_regions(seedRegions, True)
    
    seedRegions = get_all_seed_points(images, seedRegions)
        
    elapsed = timeit.default_timer() - start_time
    print(elapsed)
    
    
    # For debugging and display
    print(levels)
    
    bw = Utils.plot_levels(levels, 1024, 1024, True)    
    sp.misc.imsave('/Users/jaclynbeck/Desktop/BaramLab/objects_' + str(thresholds.size-1) + '.tif', bw*255.0/bw.max())
    
    
    s_img = Utils.plot_somas(somas, True)
    sp.misc.imsave(output_img, s_img)
    
    cont_img = Utils.plot_seed_points(points, True)
    sp.misc.imsave('/Users/jaclynbeck/Desktop/BaramLab/contours.tif',cont_img)
    