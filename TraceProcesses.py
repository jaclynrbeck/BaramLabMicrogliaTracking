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
import skimage.measure as skm
from libtiff import TIFFfile
import matplotlib.pyplot as plt
import PeaksMultithreshold as pm
import FindSomas as fs
import timeit


class Level:
    
    MIN_SOMA_SIZE = 20*20 # 10x10 region
    
    def __init__(self, k, threshold, indices):
        self.k = k
        self.threshold = round(threshold)
        self.labelledRegions = []
        self.somaLevel = False
        self.backgroundLevel = False
        self.dimensions = len(indices)
        
        if self.dimensions == 2:
            rows = sp.reshape(indices[0], (indices[0].size, 1))
            cols = sp.reshape(indices[1], (indices[1].size, 1))
            self.coordinates = sp.concatenate((rows, cols), axis=1);
        else:
            z = sp.reshape(indices[0], (indices[0].size,1))
            rows = sp.reshape(indices[1], (indices[1].size, 1))
            cols = sp.reshape(indices[2], (indices[2].size, 1))
            self.coordinates = sp.concatenate((z,rows, cols), axis=1);
            
        self.count = rows.size
        
    def z(self):
        if self.dimensions == 2:
            return 0
        
        return self.coordinates[:,0]
    
    def rows(self):
        if self.dimensions == 2:
            return self.coordinates[:,0]
        
        return self.coordinates[:,1]
    
    def cols(self):
        if self.dimensions == 2:
            return self.coordinates[:,1]
        
        return self.coordinates[:,2]
        
    def addLabelledRegion(self, coords):            
        self.labelledRegions.append(coords)
        
    def regionCount(self):
        return len(self.labelledRegions)
   
    def regionZ(self, region):
        if self.dimensions == 2:
            return 0
        
        return self.labelledRegions[region][:,0]
    
    def regionRows(self, region):
        if self.dimensions == 2:
            return self.labelledRegions[region][:,0]
        
        return self.labelledRegions[region][:,1]
    
    def regionCols(self, region):
        if self.dimensions == 2:
            return self.labelledRegions[region][:,1]
        
        return self.labelledRegions[region][:,2]
    
    def setSomaLevel(self):
        self.somaLevel = True
        
    def isSomaLevel(self):
        return self.somaLevel
    
    def setBackgroundLevel(self):
        self.backgroundLevel = True
        
    def isBackgroundLevel(self):
        return self.backgroundLevel
        
    
    """
    'Less than' function for sorting by k-index
    """
    def __lt__(self, other):
         return self.k < other.k
        
        
    """
    This is what will get printed out when using print(level) or during 
    debugging. 'thresh: count <soma/background if applicable>'
    """
    def __repr__(self):
        s = "k:\t\t" + str(self.k)
        if self.somaLevel:
            s += " (Soma)"
        elif self.backgroundLevel:
            s += " (Background)"
            
        s += "\n"
        s += "Threshold:\t" + str(int(self.threshold)) + "\n" + \
             "Pixels:\t\t" + str(self.count) + "\n" + \
             "Regions:\t" + str(self.regionCount()) + "\n\n"
            
        return s
    

def join_one_level(bw, level):
    if level.dimensions == 2:
        bw[level.rows(), level.cols()] = 255
    else:
        bw[level.z(), level.rows(), level.cols()] = 255
    
    labels = skm.label(bw, background=False, connectivity=1)
    props = skm.regionprops(labels)
    counts, edges = sp.histogram(labels, labels.max()+1)
    
    valid = sp.where(counts > level.MIN_SOMA_SIZE)
    
    # Record the valid labelled regions in the level object.
    # Black out any valid regions while leaving invalid regions in the pixel
    # map for the next level. 
    for v in valid[0]:
        if v == 0:
            continue
        
        level.addLabelledRegion(props[v-1].coords)
        
        #bw[props[v-1].coords[:,0], props[v-1].coords[:,1]] = 0
        
        # for debugging/display
        #labels[labels == v] = 255
        
    #plt.imshow(bw)
    #plt.show()
    
    #plt.imshow(labels[0]*255/(labels[0].max()+1)) # Avoid divide by zero
    #plt.show()
    
    return valid[0].size-1


def join_objects(img, levels):
    bw = sp.zeros_like(img, dtype='uint8')
    
    index = 0
    objects_added = sp.zeros((len(levels),))
    
    for L in levels:
        objects_added[index] = join_one_level(bw, L)
        index += 1

    return sum(objects_added)

        

def get_levels(images, thresholds, slice_of_interest):
    if len(images.shape) == 3:
        [z, width, height] = images.shape
    else:
        [width, height] = images.shape
        z = 1
    
    if thresholds.size == 1:
        thresholds = sp.array(thresholds, ndmin=1)
    
    thresholds = sp.concatenate(([-1], thresholds, [images.max()+1]))
    
    levels = []
    for k in range(thresholds.size-1):
        indices = sp.where((images <= thresholds[k+1]) & (images > thresholds[k]))
        levels.append(Level(k+1, thresholds[k+1], indices))
    
    levels.sort(reverse=True) # Start at highest threshold and go to lowest
    
    total_objects = join_objects(images, levels)
    
    for i in sp.arange(1,len(levels)-1):
        if (levels[i].regionCount() < levels[i-1].regionCount()) and \
            (levels[i].regionCount() >= levels[i+1].regionCount()):
                levels[i].setBackgroundLevel()

    levels[-1].setBackgroundLevel()
    print(levels)     
    
    # debuging
    bw = sp.zeros((z, width, height), dtype='uint8')
    #color = 0;
    
    for i in range(len(levels)):
        L = levels[len(levels)-i-1]
        
        if L.isBackgroundLevel():
            color = 0
        else:
            color = 20+i*5
        
        for r in range(L.regionCount()):
            bw[L.regionZ(r), L.regionRows(r), L.regionCols(r)] = color
    
    bw = bw * (255/bw.max())    
    plt.imshow(bw[slice_of_interest,:,:])
    plt.show()
    
    sp.misc.imsave('/Users/jaclynbeck/Desktop/BaramLab/objects_' + str(thresholds.size-2) + '.tif', bw[slice_of_interest,:,:])
    
    


if __name__ == '__main__':
    img_fname = '/Users/jaclynbeck/Desktop/BaramLab/Substack (1).tif' #'/Users/jaclynbeck/Desktop/BaramLab/C2-8-29-17_CRH-tdTomato+CX3CR1-GFP P8 PVN CES_Female 1 L PVN_a_.i...CX3CR1-GFP P8 PVN CES_Female 1 L PVN_a_GREEN_t1.tif'
    output_img = '/Users/jaclynbeck/Desktop/BaramLab/somas.tif'
    peak_threshold = 100
    is_stack = False
    slice_of_interest = 8 # For debugging. This slice's objects will be shown as a b/w image
    
    if is_stack == False:
        slice_of_interest = 0
        
    #[threshold, somas] = fs.find_somas(img_fname, output_img)
    
    tiff = TIFFfile(img_fname)
    samples, sample_names = tiff.get_samples()
    
    if is_stack:
        images = samples[0]
    else:
        images = samples[0][0]

    start_time = timeit.default_timer()
    
    thresholds = pm.peaks_multithreshold(images, peak_threshold)
    print(thresholds)

    
    get_levels(images, thresholds, slice_of_interest)
        
    elapsed = timeit.default_timer() - start_time
    print(elapsed)
    
    