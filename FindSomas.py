#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 13:12:56 2017

@author: jaclynbeck
"""

import scipy as sp
import skimage.measure as skm
import skimage.filters as skf
from libtiff import TIFFfile
import matplotlib.pyplot as plt
import timeit


"""
This class represents a soma in a single frame
"""
class FrameSoma:
    MIN_SOMA_RADIUS = 20.0
    MIN_SOMA_SIZE = MIN_SOMA_RADIUS*MIN_SOMA_RADIUS
    
    def __init__(self, frameNum, coords, bbox):
        self.frameNum = frameNum
        self.coordinates = coords
        self.bbox = bbox
        self.centroid = sp.round_(sp.array( (sp.mean(self.coordinates[:,0]), 
                                             sp.mean(self.coordinates[:,1])) ))
        
    def rows(self):
        return self.coordinates[:,0]
    
    def cols(self):
        return self.coordinates[:,1]
    
    def centerRow(self):
        return int(self.centroid[0])
    
    def centerCol(self):
        return int(self.centroid[1])
    
    def bboxArea(self):
        return (self.bbox[2]-self.bbox[0]) * (self.bbox[3]-self.bbox[1])
    
    
    """
    This is what will get printed out when using print(frame) or during 
    debugging. 
    """
    def __repr__(self):
        s = "Frame:\t" + str(self.frameNum) + "\n" + \
            "Centroid: " + str(int(self.centerRow())) + ", " + \
            str(int(self.centerCol())) + "\n" + \
            "Box:\t" + str(self.bbox)
            
        return s


"""
This class represents a soma as a 3D object across multiple frames  
"""
class Soma3D:
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
        

def label_objects(bw, frame):
    labels = skm.label(bw, background=False, connectivity=2)
    props = skm.regionprops(labels)
    counts, edges = sp.histogram(labels, labels.max()+1)
    
    somas = []
    
    valid = sp.where(counts > FrameSoma.MIN_SOMA_SIZE)
    
    # Create a soma object for each object that is larger than the minimum size
    for v in valid[0]:
        if v == 0: # Ignore the 'background' label, which will always be 0
            continue

        somas.append(FrameSoma(frame, props[v-1].coords, props[v-1].bbox))
    
    return somas

        

def find_somas(img_fname, output_img):
    tiff = TIFFfile(img_fname)
    samples, sample_names = tiff.get_samples()

    threshold = skf.threshold_otsu(samples[0], samples[0].max()+1)
    
    somas = []
    
    for s in range(samples[0].shape[0]):
        img = samples[0][s]
        [width, height] = img.shape
    
        bw = sp.zeros_like(img, dtype='uint8')
        bw[img > threshold] = 255

        somas += label_objects(bw, s)
        
    somas_final = match_somas(somas)        
    return [threshold, somas_final]
    


if __name__ == '__main__':
    img_fname = '/Users/jaclynbeck/Desktop/BaramLab/C2-8-29-17_CRH-tdTomato+CX3CR1-GFP P8 PVN CES_Female 1 L PVN_a_.i...CX3CR1-GFP P8 PVN CES_Female 1 L PVN_a_GREEN_DENOISED_t1.tif'
    output_img = '/Users/jaclynbeck/Desktop/BaramLab/somas.tif'
    
    start_time = timeit.default_timer()
    
    [threshold, somas] = find_somas(img_fname, output_img)
    
    elapsed = timeit.default_timer() - start_time
    print(elapsed)
    
    # For debugging and display
    s_img = sp.zeros((1024,1024))
    for soma in somas:
        s_img[soma.rows(), soma.cols()] = soma.getId()+10
        
    s_img = sp.array(s_img * (255.0 / s_img.max()), dtype='uint8')
    plt.imshow(s_img)
    plt.show()
    
    sp.misc.imsave(output_img, s_img)
    