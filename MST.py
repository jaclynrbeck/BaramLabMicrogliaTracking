#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 11:03:14 2017

@author: jaclynbeck
"""

import scipy as sp
import timeit
import matplotlib.pyplot as plt
from mst_clustering import MSTClustering


class Node:
    def __init__(self, point):
        self.point = point
        self.children = []
        self.parent = None
            
    def addChild(self, p1, p2):
        if sp.all(self.point == p1):
            n2 = Node(p2)
            n2.parent = self
            self.children.append(n2)
            
            return True
        
        for c in self.children:
            if c.addChild(p1, p2):
                return True
            
            if sp.all(c.point == p2):
                return c.addChild(p2,p1)
            
        return False
    
    
    def addParent(self, p1, p2):
        if sp.all(self.point == p2):
            self.parent = Node(p2)
            self.parent.children.append(self)
            return True
        
        if self.parent is not None:
            return self.parent.addParent(p1, p2) or self.parent.addChild(p1,p2)

    def merge(self, node):
        pass #

    def length(self):
        if self.parent is None:
            return 0
        
        return 1 + self.parent.length()
    
    def getRoot(self):
        if self.parent is None:
            return self
        
        return self.parent.getRoot()
    
    def prune(self):
        if len(self.children) > 1:
            self.children[:] = [x for x in self.children if len(x.children) > 0]
            
        for c in self.children:
            c.prune()
            
    def skeletonCoords(self):
        coords = self.point
        
        for c in self.children:
            childCoords = c.skeletonCoords()
            coords = sp.column_stack((coords,childCoords))
            
        return coords
    
    
    def printSkeleton(self, tabLevel):
        s = tabLevel*"\t" + str(self.point) + " (" + str(self.length()) + ")"
        
        if len(self.children) > 0:
            s += " -> \n"
            for c in self.children:
                s += c.printSkeleton(tabLevel+1)
            
        return s + "\n"
        
            
    def __eq__(self, other): 
        if type(other) is Node:
            return sp.all(self.point == other.point)
        else:
            return sp.all(self.point == other)
    
            
    """
    This is what will get printed out when using print() or during debugging. 
    """
    def __repr__(self):
        s = "Node: " + str(self.point) + "\n"
        if self.parent is None:
            s += "Parent: X\n"
        else:
            s += "Parent: " + str(self.parent.point) + "\n"
            
        s += "Children: "
        for c in self.children:
            s += str(c.point) + ", "
            
        return s + "\n"
    
        

def plot_mst(model, cmap='rainbow'):
    """Utility code to visualize a minimum spanning tree"""
    X = model.X_fit_
    fig, ax = plt.subplots(2, 1, figsize=(6,16), sharex=True, sharey=True)
    for axi, full_graph, colors in zip(ax, [True, False], ['lightblue', model.labels_]):
        segments = model.get_graph_segments(full_graph=full_graph)
        axi.plot(segments[0], segments[1], '-k', zorder=1, lw=1)
        #axi.scatter(X[:, 0], X[:, 1], c=colors, cmap=cmap, zorder=2)
        axi.axis('tight')
    
    ax[0].set_title('Full Minimum Spanning Tree', size=16)
    ax[1].set_title('Trimmed Minimum Spanning Tree', size=16);
    

def skeletonize(model):
    (segX,segY) = model.get_graph_segments(full_graph=False)
    numSegments = segX.shape[1]
    
    skeleton = []
    
    for i in range(1000): #numSegments):
        p1 = sp.array([segX[0,i],segY[0,i]])
        p2 = sp.array([segX[1,i],segY[1,i]])
        
        found = False
        
        for s in skeleton:
            if s.addChild(p1,p2):
                found = True
                break
                
            elif s.addParent(p1, p2):
                found = True
                break
        
        if found == False:
            n1 = Node(p1)
            n1.addChild(p1,p2)
            skeleton.append(n1)
            
        for i in range(len(skeleton)):
            skeleton[i] = skeleton[i].getRoot()  
            
    for i in sp.arange(len(skeleton)-1,0,-1):
        if skeleton[i].merge(skeleton[i-1]):
            skeleton[i-1] = skeleton[i-1].getRoot()
            skeleton.remove(skeleton[i])
    
    for s in skeleton:
        s.prune()
        
    return skeleton
        
          
def build_distance_matrix(seeds):
    numSeeds = seeds.shape[0]
    distance = sp.zeros((numSeeds,numSeeds))
    
    # Fill upper triangle
    for i in range(numSeeds):
        for j in sp.arange(i+1,numSeeds):
            distance[i,j] = sum(abs(seeds[i]-seeds[j])) # geodesic distance
            
    distance
        
        
    
    
if __name__ == '__main__':
    img = sp.misc.imread('/Users/jaclynbeck/Desktop/BaramLab/contours.tif') #'/Users/jaclynbeck/Desktop/BaramLab/objects_12_pm_roi.tif')
    
    start_time = timeit.default_timer()
    
    x,y = sp.meshgrid(range(img.shape[0]),range(img.shape[1]))
    
    x = x[img > 0]
    y = y[img > 0]
    
    xy = sp.column_stack((x.flatten(),y.flatten()))
    #xyz = sp.column_stack((x.flatten(), y.flatten(), img.flatten()))
    
    #build_distance_matrix(xy) # Takes too long
    model = MSTClustering(cutoff_scale=2, approximate=True)
    labels = model.fit_predict(xy)
    
    elapsed = timeit.default_timer() - start_time
    print(elapsed)
    
    skeleton = skeletonize(model)
    coords = skeleton[0].skeletonCoords()
    #plot_mst(model)


# TODO need to relate regions to each other with the Microglia object holding 
# all the seed points. Distance matrix only needs to be for that region
