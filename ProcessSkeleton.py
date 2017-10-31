#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 16:04:32 2017

@author: jaclynbeck

The minimum spanning tree code is heavily based on code from 
https://github.com/jakevdp/mst_clustering , but without clustering
"""

import scipy as sp
import matplotlib.pyplot as plt
from libtiff import TIFF
import timeit
import cv2
from mst_clustering import MSTClustering # installed with 'pip install mst_clustering'
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from sklearn.neighbors import kneighbors_graph
from scipy import sparse


class SkeletonRegion(object):
    __slots__ = 'coordinates', 'tree'
    
    def __init__(self, coordinates):
        self.coordinates = coordinates
        
    def setTree(self, tree):
        self.tree = tree


class DirectedNode(object):
    __slots__ = 'value', 'coordinates', 'parent', 'children', 'length'
    
    def __init__(self, value, coordinates):
        self.value = value
        self.coordinates = coordinates
        self.parent = None
        self.children = []
        self.length = 0
        
    def addParent(self, node):
        self.parent = node
        self.length = self.parent.length + 1
        
        
    def checkForPruning(self, leafNode, previousNode):
        if len(self.children) > 1:
            if leafNode.length - self.length < 10:
                return previousNode
            else:
                return None
        
        if self.parent is not None:
            return self.parent.checkForPruning(leafNode, self)
        
        return None
        
        
    def printAsTree(self, tabLevel):
        s = str(self.value) + " (" + str(self.length) + ")\n"
        for c in self.children:
            s += "  "*tabLevel + " -> " + c.printAsTree(tabLevel+1)
            
        return s
        
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
    

class Tree(object):
    # Defining all the variables ahead of time with __slots__ helps with
    # memory management and makes access quicker
    __slots__ = 'nodes', 'coordinates', 'leaves', 'centerNode'
    
    def __init__(self): 
        self.nodes = {}
        self.coordinates = {}
        self.leaves = []
        
    def addEdge(self, node1, node2, coordinates):
        if node1 not in self.nodes:
            self.nodes[node1] = []
            self.coordinates[node1] = coordinates[0]
        
        if node2 not in self.nodes:
            self.nodes[node2] = []
            self.coordinates[node2] = coordinates[1]
            
        self.nodes[node1].append(node2)
        self.nodes[node2].append(node1)
        
    def hasNode(self, node):
        return node in self.nodes

    def makeDirected(self):
        max_node = 0
        max_connections = 0
        
        for node, vals in self.nodes.items():
            if len(vals) > max_connections:
                max_node = node
                max_connections = len(vals)
                
        self.centerNode = DirectedNode(max_node, self.coordinates[max_node]) # TODO this needs to be changed to the centroid of the soma
        self.trace(self.centerNode)
        
    
    def trace(self, node):
        children = self.nodes[node.value]
        if (len(children) == 1) and (node.parent is not None) \
            and (node.parent.value == children[0]):
                self.leaves.append(node)
                return
        
        for c in children:
            if node.parent is not None and node.parent.value == c:
                continue
            
            c_node = DirectedNode(c, self.coordinates[c])
            c_node.addParent(node)
            node.children.append(c_node)
            self.trace(c_node)

    
    def prune(self):
        leaves_to_remove = []
        for node in self.leaves:
            removed = node.checkForPruning(node, node)
            if removed is not None: 
                leaves_to_remove.append(node)
                self.deleteNode(removed)
        
        for node in leaves_to_remove:
            self.leaves.remove(node)
        
        
    def draw(self):
        coords = sp.array([C for C in self.coordinates.values()])
        mn = [min(coords[:,0]), min(coords[:,1])]
        mx = [max(coords[:,0]), max(coords[:,1])]
        height = mx[0]-mn[0]+1
        width = mx[1]-mn[1]+1
        
        bw = sp.zeros((height, width), dtype="uint8")
        bw[coords[:,0]-mn[0], coords[:,1]-mn[1]] = 1
        
        plt.imshow(bw); plt.show()
        
    
    def drawOnImage(self, bw):
        coords = sp.array([C for C in self.coordinates.values()])
        bw[coords[:,0], coords[:,1]] = 1
        return bw
            
                
    def deleteNode(self, node):
        for c in node.children:
            self.deleteNode(c)
            
        if node.parent is not None:
            node.parent.children.remove(node)
        
        self.coordinates.pop(node.value)
        vals = self.nodes.pop(node.value)
        for v in vals:
            self.nodes[v].remove(node.value)
            
        del node
            
 
       
def prune_lowest_layer(skeleton):
    lowest = sp.vstack(sp.where(skeleton == 1)).T
    higher = sp.vstack(sp.where(skeleton > 1)).T
    
    pruned = []
    for i in range(lowest.shape[0]):
        root = (higher-lowest[i,:])**2
        dist = sp.sqrt(root[:,0] + root[:,1])
        if dist.min() > 20:
            pruned.append(lowest[i,:])

    pruned = sp.array(pruned)
    skeleton[pruned[:,0], pruned[:,1]] = 0
    
    return skeleton
 
    
def skeleton_to_tree(skeleton):
    coords = sp.where(skeleton > 1)
    X = sp.vstack(coords).T

    #X = check_array(X)
    n_neighbors = 20
    G = kneighbors_graph(X, n_neighbors=n_neighbors,
                         mode='distance',
                         metric='euclidean',
                         metric_params=None)
    
    tree = minimum_spanning_tree(G, overwrite=True)
    tree[tree > 5] = 0
    tree.eliminate_zeros()
    N = sparse.coo_matrix(tree)
    
    n_components, labels = connected_components(tree,
                                                directed=False)
    trees = []

    for c in range(n_components):
        tree_obj = Tree()
        label_ind = sp.where(labels == c)[0]
        
        for i in label_ind:
            node_ind = sp.where((N.row == i))[0]
            for j in node_ind:
                pts = (X[N.row[j]], X[N.col[j]])
                tree_obj.addEdge(N.row[j], N.col[j], pts)

        trees.append(tree_obj)
    
    bw_orig = sp.zeros_like(skeleton)
    bw_pruned = sp.zeros_like(skeleton)
    
    for tree in trees:
        tree.makeDirected()
        tree.drawOnImage(bw_orig)
        #tree.draw()
        #print(tree.centerNode.printAsTree(0))
        tree.prune()
        tree.drawOnImage(bw_pruned)
        #print(tree.centerNode.printAsTree(0))
        #tree.draw()
        
    plt.imshow(bw_orig); plt.show()
    plt.imshow(bw_pruned); plt.show()
    
            

    
if __name__=='__main__':
    img_fname = "/Users/jaclynbeck/Desktop/BaramLab/skeletons_max_projection.tif"
    
    tif = TIFF.open(img_fname, mode='r')
    
    start_time = timeit.default_timer()
    index = 0
    
    for img in tif.iter_images(): 
        start_time = timeit.default_timer()
        regions = skeleton_to_tree(img)
        index += 1
        elapsed = timeit.default_timer() - start_time
        print(elapsed)

        
    elapsed = timeit.default_timer() - start_time
    print(elapsed)
    
    tif.close()
