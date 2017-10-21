#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 11:03:14 2017

@author: jaclynbeck
"""

import scipy as sp
import timeit
import matplotlib.pyplot as plt
from mst_clustering import MSTClustering # installed with 'pip install mst_clustering'


class Tree(object):
    # Defining all the variables ahead of time with __slots__ helps with
    # memory management and makes access quicker
    __slots__ = 'nodes', 'distances', 'edges'
    
    def __init__(self):
        self.nodes = {}
        self.distances = {}
        self.edges = [[],[]]
    
    def addNode(self, node):
        self.nodes[node] = []
        
    def addEdge(self, node1, node2):
        if not node1 in self.nodes:
            self.addNode(node1)
            
        if not node2 in self.nodes:
            self.addNode(node2)
            
        self.nodes[node1].append(node2)
        self.nodes[node2].append(node1)
        
    def edgesAsArray(self):
        return [sp.array(self.edges[0]), sp.array(self.edges[1])]
    
    def plottableEdges(self):
        return tuple(sp.vstack(arrs) for arrs in zip(sp.array(self.edges[0]).T,
                                                     sp.array(self.edges[1]).T))
    
    def prune(self):
        nodes_to_prune = [N for N in self.nodes if len(self.nodes[N]) == 1]
        for node in nodes_to_prune:
            node2 = self.nodes[node][0]
            if len(self.nodes[node2]) > 2:
                self.nodes.pop(node, None)
                self.nodes[node2].remove(node)
                
        self.createEdges()
        
        
    def createEdges(self):
        for node1 in self.nodes:
            for node2 in self.nodes[node1]:
                self.edges[0].append(node1)
                self.edges[1].append(node2)
                
        # TODO this does create duplicate edges
            
    
        

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
    

def find_closest_point(point1, point_set, distances):
    closest_point = None
    min_dist = 9999
    
    if point1 not in distances:
        distances[point1] = {}
        
    for p in point_set:
        # TODO is searching a potentially thousands-long list any faster than
        # recalculating the distance every time? Maybe make the dict outside
        # this function
        if p not in distances[point1]:
            dist = abs(point1[0]-p[0]) + abs(point1[1]-p[1])
            distances[point1][p] = dist
        else:
            dist = distances[point1][p]
            
        if dist < min_dist:
            min_dist = dist
            closest_point = p
            
    return [min_dist, closest_point]
    

def build_tree(tree, seeds):
    point_pair = (None, None)
    min_dist = 9999
    
    for t in tree.nodes:
        [dist, point] = find_closest_point(t, seeds, tree.distances)
        if dist < min_dist:
            min_dist = dist
            point_pair = (t, point)
            
    tree.addEdge(point_pair[0], point_pair[1])
    seeds.remove(point_pair[1])
    
    if len(seeds) == 0:
        return tree
    
    return build_tree(tree, seeds)


def init_tree(seeds):
    tree = Tree()
    tree.addNode(seeds[0])
    seeds.remove(seeds[0])
    return tree


def do_mst(seed_pts, distance):
    model = MSTClustering(cutoff_scale=2, approximate=True, metric="precomputed")
    labels = model.fit_predict(distance)
    
    G = sp.sparse.coo_matrix(model.full_tree_)
    
    node1 = seed_pts[G.row]
    node2 = seed_pts[G.col]
    
    tree = Tree()
    for n1, n2 in zip(node1, node2):
        tree.addEdge(tuple(n1), tuple(n2))
    
    tree.prune()
    #tree.createEdges()
    
    return tree.plottableEdges()
    
          
def build_distance_matrix(seeds):
    numSeeds = len(seeds)
    distance = sp.zeros((numSeeds,numSeeds))
    
    # Fill upper triangle
    for i in range(numSeeds):
        for j in sp.arange(i+1,numSeeds):
            distance[i,j] = sum(abs(seeds[i]-seeds[j])) # geodesic distance
    
    # Fill lower triangle
    for j in range(numSeeds):
        for i in sp.arange(j+1,numSeeds):
            distance[i,j] = distance[j,i]
            
    return distance
        
        
    
    
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
    
    #skeleton = skeletonize(model)
    #coords = skeleton[0].skeletonCoords()
    plot_mst(model)

