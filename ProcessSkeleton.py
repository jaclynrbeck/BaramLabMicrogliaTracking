#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 16:04:32 2017

@author: jaclynbeck

The minimum spanning tree code is based on code from 
https://github.com/jakevdp/mst_clustering , but without clustering and some
extra modifications to account for somas. 
"""

import scipy as sp
from libtiff import TIFF
import timeit
import cv2
from scipy.sparse.csgraph import minimum_spanning_tree, dijkstra
from sklearn.neighbors import kneighbors_graph
from scipy import sparse
import numpy as np
import FindSomas as fs
import pickle
import os
from Objects import DirectedNode, DirectedTree, Microglia
        

"""
This function ensures that no two somas are connected by a path through the 
minimum spanning tree. Dijkstra's algorithm is used to find the shortest path
from each soma to any other point, and if there is a path between two somas,
that path is broken in the middle. 

Input: 
    tree_csr - compressed sparse matrix generated by minimum_spanning_tree
    soma_indices - list of indices into the tree array that correspond to
                   the soma centroids. 
    
Output:
    tree_csr - the modified sparse matrix tree
"""
def split_somas(tree_csr, soma_indices):
    dist, points = dijkstra(tree_csr, directed=False, 
                            indices=soma_indices, 
                            return_predecessors=True)
    
    # Using a linked-list format sparse matrix is faster for this task
    lil_tree = tree_csr.tolil()
    
    # For each soma, check if there's a path between it and another soma
    for i in range(len(soma_indices)-1):
        for j in sp.arange(i+1, len(soma_indices)):
            soma1 = soma_indices[i]
            soma2 = soma_indices[j]
            
            # If a path exists, trace it and break it at the longest gap
            if dist[i,soma2] != sp.inf and dist[i, soma2] > 0:
                path = []
                p = soma2
                while p != soma1:
                    distance = max(lil_tree[p, points[i,p]], lil_tree[points[i,p], p])
                    # Sanity check. It's possible to have already broken this
                    # path if, for example, 3 somas are connected in a line
                    if distance == 0:
                        path = []
                        break
                    
                    path.append((sp.round_(distance, 1), p))
                    p = points[i,p] # This points us to the next node in the path
            
            
                if len(path) > 0:
                    arr = sp.array(path)
                    index = sp.where(arr[:,0] == arr[:,0].max())[0]
                    break_pt = sorted(index)[int(round(len(index)/2))]
            
                    # This effectively deletes the break point from the tree
                    if break_pt < len(arr)-1:
                        lil_tree[arr[break_pt,1], arr[break_pt+1,1]] = 0
                        lil_tree[arr[break_pt+1,1], arr[break_pt,1]] = 0
    
    tree_csr = lil_tree.tocsr()
    tree_csr.eliminate_zeros()
    
    return tree_csr
    
    
"""
Extracts soma centroids and contours from the information encoded in the 
skeleton image. Centroids have pixel values of 255 and body pixels have values
of 254. 

Input:
    skeleton - MxN ndarray, grayscale skeleton image
    
Output:
    somas - list of FrameSoma objects
"""
def extract_soma_information(skeleton):
    bw = skeleton.copy()
    bw[skeleton < 254] = 0  # Only get somas
    number, labels, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=8)
    
    somas = []
    
    for i in sp.arange(1, number):
        soma_coords = sp.vstack(sp.where(labels == i)).T
        soma = fs.FrameSoma(0, soma_coords)
        somas.append(soma)
        
    return somas


"""
Translates the sparse graph representation of the tree into directed graphs.

Input: 
    tree_csr - compressed sparse matrix generated by minimum_spanning_tree
    X - Nx3 ndarray, all (row, col, value) coordinates in the skeleton image 
    centroid_indices - indices into X of the soma centroids
    
Output:
    directedTrees - list of DirectedTree objects, one per soma
"""
def create_directed_trees(tree_csr, X, centroid_indices):
    N = sparse.coo_matrix(tree_csr)  # For easier indexing into the tree
    
    directedTrees = []
    for c in centroid_indices:
        centerNode = DirectedNode(c, X[c][0:2])
        dTree = DirectedTree(centerNode)
        stack = [centerNode]
        
        while len(stack) > 0:
            node = stack.pop()
            connections = list(N.col[N.row == node.value]) + list(N.row[N.col == node.value])
            
            # Anything that is only connected to the soma center is on the
            # soma border and needs to be thrown out
            if (len(connections) == 1) and (node.parent is centerNode):
                centerNode.children.remove(node)
                continue;
                
            dTree.nodes.append(node)
            
            # If this node has only one connection, and that connection is this
            # node's parent, this node is a leaf. 
            if (len(connections) == 1) and (node.parent is not None) \
                and (node.parent.value == connections[0]):
                    dTree.leaves.append(node)
                    continue
            
            # Otherwise for each connection that isn't this node's parent, add
            # it to the stack for processing
            for conn in connections:
                if node.parent is not None and node.parent.value == conn:
                    continue
                
                c_node = DirectedNode(conn, X[conn][0:2])
                c_node.addParent(node)
                node.children.append(c_node)
                stack.append(c_node)
    
        directedTrees.append(dTree)
    
    return directedTrees

   
def skeleton_to_tree(skeleton, img, somas):    
    # Make sure soma centroids are marked   
    for soma in somas:
        skeleton[soma.centroid[0], soma.centroid[1]] = 255
        
    coords = sp.where((skeleton > 1)) 
    
    # Here we are adding a 3rd coordinate representing the color of the pixel,
    # so that the MST algorithm will preferentially connect brighter-colored
    # pixels together. Values are scaled between 0 and 1 and flipped so that
    # 0 = high color (high confidence that the skeleton is correct there) and 
    # 1 = low color (low confidence), so that during distance calculations
    # the high colors look closer. 
    values = skeleton[coords[0], coords[1]]-skeleton[skeleton > 1].min()
    values = 1 - values / values.max()
    X = sp.vstack((coords[0], coords[1], values)).T
    
    # Get a list of IDs that correspond to the soma centroids, for referencing
    # each tree's center node. 
    centroid_indices = []
    for soma in somas:
        index = sp.where((X[:,0] == soma.centroid[0]) & (X[:,1] == soma.centroid[1]))[0]
        centroid_indices.append(index[0])

    # Create a nearest neighbors graph to pass to minimum_spanning_tree
    n_neighbors = 100
    G = kneighbors_graph(X, n_neighbors=n_neighbors,
                         mode='distance',
                         metric='euclidean',
                         metric_params=None)
        
    # Edit the nearest neighbors graph to artificially force the centroid and 
    # contour points to look close together. MST will connect these first
    # before looking for other points.
    lil_G = G.tolil() # Linked list format is faster for editing specific indices 
    for soma in somas:
        centroid_index = centroid_indices[somas.index(soma)]
        
        for c in soma.contour: 
            contour_index = sp.where((X[:,0] == c[0]) & (X[:,1] == c[1]))[0]
            lil_G[centroid_index, contour_index] = 0.1
            lil_G[contour_index, centroid_index] = 0.1
    
    G = lil_G.tocsr()
    
    # Get the minimum spanning tree of the whole image
    tree_csr = minimum_spanning_tree(G, overwrite=True)
    
    # Break any connections that don't have adjacent pixels
    tree_csr[tree_csr > 50] = 0
    tree_csr.eliminate_zeros()
    
    # Break connections between somas
    tree_csr = split_somas(tree_csr, centroid_indices)
    
    directedTrees = create_directed_trees(tree_csr, X, centroid_indices)
    return directedTrees


def match_trees(videoSomas, trees):
    videoMicroglia = []
    
    for v in videoSomas:
        microglia = Microglia(v)
        for f in v.frames:            
            frameSoma  = v.frameSomas[f]
            frameTrees = trees[f]
            
            match = None
            for t in frameTrees:
                if np.all(t.centerNode.coordinates == frameSoma.centroid):
                    match = t
                    break
            
            if match is not None:
                microglia.addTreeAtFrame(match, f)
        
        microglia.matchLeaves()        
        videoMicroglia.append(microglia)
        
    return videoMicroglia


"""
Main method for this file. Translates all skeleton images in a video into
Tree objects, which are directed graphs centered around a soma centroid. Then
it tracks trees across frames and calculates the statistics for each resulting
microglia. 
"""
def process_skeleton(skeleton_fname, img_fname, metadata_fname, soma_fname, microglia_fname):
    skel_tif = TIFF.open(skeleton_fname, mode='r')
    img_tif  = TIFF.open(img_fname, mode='r')
    
    path = os.path.dirname(skeleton_fname)
    metadata_fname = os.path.join(path, metadata_fname)
    soma_fname = os.path.join(path, soma_fname)
    microglia_fname = os.path.join(path, microglia_fname)
    
    with open(soma_fname, 'rb') as f:
        videoSomas = pickle.load(f)

    frame = 0
    directedTrees = {}
    
    for skel, img in zip(skel_tif.iter_images(), img_tif.iter_images()): 
        somas = []
        for v in videoSomas:
            if frame in v.frames: somas.append(v.frameSomas[frame])
            
        trees = skeleton_to_tree(skel, img, somas)
        directedTrees[frame] = trees
        frame += 1
        
    skel_tif.close()
    img_tif.close()
    
    # Prune short branches
    for frame in directedTrees:
        for dT in directedTrees[frame]:
            dT.prune()
    
    microglia = match_trees(videoSomas, directedTrees)
        
    with open(microglia_fname, 'wb') as f:
        pickle.dump(microglia, f)

        

if __name__=='__main__':
    img_fname = "/Users/jaclynbeck/Desktop/BaramLab/videos/10-06-17_CRH-tdTomato+CX3CR1-GFP P8 PVN CES_Female 1 L PVN T2_b_4D_Female 2 L PVN T2.ims"

    
    path = os.path.dirname(img_fname)
    path = os.path.join(path, "video_processing", os.path.basename(img_fname)[0:-4])
    metadata_fname = os.path.join(path, 'img_metadata.p')
    soma_fname = os.path.join(path, 'somas.p')
    microglia_fname = os.path.join(path, 'processed_microglia.p')
    tree_fname = os.path.join(path, 'directedTrees_pruned_new.p')
    
    start_time = timeit.default_timer()
    
    #process_skeleton(skeleton_fname, img_fname, metadata_fname, soma_fname, microglia_fname)
    
    with open(soma_fname, 'rb') as f:
        videoSomas = pickle.load(f)
        
    with open(tree_fname, 'rb') as f:
        directedTrees = pickle.load(f)
        
    microglia = match_trees(videoSomas, directedTrees)
    
    with open(microglia_fname, 'wb') as f:
        pickle.dump(microglia, f)
    
    elapsed = timeit.default_timer() - start_time
    print(elapsed)


