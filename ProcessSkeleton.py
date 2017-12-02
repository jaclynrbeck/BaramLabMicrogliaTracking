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
import matplotlib.pyplot as plt
from libtiff import TIFF
import timeit
import cv2
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components, dijkstra
from sklearn.neighbors import kneighbors_graph
from scipy import sparse
import FindSomas as fs
import Utils
import pickle
import os
        

"""
This class represents a node in a directed graph.

Class variables:
    value       - Integer ID (index into the tree returned by mst)
    coordinates - (3,) ndarray of image coordinates (row, col, px val)
    parent      - DirectedNode that is the parent of this node
    children    - List of DirectedNodes that are children of this node
    length      - Number of nodes between this node and the center node of the 
                  tree. This is not physical distance. 
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
        self.coordinates = coordinates
        self.parent = None
        self.children = []
        self.length = 0
    
    """
    Sets this node's parent node
    """    
    def addParent(self, node):
        self.parent = node
        self.length = self.parent.length + sp.sqrt(sp.sum((self.parent.coordinates[0:2]-self.coordinates[0:2])**2))
        
    
    """
    Checks to see if the given branch in the tree should be pruned. A branch
    is pruned if the length (number of nodes) from leaf to root of branch is
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
This class represents a minimum spanning tree found from the skeleton in the 
image. It is undirected.

Class variables:
    nodes       - Dictionary where the keys are the node IDs and the values
                  are a list of other node IDs that each node is connected to
    coordinates - Dictionary where the keys are the node IDs and the values are
                  the real image coordinates corresponding to that node ID
    leaves      - List of DirectedNodes that are at the end of each branch
    centerNode  - DirectedNode that is the parent of all other branches. This
                  corresponds to the soma centroid if this tree "belongs" to a
                  cell. 
"""
class UndirectedTree(object):
    # Defining all the variables ahead of time with __slots__ helps with
    # memory management and makes access quicker
    __slots__ = 'nodes', 'coordinates', 'leaves', 'centerNode'
    
    
    """
    Initialization
    """
    def __init__(self): 
        self.nodes = {}
        self.coordinates = {}
        self.leaves = []
        self.centerNode = None
    
    """
    Adds an edge to the node dictionary
    
    Input:
        node1 - Integer value for node 1 (index into the tree returned by mst)
        node2 - Integer value for node 2 (index into the tree returned by mst)
        coords1 - Real image coordinates for node 1
        coords2 - Real image coordinates for node 2
    """    
    def addEdge(self, node1, node2, coords1, coords2):
        if node1 not in self.nodes:
            self.nodes[node1] = []
            self.coordinates[node1] = coords1
        
        if node2 not in self.nodes:
            self.nodes[node2] = []
            self.coordinates[node2] = coords2
            
        self.nodes[node1].append(node2)
        self.nodes[node2].append(node1)
    
    
    """
    Shortcut for checking whether this tree contains a given node
    
    Input: 
        node - the node ID to check for
        
    Output:
        True or False
    """
    def hasNode(self, node):
        return node in self.nodes

    """
    Sets the tree's center (parent) node in preparation for making it a 
    directed graph. Usually the center node corresponds to a soma centroid. 
    
    Input: 
        nodeNum - node ID of the center node
    """
    def setCenterNode(self, nodeNum):
        self.centerNode = DirectedNode(nodeNum, self.coordinates[nodeNum])
        
        
    """
    For debugging. This will draw the tree on an image that is the same size 
    as the spread of the tree. 
    """
    def draw(self):
        coords = sp.array([C for C in self.coordinates.values()])
        coords = coords[:,0:2].astype('int16')
        mn = [min(coords[:,0]), min(coords[:,1])]
        mx = [max(coords[:,0]), max(coords[:,1])]
        height = mx[0]-mn[0]+1
        width = mx[1]-mn[1]+1
        
        bw = sp.zeros((height, width), dtype="uint8")
        bw[coords[:,0]-mn[0], coords[:,1]-mn[1]] = 1
        
        plt.imshow(bw); plt.show()
        
    
    """
    For debugging. This will draw the tree on the full image containing all
    trees. 
    
    Input:
        bw - the image to draw the tree onto
    """
    def drawOnImage(self, bw):
        for key, vals in self.nodes.items():
            pt1 = self.coordinates[key]
            
            for v in vals:
                pt2 = self.coordinates[v]
                cv2.line(bw, (int(pt1[1]), int(pt1[0])), (int(pt2[1]), int(pt2[0])), 255)

        return bw
    
    
    
class DirectedTree(object):
    __slots__ = 'nodes', 'leaves', 'centerNode'
    
    def __init__(self, tree):
        self.nodes = []
        self.leaves = []
        self.centerNode = tree.centerNode
    
        self.makeDirected(tree)

    """
    Turns the undirected tree into a directed tree by following edges out 
    from the center node to the leaves. 
    """    
    def makeDirected(self, uTree):
        # If no center node was set, this tree wasn't associated with a soma
        # but we will trace it anyway. The center node is set as the node
        # with the most connections to other nodes. 
        if self.centerNode is None:
            max_node = 0
            max_connections = 0
            
            for node, vals in uTree.nodes.items():
                if len(vals) > max_connections:
                    max_node = node
                    max_connections = len(vals)
            
            self.centerNode = DirectedNode(max_node, uTree.coordinates[max_node])
        
        # Trace outward from the center node
        self.trace(uTree, self.centerNode)
        
    
    """
    Follows the given node's path through the tree by tracing through each
    child. 
    
    This is a recursive function. 
    
    Input:
        node - DirectedNode being inspected
    """
    def trace(self, uTree, node):
        self.nodes.append(node)
        connections = uTree.nodes[node.value]
        
        # If this node has only one connection, and that connection is this
        # node's parent, this node is a leaf. Stop recursion. 
        if (len(connections) == 1) and (node.parent is not None) \
            and (node.parent.value == connections[0]):
                self.leaves.append(node)
                return
        
        # Otherwise for each connection that isn't this node's parent, trace
        # that connection recursively. 
        for c in connections:
            if node.parent is not None and node.parent.value == c:
                continue
            
            c_node = DirectedNode(c, uTree.coordinates[c])
            c_node.addParent(node)
            node.children.append(c_node)
            self.trace(uTree, c_node)

    
    """
    Removes short branches (< 10 nodes long) from the tree, which are likely to
    be noise or artifacts from skeletonization. 
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
        
            
    def __getstate__(self):
        stateDict = {'nodes': self.nodes,
                     'leaves': self.leaves, 
                     'centerNode': self.centerNode}
        
        for node in stateDict['nodes']:
            if node.parent is not None:
                node.parent = stateDict['nodes'].index(node.parent)
                
            if len(node.children) > 0:
                new_children = []
                for c in node.children:
                    new_children.append(stateDict['nodes'].index(c))
                node.children = new_children

        return stateDict
    
    def __setstate__(self, stateDict):
        self.nodes = stateDict['nodes']
        self.leaves = stateDict['leaves']
        self.centerNode = stateDict['centerNode']
        
        for node in self.nodes:
            if node.parent is not None:
                node.parent = self.nodes[node.parent]
            
            if len(node.children) > 0:
                new_children = []
                for c in node.children:
                    new_children.append(self.nodes[c])
                    
                node.children = new_children
        
    
class TreeSeries(object):
    __slots__ = 'frames', 'trees'
    
    def __init__(self, frame, tree):
        self.frames = [frame]
        self.trees = {frame: tree}
        
    def addTree(self, frame, tree):
        self.frames.append(frame)
        self.trees[frame] = tree
        
        
    def isMatch(self, frame, tree):
        if frame in self.frames:
            return (False, -1)
        
        lastFrame = sorted(self.frames)[-1]
        if frame - lastFrame >= 3:
            return (False, -1)
        
        lastTree = self.trees[lastFrame]
        
        coords1 = set([tuple(C[:2]) for n, C in lastTree.coordinates.items()])
        coords2 = set([tuple(C[:2]) for n, C in tree.coordinates.items()])
        
        intersect = set.intersection(coords1, coords2)
        if len(intersect) > 0:
            return (True, len(intersect))
        
        return (False, -1)
    
    
    def distanceTo(self, mainTree):
        for f, tree in self.trees.items():
            dists = Utils.neighbors_graph(tree.coordinates, mainTree.coordinates)
            
                
    def __lt__(self, other):
        return len(self.trees) < len(other.trees)
        
    
    
        

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
    
def split_tree(tree_csr, X, img):
    threshold = sp.percentile(img, 20)
    N = sparse.coo_matrix(tree_csr)  # For easier indexing into the tree
    
    # Using a linked-list format sparse matrix is faster for this task
    lil_tree = tree_csr.tolil()
    
    for p1, p2 in zip(N.row, N.col):
        if lil_tree[p1,p2] < 2:
            continue
        
        point1 = X[p1]
        point2 = X[p2]
        
        if int(point1[0]) == int(point2[0]):
            sign = sp.sign(point2[1]-point1[1])
            cols = sp.arange(point1[1], point2[1], sign).astype('int16') 
            rows = sp.full(cols.shape, point1[0].astype('int16'))
            
        else:
            m = float(point1[1]-point2[1]) / float(point1[0]-point2[0])
            b = sp.mean( [point1[1]-m*point1[0], point2[1]-m*point2[0]] )
        
            sign = sp.sign(point2[0]-point1[0])
            rows = sp.arange(point1[0], point2[0], sign).astype('int16')
            cols = sp.round_(m*rows + b).astype('int16')
        
        for r,c in zip(rows, cols):
            if img[r,c] < threshold:
                lil_tree[p1,p2] = 0
                lil_tree[p2,p1] = 0
                break
    
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
Translates the sparse graph representation of the tree into Tree objects.

Input: 
    tree_csr - compressed sparse matrix generated by minimum_spanning_tree
    X - Nx3 ndarray, all (row, col, value) coordinates in the skeleton image 
    centroid_indices - indices into X of the soma centroids
    
Output:
    mainTrees - list of Tree objects that have somas assigned to them
    orphans - list of Tree objects that do not have somas assigned to them
"""
def create_undirected_trees(tree_csr, X, centroid_indices):
    N = sparse.coo_matrix(tree_csr)  # For easier indexing into the tree
    
    n_components, labels = connected_components(tree_csr, directed=False)
    trees = []

    for c in range(n_components):
        tree_obj = UndirectedTree()
        label_ind = sp.where(labels == c)[0]
        
        for i in label_ind:
            node_ind = sp.where((N.row == i))[0]
            for j in node_ind:
                pt1 = X[N.row[j]]
                pt2 = X[N.col[j]]

                tree_obj.addEdge(N.row[j], N.col[j], pt1, pt2)

        if len(tree_obj.nodes) > 0:
            trees.append(tree_obj)
    
    # For any trees that touch a soma, set the soma centroid as the tree center
    for mst in trees:
        for c in centroid_indices:
            if mst.hasNode(c):
                mst.setCenterNode(c) 
    
    # Now there should be one tree per soma
    mainTrees = [T for T in trees if T.centerNode is not None]
    orphans = [T for T in trees if T.centerNode is None]
    
    return mainTrees, orphans
 

   
def skeleton_to_tree(skeleton, img, somas):
    # Get rid of the soma bodies, leaving only the contour
    for soma in somas:
        skeleton[soma.rows(), soma.cols()] = 0
        skeleton[soma.contourRows(), soma.contourCols()] = 254
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
        centroid_index = sp.where((X[:,0] == soma.centroid[0]) & (X[:,1] == soma.centroid[1]))[0]
        
        for c in soma.contour: 
            contour_index = sp.where((X[:,0] == c[0]) & (X[:,1] == c[1]))[0]
            lil_G[centroid_index, contour_index] = 0.1
            lil_G[contour_index, centroid_index] = 0.1
    
    G = lil_G.tocsr()
    
    # Get the minimum spanning tree of the whole image
    tree_csr = minimum_spanning_tree(G, overwrite=True)

    #tree_csr = split_tree(tree_csr, X, img)
    
    # Break any connections that don't have adjacent pixels
    tree_csr[tree_csr > 50] = 0
    tree_csr.eliminate_zeros()
    
    # Break connections between somas
    tree_csr = split_somas(tree_csr, centroid_indices)
    
    mainTrees, orphans = create_undirected_trees(tree_csr, X, centroid_indices)
    
    return mainTrees, orphans


def match_orphans(orphans):
    treeSeries = []
    
    for f, trees in orphans.items():
        for orphan in trees:
            matches = []
            for t in treeSeries:
                isMatch, overlap = t.isMatch(f, orphan)
                if isMatch:
                    matches.append((overlap, t))
                    
            if len(matches) == 0:
                treeSeries.append(TreeSeries(f, orphan))
                
            elif len(matches) == 1:
                matches[0][1].addTree(f, orphan)
                
            else:
                matches.sort()
                matches[-1][1].addTree(f, orphan)
                
    return treeSeries
    

def combine_orphan_trees(mainTrees, orphans):
    treeSeries = match_orphans(orphans) # TODO we need to do this on the actual image, not the skeleton data
    
    for tree in treeSeries:
        touching = []
        
        for main in mainTrees:
            (touches, dist) = tree.distanceTo(mainTrees[main])
            if touches:
                touching.append((dist, main))
                
    

"""
Main method for this file. Translates all skeleton images in a video into
Tree objects, which are directed graphs centered around a soma centroid. 
"""
def process_skeleton(skeleton_fname, img_fname, soma_fname, tree_fname):
    skel_tif = TIFF.open(skeleton_fname, mode='r')
    img_tif  = TIFF.open(img_fname, mode='r')
    
    path = os.path.dirname(skeleton_fname)
    soma_fname = path + "/" + soma_fname
    tree_fname = path + "/" + tree_fname
    
    with open(soma_fname, 'rb') as f:
        videoSomas = pickle.load(f)

    frame = 0
    mainTrees = {}
    orphanTrees = {}
    
    for skel, img in zip(skel_tif.iter_images(), img_tif.iter_images()): 
        somas = []
        for v in videoSomas:
            if frame in v.frames: somas.append(v.frameSomas[frame])
            
        trees, orphans = skeleton_to_tree(skel, img, somas)
        mainTrees[frame] = trees
        orphanTrees[frame] = orphans
        frame += 1
        
    skel_tif.close()
    img_tif.close()
    
    #mainTrees = combine_orphan_trees(mainTrees, orphanTrees)
    
    directedTrees = {}
    # Turn the trees into directed graphs and prune short branches
    for frame in mainTrees:
        directedTrees[frame] = []
        for T in mainTrees[frame]:
            dT = DirectedTree(T)
            dT.prune()
            directedTrees[frame].append(dT)
    
    with open(tree_fname, 'wb') as f:
        pickle.dump(directedTrees, f)
        
    return directedTrees, orphanTrees

        

if __name__=='__main__':
    skeleton_fname = "/Users/jaclynbeck/Desktop/BaramLab/videos/A_LPVN_T1_08202017/processed_skeleton_max_projection_deconvolved_iter5.tif"
    img_fname = "/Users/jaclynbeck/Desktop/BaramLab/videos/A_LPVN_T1_08202017/processed_max_projection_deconvolved_5iter.tif"
    soma_fname = "/Users/jaclynbeck/Desktop/BaramLab/videos/A_LPVN_T1_08202017/processed_somas.p"
    tree_fname = "/Users/jaclynbeck/Desktop/BaramLab/videos/A_LPVN_T1_08202017/tree_output.p"
    start_time = timeit.default_timer()
    
    trees, orphans = process_skeleton(skeleton_fname, img_fname, soma_fname, tree_fname)
        
    elapsed = timeit.default_timer() - start_time
    print(elapsed)


