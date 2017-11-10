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
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components, dijkstra
from sklearn.neighbors import kneighbors_graph
from scipy import sparse
import FindSomas as fs
        

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
        self.length = self.parent.length + 1
        
    
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
image. It starts as undirected but then is made to be a directed graph once
all edges are added. 

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
class Tree(object):
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
    Turns the undirected tree into a directed tree by following edges out 
    from the center node to the leaves. 
    """    
    def makeDirected(self):
        # If no center node was set, this tree wasn't associated with a soma
        # but we will trace it anyway. The center node is set as the node
        # with the most connections to other nodes. 
        if self.centerNode is None:
            max_node = 0
            max_connections = 0
            
            for node, vals in self.nodes.items():
                if len(vals) > max_connections:
                    max_node = node
                    max_connections = len(vals)
            
            self.centerNode = DirectedNode(max_node, self.coordinates[max_node])
        
        # Trace outward from the center node
        self.trace(self.centerNode)
        
    
    """
    Follows the given node's path through the tree by tracing through each
    child. 
    
    This is a recursive function. 
    
    Input:
        node - DirectedNode being inspected
    """
    def trace(self, node):
        connections = self.nodes[node.value]
        
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
            
            c_node = DirectedNode(c, self.coordinates[c])
            c_node.addParent(node)
            node.children.append(c_node)
            self.trace(c_node)

    
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
        
        # Remove this node from the coordinates list and remove references to 
        # it in the list of nodes
        self.coordinates.pop(node.value)
        vals = self.nodes.pop(node.value)
        for v in vals:
            self.nodes[v].remove(node.value)
        
        # Finally, delete the object to free up memory
        del node
        
        
    """
    For debugging. This will draw the tree on an image that is the same size 
    as the spread of the tree. 
    """
    def draw(self):
        coords = sp.array([C for C in self.coordinates.values()])
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
            
            # If a path exists, trace it and break it in the middle
            if dist[i,soma2] != sp.inf and dist[i, soma2] > 0:
                path = []
                p = soma2
                while p != soma1:
                    # Sanity check. It's possible to have already broken this
                    # path if, for example, 3 somas are connected in a line
                    if (lil_tree[p, points[i,p]] == 0) and (lil_tree[points[i,p], p] == 0):
                        path = []
                        break
                    
                    path.append(p)
                    p = points[i,p] # This points us to the next node in the path
            
            
                if len(path) > 0:
                    break_pt = int(len(path)/2) # TODO better/smarter break point?
            
                    # This effectively deletes the break point from the tree
                    lil_tree[path[break_pt], path[break_pt-1]] = 0
                    lil_tree[path[break_pt-1], path[break_pt]] = 0
                    lil_tree[path[break_pt], path[break_pt+1]] = 0
                    lil_tree[path[break_pt+1], path[break_pt]] = 0
    
    tree_csr = lil_tree.tocsr()
    tree_csr.eliminate_zeros()
    
    return tree_csr
    

"""
Extracts soma centroids and contours from the information encoded in the 
skeleton image. Centroids have pixel values of 255 and contours have pixel
values of 200. 

Input:
    skeleton - MxN ndarray, grayscale skeleton image
    
Output:
    somas - list of FrameSoma objects
"""
def extract_soma_information(skeleton):
    bw = skeleton.copy()
    bw[skeleton != 200] = 0     # Only get contours
    number, labels, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=8)
    
    img_centroids = sp.vstack(sp.where(skeleton == 255)).T
    
    somas = []
    
    for i in sp.arange(1, number):
        contour_ij = sp.vstack(sp.where(labels == i)).T
        
        # This FrameSoma isn't really complete. We are using the contour as
        # its internal coordinates as well as its contour coordinates, but for
        # the purposes of processing the skeleton that's all we need. 
        soma = fs.FrameSoma(0, contour_ij, contour=contour_ij)
        
        # Manually set the soma centroid to the point marked in the image, 
        # since we did not give the FrameSoma enough coordinates to calculate
        # it accurately. 
        # "centroids[i]" needs to be reversed because it's in terms of x,y 
        # instead of row, col order. 
        dist = sp.sqrt(sp.sum((img_centroids - centroids[i][::-1])**2, axis=1))
        soma.centroid = img_centroids[sp.argmin(dist)]
        somas.append(soma)
        
    return somas


"""
Translates the sparse graph representation of the tree into Tree objects.

Input: 
    tree_csr - compressed sparse matrix generated by minimum_spanning_tree
    X - Nx3 ndarray, all (row, col, value) coordinates in the skeleton image 
    centroid_indices - indices into X of the soma centroids
    
Output:
    trees - list of Tree objects
"""
def create_undirected_trees(tree_csr, X, centroid_indices):
    N = sparse.coo_matrix(tree_csr)  # For easier indexing into the tree
    
    n_components, labels = connected_components(tree_csr, directed=False)
    trees = []

    for c in range(n_components):
        tree_obj = Tree()
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
    trees = [T for T in trees if T.centerNode is not None]
    
    return trees
 

   
def skeleton_to_tree(skeleton, somas=None):   
    if somas is None:
        somas = extract_soma_information(skeleton)
    
    coords = sp.where((skeleton > 1)) 
    
    # Here we are adding a 3rd coordinate representing the color of the pixel,
    # so that the MST algorithm will preferentially connect brighter-colored
    # pixels together. Values are scaled between 0 and 1 and flipped so that
    # 0 = high color (high confidence that the skeleton is correct there) and 
    # 1 = low color (low confidence), so that during distance calculations
    # the high colors look closer. 
    values = skeleton[coords[0], coords[1]]-skeleton[skeleton > 1].min()
    values = 1-values / values.max()
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
    
    # Break any connections that don't have adjacent pixels
    tree_csr[tree_csr > 2] = 0
    tree_csr.eliminate_zeros()
    
    # Break connections between somas
    tree_csr = split_somas(tree_csr, centroid_indices)
    
    trees = create_undirected_trees(tree_csr, X, centroid_indices)
    
    #bw_orig = sp.zeros_like(skeleton)
    #bw_pruned = sp.zeros_like(skeleton)
    
    # Turn the trees into directed graphs and prune short branches
    for T in trees:
        T.makeDirected()
        #tree.drawOnImage(bw_orig)
        #print(tree.centerNode.printAsTree(0))
        T.prune()
        #tree.drawOnImage(bw_pruned)
        #print(tree.centerNode.printAsTree(0))
        
    #plt.imshow(bw_orig); plt.show()
    #plt.imshow(bw_pruned); plt.show()
    
    return trees
    
            

    
if __name__=='__main__':
    skeleton_fname = "/Users/jaclynbeck/Desktop/BaramLab/tst2.tif"
    img_fname = "/Users/jaclynbeck/Desktop/BaramLab/averaged_max_projection.tif"        
    
    img_tif = TIFF.open(img_fname, mode='r')
    skel_tif = TIFF.open(skeleton_fname, mode='r')
    
    start_time = timeit.default_timer()
    frame = 0
    
    trees = {}
    
    for skel in skel_tif.iter_images(): 
        #start_time = timeit.default_timer()
        trees[frame] = skeleton_to_tree(skel)
        frame += 1
        #elapsed = timeit.default_timer() - start_time
        #print(elapsed)
        
    elapsed = timeit.default_timer() - start_time
    print(elapsed)

    skel_tif.close()
