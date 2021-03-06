#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 15:19:03 2017

@author: jaclynbeck
"""

import cv2
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse


"""
Contrast Limited Adaptive Histogram equalization to increase contrast
"""
def equalize_img(img):
    if img.dtype != 'uint8':
        img = (img * 255.0 / img.max()).astype('uint8')

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(img)

    return cl1 


"""
Denoise using a bilateral filter to preserve edges
"""
def denoise_img(img):
    img = sp.array(img, dtype=sp.float32)   
    return cv2.bilateralFilter(img, 5, 75, 75)


"""
Equalizes img histogram, denoises the image, and does a preliminary
adaptive thresholding to remove low-value pixels
"""
def preprocess_img(img):
    img = equalize_img(img)
    img = denoise_img(img).astype('uint8')
    
    return img
    

def bbox_has_overlap(bbox1, bbox2):
    if bbox1[2] < bbox2[0] or bbox1[3] < bbox2[1]:
        return False
        
    if bbox1[0] > bbox2[2] or bbox1[1] > bbox2[3]:
        return False
        
    return True


def bbox_overlap_area(bbox1, bbox2):
    if not bbox_has_overlap(bbox1, bbox2):
        return 0
    
    xmin = max(bbox1[0], bbox2[0])
    ymin = max(bbox1[1], bbox2[1])
    xmax = min(bbox1[2], bbox2[2])
    ymax = min(bbox1[3], bbox2[3])
    
    return (xmax-xmin)*(ymax-ymin)


def bbox_area(bbox):
    return (bbox[2]-bbox[0])*(bbox[3]-bbox[1])


def bbox_significant_overlap(bbox1, bbox2, overlap_percent):
    overlap = bbox_overlap_area(bbox1, bbox2)
        
    if overlap == 0:
        return False
    
    # Overlap needs to be significant
    if (overlap / bbox_area(bbox1) > overlap_percent) \
        or (overlap / bbox_area(bbox2) > overlap_percent):
            return True
        
    return False


def distance(p1, p2):
    return sp.sqrt(sp.sum((p2 - p1)**2, axis=1))
    

def neighbors_graph(points1, points2):
    dist_matrix = sp.zeros((len(points1), len(points2)))
    
    for p in range(len(points1)):
        dist_matrix[p,:] = distance(points1[p], points2)
        
    return dist_matrix


def find_matches(points1, points2):
    matches = []
    dist_matrix = neighbors_graph(points1, points2)
        
    closest1 = sp.argmin(dist_matrix, axis=1)
    closest2 = sp.argmin(dist_matrix, axis=0)
    
    for c in range(len(closest1)):
        if closest2[closest1[c]] == c:
            matches.append((c, closest1[c], dist_matrix[c,closest1[c]]))
        
    return matches
    
    
def nearest_neighbor(point, other_points):
    dist = distance(point, other_points)
    closest_index = sp.argmin(dist)
    
    return (other_points[closest_index], dist[closest_index], closest_index)

def depth_search(graph, index, parent, visited):
    visited[index] = max(visited[index], 1)

    dists = np.vstack(sparse.find(graph[index]))
    inds = np.where(dists[2] < 2)[0]
    children = dists[1, inds].astype('int')
    
    cycle = []
    
    # No children = no cycle
    if len(children) == 0:
        return []
    
    for child in children:
        # Don't traverse backward
        if child == parent:
            continue
        
        # We already found a cycle containing the child node
        if visited[child] == 2:
            found = [Cyc for Cyc in cycle if index in Cyc[0] and child in Cyc[0]]
            
            # Already explored this path
            if len(found) > 0:
                continue
            
            # Else fall through to next condition
        
        # There is a cycle including this node
        if visited[child] > 0:
            cycle.append(([child, index], True))
            visited[child] = 2
            visited[index] = 2
            
        else:
            child_cycle = depth_search(graph, child, index, visited)
            
            for CC in range(len(child_cycle)):
                entry = child_cycle[CC]
                if entry[1] == True:
                    if index in entry[0]:
                        entry = (entry[0], False)
                        child_cycle[CC] = entry
                    else:
                        entry[0].append(index)
                    visited[index] = 2
                
            cycle.extend(child_cycle)
    
    return cycle
    
    
def find_cycles(graph, coordinates):
    visited = [0] * len(coordinates)
    cycles = []
    
    for C in range(len(coordinates)):
        if visited[C] > 0:
            continue
        
        cycle = depth_search(graph, C, None, visited)
        cycles.extend(cycle)
        
    return cycles
        

    
def plot_levels(levels, img_size, display=False):
    bw = sp.zeros(img_size, dtype='uint16')
    
    for i in range(len(levels)):
        L = levels[len(levels)-i-1]
        
        if L.isBackgroundLevel():
            color = 0
        else:
            color = L.threshold+1
        
        #bw[L.rows(),L.cols()] = color
        for region in L.regions:
            bw[region.rows(), region.cols()] = color 
    
    if display:
        plt.imshow(bw*255.0/bw.max())
        plt.show()
    
    return bw


def plot_somas(somas, img_size, display=False):
    s_img = sp.zeros(img_size, dtype='uint8')
    color = 0
    for soma in somas:
        s_img[soma.rows(), soma.cols()] = 20 + 5*color
        color += 1
        
    s_img = sp.array(s_img * (255.0 / s_img.max()), dtype='uint8')
    
    if display:
        plt.imshow(s_img)
        plt.show()
    
    return s_img


def plot_seed_regions(seedRegions, img_size, display=False):
    s_img = sp.zeros(img_size, dtype='uint8')
    
    for seed in seedRegions:
        s_img[seed.rows(), seed.cols()] = 128
        for soma in seed.somas:
            s_img[soma.rows(),soma.cols()] = 255
        
    s_img = sp.array(s_img * (255.0 / s_img.max()), dtype='uint8')
    
    if display:
        plt.imshow(s_img)
        plt.show()
    
    return s_img
    


def plot_seed_points(seedRegions, img_size, display=False):
    bw = sp.zeros(img_size, dtype='uint8')
    
    for seed in seedRegions:
        for pts in seed.seedPoints:
            bw[pts[:,0],pts[:,1]] = 255
    
    if display:
        plt.imshow(bw)
        plt.show()
        
    return bw


def plot_cell_regions(regions, somas, img_size, display=False):
    bw = sp.zeros(img_size, dtype='uint8')
    
    for region in regions:
        bw[region.rows(), region.cols()] = 128
    
    for soma in somas:
        bw[soma.rows(), soma.cols()] = 255
    
    if display:
        plt.imshow(bw)
        plt.show()
        
    return bw


def plot_skeleton(regions, somas, img_size, display=False):
    bw = sp.zeros(img_size, dtype='uint8')
    
    for region in regions:
        bw[region.skeletonRows(), region.skeletonCols()] = 255
        
    for soma in somas:
        bw[soma.rows(), soma.cols()] = 255
    
    if display:
        plt.imshow(bw)
        plt.show()
        
    return bw


"""
Input:
    tree_csr - sparse matrix in compressed sparse row format 
                (output from minimum_spanning_tree)
    points - array of actual coordinates in the image that make up the nodes
             in tree_csr. See variable "X" in skeleton_to_tree
    img_size - tuple of the image size (i.e. (1024,1024))
    display - whether to plot or not
"""
def plot_tree_csr(tree_csr, points, img_size, display=False):
    bw = np.zeros(img_size, dtype='uint8')
    N = sparse.coo_matrix(tree_csr)
    
    X = points[:,0:2]
    X = X[:,::-1]
    
    for p1, p2 in zip(N.row, N.col): 
        bw = cv2.line(bw, tuple(X[p1].astype('int16')), tuple(X[p2].astype('int16')), 255)
        
    if display:
        plt.imshow(bw)
        plt.show()
        
    return bw