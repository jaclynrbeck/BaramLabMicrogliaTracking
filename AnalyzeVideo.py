#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 22:02:48 2017

@author: jaclynbeck
"""

import ProcessSkeleton as ps
import Utils
import scipy as sp
import cv2

class ProcessTip(object):
    __slots__ = 'tips', 'tipIDs', 'directionX', 'directionY', 'magnitude', \
                'length'
    
    def __init__(self, tipID1, tipID2, frame1, frame2, leaf1, leaf2):
        self.tipIDs = [tipID1, tipID2]
        self.tips = {frame1: leaf1, frame2: leaf2}

    def addFrame(self, tipID, frame, leaf):
        self.tipIDs.append(tipID)
        self.tips[frame] = leaf
        
    def getFrames(self):
        return [f for f in self.tips.keys()]
        
    def calculateData(self, numFrames):
        self.directionX = sp.full((numFrames,), None)
        self.directionY = sp.full((numFrames,), None)
        self.magnitude  = sp.full((numFrames,), None)
        self.length = sp.full((numFrames,), None)
    
        frames = [k for k in self.tips.keys()]
        self.length[0] = self.tips[frames[0]].length
        
        for i in range(len(frames)-1):
            f1 = frames[i]
            f2 = frames[i+1]

            tip1 = self.tips[f1]
            tip2 = self.tips[f2]
            
            self.length[f2] = tip2.length
                
            direction = tip2.coordinates[0:2] - tip1.coordinates[0:2]
            self.directionX[f2] = direction[1]
            self.directionY[f2] = direction[0]
            self.magnitude[f2]  = sp.sqrt(sp.sum(direction**2))    
            
    
    
class Microglia(object):
    __slots__ = 'trees', 'somas', 'processTips', 'somaDirectionX', \
                'somaDirectionY', 'somaMagnitude', 'somaArea', \
                'domainArea', 'processTips', 'numberOfProcesses', \
                'numberOfMainBranches'
    
    def __init__(self, somas, trees, initial_tree, tree_relations):
        self.somas = somas
        self.processTips = []
        
        frame, index = initial_tree
        self.trees = {frame: trees[frame][index]}
        
        for relation in tree_relations:
            frame, index = relation
            self.trees[frame] = trees[frame][index]
            
        self.matchLeaves()
            
            
    def calculateSomaMovement(self, numFrames):
        self.somaDirectionX = sp.full((numFrames,), None)
        self.somaDirectionY = sp.full((numFrames,), None)
        self.somaMagnitude  = sp.full((numFrames,), None)
        self.somaArea = sp.full((numFrames,), None)
        
        frames = [k for k in self.trees.keys()]
        for i in range(len(frames)-1):
            f1 = frames[i]
            f2 = frames[i+1]
            c1 = self.trees[f1].centerNode.coordinates
            c2 = self.trees[f2].centerNode.coordinates
        
            direction = c2[0:2] - c1[0:2]
            self.somaDirectionX[f2] = direction[1]
            self.somaDirectionY[f2] = direction[0]
            self.somaMagnitude[f2] = sp.sqrt(sp.sum(direction**2))
            #self.somaArea[f2] = cv2.contourArea(soma.contour)
            
            
    def matchLeaves(self):
        frames = [k for k in self.trees.keys()]
        
        for i in range(len(frames)-1):
            f1 = frames[i]
            f2 = frames[i+1]

            leaves1 = self.trees[f1].leaves
            leaves2 = self.trees[f2].leaves
            
            coords1 = [L.coordinates for L in leaves1]
            coords2 = [L.coordinates for L in leaves2]
            
            matches = Utils.find_matches(coords1, coords2)
            
            for m in matches:
                found = False
                for p in self.processTips:
                    if (f1, m[0]) in p.tipIDs:
                        p.addFrame((f2, m[1]), f2, leaves2[m[1]])
                        found = True
                        break
                    
                if found == False: 
                    self.processTips.append(ProcessTip((f1, m[0]), (f2, m[1]), 
                                                       f1, f2, 
                                                       leaves1[m[0]], 
                                                       leaves2[m[1]]))
                    
    
    def calculateLeafData(self, numFrames):
        self.domainArea = sp.full((numFrames,), None)
        self.numberOfProcesses = sp.zeros((numFrames,))
        self.numberOfMainBranches = sp.full((numFrames,), None)
        
        for p in self.processTips:
            p.calculateData(numFrames)
            frames = p.getFrames()
            self.numberOfProcesses[frames] += 1
        
        self.numberOfProcesses[self.numberOfProcesses == 0] = None
        
        for frame, tree in self.trees.items():
            coords = sp.array([C[0:2] for key, C in tree.coordinates.items()])
            hull = cv2.convexHull(coords.astype('int32'))
            self.domainArea[frame] = cv2.contourArea(hull)
            self.numberOfMainBranches[frame] = len(tree.centerNode.children)
        
        
    def dataToCsv(self):
        valid = [v for v in self.somaDirectionX if v is not None]
        if len(valid) < 30: # Exclude microglia that aren't in 30 or more frames
            return None, None
        
        somaData = {"DirectionX": self.arrayToCsvString(self.somaDirectionX), 
                    "DirectionY": self.arrayToCsvString(self.somaDirectionY), 
                    "Magnitude": self.arrayToCsvString(self.somaMagnitude),
                    "Area": self.arrayToCsvString(self.somaArea)}
        
        processData = {"DirectionX": "", "DirectionY": "", 
                       "Magnitude": "", "Length": "", "DomainArea": "",
                       "NumberOfProcesses": "", "NumberOfMainBranches": ""}
        
        for p in self.processTips:
            valid = [v for v in p.directionX if v is not None]
            if len(valid) < 5: # Exclude tips that aren't in 5 or more frames
                continue
            
            if len(processData["DirectionX"]) > 0:
                processData["DirectionX"] += '\n,'
                processData["DirectionY"] += '\n,'
                processData["Magnitude"]  += '\n,'
                processData["Length"] += '\n,'
                
            processData["DirectionX"] += self.arrayToCsvString(p.directionX)
            processData["DirectionY"] += self.arrayToCsvString(p.directionY)
            processData["Magnitude"] += self.arrayToCsvString(p.magnitude)
            processData["Length"] += self.arrayToCsvString(p.length)
        
        
        processData["DomainArea"] = self.arrayToCsvString(self.domainArea)
        processData["NumberOfProcesses"] = self.arrayToCsvString(self.numberOfProcesses)
        processData["NumberOfMainBranches"] = self.arrayToCsvString(self.numberOfMainBranches)
            
        return somaData, processData
    
    
    def arrayToCsvString(self, arr):
        string = ""
        for a in arr:
            if a is None:
                string += ","
            else:
                string += str(a) + ","
            
        return string
                

def match_trees(centroids):
    tree_relations = {}
    
    for i in range(len(centroids)-1):
        c1 = centroids[i]
        c2 = centroids[i+1]
        
        matches = Utils.find_matches(c1, c2)
        
        for m in matches:
            found = False
            #trees[i][m[0]].draw()
            #trees[i+1][m[1]].draw()
            for key, items in tree_relations.items():
                if (i,m[0]) in items:
                    tree_relations[key].append((i+1,m[1]))
                    found = True
                    break
                
            if found == False:    
                tree_relations[(i,m[0])] = [(i+1,m[1])]
                    
    return tree_relations


def write_csv(data, numFrames, out_file):
    csv = open(out_file, 'w')
    
    str_dict = {}
    str_dict["heading"] = "Frame," + ",".join([str(i) for i in range(numFrames)])
    
    for d in data:
        for key, item in d.items():
            if key not in str_dict:
                str_dict[key] = key 
            
            str_dict[key] += "," + item + "\n"
                    
    for key, items in str_dict.items():
        csv.writelines(items + "\n\n\n\n")
        
    csv.close() 


if __name__ == "__main__":
    skeleton_fname = "/Users/jaclynbeck/Desktop/BaramLab/videos/D_LPVN_T1_06132017/processed_skeleton_max_projection.tif"
    output_dir = "/Users/jaclynbeck/Desktop/BaramLab/videos/D_LPVN_T1_06132017/"
    trees = ps.process_skeleton(skeleton_fname)
    
    numFrames = len(trees)
    centroids = []
    
    for i in range(numFrames):
        centroids.append([T.centerNode.coordinates for T in trees[i]])
        
    tree_relations = match_trees(centroids)
    
    microglia = []
    somaData = []
    processData = []
    for key, item in tree_relations.items():
        microglia.append(Microglia(None, trees, key, item))
        
    for m in microglia:
        m.calculateSomaMovement(numFrames)
        m.calculateLeafData(numFrames)
        
        somaCsv, processCsv = m.dataToCsv()
        if somaCsv is not None and processCsv is not None:
            somaData.append(somaCsv)
            processData.append(processCsv)
    
    write_csv(somaData, numFrames, output_dir + "soma_data.csv")
    write_csv(processData, numFrames, output_dir + "process_data.csv")
    
