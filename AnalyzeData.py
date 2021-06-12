#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 22:02:48 2017

@author: jaclynbeck
"""

import pickle
import timeit
import os
import numpy as np
    

def write_csv(data, numFrames, out_file):
    data = [d for d in data if len(d) > 1]
    data = np.array(data, dtype=object).T
    lines = ""
    
    # If this is a list of process data, it will have sublists within each
    # list and needs extra processing
    new_data = None
    if len(data.shape) == 1:
        for d in data:
            if len(d) == 0:
                continue
            
            stack = np.vstack(d).T
                
            if new_data is None:
                new_data = stack
            else:
                new_data = np.concatenate((new_data, stack), axis=1)
    
        data = new_data
    
    if data is not None:
        for d in data:
            lines += ",".join([str(n) for n in d]).replace("None","").replace("nan","") + "\n"
    
        csv = open(out_file, 'w')
        csv.writelines(lines)
        csv.close() 
    
    
def analyze_microglia(microglia_fname, metadata_fname):
    path = os.path.dirname(microglia_fname)
    raw_path = os.path.join(path, "raw_data")
    if not os.path.isdir(raw_path):
        os.mkdir(raw_path)
        
    microglia_csv_base = os.path.join(raw_path, "microglia_identity.csv")
    soma_csv_base = os.path.join(raw_path, "soma_")
    process_csv_base = os.path.join(raw_path, "process_")
    average_csv_base = os.path.join(raw_path, "average_")
    
    with open(microglia_fname, 'rb') as f:
        microglia = pickle.load(f)
        
    with open(metadata_fname, 'rb') as f:
        metadata = pickle.load(f)

    microglia_ids = []
    soma_dict = {}
    process_dict = {}
    average_dict = {}
    
    numFrames = int(np.ceil(30*60 / np.mean(metadata.timeDeltas))) # 30 minutes
    window_size = int(round(10*60 / np.mean(metadata.timeDeltas))) # 10 minutes
        
    
    for m in microglia:
        m.calculateSomaMovement(metadata, numFrames)
        m.calculateLeafData(metadata, numFrames)
        
        mID = microglia.index(m)
        idCsv, somaCsv, processCsv, averageCsv = m.dataToDict(mID, window_size)
        
        if idCsv is not None and somaCsv is not None \
            and processCsv is not None and averageCsv is not None:
            microglia_ids.append(idCsv)
                    
            for key, item in somaCsv.items():
                if key not in soma_dict:
                    soma_dict[key] = []
                soma_dict[key].append(item)
                
            for key, item in processCsv.items():
                if key not in process_dict:
                    process_dict[key] = []
                process_dict[key].append(item)
                
            for key, item in averageCsv.items():
                if key not in average_dict:
                    average_dict[key] = []
                average_dict[key].append(item)
    
    write_csv(microglia_ids, numFrames, microglia_csv_base)
    
    for key, item in soma_dict.items():
        write_csv(item, numFrames, soma_csv_base + key + ".csv")
        
    for key, item in process_dict.items():
        write_csv(item, numFrames, process_csv_base + key + ".csv")
        
    for key, item in average_dict.items():
        write_csv(item, numFrames, average_csv_base + key + ".csv")
        
        
def postprocess_data(microglia_fname, included_microglia):
    path = os.path.dirname(microglia_fname)
    raw_path = os.path.join(path, "raw_data")
    filtered_path = os.path.join(path, "filtered_data")
    if not os.path.isdir(filtered_path):
        os.mkdir(filtered_path)
        
    files = os.listdir(raw_path)
    
    for file in files:
        if file[0] != '.' and file[0] != '~' and file[-3:] == 'csv':
            with open(os.path.join(raw_path, file), 'r') as readfile:
                data = [line.split(',') for line in readfile.readlines()]
                row1 = [float(x) for x in data[0] if x != '\n']
                
                indices = []
                for r in range(len(row1)):
                    if int(row1[r]) in included_microglia:
                        indices.append(r)
                        
                data = [",".join(d) for d in np.array(data)[:,indices]]
                
                for d in range(len(data)):
                    if data[d][-1] != '\n':
                        data[d] += '\n'
                
                with open(os.path.join(filtered_path, file), 'w') as writefile:
                    writefile.writelines(data)
                    
                    

if __name__ == "__main__":
    metadata_fname  = '/mnt/storage/BaramLabFiles/7-20-17_CRH-tdTomato+CX3CR1-GFP P8 PVN CES/video_processing/7-20-17_CRH-tdTomato+CX3CR1-GFP P8 PVN CES_Male 3 L PVN T1_b_4D_Male 3 L PVN T1/img_metadata.p'
    microglia_fname = '/mnt/storage/BaramLabFiles/7-20-17_CRH-tdTomato+CX3CR1-GFP P8 PVN CES/video_processing/7-20-17_CRH-tdTomato+CX3CR1-GFP P8 PVN CES_Male 3 L PVN T1_b_4D_Male 3 L PVN T1/processed_microglia.p'
    
    start_time = timeit.default_timer()
    
    analyze_microglia(microglia_fname, metadata_fname)
    
    included_microglia = [1,3,6,8,9,10,11,12,13,14,15,16,17,18,19,22,23,25,31,33]
    postprocess_data(microglia_fname, included_microglia)
        
    elapsed = timeit.default_timer() - start_time
    print(elapsed)

    
