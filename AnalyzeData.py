#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 22:02:48 2017

@author: jaclynbeck
"""

import pickle
import timeit
import os
import scipy as sp
    

def write_csv(data, numFrames, out_file):
    csv = open(out_file, 'w')
    
    data = sp.array(data).T
    lines = ""
    
    # If this is a list of process data, it will have sublists within each
    # list and needs extra processing
    new_data = None
    if len(data.shape) == 1:
        for d in data:
            stack = sp.vstack(d).T
            if new_data is None:
                new_data = stack
            else:
                new_data = sp.concatenate((new_data, stack), axis=1)
    
        data = new_data
    
    for d in data:
        lines += ",".join([str(n) for n in d]).replace("None","").replace("nan","0") + "\n"
    
    csv.writelines(lines)
        
    csv.close() 
    
    
def analyze_microglia(microglia_fname, metadata_fname):
    path = os.path.dirname(microglia_fname)
    raw_path = path + "/raw_data"
    if not os.path.isdir(raw_path):
        os.mkdir(raw_path)
        
    microglia_csv_base = raw_path + "/microglia_identity.csv"
    soma_csv_base = raw_path + "/soma_"
    process_csv_base = raw_path + "/process_"
    
    with open(microglia_fname, 'rb') as f:
        microglia = pickle.load(f)
        
    with open(metadata_fname, 'rb') as f:
        metadata = pickle.load(f)

    microglia_ids = []
    soma_dict = {}
    process_dict = {}
    
    numFrames = len(metadata.imgTimes)
        
    for m in microglia:
        m.calculateSomaMovement(metadata, numFrames)
        m.calculateLeafData(metadata, numFrames)
        
        mID = microglia.index(m)
        idCsv, somaCsv, processCsv = m.dataToDict(mID)
        
        if idCsv is not None and somaCsv is not None and processCsv is not None:
            microglia_ids.append(idCsv)
                    
            for key, item in somaCsv.items():
                if key not in soma_dict:
                    soma_dict[key] = []
                soma_dict[key].append(item)
                
            for key, item in processCsv.items():
                if key not in process_dict:
                    process_dict[key] = []
                process_dict[key].append(item)
    
    write_csv(microglia_ids, numFrames, microglia_csv_base)
    
    for key, item in soma_dict.items():
        write_csv(item, numFrames, soma_csv_base + key + ".csv")
        
    for key, item in process_dict.items():
        write_csv(item, numFrames, process_csv_base + key + ".csv")


if __name__ == "__main__":
    metadata_fname  = '/Volumes/Baram Lab/2-photon Imaging/7-30-17_CRH-tdTomato+CX3CR1-GFP P8 PVN Ctrl/video_processing/7-30-17_CRH-tdTomato+CX3CR1-GFP P8 PVN Ctrl_Male 1 R PVN T1_b_4D/img_metadata.p'
    microglia_fname = '/Volumes/Baram Lab/2-photon Imaging/7-30-17_CRH-tdTomato+CX3CR1-GFP P8 PVN Ctrl/video_processing/7-30-17_CRH-tdTomato+CX3CR1-GFP P8 PVN Ctrl_Male 1 R PVN T1_b_4D/processed_microglia.p'
    
    start_time = timeit.default_timer()
    
    analyze_microglia(microglia_fname, metadata_fname)
        
    elapsed = timeit.default_timer() - start_time
    print(elapsed)

    
