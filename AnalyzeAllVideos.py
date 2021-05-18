#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run this file to batch process multiple videos at once. This file ingests a
comma-separated values (.csv) file to determine which files get which kinds
of processing. It then calls the appropriate processing functions on those
files. See the description of the VideoData class for csv format. 

Created on Thu Nov 30 14:06:41 2017

@author: jaclynbeck
"""

import DeconvolveImages2D as dc
import TraceProcesses2D as tp
import ProcessSkeleton as ps
import AnalyzeData as ad
import os
import sys
import timeit
import csv


"""
This class holds all the information from a line in the csv file in a more
human-readable way. 

The csv file must have the following fields, in order, separated by commas:
    psf_file:        Full file path of the point spread function (PSF) file to 
                     use for deconvolution. 
                         - PSF file must be a .png
    ims_file:        Full file path of the .ims file to process
    metadata_file:   local file name where metadata will be saved. 
                         - File must end in .p
    dc_output:       local file name to save the deconvolved image stack. 
                         - File must end in .tif
    skeleton_output: local file name to save the skeleton stack. 
                         - File must end in .tif
    soma_output:     local file name to save the soma information. 
                         - File must end in .p
    microglia_output: local file name to save the microglia information. 
                         - File must end in .p
    deconvolve:      True or False, whether to deconvolve the .ims file
    skeletonize:     True or False, whether to skeletonize the deconvolved 
                     image
    process_skeleton: True or False, whether to turn the skeleton into tree
                     information and collect stats on microglia
    analyze:         True or False, whether to gather stats about the 
                     skeletonized data
    postprocess:     True or False, whether to create some additional CSVs with
                     only a subset of the microglia data 
                          - (see included_microglia)
    threshold:       integer, usually 10 to 15. Use the top X% of pixel values 
                     for skeletonization. 
                         - If this is left blank, the image will not be 
                           skeletonized even if skeletonize == True
    deconvolutions:  integer, number of deconvolution interations to run on
                     the .ims file. Try 10 first, then 20 if 10 is too noisy. 
                         - If this is left blank, the image will not be 
                           deconvolved even if deconvolved == True
    included_microglia: a comma-separated list of microglia ID's that should be
                        included in post-processing
    
"""
class VideoData(object):
    __slots__ = 'psf_file', 'ims_file', 'metadata_file', 'dc_output', \
                'skeleton_output', 'soma_output', 'microglia_output', \
                'deconvolve', 'skeletonize', 'process_skeleton', 'analyze', \
                'postprocess', 'threshold', 'deconvolutions', \
                'included_microglia'
    
    """
    Initialization. 
    
    Input: 
        csv_line - string. a single line from the csv file.
    """            
    def __init__(self, csv_dict):
        if len(csv_dict) >= 11:
            self.psf_file        = csv_dict['PSF']
            self.ims_file        = csv_dict['IMS']
            self.metadata_file   = csv_dict['Metadata']
            self.dc_output       = csv_dict['Deconvolved Output']
            self.skeleton_output = csv_dict['Skeleton Output']
            self.soma_output     = csv_dict['Soma Output']
            self.microglia_output = csv_dict['Microglia Output']
            self.deconvolve      = csv_dict['Deconvolve'].lower().strip()
            self.skeletonize     = csv_dict['Skeletonize'].lower().strip()
            self.process_skeleton = csv_dict['Process Skeleton'].lower().strip()
            self.analyze         = csv_dict['Analyze'].lower().strip()
            self.postprocess     = csv_dict['Postprocess'].lower().strip()
            
            if len(csv_dict['Skeleton Threshold']) > 0:
                self.threshold   = 100-int(csv_dict['Skeleton Threshold'])
            else:
                self.threshold   = 100
                self.skeletonize = "false"
                
            if len(csv_dict['Deconvolutions']) > 0:
                self.deconvolutions = int(csv_dict['Deconvolutions'])
            else:
                self.deconvolutions = 0
                self.deconvolve = "false"
            
            if len(csv_dict['Included Microglia']) > 0:
                self.included_microglia = [int(s) for s in csv_dict['Included Microglia'].split(',')]
            
    
    """
    For debugging. This will get printed out by calling print(v)
    """        
    def __repr__(self):
        s = "PSF File: " + self.psf_file + "\n" + \
            "IMS File: " + self.ims_file + "\n" + \
            "Metadata File: " + self.metadata_file + "\n" + \
            "Deconvolved File: " + self.dc_output + "\n" + \
            "Skeleton File: " + self.skeleton_output + "\n" + \
            "Soma File: " + self.soma_output + "\n" + \
            "Tree File: " + self.microglia_output + "\n" + \
            "Skeleton Threshold: " + str(self.threshold) + "\n" + \
            "Deconvolutions: " + str(self.deconvolutions) + "\n" + \
            "Deconvolve: " + self.deconvolve + "\n" + \
            "Skeletonize: " + self.skeletonize + "\n" + \
            "Process Skeleton: " + self.process_skeleton + "\n" + \
            "Analyze: " + self.analyze + "\n"
            
        return s
    

"""
Ingests a comma-separated values file and creates a VideoData object for each
line.

Input:
    csv_file: full file path to the csv file
    
Output:
    csv_data: list of VideoData objects
"""
def read_csv(csv_file):
    csv_data = []
    
    with open(csv_file, 'r') as openfile:
        data = csv.DictReader(openfile)
        for line in data:
            csv_data.append(VideoData(line))
    
    return csv_data
    

"""
Calls the deconvolve function after doing some error checking of file names.

Inputs: 
    v - a VideoData object
    
Outputs:
    True if successful, False if not
"""    
def call_deconvolve(v):
    # If the .ims file doesn't exist
    if not os.path.isfile(v.ims_file):
        return False
    
    start_time = timeit.default_timer()
    
    # Deconvolve
    dc.deconvolve_images_2D(v.ims_file, v.dc_output, v.psf_file, \
                            v.metadata_file, 1, v.deconvolutions)
    
    elapsed = timeit.default_timer() - start_time
    print("Deconvolve: " + str(elapsed))
    
    # Check to see if the output file exists
    path = os.path.dirname(v.ims_file)
    path = os.path.join(path, "video_processing", os.path.basename(v.ims_file)[0:-4])
    
    return os.path.isfile(os.path.join(path, v.dc_output))


"""
Calls skeletonize and process skeleton after doing some error checking of file 
names. If the preprequisite files do not exist (i.e. no deconvolved file to
skeletonize exists), it deconvolves the .ims file first. 

Input: 
    v - a VideoData object
    
Output: 
    True if successful, False if not
"""
def call_skeletonize(v):
    success = True
    
    path = os.path.dirname(v.ims_file)
    path = os.path.join(path, "video_processing", os.path.basename(v.ims_file)[0:-4])
    
    dc_output = os.path.join(path, v.dc_output)
    
    try:
        # If the deconvolved file doesn't exist, deconvolve the .ims file first
        if not os.path.exists(dc_output):
            success = call_deconvolve(v)
        
        start_time = timeit.default_timer()
        
        # Skeletonize
        if success:
            tp.trace_all_images(dc_output, v.skeleton_output, v.soma_output,
                                v.threshold)
            
    except: 
        success = False
        print(sys.exc_info())
    
    elapsed = timeit.default_timer() - start_time
    print("Skeletonize: " + str(elapsed))
    
    # Check if the skeleton file is there
    return os.path.isfile(os.path.join(path, v.skeleton_output))


def call_process_skeleton(v):
    success = True
    
    path = os.path.dirname(v.ims_file)
    path = os.path.join(path, "video_processing", os.path.basename(v.ims_file)[0:-4])
    
    dc_output = os.path.join(path, v.dc_output)
    skeleton_output = os.path.join(path, v.skeleton_output)
    
    try:
        # If the skeleton file doesn't exist, skeletonize the file first
        if not os.path.exists(skeleton_output):
            success = call_skeletonize(v)
        
        start_time = timeit.default_timer()
        
        if success:
            ps.process_skeleton(skeleton_output, dc_output, v.metadata_file,
                                v.soma_output, v.microglia_output)
    except: 
        success = False
        print(sys.exc_info())
    
    elapsed = timeit.default_timer() - start_time
    print("Process Skeleton: " + str(elapsed))
    
    # Check if the microglia file is there
    return os.path.isfile(os.path.join(path, v.microglia_output))


"""
Calls analyze on the processed skeleton after doing some error checking of file 
names. If the preprequisite files do not exist (i.e. no skeleton file to
analyze exists), it skeletonizes the image first. 

Input: 
    v - a VideoData object
    
Output: 
    True if successful, False if not
"""
def call_analyze(v):
    success = True
    
    path = os.path.dirname(v.ims_file)
    path = os.path.join(path, "video_processing", os.path.basename(v.ims_file)[0:-4])
    
    microglia_output = os.path.join(path, v.microglia_output)
    metadata_file = os.path.join(path, v.metadata_file)
    
    # If the microglia file doesn't exist, process the skeleton first
    if not os.path.exists(microglia_output):
        success = call_process_skeleton(v)
    
    start_time = timeit.default_timer()
    
    if success:
        ad.analyze_microglia(microglia_output, metadata_file)
    
    elapsed = timeit.default_timer() - start_time
    print("Analyze: " + str(elapsed))
    
    return success


def call_postprocess(v):
    success = True
    
    path = os.path.dirname(v.ims_file)
    path = os.path.join(path, "video_processing", os.path.basename(v.ims_file)[0:-4])
    microglia_output = os.path.join(path, v.microglia_output)
    
    # If this video hasn't been analyzed, do that first
    if not os.path.exists(os.path.join(path, "raw_data")):
        success = call_analyze(v)
    
    start_time = timeit.default_timer()
    
    if success:
        ad.postprocess_data(microglia_output, v.included_microglia)
    
    elapsed = timeit.default_timer() - start_time
    print("Postprocess: " + str(elapsed))
    
    return success
    

"""
Main function to run. This function ingests a comma-separated values file and,
for each line that requires some kind of processing, calls the appropriate
processing functions on that line. If any files fail, they are printed out
at the end of the function. 
"""
if __name__ == '__main__':
    csv_file = "/home/jaclynbeck/Desktop/BaramLabFiles/videoData.csv"
    #csv_file = "/mnt/storage/BaramLabFiles/CX3CR1-Gq-DREADD mCitrine Videos/Peter's Analysis/Batch 1/51-101/videoData.csv"
    
    # Ingest the csv, then use only the data for which at least one of 
    # "deconvolve", "skeletonize", or "analyze" is set to true. 
    video_data = read_csv(csv_file)
    
    video_data = [v for v in video_data if v.deconvolve == "true" \
                                          or v.skeletonize == "true" \
                                          or v.process_skeleton == "true" \
                                          or v.analyze == "true" \
                                          or v.postprocess == "true"]

    ### Optional -- split the data into odds and evens and run "odds" in one
    # terminal and "evens" in another so it gets done almost twice as fast.
    # Comment this section out if not needed. 
    odds = video_data[1::2]
    evens = video_data[0::2]
    
    #video_data = odds
    ### Optional
    
    failures = []
    
    # For each VideoData object
    for v in video_data:
        print("\nProcessing " + v.ims_file)
        
        # Deconvolve it
        if v.deconvolve == "true":
            success = call_deconvolve(v)
            if not success: 
                failures.append(v)
                continue
        
        # Skeletonize it
        if v.skeletonize == "true":
            success = call_skeletonize(v)
            if not success:
                failures.append(v)
                continue
            
        # Process the skeleton
        if v.process_skeleton == "true":
            success = call_process_skeleton(v)
            if not success:
                failures.append(v)
                continue
        
        # Analyze it
        if v.analyze == "true":
            success = call_analyze(v)
            if not success:
                failures.append(v)
                continue
        
        # Postprocess it
        if v.postprocess == "true":
            success = call_postprocess(v)
            if not success:
                failures.append(v)
                continue
    
    # Print any failures
    if len(failures) > 0:
        print("\nFailed files: \n")
        for f in failures:
            print("\t" + f.ims_file + "\n")
    