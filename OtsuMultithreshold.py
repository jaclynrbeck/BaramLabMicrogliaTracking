# -*- coding: utf-8 -*-
"""
@author: jaclynbeck
"""

import scipy as sp
import scipy.optimize as opt
import skimage.filters as skf
import matplotlib.pyplot as plt
import timeit

def sigma_b(ks, p_i, L):
    sigma_b = 0

    if sp.any(ks < 1) or sp.any(ks > L) or sp.unique(sp.round_(ks)).size != ks.size:
        return 1
    
    k = sp.zeros((ks.size+2,), dtype='int16')
    k[0] = 0
    k[1:-1] = sp.round_(ks)
    k[-1] = L
    
    for i in range(k.size-1):    
        k_range = sp.arange(k[i]+1,k[i+1]+1)
        p_k = p_i[ k_range-1 ]
        w_k  = sum( p_k )
        
        if w_k == 0: # No divisions by 0
            return 1
        
        mu_k = sum( k_range * p_k ) / w_k
        sigma_b += w_k * mu_k * mu_k #pow((mu_k - mu_t),2) 

    return -sigma_b # Minimize the negative of this function
    
    
def otsu_multithreshold(img, num_thresholds, max_val=0, opt_type='local'):
    L = img.max() + 1
    
    if L == 256:
        img = img - img.min()
        img = sp.round_(img/img.max()*255)
    
    counts, binEdges = sp.histogram(img, L)
    p_i = counts / sum(counts)
    
    if max_val == 0:
        midpoint = skf.threshold_otsu(img, L)
    
        k_half1 = sp.linspace(0,int(midpoint),int(num_thresholds/2)+2)[1:]
        k_half2 = sp.linspace(int(midpoint)+1,L,int(num_thresholds/2)+2)[1:]
        
        k0 = sp.concatenate((k_half1,k_half2))
        k0 = k0[0:num_thresholds]
    
    else:
        k0 = sp.linspace(0, max_val, num_thresholds+2)
        k0 = k0[1:num_thresholds+1]
    
    start_time = timeit.default_timer()
    
    if opt_type == 'global':
        rranges = [ [i+1, L-num_thresholds+i+1] for i in range(num_thresholds) ]
        xopt = opt.differential_evolution(sigma_b, rranges, args=(p_i, L), 
                                          popsize=15, mutation=(0.5, 1.0), 
                                          recombination=0.3, polish=False)
        xopt = xopt.x
        
    else:
        xopt = opt.fmin_powell(sigma_b, x0=k0, args=(p_i,L), xtol=1.0)
    
    elapsed = timeit.default_timer() - start_time
    print(elapsed)
    
    if xopt.size > 1:
        xopt = sp.sort(xopt)
        
    return [sigma_b(xopt, p_i, L), sp.round_(xopt-1)] # make xopt 0-indexed instead of 1-indexed

if __name__ == '__main__':
    img_fname = '/Users/jaclynbeck/Desktop/BaramLab/Substack (1).tif'
    num_thresholds = 12; 
    
    img = sp.misc.imread(img_fname)
    [maxSig, thresholds] = otsu_multithreshold(img, num_thresholds, 0, 'local')
    print(maxSig)
    print(thresholds)
    
    img = sp.misc.imread(img_fname)
    L = img.max() + 1; 
    
    if L == 256:
        img = img - img.min()
        img = sp.round_(img/img.max()*255)
        
    k = sp.concatenate(([0], thresholds, [L]))
    values = sp.round_(sp.linspace(0, L-1, num_thresholds+1))
    bw = sp.zeros_like(img, dtype='uint8')
        
    for i in sp.arange(1,k.size-1):
        bw[(img < k[i+1]) & (img >= k[i])] = 10+values[i];

    plt.imshow(bw/bw.max())
    plt.show()
    
    sp.misc.imsave('/Users/jaclynbeck/Desktop/BaramLab/pyGliaMask.tif', bw*255/bw.max())