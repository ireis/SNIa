import pandas
import matplotlib.pyplot as plt
import numpy
import specplotlib
from astropy.table import Table
%matplotlib inline

import os
from collections import Counter
import time

import glob 

from scipy.interpolate import interp1d
import numpy
from sklearn.preprocessing import Imputer
from  scipy import signal


def select_spec_from_peak(kpra_df, flux_matrix, tol):
    """
    pick the nearest to peak spectra from the kpra spectra db (+-7 days from peak) that has
    not_nan_flux/all_flux > tolarence parameter.
    
    return
    ------
    The indexes of required spectra
    """

unq_sn = list(set(kpra_table['sn_kpr'])) # unique sn list

# list of lists of indexes per sn: idx[j] - all idxs that fit unq_sn[j]
idx = [[] for x in range(len(unq_sn))]
for j in range (0,len(unq_sn)):
    idx[j] = [i for i, e in enumerate(sn) if e == unq_sn[j]]
 
final_idx = []
phase = kpra_table['phase_kpr']

for j in range (len(unq_sn)): # calculating for the j sn
    temp_phase=[]
    sorted_phase=[]
    for i in range(0,len(idx[j])):  # sorted list of spectra phases
        temp_phase.append(phase[idx[j][i]])
        sorted_phase.append(phase[idx[j][i]])   
    sorted_phase.sort(key=abs)

    
    # finding the best spectra according to tol - best_idx
    k = 0                         
    spec_idx = temp_phase.index(sorted_phase[k])
    # print spec_idx
    while (sum(~numpy.isnan(F_good[idx[j][spec_idx]])) / float(F_good.shape[1])) < tol: # not nan/total < tolarence
        k = k+1
        if k == (len(sorted_phase)):
            spec_idx = None
            #final_idx.append(numpy.nan)
            break
        spec_idx = temp_phase.index(sorted_phase[k])
    if  spec_idx is None:
        final_idx.append(numpy.nan)
    else:
        final_idx.append(idx[j][spec_idx])
    #print 'for sn' + unq_sn[j] + ' it is the ' + str(final_idx[j]) + ' spectrum in list'
    
    return final_idx
    