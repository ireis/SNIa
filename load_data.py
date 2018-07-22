import pandas
import os
import numpy
from collections import Counter

import matplotlib
def load_SN_spec_df():
    sn_df = pandas.read_csv('data/cfaspec_snIa/cfasnIa_param.dat', skiprows=42,sep='\s+')
    sn_df = sn_df.drop(0)
    sn_df = sn_df.set_index('#SN')

    sn_spec_df = pandas.read_csv('data/cfaspec_snIa/cfasnIa_mjdspec.dat', sep='\s+')

    nspec = sn_spec_df.shape[0]
    mjd_from_peak_list = []
    for i in range(nspec):
        fname = sn_spec_df['#Filename'].iloc[i]
        name = fname.split('-')[0]
        name = str(name)[2:]
        if name in sn_df.index:
            mjd_peak = sn_df['tmax(B)'].loc[name]
            # print(mjd_peak)
            mjd = sn_spec_df['MJD'].iloc[i]
            if mjd_peak < 89999:
                # print(mjd_peak, mjd)
                mjd_from_peak_list += [mjd - mjd_peak]
            else:
                mjd_from_peak_list += [99999.9]
        else:
            mjd_from_peak_list += [-99999.9]

    sn_spec_df['t_from_peak'] = mjd_from_peak_list

    inst_list = []
    for fname in sn_spec_df['#Filename'].values:
        inst_list += [fname.split('-')[2][:-4]]

    sn_spec_df['inst_list'] = inst_list
    name_list = []

    for fname in sn_spec_df['#Filename'].values:
        name_list += [fname.split('-')[0]]
    sn_spec_df['SN_name'] = name_list

    counts = Counter(sn_spec_df['SN_name'])
    names = [s[2:] for s in list(counts.keys()) if s[2:] in sn_df.index]
    n_spec = [int(n) for n, k in zip(list(counts.values()), list(counts.keys())) if k[2:] in sn_df.index]
    sn_df.set_value(index=names, value=n_spec, col='n_spec')


    return sn_df, sn_spec_df


def load_single_spectrum(filename):
    folder = filename.split('-')[0]
    path = 'data/cfaspec_snIa/' + folder + '/' + filename
    #print(path)
    w, s, ds = 0, 0, 0
    if os.path.isfile(path):
        test = numpy.loadtxt(path)
        w = test[:,0]
        s = test[:,1]
        ds = test[:,2]
        #print(w.shape, w.min(), w.max())
    return w,s,ds



