import pandas
import os
import numpy
import matplotlib
def load_SN_spec_df():
    tbl = pandas.read_csv('data/cfaspec_snIa/cfasnIa_param.dat', skiprows=42,sep='\s+')
    tbl = tbl.drop(0)
    tbl = tbl.set_index('#SN')

    spc = pandas.read_csv('data/cfaspec_snIa/cfasnIa_mjdspec.dat', sep='\s+')

    nspec = spc.shape[0]
    mjd_from_peak_list = []
    for i in range(nspec):
        fname = spc['#Filename'].iloc[i]
        name = fname.split('-')[0]
        name = str(name)[2:]
        if name in tbl.index:
            mjd_peak = tbl['tmax(B)'].loc[name]
            # print(mjd_peak)
            mjd = spc['MJD'].iloc[i]
            if mjd_peak < 89999:
                # print(mjd_peak, mjd)
                mjd_from_peak_list += [mjd - mjd_peak]
            else:
                mjd_from_peak_list += [99999.9]
        else:
            mjd_from_peak_list += [-99999.9]

    spc['t_from_peak'] = mjd_from_peak_list

    inst_list = []
    for fname in spc['#Filename'].values:
        inst_list += [fname.split('-')[2][:-4]]

    name_list = []

    for fname in spc['#Filename'].values:
        name_list += [fname.split('-')[0]]
    spc['SN_name'] = name_list


    return spc


def load_single_spectrum(filename):
    folder = filename.split('-')[0]
    path = 'data/cfaspec_snIa/' + folder + '/' + filename
    print(path)
    if os.path.isfile(path):
        test = numpy.loadtxt(path)
        w = test[:,0]
        s = test[:,1]
        ds = test[:,2]
        print(w.shape, w.min(), w.max())
    return w,s,ds



