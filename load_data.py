import pandas
import os
import numpy
from collections import Counter
import preprocess_spectra
from astroquery.simbad import Simbad
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import WMAP9 as cosmo
import time


def get_ra_dec(SN_df):

    nof_objects = SN_df.shape[0]
    ra_arr = numpy.zeros(nof_objects)
    dec_arr = numpy.zeros(nof_objects)


    for idx, name in enumerate(SN_df.index):
        if ra_arr[idx] == 0:
            try:
                sn = 'sn' + name
                result_table = Simbad.query_object(sn)
                ra, dec = result_table['RA'], result_table['DEC']
                c = SkyCoord(ra = ra[0], dec = dec[0],unit=(u.hourangle, u.deg), frame='icrs')
                ra, dec = c.ra.value, c.dec.value
            except:
                try:
                    # Sometimes even SIMBAD needs a rest
                    print('Sleeping', sn)
                    time.sleep(180)
                    result_table = Simbad.query_object(sn)
                    ra, dec = result_table['RA'], result_table['DEC']
                    c = SkyCoord(ra = ra[0], dec = dec[0],unit=(u.hourangle, u.deg), frame='icrs')
                    ra, dec = c.ra.value, c.dec.value
                except:
                    break
            ra_arr[idx] = ra
            dec_arr[idx] = dec


    SN_ra_dec = pandas.DataFrame(index=SN_df.index)
    SN_ra_dec['RA'] = ra_arr
    SN_ra_dec['DEC'] = dec_arr
    SN_ra_dec.to_csv('SN_ra_dec.csv')
    print('saved SN_ra_dec.csv')

    return SN_ra_dec

def load_lc_df():
    lc_df_CfA3 = pandas.read_csv('data/CfA3.tsv', skiprows=77, sep='\t')
    lc_df_CfA3 = lc_df_CfA3.drop(0)
    lc_df_CfA3 = lc_df_CfA3.drop(1)
    lc_df_CfA3['SN'] = lc_df_CfA3['SN'].str.rstrip(' ')
    lc_df_CfA3 = lc_df_CfA3.set_index('SN')
    lc_df_CfA3 = lc_df_CfA3.assign(source='CfA3')

    lc_df_CfA2 = pandas.read_excel('data/CfA2.xlsx')
    lc_df_CfA2 = lc_df_CfA2.set_index('SN')
    lc_df_CfA2['SN'] = lc_df_CfA2.index
    lc_df_CfA2['SN'] = lc_df_CfA2['SN'].str.rstrip('.')
    lc_df_CfA2 = lc_df_CfA2.set_index('SN')
    lc = lc_df_CfA2['Δm15(B)'].str.split('±', expand=True)
    lc = lc.rename(index=str, columns={0: "Δm15(B)", 1: "dΔm15(B)"})
    lc_df_CfA2 = lc_df_CfA2.drop(['Δm15(B)'], axis=1)
    lc_df_CfA2_temp = pandas.concat([lc_df_CfA2, lc], axis=1, sort=False)

    lc_df_CfA2_rd = pandas.read_csv('data/CfA2radec.tsv', skiprows=35, sep='\t')
    lc_df_CfA2_rd = lc_df_CfA2_rd.drop(0)
    lc_df_CfA2_rd = lc_df_CfA2_rd.drop(1)
    lc_df_CfA2_rd['SN'] = lc_df_CfA2_rd['SN'].str.rstrip(' ')
    lc_df_CfA2_rd = lc_df_CfA2_rd.set_index('SN')

    lc_df_CfA2_final = pandas.concat([lc_df_CfA2_temp, lc_df_CfA2_rd], axis=1, sort=False)
    lc_df_CfA2_final=lc_df_CfA2_final.assign(source='CfA2')

    #lc_df_CfA1 = pandas.read_csv('data/CfA1.tsv', skiprows=52, sep='\t')
    #lc_df_CfA1 = lc_df_CfA1.drop(0)
    #lc_df_CfA1 = lc_df_CfA1.drop(1)
    #lc_df_CfA1 = lc_df_CfA1.set_index('SN')

    lc_df = pandas.concat([lc_df_CfA3, lc_df_CfA2_final], axis=1, sort=False)

    return lc_df, lc_df_CfA2_final, lc_df_CfA3

def set_nan_to_cells(your_df, exception):     # change all cells that are not floats to numpy nan (float) - send header as string!
    for column in your_df.columns:
        test = 0
        for m in exception:
            if column is m:
                test = 1
        if test == 0:
            for l in your_df.index:
                try:
                    your_df.loc[l,column] = float(your_df[column].loc[l])  ##itamar shouls we worry abput int? + change to a loop on all columns
                except:
                    your_df.loc[l,column] = numpy.nan

    return your_df

"""
def create_lc_param_table(df_list,columns_to_merge,headers_for_columns): #send headers as strings!
    for n in your_df.index:
        if not numpy.isnan(float(your_df.loc[n,column_cfa3])):
            B_mag.append(float(your_df.loc[n,column_cfa3])) #first take from CfA3
            B_src.append('CfA3')
        else:
            B_mag.append(float(your_df.loc[n,column_cfa2]))
            B_src.append('CfA2')

    return lc_param
"""

def get_intrinsic_mag(sn_name, Z, B, AB):
    MB = pandas.DataFrame(index=sn_name, columns=['MB'])
    MB_arr = numpy.zeros(len(sn_name))
    for i, n in enumerate(sn_name):
        try:
            mb = B.loc[n]
            d = cosmo.luminosity_distance(Z.loc[n]) #
            ab = AB.loc[n]
            mu = 5*numpy.log10(d/(10 * u.pc))
            MB_arr[i] = mb - mu - ab
        except:
            MB_arr[i] = numpy.nan
    MB['MB'] = MB_arr
    return MB#Mpandas.DataFrame(index=sn_name, data=MB)


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

def all_spectra_idxs(SN_df, SN_spec_df):
    """
    Returns indices of the most close to peak spectra, for SN with spectra within 5 days from peak light

    :return:
    """
    sn_name_all = []
    spec_idx_list_use = []
    t = []
    for sn_idx, sn in enumerate(SN_df.index):
        snname = 'sn' + sn
        sn_spec_idx = SN_spec_df[SN_spec_df['SN_name'] == snname].index

        for i in sn_spec_idx:
            t += [SN_spec_df['t_from_peak'].loc[i]]
            spec_idx_list_use += [i]
            sn_name_all += [sn]

    spec_idx_list_use = numpy.array(spec_idx_list_use).astype(int)


    return spec_idx_list_use, sn_name_all, t


def near_max_spectra_idxs(SN_df, SN_spec_df):
    """
    Returns indices of the most close to peak spectra, for SN with spectra within 5 days from peak light

    :return:
    """
    spec_idx_list_use = numpy.ones(SN_df.shape[0]) * (-1)
    for sn_idx, sn in enumerate(SN_df.index):
        snname = 'sn' + sn
        sn_spec_idx = SN_spec_df[SN_spec_df['SN_name'] == snname].index

        for i in sn_spec_idx:

            t = SN_spec_df['t_from_peak'].loc[i]
            tmin = 10

            if ((t > -5) & (t < 5)):
                if abs(t) < tmin:
                    spec_idx_list_use[sn_idx] = i
    spec_idx_list_use = spec_idx_list_use.astype(int)

    sn_name = []
    sn_spec_idx = []
    sn_spec_time = []
    for idx, spec_idx in enumerate(spec_idx_list_use):
        if spec_idx >= 0:
            sn_name += [SN_df.index[idx]]
            sn_spec_idx += [spec_idx]
            sn_spec_time += [SN_spec_df['t_from_peak'].loc[spec_idx]]

    return spec_idx_list_use, sn_name, sn_spec_idx, sn_spec_time

def get_vanilla_spectra_matrix(file_name_list, sn_spec_idx):

    nof_objects = len(sn_spec_idx)

    w_list = []
    s_list = []
    ds_list = []
    for idx in sn_spec_idx:
        w, s, ds = load_single_spectrum(file_name_list[idx], )
        w_list += [w]
        s_list += [s]
        ds_list += [ds]
    spec_len = [len(w) for w in w_list]

    W = numpy.zeros([nof_objects, max(spec_len)])
    X = numpy.zeros([nof_objects, max(spec_len)])
    dX = numpy.zeros([nof_objects, max(spec_len)])

    for i in range(nof_objects):
        W[i, :spec_len[i]] = w_list[i]
        X[i, :spec_len[i]] = s_list[i]
        dX[i, :spec_len[i]] = ds_list[i]

    return W, X, dX, spec_len

def pp_spectra_mat(SN_df, sn_name, spec_len, W, X, dX):

    nof_objects = X.shape[0]
    E_bv = numpy.zeros(nof_objects)
    print('FIXME: No E(B-V), using zeros')
    W, X = preprocess_spectra.clean_and_deredd_spectra(W, X, dX, E_bv)

    redshift = SN_df['zhel'].loc[sn_name].values

    W_z = numpy.zeros(W.shape)
    for i in range(nof_objects):
        W_z[i] = preprocess_spectra.de_redshift(W[i], redshift[i])

    print('FIXME(?): using hard coded wavelength grid')

    idx = numpy.argmin(abs(spec_len - numpy.nanmedian(spec_len)))
    common_wave = W_z[idx, :spec_len[idx]]

    X_NORM = numpy.zeros(X.shape)
    for i in range(nof_objects):
        X_NORM[i] = preprocess_spectra.norm_spectrum(spec=X[i])

    X_SG = preprocess_spectra.same_grid(common_wave, W_z, X_NORM)

    X_SG, CW = preprocess_spectra.impute_spec(X_SG, common_wave)


    return X_SG, CW


def near_max_spectra_matrix(SN_df, SN_spec_df):
    """
    Returns a matrix with processed spectra that is ready for use
    Only for SN with most close to peak spectra, for SN with spectra within 5 days from peak light
    :return:
    """
    spec_idx_list_use, sn_name, sn_spec_idx, sn_spec_time = near_max_spectra_idxs(SN_df, SN_spec_df)
    #nof_objects = spec_idx_list_use[spec_idx_list_use > 0].shape[0]

    file_name_list = SN_spec_df['#Filename'].values
    W, X, dX, spec_len = get_vanilla_spectra_matrix(file_name_list, sn_spec_idx)

    X_SG, CW = pp_spectra_mat(SN_df, sn_name, spec_len, W, X, dX)



    return X_SG, CW, sn_name, sn_spec_idx, sn_spec_time


def full_spectra_matrix(SN_df, SN_spec_df):
    """
    Returns a matrix with processed spectra that is ready for use
    Only for SN with most close to peak spectra, for SN with spectra within 5 days from peak light
    :return:
    """
    sn_spec_idx, sn_name, time_from_peak  = all_spectra_idxs(SN_df, SN_spec_df)
    #nof_objects = spec_idx_list_use[spec_idx_list_use > 0].shape[0]

    file_name_list = SN_spec_df['#Filename'].values
    W, X, dX, spec_len = get_vanilla_spectra_matrix(file_name_list, sn_spec_idx)

    X_SG, CW = pp_spectra_mat(SN_df, sn_name, spec_len, W, X, dX)



    return X_SG, CW, sn_name, sn_spec_idx, time_from_peak

