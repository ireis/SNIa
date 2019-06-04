import pandas
import matplotlib.pyplot as plt
import numpy
import specplotlib
from astropy.table import Table

from sklearn.preprocessing import Imputer

# function to load a sinfle file spectrum from the kpra table according to filename

def load_single_spectrum(filename,source):
    if source == 'cfa':
        search = '/Users/natalie/git/SNIa/kpra/kaepora/data/spectra/*/*/' + filename
        path = glob.glob(search)
        #print(path)
    else:
        search = '/Users/natalie/git/SNIa/kpra/kaepora/data/spectra/*/' + filename
        path = glob.glob(search)
        #print(path)

    w, f = 0, 0
    if os.path.isfile(path[0]):
        test = numpy.loadtxt(path[0])
        w = test[:,0]
        f = test[:,1]
        #print(w.shape, w.min(), w.max())
        
    return w,f

def get_raw_spectra_matrix(df):
    
    file_name_list = df['file_kpr'].values 
    source_name_list = df['src_kpr'].values
    sn_name = df['sn_kpr'].values
    phase = df['phase_kpr'].values
    #sn_spec_idx = df['idx_kpr'].values
    

    nof_objects = len(sn_name)
    sn_spec_idx = numpy.arange(nof_objects)

    wl_list = []
    fl_list = []
    for idx in sn_spec_idx:
        #print(idx, file_name_list[idx] )
        wl, fl = load_single_spectrum(file_name_list[idx],source_name_list[idx])
        wl_list += [wl]
        fl_list += [fl]
    spec_len = [len(wl) for wl in wl_list]

    W = numpy.zeros([nof_objects, max(spec_len)])
    F = numpy.zeros([nof_objects, max(spec_len)])

    for i in range(nof_objects):
        W[i, :spec_len[i]] = wl_list[i]
        F[i, :spec_len[i]] = fl_list[i]

    return W, F, sn_name, phase, spec_len

def zero_to_nan(arr):
    """
    Replace zeros with nans
    Note: doing this because there are empty entries in the matrix which are set to zero (Due to different objects having different number of pixels)
    :param arr:
    :return:
    """
    arr[arr == 0] = numpy.nan
    return arr

def same_grid_single(wave_common, wave_orig, flux_orig):
    """
    Putting a single spectrum on the common wavelength grid
    """
    is_finite = numpy.isfinite(wave_orig) & numpy.isfinite(flux_orig)
    spec = numpy.interp(wave_common, wave_orig, flux_orig, left=numpy.nan, right=numpy.nan)

    return spec


def same_grid(common_wave, sn_waves, sn_fluxs):
    """
    Putting all spectra on the same wavelength grid
    """
    print('Putting all spectra on the same grid with min lambda = ', common_wave.min(), 'and max lambda = ', common_wave.max())

    specs_same_grid = numpy.zeros([sn_fluxs.shape[0], common_wave.shape[0]])
    for i in range(sn_waves.shape[0]):
        specs_same_grid[i] = same_grid_single(common_wave, sn_waves[i], sn_fluxs[i])

    return specs_same_grid


def norm_spectrum(spec):
    """
    Normalize spectrum - divide by median (clipped to one)
    """
    spec_norm = numpy.nanmedian(spec)
    #if spec_norm >= 1:
    spec = (spec / spec_norm)
    #else:
    #    spec = spec + (1 - spec_norm)

    return spec


def good_spectra_mat(W, F, spec_len, wmin, wmax, nw = None):
    """
    put NANs where zeroes are
    do common grid between w_min, w_max, nw = how many wl between, default is 1 ang jumps
    interpulate between
    """
    nof_objects = F.shape[0]
    
    # put nans where zeros are
    W = zero_to_nan(W)
    F = zero_to_nan(F)
    
    # put on same grid
    #idx = numpy.argmin(abs(spec_len - numpy.nanmedian(spec_len))) #nanmedian - the median along the specified axis, while ignoring NaNs
    if nw is None:
        nw = int(wmax - wmin)
    common_wave = numpy.linspace(wmin,wmax,nw)
    F_grid = same_grid(common_wave, W, F) # preprocess_spectra.same_grid(common_wave, W, F)
    
    # normalize according to median
    F_norm = numpy.zeros(F_grid.shape)
    for i in range(0,nof_objects):
        F_norm[i] = norm_spectrum(spec = F_grid[i])

    return common_wave, F_norm

def select_spec_from_peak(kpra_table, flux_matrix, tol, sn_names):
        
    """
    pick the nearest to peak spectra from the kpra spectra db (+-7 days from peak) that has
    not_nan_flux/all_flux > tolarence parameter.
    
    return
    ------
    The indexes of required spectra
    """

    #unq_sn = list(set(kpra_table['sn_kpr'])) # unique sn list --> we get this with sn_names from db

    # list of lists of indexes per sn: idx[j] - all idxs that fit unq_sn[j]
    idx = [[] for x in range(len(sn_names))]
    for j in range (0,len(sn_names)):
        idx[j] = [i for i, e in enumerate(kpra_table['sn_kpr'].values) if e == sn_names[j]]

    final_idx = []
    phase = kpra_table['phase_kpr']

    for j in range (len(sn_names)): # calculating for the j sn
        temp_phase=[]
        sorted_phase=[]
        for i in range(0,len(idx[j])):  # sorted list of spectra phases
            temp_phase.append(phase[idx[j][i]])
            sorted_phase.append(phase[idx[j][i]])   
        sorted_phase.sort(key=abs)

        # finding the best spectra according to tol - best_idx
        k = 0                         
        spec_idx = temp_phase.index(sorted_phase[k])
        while (sum(~numpy.isnan(flux_matrix[idx[j][spec_idx]])) / float(flux_matrix.shape[1])) < tol: # not nan/total < tolarence
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

def find_nearest(array, value):
    """
    finds the index and values of the nearest value in an array given a value to search
    """
    array = numpy.asarray(array)
    idx = (numpy.abs(array - value)).argmin()
    
    return idx, array[idx]

def select_spec_from_peak_by_bin (kpra_table, flux_matrix, w_matrix, tol, sn_names, min_wl, max_wl):
        
    """
    pick the nearest to peak spectra from the kpra spectra db (+-7 days from peak) that has
    not_nan_flux/all_flux > tolarence parameter within the (min_wl,max_wl) range.
    
    return
    ------
    The indexes of required spectra
    """

    # find the indexes where the min_wl and max_wl are
    min_idx, min_val = find_nearest(w_matrix, min_wl)
    max_idx, max_val = find_nearest(w_matrix, max_wl)
    
    # list of lists of indexes per sn: idx[j] - all idxs that fit unq_sn[j]
    idx = [[] for x in range(len(sn_names))]
    for j in range (0,len(sn_names)):
        idx[j] = [i for i, e in enumerate(kpra_table['sn_kpr'].values) if e == sn_names[j]]

    final_idx = []
    phase = kpra_table['phase_kpr']

    for j in range (len(sn_names)): # calculating for the j sn
        temp_phase=[]
        sorted_phase=[]
        for i in range(0,len(idx[j])):  # sorted list of spectra phases
            temp_phase.append(phase[idx[j][i]])
            sorted_phase.append(phase[idx[j][i]])   
        sorted_phase.sort(key=abs)

        # finding the best spectra according to tol - best_idx
        k = 0                         
        spec_idx = temp_phase.index(sorted_phase[k])
        while (sum(~numpy.isnan(flux_matrix[idx[j][spec_idx]][min_idx:max_idx])) / float(abs(max_idx-min_idx))) < tol: # not nan/total < tolarence
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

def create_umap_flux_mat (F_kpr, sn_names, idx, idx_min, idx_max):
    """
    create a matrix with spectra for good sne and nans for not
    the flux mat returned will include only the flux in the choosen bin between min and max wl values
    the use_inds parameter returned is a list of all idx which have spectra and are not nans
    """
    idx = numpy.array(idx)
    F_mat = numpy.zeros([len(sn_names), abs(idx_max-idx_min)])
    use_inds = []
    for i, i_all in enumerate(idx):
        if numpy.isfinite(i_all):
            i_all = int(i_all)
            F_mat[i] = F_kpr[i_all,idx_min:idx_max]
            use_inds += [i]
        else:
            F_mat[i] = numpy.zeros(abs(idx_max-idx_min))*numpy.nan
    use_inds = numpy.array(use_inds)
    
    return F_mat, use_inds

def impute_spec(specs, wave): 
    """
    where there is nan in specific wl switch the value to the median of all other sne in same wl
    """
    imp = Imputer(missing_values='NaN', strategy='median', axis=0)
    imp = imp.fit(specs)
    specs_imp = imp.transform(specs)
    wave_imp = imp.transform(wave.reshape(1, -1)).T

    return specs_imp, wave_imp

def norm_by_bin(spec):
    """
    normalize according to median of each bin spectra seperately
    """
    nof_objects = spec.shape[0]
    spec_norm = numpy.zeros(spec.shape)
    for i in range(0,nof_objects):
        spec_norm[i] = norm_spectrum(spec = spec[i])

    return spec_norm
    