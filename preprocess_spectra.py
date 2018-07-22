from scipy.interpolate import interp1d
import numpy
from sklearn.preprocessing import Imputer
from scipy import signal

import sfdmap
global m
m = sfdmap.SFDMap('dustmap/')



def deredden_spectrum(wl, spec, E_bv):
    """
    function dereddens a spectrum based on the given extinction_g value and Fitzpatric99 model
    IMPORTANT: the spectrum should be in the observer frame (do not correct for redshift)
    """
    # dust model
    wls = numpy.array([ 2600,  2700,  4110,  4670,  5470,  6000, 12200, 26500])
    a_l = numpy.array([ 6.591,  6.265,  4.315,  3.806,  3.055,  2.688,  0.829,  0.265])
    f_interp = interp1d(wls, a_l, kind="cubic")

    a_l_all = f_interp(wl)
    #E_bv = extinction_g / 3.793
    A_lambda = E_bv * a_l_all
    spec_real = spec * 10 ** (A_lambda / 2.5)

    return spec_real


def remove_bad_pixels(spec, ds):
    """
    Puts to Nan pixels with zero inverse variance
    :param spec:
    :param ivar:
    :return:
    """
    spec[ds == 9999999999] = numpy.nan

    return spec


def zero_to_nan(arr):
    """
    Replace zeros with nans
    Note: doing this because there are empty entries in the matrix which are set to zero (Due to different objects having different number of pixels)
    :param arr:
    :return:
    """
    arr[arr == 0] = numpy.nan
    return arr


def norm_spectrum(spec):
    """
    Normalize spectrum - divide by median (clipped to one)
    :param spec:
    :return:
    """
    spec_norm = numpy.nanmedian(spec)
    if spec_norm >= 1:
        spec = (spec / spec_norm)
    else:
        spec = spec + (1 - spec_norm)

    return spec


def clean_and_deredd_spectrum_single(wave, spec, ds, E_bv):
    """
    This function removes bad pixels, de-reds, normalize and runs median filter on a single object
    """

    """
    Remove bad pixels (ds = 0 => must remove )
    """
    spec = remove_bad_pixels(spec, ds)

    wave = zero_to_nan(wave)
    spec = zero_to_nan(spec)

    """
    De-Red
    """
    not_nan_inds = numpy.where(~numpy.isnan(spec))
    specs_dered_tmp = deredden_spectrum(wave[not_nan_inds], spec[not_nan_inds], E_bv)
    spec[not_nan_inds] = specs_dered_tmp

    """
    Median Filter
    """
    spec = signal.medfilt(spec, 5)

    return wave, spec


def clean_and_deredd_spectra(waves, specs, ds, E_bv):
    """
    This function removes bad pixels, de-reds, normalize and runs median filter on the spectra
    """
    objnum = specs.shape[0]

    for i in range(objnum):
        waves[i], specs[i] = clean_and_deredd_spectrum_single(waves[i], specs[i], ds[i], E_bv[i])

    return waves, specs



def same_grid_single(wave_common, wave_orig, spec_orig):
    """
    Putting a single spectrum on the common wavelength grid
    """
    spec = numpy.interp(wave_common, wave_orig, spec_orig, left=numpy.nan, right=numpy.nan)

    return spec


def same_grid(wave, waves, specs):
    """
    Putting all spectra on the same wavelength grid
    """
    print('Putting all spectra on the same grid wit min lambda = ', wave.min(), 'and max lambda = ', wave.max())

    specs_same_grid = numpy.zeros([specs.shape[0], wave.shape[0]])
    for i in range(waves.shape[0]):
        specs_same_grid[i] = same_grid_single(wave, waves[i], specs[i])

    return specs_same_grid

def check_spectra(spec):
    """
    Check if spectra is good to use.
    If fits file was not downloaded properly spectra will be all nans
    too many bad pixels, or too little coverage on common grip will also be removed
    """

    nof_pixels = len(spec)
    nof_good_pixels = numpy.sum(numpy.isnan(spec))

    is_good = float(nof_good_pixels)/float(nof_pixels) > 0.75

    return is_good

def impute_spec(specs, wave):

    imp = Imputer(missing_values='NaN', strategy='median', axis=0)
    imp = imp.fit(specs)
    specs_imp = imp.transform(specs)
    wave_imp = imp.transform(wave.reshape(1, -1)).T

    return specs_imp, wave_imp


def de_redshift(wave, z):
    """
    Switch to rest frame wave length
    """
    wave = wave / (1 + z)
    return wave

def E_bv_get(object_id):
    """
    E_bv for a single object
    :param m:
    :param object_id:
    :return:
    """

    ra = float(object_id[8])
    dec = float(object_id[9])
    E_bv = m.ebv(ra, dec)

    return E_bv

def E_bv_get_all(objects_ids_list):
    """
    Get the E_bv value for each object in objects_ids_list
    The data is from the sfdmap package
    :param objects_ids_list:
    :return E_bv:
    """
    E_bv = numpy.zeros(len(objects_ids_list))

    for i, object_id in enumerate(objects_ids_list):
        E_bv[i] = E_bv_get(object_id)

    return E_bv