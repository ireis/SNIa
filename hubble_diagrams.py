import pandas
import numpy
import matplotlib.pyplot as plt
import specplotlib
from tqdm import tqdm_notebook as tqdm
import pfit
from importlib import reload
reload(pfit)

# functions to find alpha and betha
from scipy import optimize

# chi of distance modulus from cosmolgy vs. fit




def plot_groups(gs, X, dX, Y, dY):

    plt.figure(figsize=(15,12))
    y_plt = Y.copy()
    dy_plt = dY.copy()

    plt.subplot(211)
    x_plt = X[:,0].copy()
    dx_plt = dX[:,0].copy()
    plt.errorbar(x=x_plt, y=y_plt, yerr=dy_plt, xerr=dx_plt, c='gray', alpha = 0.5,  fmt='o')
    for g in gs:
        plt.errorbar(x=x_plt[g], y=y_plt[g], yerr=dy_plt[g], xerr=dx_plt[g],  fmt='o')
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.xlabel(r'$s_salt$', fontsize = 20)
    plt.ylabel(r'$m_B intrinsic$', fontsize = 20)

    plt.subplot(212)
    x_plt = X[:,1].copy()
    dx_plt = dX[:,1].copy()
    plt.errorbar(x=x_plt, y=y_plt, yerr=dy_plt, xerr=dx_plt, c='gray', alpha = 0.5,  fmt='o')
    for g in gs:
        plt.errorbar(x=x_plt[g], y=y_plt[g], yerr=dy_plt[g], xerr=dx_plt[g],  fmt='o')
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.xlabel(r'$c_salt$', fontsize = 20)
    plt.ylabel(r'$m_B intrinsic$', fontsize = 20)

    plt.tight_layout()
    plt.show()

    return


def group_fit(g, X, dX, Y, dY, fitter = 'SALT',flag_plot = True, flag_color = 'default'):

    x_ = X[g].copy()
    dx_ = dX[g].copy()
    y_ = Y[g].copy()
    dy_ = dY[g].copy()
    if fitter == 'SALT':
        res = pfit.lin_fit_salt(x_, dx_, y_, dy_)
    elif fitter == 'MLCS':
        res = pfit.lin_fit_mlcs(x_, dx_, y_, dy_)

    if flag_plot:

        plt.figure(figsize=(15,12))
        y_plt = Y.copy()
        dy_plt = dY.copy()

        plt.subplot(211)
        x_plt = X[:,0].copy()
        dx_plt = dX[:,0].copy()
        plt.errorbar(x=x_plt, y=y_plt, yerr=dy_plt, xerr=dx_plt, c='gray', alpha = 0.5,  fmt='o')
        if flag_color != 'default' :
            plt.errorbar(x=x_plt[g], y=y_plt[g], yerr=dy_plt[g], xerr=dx_plt[g],  fmt='o', c = flag_color)

        plt.subplot(212)
        x_plt = X[:,1].copy()
        dx_plt = dX[:,1].copy()
        plt.errorbar(x=x_plt, y=y_plt, yerr=dy_plt, xerr=dx_plt, c='gray', alpha = 0.5,  fmt='o')
        if flag_color != 'default' :
            plt.errorbar(x=x_plt[g], y=y_plt[g], yerr=dy_plt[g], xerr=dx_plt[g],  fmt='o', c = flag_color)
        plt.show()

    return res

def  get_distance_modulos(SN_df, z = 'CMB', cosmo = 'WMAP9'):

    if cosmo == 'WMAP9':
        from astropy.cosmology import WMAP9 as cosmo
    else:
        from astropy.cosmology import WMAP7 as cosmo

    if z == 'CMB':
        z_distmod = cosmo.distmod(SN_df['z_cmb_salt'].values)
    else:
        z_distmod = cosmo.distmod(SN_df['zhel_spec'].values)

    return z_distmod

def get_xy_salt(SN_df, z_distmod):

    # fitting the linear fit for the shoocen groups
    has_fitter = numpy.where((numpy.isfinite(SN_df['s_salt'].values)))[0]

    Y = z_distmod.value - SN_df['m_B_salt'].values.copy()
    X = numpy.vstack([SN_df['s_salt'].values, SN_df['c_salt'].values]).T

    dY = numpy.sqrt(SN_df['mu_err_salt'].values.copy()**2 + SN_df['m_B_err_salt'].values.copy()**2)
    dX = numpy.vstack([SN_df['s_err_salt'].values, SN_df['c_err_salt'].values]).T

    return has_fitter, X, dX, Y, dY

def get_xy_salt2(SN_df):

    raise(ValueError('{} Not implemented it'.format(get_xy_salt2)))

    return


def get_xy_mlcs31(SN_df, z_distmod):

    # fitting the linear fit for the shoocen groups
    has_fitter = numpy.where((numpy.isfinite(SN_df['Delta_ml31'].values) & (numpy.isfinite(SN_df['m_B_salt'].values) ))  )[0]

    Y = z_distmod.value - SN_df['m_B_salt'].values.copy()
    X = numpy.vstack([SN_df['Delta_ml31'].values, SN_df['A_V_ml31'].values]).T

    dY = numpy.sqrt(SN_df['mu_err_salt'].values.copy()**2 + SN_df['m_B_err_salt'].values.copy()**2)
    dX = numpy.vstack([SN_df['Delta_err_ml31'].values, SN_df['A_V_err_ml31'].values]).T

    return has_fitter, X, dX, Y, dY



def fit_nn(fitter = 'SALT', nof_nn = 8, SN_df=None, s_umap=None):

    nof_objects = SN_df.shape[0]

    z_distmod = get_distance_modulos(SN_df)

    if fitter == 'SALT':
        has_fitter, X, dX, Y, dY = get_xy_salt(SN_df, z_distmod)
    elif fitter == 'MLCS':
        has_fitter, X, dX, Y, dY = get_xy_mlcs31(SN_df, z_distmod)

    Yf = Y[has_fitter].copy()
    Xf = X[has_fitter].copy()
    dYf = dY[has_fitter].copy()
    dXf = dX[has_fitter].copy()

    from sklearn.metrics.pairwise import euclidean_distances
    d_umap = euclidean_distances(s_umap, s_umap)

    alpha = numpy.ones(nof_objects)*numpy.nan
    beta = numpy.ones(nof_objects)*numpy.nan
    m0 = numpy.ones(nof_objects)*numpy.nan

    nnn = nof_nn
    has_fitter_dmat = d_umap[has_fitter,:][:,has_fitter].copy()
    has_fitter_dmat_asort = numpy.argsort(has_fitter_dmat, axis = 1)[:,:nnn]

    for i in tqdm(range(len(has_fitter))): #the use in tqdm is in order to plot the timing bar
        g                 = has_fitter_dmat_asort[i]
        inter, a, b       = group_fit(g, Xf, dXf, Yf, dYf, fitter = fitter, flag_plot=False)
        alpha[has_fitter[i]] = a
        beta[has_fitter[i]]  = b
        m0[has_fitter[i]]    = inter

    return m0, alpha, beta










    return res
