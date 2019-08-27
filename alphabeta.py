import pandas
import numpy
import matplotlib.pyplot as plt
import specplotlib
from tqdm import tqdm_notebook as tqdm
from importlib import reload

# functions to find alpha and betha
from scipy import optimize
import pfit
reload(pfit)

def  get_distance_modulos(z_n, cosmo = 'WMAP9'):

    if cosmo == 'WMAP9':
        from astropy.cosmology import WMAP9 as cosmo
    else:
        from astropy.cosmology import WMAP7 as cosmo

    z_distmod = cosmo.distmod(z_n)
    
    return z_distmod

def get_xy_salt2(z_n, mb_n, x1_n, c_n, distmod, mb_err_n, x1_err_n, c_err_n):

    # fitting the linear fit for the choosen groups
    has_fitter = numpy.where(numpy.isfinite(x1_n))[0]

    Y = distmod.value - mb_n.copy()
    X = numpy.vstack([x1_n.copy(), c_n.copy()]).T

    dY = numpy.sqrt(mb_err_n.copy()**2) # should have also error in distmod - do it later  
    dX = numpy.vstack([x1_err_n.copy()**2, c_err_n.copy()**2]).T

    return has_fitter, X, dX, Y, dY


def fit_plot(X, dX, Y, dY, mb_n, m0, alpha, beta, Wclass, umap, name, g = None, low_prec=5, high_prec=95): 
    # if flag_color is not default that means there is also a choosen group g and that we need to mark in plot
    
    has_fitter = numpy.where(numpy.isfinite(X[:,0]))[0]
    x_has = X[has_fitter].copy()
    dx_has = dX[has_fitter].copy()
    mb_has = mb_n[has_fitter].copy()
    y_has = Y[has_fitter].copy()
    dy_has = dY[has_fitter].copy()
    
    x_map = umap[:, 0].copy()
    y_map = umap[:, 1].copy()
    
    a_max_id = numpy.where(alpha == numpy.nanmax(alpha))[0][0]
    a_min_id = numpy.where(alpha == numpy.nanmin(alpha))[0][0]
    b_max_id = numpy.where(beta == numpy.nanmax(beta))[0][0]
    b_min_id = numpy.where(beta == numpy.nanmin(beta))[0][0]
    m0_max_id = numpy.where(m0 == numpy.nanmax(m0))[0][0]
    m0_min_id = numpy.where(m0 == numpy.nanmin(m0))[0][0]
    nnn = 10
    
    from sklearn.metrics.pairwise import euclidean_distances
    d_umap = euclidean_distances(umap, umap)
    d_umap_asort = numpy.argsort(d_umap, axis = 1)[:,:nnn]
    
    g_a_max = d_umap_asort[a_max_id]
    g_a_max_has = numpy.intersect1d(g_a_max, has_fitter)
    x_a_max_has = X[g_a_max_has].copy()
    dx_a_max_has = dX[g_a_max_has].copy()
    y_a_max_has = Y[g_a_max_has].copy()
    dy_a_max_has = dY[g_a_max_has].copy()
    
    g_a_min = d_umap_asort[a_min_id]
    g_a_min_has = numpy.intersect1d(g_a_min, has_fitter)
    x_a_min_has = X[g_a_min_has].copy()
    dx_a_min_has = dX[g_a_min_has].copy()
    y_a_min_has = Y[g_a_min_has].copy()
    dy_a_min_has = dY[g_a_min_has].copy()
    
    g_b_max = d_umap_asort[b_max_id]
    g_b_max_has = numpy.intersect1d(g_b_max, has_fitter)
    x_b_max_has = X[g_b_max_has].copy()
    dx_b_max_has = dX[g_b_max_has].copy()
    y_b_max_has = Y[g_b_max_has].copy()
    dy_b_max_has = dY[g_b_max_has].copy()
    
    g_b_min = d_umap_asort[b_min_id]
    g_b_min_has = numpy.intersect1d(g_b_min, has_fitter)
    x_b_min_has = X[g_b_min_has].copy()
    dx_b_min_has = dX[g_b_min_has].copy()
    y_b_min_has = Y[g_b_min_has].copy()
    dy_b_min_has = dY[g_b_min_has].copy()
    
    g_m0_max = d_umap_asort[m0_max_id]
    g_m0_max_has = numpy.intersect1d(g_m0_max, has_fitter)
    mb_m0_max_has = mb_n[g_m0_max_has].copy()
    y_m0_max_has = Y[g_m0_max_has].copy()
    
    g_m0_min = d_umap_asort[m0_min_id]
    g_m0_min_has = numpy.intersect1d(g_m0_min, has_fitter)
    mb_m0_min_has = mb_n[g_m0_min_has].copy()
    y_m0_min_has = Y[g_m0_min_has].copy()
    
    if g is not None: 
        # it is necessary to cut according to has_fitter and g only in the relations, not in umap,                                       there we present everything not just has fitter 
        g_has_fitter = numpy.intersect1d(g, has_fitter)
        x_g_has = X[g_has_fitter].copy()
        dx_g_has = dX[g_has_fitter].copy()
        mb_g_has = mb_n[g_has_fitter].copy()
        y_g_has = Y[g_has_fitter].copy()
        dy_g_has = dY[g_has_fitter].copy()
    
    # plot the figures
    fig = plt.figure(figsize=(15,50))
    plt.suptitle('%s' %name, fontsize=30, horizontalalignment='center', verticalalignment='top', y=1.01)

    plt.subplot(10,2,1)
    plt.errorbar(x=x_has[:,0], y=y_has, yerr=dy_has, xerr=dx_has[:,0], c='gray', alpha = 0.5,  fmt='o')
    plt.errorbar(x=x_a_max_has[:,0], y=y_a_max_has, yerr=dy_a_max_has, xerr=dx_a_max_has[:,0], c='green', alpha = 0.5,  fmt='o', label = 'max_alpha')
    plt.errorbar(x=x_a_min_has[:,0], y=y_a_min_has, yerr=dy_a_min_has, xerr=dx_a_min_has[:,0], c='red', alpha = 0.5,  fmt='o', label = 'min_alpha')
    if g is not None:
        plt.errorbar(x=x_g_has[:,0], y=y_g_has, yerr=dy_g_has, xerr=dx_g_has[:,0], fmt='o', c='blue')
    plt.xlabel('x1')
    plt.ylabel('M_abs')
    plt.legend()
    
    plt.subplot(10,2,2)
    c_ = alpha.copy()
    low_cut = numpy.nanpercentile(c_, q=low_prec)
    high_cut = numpy.nanpercentile(c_, q=high_prec)
    isfin = numpy.isfinite(c_)
    c_isfin = c_[isfin]
    c_isfin[c_isfin < low_cut] = low_cut
    c_isfin[c_isfin > high_cut] = high_cut
    c_[isfin] = c_isfin    
    plt.scatter(x_map, y_map, color = 'gray') # background = all, no matter if have color param
    plt.scatter(x_map, y_map, c = c_, cmap = 'viridis') # all with color
    plt.colorbar()
    plt.scatter(x_map[g_a_max], y_map[g_a_max], s=100, facecolors='none', edgecolors='green', label='max_alpha')
    plt.scatter(x_map[g_a_min], y_map[g_a_min], s=100, facecolors='none', edgecolors='red', label='min_alpha')
    if g is not None:
        plt.scatter(x_map[g], y_map[g], s=100, facecolors='none', edgecolors='black', label='group')
    plt.title('alpha')
    plt.legend()

    plt.subplot(10,2,3)
    plt.errorbar(x=x_has[:,1], y=y_has, yerr=dy_has, xerr=dx_has[:,1], c='gray', alpha = 0.5,  fmt='o')
    plt.errorbar(x=x_b_max_has[:,1], y=y_b_max_has, yerr=dy_b_max_has, xerr=dx_b_max_has[:,1], c='green', alpha = 0.5,  fmt='o', label = 'max_beta')
    plt.errorbar(x=x_b_min_has[:,1], y=y_b_min_has, yerr=dy_b_min_has, xerr=dx_b_min_has[:,1], c='red', alpha = 0.5,  fmt='o', label = 'min_beta')
    if g is not None:
        plt.errorbar(x=x_g_has[:,1], y=y_g_has, yerr=dy_g_has, xerr=dx_g_has[:,1], fmt='o', c='blue')
    plt.xlabel('c')
    plt.ylabel('M_abs')
    plt.legend()
    
    plt.subplot(10,2,4)
    c_ = beta.copy()
    low_cut = numpy.nanpercentile(c_, q=low_prec)
    high_cut = numpy.nanpercentile(c_, q=high_prec)
    isfin = numpy.isfinite(c_)
    c_isfin = c_[isfin]
    c_isfin[c_isfin < low_cut] = low_cut
    c_isfin[c_isfin > high_cut] = high_cut
    c_[isfin] = c_isfin    
    plt.scatter(x_map, y_map, color = 'gray') # background = all, no matter if have color param
    plt.scatter(x_map, y_map, c = c_, cmap = 'Spectral') # all with color
    plt.colorbar()
    plt.scatter(x_map[g_b_max], y_map[g_b_max], s=100, facecolors='none', edgecolors='green', label='max_beta')
    plt.scatter(x_map[g_b_min], y_map[g_b_min], s=100, facecolors='none', edgecolors='red', label='min_beta')
    if g is not None:
        plt.scatter(x_map[g], y_map[g], s=100, facecolors='none', edgecolors='black', label='group')
    plt.title('beta')
    plt.legend()
    
    plt.subplot(10,2,5)
    plt.scatter(x=mb_has, y=y_has, c='gray', alpha = 0.5)
    plt.scatter(x=mb_m0_max_has, y=y_m0_max_has, c='green', alpha = 0.5, label = 'max_M0')
    plt.scatter(x=mb_m0_min_has, y=y_m0_min_has, c='red', alpha = 0.5, label = 'min_M0')
    if g is not None:
        plt.scatter(x=mb_g_has, y=y_g_has, c='blue')
    plt.xlabel('Mb_obs')
    plt.ylabel('M_abs')
    plt.legend()

    plt.subplot(10,2,6)
    c_ = m0.copy()
    low_cut = numpy.nanpercentile(c_, q=low_prec)
    high_cut = numpy.nanpercentile(c_, q=high_prec)
    isfin = numpy.isfinite(c_)
    c_isfin = c_[isfin]
    c_isfin[c_isfin < low_cut] = low_cut
    c_isfin[c_isfin > high_cut] = high_cut
    c_[isfin] = c_isfin    
    plt.scatter(x_map, y_map, color = 'gray') # background = all, no matter if have color param
    plt.scatter(x_map, y_map, c = c_, cmap = 'inferno') # all with color
    plt.colorbar()
    plt.scatter(x_map[g_m0_max], y_map[g_m0_max], s=100, facecolors='none', edgecolors='green', label='max_M0')
    plt.scatter(x_map[g_m0_min], y_map[g_m0_min], s=100, facecolors='none', edgecolors='red', label='min_M0')
    if g is not None:
        plt.scatter(x_map[g], y_map[g], s=100, facecolors='none', edgecolors='black', label='group')
    plt.title('M0')
    plt.legend()

    plt.subplot(10,2,7)
    c_ = X[:,0].copy()
    low_cut = numpy.nanpercentile(c_, q=low_prec)
    high_cut = numpy.nanpercentile(c_, q=high_prec)
    isfin = numpy.isfinite(c_)
    c_isfin = c_[isfin]
    c_isfin[c_isfin < low_cut] = low_cut
    c_isfin[c_isfin > high_cut] = high_cut
    c_[isfin] = c_isfin    
    plt.scatter(x_map, y_map, color = 'gray') # background = all, no matter if have color param
    plt.scatter(x_map, y_map, c = c_, cmap = 'viridis') # all with color
    plt.colorbar()
    #plt.scatter(x_map[g_a_max], y_map[g_a_max], s=100, facecolors='none', edgecolors='green', label='max_alpha')
    #plt.scatter(x_map[g_a_min], y_map[g_a_min], s=100, facecolors='none', edgecolors='red', label='min_alpha')
    if g is not None:
        plt.scatter(x_map[g], y_map[g], s=100, facecolors='none', edgecolors='black', label='group')
    plt.title('x1')
    #plt.legend()
    
    plt.subplot(10,2,8)
    c_ = X[:,1].copy()
    low_cut = numpy.nanpercentile(c_, q=low_prec)
    high_cut = numpy.nanpercentile(c_, q=high_prec)
    isfin = numpy.isfinite(c_)
    c_isfin = c_[isfin]
    c_isfin[c_isfin < low_cut] = low_cut
    c_isfin[c_isfin > high_cut] = high_cut
    c_[isfin] = c_isfin    
    plt.scatter(x_map, y_map, color = 'gray') # background = all, no matter if have color param
    plt.scatter(x_map, y_map, c = c_, cmap = 'Spectral') # all with color
    plt.colorbar()
    #plt.scatter(x_map[g_b_max], y_map[g_b_max], s=100, facecolors='none', edgecolors='green', label='max_beta')
    #plt.scatter(x_map[g_b_min], y_map[g_b_min], s=100, facecolors='none', edgecolors='red', label='min_beta')
    if g is not None:
        plt.scatter(x_map[g], y_map[g], s=100, facecolors='none', edgecolors='black', label='group')
    plt.title('c')
    #plt.legend()
    
    plt.subplot(10,2,9)
    c_ = Y.copy()
    low_cut = numpy.nanpercentile(c_, q=low_prec)
    high_cut = numpy.nanpercentile(c_, q=high_prec)
    isfin = numpy.isfinite(c_)
    c_isfin = c_[isfin]
    c_isfin[c_isfin < low_cut] = low_cut
    c_isfin[c_isfin > high_cut] = high_cut
    c_[isfin] = c_isfin    
    plt.scatter(x_map, y_map, color = 'gray') # background = all, no matter if have color param
    plt.scatter(x_map, y_map, c = c_, cmap = 'inferno') # all with color
    plt.colorbar()
    #plt.scatter(x_map[g_m0_max], y_map[g_m0_max], s=100, facecolors='none', edgecolors='green', label='max_M0')
    #plt.scatter(x_map[g_m0_min], y_map[g_m0_min], s=100, facecolors='none', edgecolors='red', label='min_M0')
    if g is not None:
        plt.scatter(x_map[g], y_map[g], s=100, facecolors='none', edgecolors='black', label='group')
    plt.title('M_abs')
    #plt.legend()
   
    plt.subplot(10,2,10)
    c_ = numpy.random.rand(len(x_map)) 
    plt.scatter(x_map, y_map, c = c_, cmap = 'Greys', s = 10)
    plt.colorbar()
    inds = numpy.where(Wclass == bytes('N', 'utf-8'))
    plt.scatter(x_map[inds], y_map[inds], s= 100, marker='*', label = 'N')
    inds = numpy.where(Wclass == bytes('91bg', 'utf-8'))
    plt.scatter(x_map[inds], y_map[inds], s= 100, label = '91bg')
    inds = numpy.where(Wclass == bytes('HV', 'utf-8'))
    plt.scatter(x_map[inds], y_map[inds], s= 100, marker='s', label = 'HV')
    inds = numpy.where(Wclass == bytes('91T', 'utf-8'))
    plt.scatter(x_map[inds], y_map[inds], s= 100, marker='^', label = '91T')
    if g is not None:
        plt.scatter(x_map[g], y_map[g], s=100, facecolors='none', edgecolors='black', label='group')
    plt.legend()
    

    plt.tight_layout()
    plt.show()

def fit_nn_salt2(z_n, mb_n, x1_n, c_n, mb_err_n, x1_err_n, c_err_n, nof_nn = 8, all_umap=None):

    nof_objects = len(z_n)
    distmod = get_distance_modulos(z_n)
    
    has_fitter, X, dX, Y, dY = get_xy_salt2(z_n, mb_n, x1_n, c_n, distmod, mb_err_n, x1_err_n, c_err_n)

    Yf = Y[has_fitter].copy()
    Xf = X[has_fitter].copy()
    dYf = dY[has_fitter].copy()
    dXf = dX[has_fitter].copy()

    from sklearn.metrics.pairwise import euclidean_distances
    d_umap = euclidean_distances(all_umap, all_umap)

    alpha = numpy.ones(nof_objects)*numpy.nan
    beta = numpy.ones(nof_objects)*numpy.nan
    M0 = numpy.ones(nof_objects)*numpy.nan

    nnn = nof_nn
    has_fitter_dmat = d_umap[has_fitter,:][:,has_fitter].copy()
    has_fitter_dmat_asort = numpy.argsort(has_fitter_dmat, axis = 1)[:,:nnn]

    for i in tqdm(range(len(has_fitter))): #the use in tqdm is in order to plot the timing bar
        g                 = has_fitter_dmat_asort[i]
        inter, a, b       = pfit.lin_fit_salt2(Xf[g], dXf[g], Yf[g], dYf[g])
        alpha[has_fitter[i]] = a
        beta[has_fitter[i]]  = b
        M0[has_fitter[i]]    = inter

    return M0, alpha, beta, X, dX, Y, dY, has_fitter