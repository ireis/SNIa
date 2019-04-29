from scipy import optimize
import numpy
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm_notebook as tqdm


def func(z, *p):
    m0, a, b = z
    x, dx, y, dy = p
    chi2 = (m0 + x[:,0]*a + x[:,1]*b - y)**2
    chi2 = chi2/( (dx[:,0]*a)**2 + (dx[:,1]*b)**2 + dy**2 )
    return numpy.sum(chi2)



def lin_fit(x, dx, y, dy):
    p = (x, dx, y, dy)
    rranges = (slice(19, 19.5, 0.01), slice(0, 0.2, 0.01), slice(-3, -1.5, 0.05))
    resbrute = optimize.brute(func, rranges, args=p, full_output=True, finish=optimize.fmin)

    return resbrute[0]

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
    plt.xlabel(r'$x_1$', fontsize = 20)
    plt.ylabel(r'$M_B$', fontsize = 20)

    plt.subplot(212)
    x_plt = X[:,1].copy()
    dx_plt = dX[:,1].copy()
    plt.errorbar(x=x_plt, y=y_plt, yerr=dy_plt, xerr=dx_plt, c='gray', alpha = 0.5,  fmt='o')
    for g in gs:
        plt.errorbar(x=x_plt[g], y=y_plt[g], yerr=dy_plt[g], xerr=dx_plt[g],  fmt='o')
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.xlabel(r'$c$', fontsize = 20)
    plt.ylabel(r'$M_B$', fontsize = 20)

    plt.tight_layout()
    plt.show()

    return


def group_fit(g, X, dX, Y, dY, flag_plot = True, flag_color = 'default'):

    x_ = X[g].copy()
    dx_ = dX[g].copy()
    y_ = Y[g].copy()
    dy_ = dY[g].copy()
    res = lin_fit(x_, dx_, y_, dy_)

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


def embed_nn_fit(embed, SN_df):

    d_umap = euclidean_distances(embed, embed)
    has_lc_fit = numpy.where(  numpy.isfinite(SN_df['c'].values) & numpy.isfinite(SN_df['m_B'].values)  )[0]

    Y = SN_df['m_B'].values.copy() - SN_df['mu'].values
    X = numpy.vstack([SN_df['x_1'].values, SN_df['c'].values]).T

    dY = numpy.sqrt(SN_df['+/-.4_y'].values.copy()**2  + SN_df['+/-.1_y'].values.copy()**2)
    dX = numpy.vstack([SN_df['+/-.2_y'].values, SN_df['+/-.3_y'].values]).T

    Yf = Y[has_lc_fit].copy()
    Xf = X[has_lc_fit].copy()
    dYf = dY[has_lc_fit].copy()
    dXf = dX[has_lc_fit].copy()

    nof_objects = embed.shape[0]
    alpha = numpy.ones(nof_objects)*numpy.nan
    beta = numpy.ones(nof_objects)*numpy.nan
    m0 = numpy.ones(nof_objects)*numpy.nan

    nnn = 10
    has_lc_fit_dmat = d_umap[has_lc_fit,:][:,has_lc_fit].copy()
    has_lc_fit_dmat_asort = numpy.argsort(has_lc_fit_dmat, axis = 1)[:,:nnn]

    for i in tqdm(range(len(has_lc_fit))):
        g                    = has_lc_fit_dmat_asort[i]
        inter, a, b          = group_fit(g, Xf, dXf, Yf, dYf, flag_plot=False)
        alpha[has_lc_fit[i]] = a
        beta[has_lc_fit[i]]  = b
        m0[has_lc_fit[i]]    = inter

    return alpha, beta, m0
