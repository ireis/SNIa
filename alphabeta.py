# functions to find alpha and betha
from scipy import optimize

# chi of distance modulus from cosmolgy vs. fit
def func(z, *p):
    m0, a, b = z # this is absolute M
    x, dx, y, dy = p
    chi2 = (-m0 + (x[:,0]-1)*a - x[:,1]*b - y)**2
    chi2 = chi2/( (dx[:,0]*a)**2 + (dx[:,1]*b)**2 + dy**2 )
    return numpy.sum(chi2)


# do the fit for m0,a,b (in this order) according to ranges of slice
def lin_fit(x, dx, y, dy):
    p = (x, dx, y, dy)
    rranges = (slice(-20, -19, 0.05), slice(0.5, 2.5, 0.05), slice(2, 3, 0.05))
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

# fitting the linear fit
has_m_b = numpy.where(  (numpy.isfinite(SN_df_w_salt_w_class['s_salt'].values)))[0]

Y = z_CMB_distmod.value - SN_df_w_salt_w_class['m_B_salt'].values.copy()
X = numpy.vstack([SN_df_w_salt_w_class['s_salt'].values, SN_df_w_salt_w_class['c_salt'].values]).T

dY = numpy.sqrt(SN_df_w_salt_w_class['mu_err_salt'].values.copy()**2  + SN_df_w_salt_w_class['m_B_err_salt'].values.copy()**2)
dX = numpy.vstack([SN_df_w_salt_w_class['s_err_salt'].values, SN_df_w_salt_w_class['c_err_salt'].values]).T

Yf = Y[has_m_b].copy()
Xf = X[has_m_b].copy()
dYf = dY[has_m_b].copy()
dXf = dX[has_m_b].copy()

from sklearn.metrics.pairwise import euclidean_distances
d_umap = euclidean_distances(s_umap, s_umap)

alpha = numpy.ones(nof_objects)*numpy.nan
beta = numpy.ones(nof_objects)*numpy.nan
m0 = numpy.ones(nof_objects)*numpy.nan

nnn = 8
has_m_b_dmat = d_umap[has_m_b,:][:,has_m_b].copy()
has_m_b_dmat_asort = numpy.argsort(has_m_b_dmat, axis = 1)[:,:nnn]

for i in tqdm(range(len(has_m_b))):
    g                 = has_m_b_dmat_asort[i]
    inter, a, b       = group_fit(g, Xf, dXf, Yf, dYf, flag_plot=False)
    alpha[has_m_b[i]] = a
    beta[has_m_b[i]]  = b
    m0[has_m_b[i]]    = inter