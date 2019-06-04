import numpy
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from tqdm import tqdm_notebook as tqdm
#from tqdm import tqdm
import umap
cmap = get_cmap('viridis_r')

line_names = ['Si II', 'C II']
line_w =     [6355,  6580]

def reorder_spectra_mat(spectra_matrix, indecies_to_plot, order_by):
    nof_objects = len(indecies_to_plot)
    if len(order_by) > nof_objects:
        order_by = order_by[indecies_to_plot]
    order = numpy.argsort(order_by)
    spectra_matrix = spectra_matrix[indecies_to_plot]

    plot_matrix = spectra_matrix[order]
    return plot_matrix


def sequencer_plot_smooth(CW, spectra_matrix, indecies_to_plot, order_by, smooth=10):
    plot_matrix = reorder_spectra_mat(spectra_matrix.copy(), indecies_to_plot, order_by)
    nof_objects = len(indecies_to_plot)

    for i in range(nof_objects - smooth):
        plot_matrix[i] = numpy.mean(plot_matrix[i:i + smooth], axis=0)

    plt.figure(figsize=(10, 7))
    plt.imshow(plot_matrix[:i], aspect='auto', vmin=0.5, vmax=2, cmap='BrBG')
    plt.colorbar()
    # plt.xlim([228,581])
    #plt.xticks(numpy.array([numpy.argmax(CW > 1216),
    #                        numpy.argmax(CW > 1400),
    #                        numpy.argmax(CW > 1549),
    ##                        numpy.argmax(CW > 1909)]),
    #           ['Lya, NV', 'SiIV, OIV]', 'CIV', 'CIII]'],
    #           fontsize=15)
    plt.show()

    return plot_matrix[:i]


def ladder_plot_smooth(CW, spectra_matrix, indecies_to_plot, order_by, nof_spectra, delta=1):
    plot_matrix = reorder_spectra_mat(spectra_matrix.copy(), indecies_to_plot, order_by)

    n_groups = nof_spectra
    l = int(len(indecies_to_plot) / n_groups) * n_groups
    groups = numpy.split(plot_matrix[:l], n_groups)
    plt.figure(figsize=(10, 10))
    for g_idx, g in enumerate(groups):
        d = delta * g_idx
        x_plt = numpy.nanmedian(g, axis=0)
        plt.plot(CW, x_plt + d, c=cmap(g_idx / n_groups))
    plt.axvline(x = 6143.68)
    plt.axvline(x = 4869.66)
    plt.axvline(x = 4308.88)
    plt.show()
    return


def embedding_plot_groups(embed, SN_df):
    x = embed[:, 0]
    y = embed[:, 1]

    plt.figure(figsize=(10,15))

    plt.subplot(211)
    plt.title("BClass" )

    inds = numpy.where(SN_df['BClass_spec'] == 'CN')
    plt.scatter(x[inds], y[inds], marker='*', label = 'CN')

    inds = numpy.where(SN_df['BClass_spec'] == 'CL')
    plt.scatter(x[inds], y[inds], s= 50, marker='s', label = 'CL')

    inds = numpy.where(SN_df['BClass_spec'] == 'BL')
    plt.scatter(x[inds], y[inds], label = 'BL')

    inds = numpy.where(SN_df['BClass_spec'] == 'SS')
    plt.scatter(x[inds], y[inds], marker='^', label = 'SS')
    #plt.scatter(x, y, c = t)

    plt.legend()


    plt.subplot(212)
    plt.title("WClass" )

    plt.scatter(x, y, color = 'gray', s = 10)
    inds = numpy.where(SN_df['WClass_spec'] == 'N')
    plt.scatter(x[inds], y[inds], s= 50, marker='*', label = 'N')

    inds = numpy.where(SN_df['WClass_spec'] == '91bg')
    plt.scatter(x[inds], y[inds], marker='s', label = '91bg')

    inds = numpy.where(SN_df['WClass_spec'] == 'HV')
    plt.scatter(x[inds], y[inds], label = 'HV')

    inds = numpy.where(SN_df['WClass_spec'] == '91T')
    plt.scatter(x[inds], y[inds], marker='^', label = '91T')


    plt.legend()

    plt.tight_layout()
    plt.show()

    return


def embedding_plot_spec_meas(embed, SN_DF, s=50):
    x = embed[:, 0]
    y = embed[:, 1]

    plt.figure(figsize=(10,20))
    plt.subplot(311)
    plt.title("v6355" )

    plt.scatter(x, y, color = 'gray', s = s)
    #plt.scatter(x[iax_in_df_idx], y[iax_in_df_idx], color ='orange', s= 200, marker='*')
    #plt.scatter(x, y, c = t)
    t = (SN_DF['EW5972_spec'].values).copy() / (SN_DF['EW6355_spec'].values).copy()
    t = SN_DF['v6355_spec'].copy()
    cut = -8000
    t[t > cut] = cut
    plt.scatter(x, y, c = t)
    plt.colorbar()
    #plt.colorbar()


    plt.subplot(312)
    plt.title("EW5972" )

    plt.scatter(x, y, color = 'gray', s = s)
    #plt.scatter(x[iax_in_df_idx], y[iax_in_df_idx], color ='orange', s= 200, marker='*')
    #plt.scatter(x, y, c = t)
    t = SN_DF['EW5972_spec'].values.copy()
    cut = -8000
    #t[t > cut] = cut
    plt.scatter(x, y, c = t)
    plt.colorbar()



    plt.subplot(313)
    plt.title("EW6355" )
    plt.scatter(x, y, color = 'gray', s = s)
    #plt.scatter(x[iax_in_df_idx], y[iax_in_df_idx], color ='orange', s= 200, marker='*')
    #plt.scatter(x, y, c = t)
    t =  (SN_DF['EW6355_spec'].values).copy()
    cut = -8000
    #t[t > cut] = cut
    plt.scatter(x, y, c = t)
    plt.colorbar()

    plt.tight_layout()
    plt.show()

    return


def embedding_plot_lc_params(embed, SN_DF):
    x = embed[:, 0]
    y = embed[:, 1]

    plt.figure(figsize=(10,15))


    plt.subplot(311)
    plt.title("Intrinsic M" )

    plt.scatter(x, y, color = 'gray', s = 10)
    t = (SN_DF['M_B_spec'].values).copy()
    cut = -80
    t[t < cut] = numpy.nan
    plt.scatter(x, y, c = t)
    #plt.gca().invert_yaxis()
    plt.colorbar()
    #plt.colorbar()

    plt.subplot(312)
    plt.title("DM15" )

    plt.scatter(x, y, color = 'gray', s = 10)
    #plt.scatter(x[iax_in_df_idx], y[iax_in_df_idx], color ='orange', s= 200, marker='*')
    #plt.scatter(x, y, c = t)
    t = SN_DF['Dm15_spec'].values.copy()
    cut = 3
    t[t > cut] = numpy.nan
    plt.scatter(x, y, c = t, cmap = 'Spectral')
    plt.colorbar()


    plt.subplot(313)
    plt.title("Color" )

    plt.scatter(x, y, color = 'gray', s = 10)
    #plt.scatter(x[iax_in_df_idx], y[iax_in_df_idx], color ='orange', s= 200, marker='*')
    #plt.scatter(x, y, c = t)
    t = SN_DF['B-V_spec'].values.copy()
    #cut = 3
    #t[t > cut] = numpy.nan
    plt.scatter(x, y, c = t, cmap = 'plasma')
    plt.colorbar()


    plt.tight_layout()
    plt.show()

    return

def embedding_plot_lc_params(embed, SN_DF):
    x = embed[:, 0]
    y = embed[:, 1]

    plt.figure(figsize=(10,15))


    plt.subplot(311)
    plt.title("Intrinsic M" )

    plt.scatter(x, y, color = 'gray', s = 10)
    t = (SN_DF['M_B_spec'].values).copy()
    cut = -80
    t[t < cut] = numpy.nan
    plt.scatter(x, y, c = t)
    #plt.gca().invert_yaxis()
    plt.colorbar()
    #plt.colorbar()

    plt.subplot(312)
    plt.title("DM15" )

    plt.scatter(x, y, color = 'gray', s = 10)
    #plt.scatter(x[iax_in_df_idx], y[iax_in_df_idx], color ='orange', s= 200, marker='*')
    #plt.scatter(x, y, c = t)
    t = SN_DF['Dm15_spec'].values.copy()
    cut = 3
    t[t > cut] = numpy.nan
    plt.scatter(x, y, c = t, cmap = 'Spectral')
    plt.colorbar()


    plt.subplot(313)
    plt.title("Color" )

    plt.scatter(x, y, color = 'gray', s = 10)
    #plt.scatter(x[iax_in_df_idx], y[iax_in_df_idx], color ='orange', s= 200, marker='*')
    #plt.scatter(x, y, c = t)
    t = SN_DF['B-V_spec'].values.copy()
    #cut = 3
    #t[t > cut] = numpy.nan
    plt.scatter(x, y, c = t, cmap = 'plasma')
    plt.colorbar()


    plt.tight_layout()
    plt.show()

    return

def embedding_plot_salt2_lc_params(embed, SN_DF):
    x = embed[:, 0]
    y = embed[:, 1]

    plt.figure(figsize=(10,15))


    plt.subplot(311)
    plt.title("Intrinsic M" )

    plt.scatter(x, y, color = 'gray', s = 10)
    t = (SN_DF['m_B_salt2'].values).copy() - (SN_DF['mu'].values).copy()
    cut = -80
    t[t < cut] = numpy.nan
    plt.scatter(x, y, c = t)
    #plt.gca().invert_yaxis()
    plt.colorbar()
    #plt.colorbar()

    plt.subplot(312)
    plt.title("x_1" )

    plt.scatter(x, y, color = 'gray', s = 10)
    #plt.scatter(x[iax_in_df_idx], y[iax_in_df_idx], color ='orange', s= 200, marker='*')
    #plt.scatter(x, y, c = t)
    t = SN_DF['x_1_salt2'].values.copy()
    #cut = 3
    #t[t > cut] = numpy.nan
    plt.scatter(x, y, c = t, cmap = 'Spectral')
    plt.colorbar()


    plt.subplot(313)
    plt.title("c" )

    plt.scatter(x, y, color = 'gray', s = 10)
    #plt.scatter(x[iax_in_df_idx], y[iax_in_df_idx], color ='orange', s= 200, marker='*')
    #plt.scatter(x, y, c = t)
    t = SN_DF['c_salt2'].values.copy()
    #cut = 3
    #t[t > cut] = numpy.nan
    plt.scatter(x, y, c = t, cmap = 'plasma')
    plt.colorbar()


    plt.tight_layout()
    plt.show()

    return

def get_title(fitter = 'SALT'):
    if fitter == 'SALT':
        ttl = r'$M_{B} = M + \alpha \times (s-1) - \beta \times c$'
    elif fitter == 'MLCS':
        ttl = r'$M_{B} = M + \alpha \times \Delta + \beta \times A_{v}$'
    else:
        ttl = ''
    return ttl

def embedding_plot_fit_params(embed, m0, alpha, beta, s = 50, fitter='SALT'):
    x = embed[:, 0]
    y = embed[:, 1]

    plt.figure(figsize=(10,20))

    ttl = get_title(fitter)



    plt.subplot(311)
    plt.title(ttl + r' --- $\alpha$', fontsize = 20 )

    plt.scatter(x, y, color = 'gray', s = 10)

    t = alpha
    plt.scatter(x, y, c = t, cmap = 'coolwarm', s = s)
    plt.colorbar()


    plt.subplot(312)
    plt.title(ttl + r' --- $\beta$', fontsize = 20 )

    plt.scatter(x, y, color = 'gray', s = 10)
    t = beta
    plt.scatter(x, y, c = t, cmap = 'Spectral', s = s)
    plt.colorbar()

    plt.subplot(313)
    plt.title(ttl + " --- M", fontsize = 20)

    plt.scatter(x, y, color = 'gray', s = 10)
    t = (m0).copy()
    
    plt.scatter(x, y, c = t, s = s, cmap = 'viridis_r')
    plt.colorbar()



    plt.tight_layout()
    plt.show()

    return

def umap_param_scan(X, metric = 'euclidean', save=False,c = None):

    prep_list = [3, 5, 10, 25] #how many "nearest neighbors" to consider
    learning_rate = [0.01, 0.1, 0.25, 0.5] #which gap to take between calculations of minima (resolution)
    plt.figure(figsize = (10,10))

    count = 1
    for prep_val in tqdm(prep_list):
        for learn_val in learning_rate:
            t_umap = umap.UMAP(n_neighbors=prep_val, metric=metric,
                          min_dist=learn_val).fit_transform(X)

            x = t_umap[:, 0]
            y = t_umap[:, 1]
            plt.subplot(4, 4, count)
            plt.title("nn=%s, md=%s" % (prep_val, learn_val))
            if c is None:
                plt.scatter(x, y, s=6)
            else:
                uc = numpy.unique(c)
                uc = uc[numpy.isfinite(uc)]
                for uc_ in uc:
                    inds = numpy.where(c == uc_)[0]
                    plt.scatter(x[inds], y[inds], s=20)
            plt.axis('off')
            if save:
                numpy.save('umap_{}_{}'.format(prep_val, learn_val), t_umap)
            
            count += 1

    plt.show()

    return

def embedding_plot_salt_fit_params(s_umap, alpha, beta, m0, s = 50):
        
    x = s_umap[:, 0]
    y = s_umap[:, 1]

    plt.figure(figsize=(10,20))


    plt.subplot(311)
    plt.title(r'$M_{B} = M + \alpha \times (s-1) - \beta \times c$ --- $\alpha$', fontsize = 20 )

    plt.scatter(x, y, color = 'gray', s = 10)
    #plt.scatter(x[iax_in_df_idx], y[iax_in_df_idx], color ='orange', s= 200, marker='*')
    #plt.scatter(x, y, c = t)
    t = alpha.copy()
    #t = SN_df_w_salt_w_class['c'].values.copy()
    #cut = 0.2
    #t[t > cut] = cut
    plt.scatter(x, y, c = t, cmap = 'inferno', s = 100)
    plt.colorbar()
    plt.tight_layout()

    plt.subplot(312)
    plt.title(r'$M_{B} = M + \alpha \times (s-1) - \beta \times c$ --- $\beta$', fontsize = 20 )


    plt.scatter(x, y, color = 'gray', s = 10)
    #plt.scatter(x[iax_in_df_idx], y[iax_in_df_idx], color ='orange', s= 200, marker='*')
    #plt.scatter(x, y, c = t)
    t = beta.copy()
    #cut = 6
    #t[t < cut] = cut
    #cut = 0.2
    #t[t > cut] = cut
    plt.scatter(x, y, c = t, cmap = 'Spectral', s = 100)
    plt.colorbar()

    plt.subplot(313)
    plt.title(r'$M_{B} = M + \alpha \times (s-1) - \beta \times c$ --- $M$', fontsize = 20 )

    plt.scatter(x, y, color = 'gray', s = 10)
    #plt.scatter(x[iax_in_df_idx], y[iax_in_df_idx], color ='orange', s= 200, marker='*')
    #plt.scatter(x, y, c = t)
    t = m0.copy()
    #cut = 6
    #t[t < cut] = cut
    #cut = 19.25
    #t[t < cut] = cut
    plt.scatter(x, y, c = t, cmap = 'viridis_r', s = 100)

    plt.colorbar()
    plt.tight_layout()
    plt.show()
    
    return


def embedding_plot_time_params(s_umap,  m0, ss):
    
    x = s_umap[:, 0]
    y = s_umap[:, 1]
    
    plt.figure(figsize=(10,7))
    plt.title(r'time from peak', fontsize = 20)

    plt.scatter(x, y, color = 'gray', s = 10)
    #plt.scatter(x[iax_in_df_idx], y[iax_in_df_idx], color ='orange', s= 200, marker='*')
    #plt.scatter(x, y, c = t)
    t = numpy.load('sn_spec_time.npy') 
    #cut = 6
    #t[t < cut] = cut
    #cut = 19.25
    #t[t < cut] = cut
    plt.scatter(x, y, c = t, cmap = 'viridis_r', s=ss)

    plt.colorbar()
    plt.tight_layout()
    plt.show()
    
    return

