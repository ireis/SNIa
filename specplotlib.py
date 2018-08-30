import numpy
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
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