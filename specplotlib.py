import numpy
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
cmap = get_cmap('viridis_r')

def reorder_spectra_mat(spectra_matrix, indecies_to_plot, order_by):
    nof_objects = len(indecies_to_plot)
    if len(order_by) > nof_objects:
        order_by = order_by[indecies_to_plot]
    order = numpy.argsort(order_by)
    spectra_matrix = spectra_matrix[indecies_to_plot]

    plot_matrix = spectra_matrix[order]
    return plot_matrix


def sequencer_plot_smooth(spectra_matrix, indecies_to_plot, order_by, smooth=10):
    plot_matrix = reorder_spectra_mat(spectra_matrix.copy(), indecies_to_plot, order_by)
    nof_objects = len(indecies_to_plot)

    for i in range(nof_objects - smooth):
        plot_matrix[i] = numpy.mean(plot_matrix[i:i + smooth], axis=0)

    plt.figure(figsize=(10, 7))
    plt.imshow(plot_matrix[:i], aspect='auto', vmin=0.5, vmax=2, cmap='BrBG')
    plt.colorbar()
    # plt.xlim([228,581])
    plt.show()

    return plot_matrix[:i]



def ladder_plot_smooth(spectra_matrix, indecies_to_plot, order_by, nof_spectra, delta=1):
    plot_matrix = reorder_spectra_mat(spectra_matrix.copy(), indecies_to_plot, order_by)

    n_groups = nof_spectra
    l = int(len(order_by) / n_groups) * n_groups
    groups = numpy.split(plot_matrix[:l], n_groups)
    plt.figure(figsize=(10, 10))
    for g_idx, g in enumerate(groups):
        d = delta * g_idx
        x_plt = numpy.nanmedian(g, axis=0)
        plt.plot(x_plt + d, c=cmap(g_idx / n_groups))
    plt.show()
    return