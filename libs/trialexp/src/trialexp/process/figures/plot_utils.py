from scipy.cluster.hierarchy import dendrogram
import matplotlib.pylab as plt 
import numpy as np
import PIL
from io import BytesIO

def create_plot_grid(n_plots, ncol, figsize_plot=(3,3), **kwargs):
    nrow = n_plots//ncol + 1
    fig = plt.figure(figsize=(figsize_plot[0]*ncol, figsize_plot[1]*nrow), **kwargs)
    axes = []
    for i in range(n_plots):
        ax = fig.add_subplot(nrow, ncol, i+1)
        axes.append(ax)
    return fig, axes

def make_linkage(model):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)
    
    return linkage_matrix
    

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    linkage_matrix = make_linkage(model)

    # Plot the corresponding dendrogram
    return dendrogram(linkage_matrix, **kwargs)
    
    
def combine_figures(figs, ncol=6):
    #combine figures together, figures are assumed to be the same size
    if len(figs)>0:
        nrow = np.ceil(len(figs)/ncol).astype(int)
        width, height = figs[0].canvas.get_width_height()
        combined = PIL.Image.new('RGB', (width*ncol, height*nrow),'white')
        
        for i, fig in enumerate(figs):
            
            # save the figure as buf
            buf = BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            image = PIL.Image.open(buf)

            # paste to final images
            combined.paste(image, (width*(i%ncol), height*(i//ncol)))
    else:
        combined = PIL.Image.new('RGB', (100,100),'white')

    return combined


def plot_pie_chart(data, col, palette=None):
    ax = plt.gca()
    count = data.groupby(col)[col].count()
    labels = count.index
    sizes = count.values
    if palette is not None:
        colors = [palette[lbl] for lbl in palette]
    else:
        colors = plt.cm.tab20.colors
        
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors)