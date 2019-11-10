import numpy as np

import matplotlib.pyplot as plt

from skimage import img_as_float
from skimage.exposure import cumulative_distribution 


def plot_image(images, titles=None, cmap=None):
    
    if isinstance(images, list) or isinstance(images, tuple):
        pass
    else:
        images = [images]
    
    n_images = len(images)
    
    fig = plt.figure(figsize=(6 * n_images, 6))
    
    for i, image in enumerate(images):
        axis = fig.add_subplot(1, n_images, 1+i)
    
        # If we have a colour image, do not specify a colormap
        if image.ndim == 3:
            if titles is not None:
                title = titles[i]
            else:
                title = 'Colour image of shape {}'.format(image.shape)
            axis.imshow(image)
        else:
            if titles is not None:
                title = titles[i]
            else:
                title = 'Greyscale image of shape {}'.format(image.shape)
            # Else, use the colormap supplied
            if cmap is not None:
                axis.imshow(image, cmap=cmap)

            # Else use a default black and white colourmap
            else:
                axis.imshow(image, cmap=plt.cm.gray)
        
        # Remove the axes and use a tight layout
        axis.set_title(title)
        axis.set_axis_off()
        
    fig.tight_layout()
    
    # Display the image
    plt.show()


def plot_image_and_hist(images, titles, bins=256):
    """Plot an image along with its histogram and cumulative histogram.

    """
    
    n_images = len(images)
    
    # Display results
    fig = plt.figure(figsize=(4 * n_images, 8))
    axes = np.zeros((2, n_images), dtype=np.object)
    
    for i in range(n_images):
        
        if i == 0:
            axes[0, i] = fig.add_subplot(2, n_images, 1)
        else:
            axes[0, i] = fig.add_subplot(2, n_images, 1+i, sharex=axes[0,0], sharey=axes[0,0])

        axes[1, i] = fig.add_subplot(2, n_images, n_images+1+i)
    
    for i, image in enumerate(images):

        image = img_as_float(image)
            
        ax_img, ax_hist = axes[:, i]
        ax_cdf = ax_hist.twinx()
            
        ax_img.set_title(titles[i])
        ax_img.imshow(image, cmap=plt.cm.gray)
        ax_img.set_axis_off()

        # Display histogram
        ax_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
        ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
        ax_hist.set_xlabel('Pixel intensity')
        ax_hist.set_xlim(0, 1)
        ax_hist.set_yticks([])
        
        if i == 0:
            y_min, y_max = ax_hist.get_ylim()
            ax_hist.set_ylabel('Number of pixels')
            ax_hist.set_yticks(np.linspace(0, y_max, 5))
        
        # Display cumulative distribution
        img_cdf, bins = cumulative_distribution(image, bins)
        ax_cdf.plot(bins, img_cdf, 'r')
        ax_cdf.set_yticks([])
    
    ax_cdf.set_ylabel('Fraction of total intensity')
    ax_cdf.set_yticks(np.linspace(0, 1, 5))

    # prevent overlap of y-axis labels
    fig.tight_layout()
    plt.show()
