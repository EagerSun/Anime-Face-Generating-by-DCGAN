import numpy as np
import math 
import matplotlib.pyplot as plt

def plot_images(images):
    '''
    images(np.array) is the array includes the anime faces.
    This function is intended to plot a 10*10 subplots where each subplot is a anime face.
    '''
    images = images * 0.5 + 0.5
    if len(images) > 100:
        idx = np.arange(0, len(images))
        np.shuffle(idx)
        idx_pick = idx[:100]
        images_ = images[idx_pick]
    else:
        images_ = images
  
    num_grid = math.ceil(math.sqrt(len(images_)))
    fig, axes = plt.subplots(num_grid, num_grid, figsize = (10, 10))
    for i, ax in enumerate(axes.flat):
        if i < len(images_):
            ax.imshow(images_[i, :, :, :])
            ax.axis('off')
            ax.set_xticks([])
            ax.set_yticks([])

    plt.show()