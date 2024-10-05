import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


# Global variables to hold state
im = None
ax = None
cb = None
frame_index = 0

def try_init(data):
    global im, ax, cb
    if ax is not None:
        return
    plt.ion()
    ax = plt.gca()
    shape = data.shape
    if len(shape) == 1:
        [im,] = ax.plot(range(shape[0]), data)
        ax.set_ylim([np.min(data), np.max(data)])
    else:
        norm = Normalize(vmin=np.min(data), vmax=np.max(data))
        im = ax.imshow(data, aspect='auto', cmap='viridis', norm = norm)
        ax.xaxis.set_ticks_position('bottom')
        ax.invert_yaxis()
        cb = plt.colorbar(im, ax=ax)
        cb.set_label('Intensity')  # 设置 colorbar 标签

update_count = 0
def update(data, sync_mode=False):
    global frame_index, update_count
    update_count = update_count + 1
    # if update_count % 10 != 0: return
    [xsize, ysize] = data.shape
    lb, ub = np.min(data), np.max(data)
    print(f"frame-{frame_index}:\n    min:{lb}\n    max:{ub}\n    sum:{np.sum(data)}\n")
    frame_index = frame_index + 1
    # Update the image data (use a 1D slice of the matrix for simplicity)
    # data = data[xsize // 2]
    # ysize = 1
    
    try_init(data)
    
    if ysize == 1: 
        im.set_ydata(data)
    else:
        # data = (data - lb)/(ub -lb)
        im.set_array(data)
    
    # im.norm = Normalize(vmin=lb, vmax=ub)
    # cb.update_normal(im) 
    plt.draw()
    
    if sync_mode:
        plt.ioff()
        plt.show(block=True)
    else:
        plt.pause(0.1)

# Function to clean up and close the plot
def close_plot():
    plt.close()

def regist_on_close(callback_on_close):
    cid = ax.figure.canvas.mpl_connect('close_event', callback_on_close)