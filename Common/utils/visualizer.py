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
    # print(f"frame-{frame_index}:\n    min:{lb}\n    max:{ub}\n    sum:{np.sum(data)}\n")
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

mouse_pressed = False
start_x, start_y = None, None
move_x, move_y= None, None
def regist_click_and_motion(click = None, motion = None):
    def clamp(value, min_value, max_value):
        return max(min_value, min(value, max_value))
    def to_data_coord(px, py):
        data_x, data_y = ax.transData.inverted().transform((px, py))
        data_x = clamp(data_x, ax.viewLim.x0, ax.viewLim.x1)
        data_y = clamp(data_y, ax.viewLim.y0, ax.viewLim.y1)
        move_x, move_y = data_x, data_y
        return data_x, data_y   
    def on_button_press(event):
        if not event.inaxes: return 
        global mouse_pressed, start_x, start_y, move_x, move_y
        mouse_pressed = True
        start_x, start_y = event.xdata, event.ydata
        move_x, move_y = start_x, start_y 
        if None != click : 
            click(int(event.button), int(0), float(start_x), float(start_y))
        # else:
            print(f'press   from point ({start_x:.2f}, {start_y:.2f})')
    def on_button_release(event):
        global mouse_pressed, start_x, start_y, move_x, move_y
        if not mouse_pressed: return
        data_x, data_y = to_data_coord(event.x, event.y)
        if None != click : 
            click(int(event.button), int(1), float(data_x), float(data_y))
        #else:
            print(f'release from point ({data_x:.2f}, {data_y:.2f})')
        mouse_pressed = False
        start_x, start_y = None, None
        move_x, move_y = None,None
    def on_motion(event):
        global move_x, move_y
        if not mouse_pressed or not event.inaxes: return 
        dx = event.xdata - move_x
        dy = event.ydata - move_y
        if None != motion : 
            motion(float(move_x), float(move_y), float(dx), float(dy))
        else : 
            print(f'move    from point ({move_x:.2f}, {move_y:.2f}) \t dir ({dx:.2f}, {dy:.2f}) ')
        move_x, move_y = event.xdata, event.ydata
        
    ax.figure.canvas.mpl_connect('button_press_event', on_button_press)
    ax.figure.canvas.mpl_connect('motion_notify_event', on_motion)
    ax.figure.canvas.mpl_connect('button_release_event', on_button_release)

    
def regist_on_close(callback_on_close):
    cid = ax.figure.canvas.mpl_connect('close_event', callback_on_close)
def regist_mouse_event(click, motion):
    regist_click_and_motion(click, motion)
