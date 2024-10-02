import numpy as np
import matplotlib.pyplot as plt

frame_index = 0
im = None
ax = None

def try_init(data):
    global im
    global ax
    if None != ax: return
    plt.ion()
    ax = plt.gca()
    shape = data.shape
    # 如果是1D数据，将其扩展为2D形式进行显示
    if len(shape) == 1:
        [im,] = ax.plot(range(shape[0]), data)
        ax.set_ylim([np.min(data), np.max(data)])
        # im = ax.imshow(np.zeros((shape[0], 1)), aspect='auto', cmap='hot')
    else:
        im, = ax.imshow(data, aspect='auto', cmap='hot')
        ax.xaxis.set_ticks_position('bottom')
        ax.invert_yaxis()
        plt.colorbar(im)

def normization(data):
    min_data = np.min(data)
    max_data = np.max(data)
    normalized_data = ((data - min_data) / (max_data - min_data))
    return normalized_data 
def update(data):
    global frame_index
    [xsize, ysize] = data.shape 
    
    print("frame-%d:\n    min:%f\n    max:%f\n    sum:%f\n" % (
            frame_index,
            np.min(np.min(data)),
            np.max(data),
            np.sum(data)
        )
    )
    # 更新图像数据
    data = data[xsize//2]
    ysize  = 1
    
    try_init(data)
    # 如果是1D数据，调整显示格式
    if ysize == 1:
        im.set_ydata(data)
        # im.set_array(data.reshape(1, -1))  # 确保是2D形状
    else:
        im.set_array(data)
        
    plt.draw()
    plt.pause(0.1)  # 暂停一点时间以刷新显示
    frame_index += 1
