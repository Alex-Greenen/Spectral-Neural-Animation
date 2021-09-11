import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
from mpl_toolkits.mplot3d import art3d     

import torch
import typing
import numpy as np

from ProcessData.Utils import * 
from ProcessData.Skeleton import * 

def skeleton_plot(pose_tm1:torch.Tensor, poseDimDiv:typing.List[int], skeleton:Skeleton, rotation:rotType, save_filename:str = None):

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    R, _, _, _ = torch.split(pose_tm1, poseDimDiv, dim=-1)
    R = reformat(R)
    R = correct(R, rotation)
    X = np.reshape(getX(R, skeleton, rotation).clone().detach().cpu().numpy()[0], (-1,3))
    for i in range(len(X)):
        p = skeleton._parents[i]
        if p != -1:
            ax.plot([X[i,0], X[p,0]], [X[i,1], X[p,1]], [X[i,2], X[p,2]], alpha = 1.0, c='black')

    ax.set_xlim3d(-100, 100)
    ax.set_ylim3d(-100, 100)
    ax.set_zlim3d(0, 200)

    if save_filename is not None :
        plt.savefig(save_filename, bbox_inches="tight",transparent=False)

    plt.show()

def motion_animation(X_in:torch.Tensor, parents, epoch:int, folder:str, zoom:float = 1) -> animation.FuncAnimation:
    """
    To Use:
        from IPython.display import HTML
        import matplotlib.animation as animation
        from matplotlib.animation import FuncAnimation, PillowWriter
        %matplotlib qt
        anim = motion_animation(X, parents)
        anim.save("__name__of__anim__.gif", writer = PillowWriter(fps = 30))
        HTML(anim.tojshtml())
    """

    X = X_in.detach().numpy()

    fig = plt.figure()
    #ax = fig.add_subplot(projection='3d')
    ax = p3.Axes3D(fig)
    fig.add_axes(ax)

    lines1 = np.zeros((len(parents)-1, 2, 3))

    counter = 0
    zxy = [1,2,0]
    for i in range(len(parents)):
        p = parents[i]
        if p != -1:
            lines1[counter,0,:] = X[0, i, zxy]
            lines1[counter,1,:] = X[0, p, zxy]
            counter += 1

    lc1 = art3d.Line3DCollection(lines1, linewidths=2, color='g')
    ax.add_collection3d(lc1)

    ax.set_xlim3d(-200 * zoom, 200 * zoom)
    ax.set_ylim3d(-200 * zoom, 200 * zoom)
    ax.set_zlim3d(0, 400 * zoom)
    
    ax.set_xlabel('X', fontsize=30)
    ax.set_ylabel('Y', fontsize=30)
    ax.set_zlabel('Z', fontsize=30)


    def update_lines(num):

        segments1 = np.zeros((len(parents)-1, 2, 3))

        counter = 0
        for i in range(len(parents)):
            p = parents[i]
            if p != -1:
                segments1[counter,0,:] = X[num, i, zxy]
                segments1[counter,1,:] = X[num, p, zxy]
                counter += 1

        lc1.set_segments(segments1)

    anim = animation.FuncAnimation(fig, update_lines, frames = len(X),interval=30, blit=False)
    anim.save(folder + "Animation_{}.gif".format(epoch), writer = animation.PillowWriter(fps = 30))
    
    return anim

    