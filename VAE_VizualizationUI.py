import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D
from torch.utils import data

from ProcessData.Utils import *
from ProcessData.Skeleton import Skeleton
from VAE.VAE import VAE
from VAE.DataBase import DataBase
import Format

name = Format.name
model_dir = 'VAE/VAE_{}.pymodel'.format(name)
rotation = Format.rotation
latent_dim = Format.latentDim

fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
ax.set_xlim3d(-100, 100)
ax.set_ylim3d(-100, 100)
ax.set_zlim3d(0, 200)
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

database = DataBase('TrainingData')
latent = torch.zeros(latent_dim).type(torch.FloatTensor).unsqueeze(0)
axcolor = 'lightgoldenrodyellow'
latentaxs = []
sliders = []
#jointScatter = ax.scatter([1]*22, [1]*22, [1]*22, c='g')

limbPlots = [ax.plot([0.,0.], [0.,0.], [0.,0.], c='g')[0] for i in range(21)]

skeleton = Format.skeleton

model = VAE(database.featureDim, database.poseDim, latent_dim)
model.load_state_dict(torch.load(model_dir))
model.eval()

def update(val):
    latent = torch.tensor([s.val for s in sliders]).type(torch.FloatTensor).unsqueeze(0)

    R = torch.split(model.decoder(latent), database.poseDimDiv, dim=-1)[0]
    R = reformat(R)
    R = correct(R, rotation)
    X = np.reshape(getX(R, skeleton, rotation, pose_offset = torch.tile(torch.tensor([50, 0, 0]), (1,1))).clone().detach().cpu().numpy()[0], (-1,3))

    #jointScatter.set_data(X[:,2], X[:,1]) # set x, y
    #jointScatter.set_3d_properties(X[:,0]) # set z
    counter = 0
    for i in range(len(X)):
        p = skeleton._parents[i]
        if p != -1:
            limbPlots[counter].set_data(np.array([X[i,2], X[p,2]]), np.array(([X[i,1], X[p,1]])))
            limbPlots[counter].set_3d_properties(np.array([X[i,0], X[p,0]]))
            counter +=1
    fig.canvas.draw_idle()

for i in range(latent_dim):
    latentaxs.append(fig.add_axes([0.55, (i+1)/(latent_dim+1), 0.4, 1. / (latent_dim*2)], facecolor=axcolor))
    sliders.append(Slider(latentaxs[-1], 'L'+str(i+1), -2, 2, valinit=0))
    sliders[-1].on_changed(update)


resetax = plt.axes([0.05, 0.05, 0.1, 0.04])
resetButton = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

def reset(event):
    for i in range(latent_dim):
        sliders[i].reset()

resetButton.on_clicked(reset)
sampletax = plt.axes([0.05, 0.1, 0.1, 0.04])
sampleButton = Button(sampletax, 'Sample', color=axcolor, hovercolor='0.975')

def sample(event):
    for i in range(latent_dim):
        sliders[i].set_val(np.random.normal() * 0.2)

sampleButton.on_clicked(sample)

update(0)
plt.show()