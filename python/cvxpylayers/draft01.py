# general polytope projections
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import cvxpy
from tqdm import tqdm

import cvxpylayers.torch

plt.ion()

hf_to_np = lambda x: x.detach().cpu().numpy()


def naive_get_all_vertex_of_2d_polygon(matA, matb, interior_x):
    # Ax <= b
    # https://or.stackexchange.com/q/4540
    matA = matA.astype(np.float64) #np.float32 may not meet the precision (1e-10) set below
    matb = matb.astype(np.float64)
    interior_x = np.asarray(interior_x)
    assert (matA.ndim==2) and (matA.shape[1]==2) and (matb.ndim==1)
    assert (interior_x.ndim==1) and (interior_x.shape[0]==2)
    num_constraint = matA.shape[0]
    tmp0 = [(i,j) for i in range(num_constraint-1) for j in range(i+1,num_constraint)]
    vertex_list = []
    for i,j in tmp0:
        tmp0 = np.linalg.solve(matA[[i,j]], matb[[i,j]])
        if np.all(matA @ tmp0 - matb < 1e-10):
            vertex_list.append(tmp0)
    vertex_list = np.stack(vertex_list)
    ind0 = np.argsort(np.angle((vertex_list[:,0] - interior_x[0]) + (vertex_list[:,1] - interior_x[1])*1j))
    ret = vertex_list[ind0]
    return ret


def plt_plot_polygon(**kwargs):
    assert len(kwargs)>=1
    fig,ax = plt.subplots()
    for k,v in kwargs.items():
        ax.fill(v[:,0], v[:,1], label=k, alpha=0.3)
    ax.legend()
    fig.tight_layout()


numX = 2 #2 is necessary for plot
num_constraint = 10

cvxG = cvxpy.Parameter((num_constraint, numX))
cvxh = cvxpy.Parameter(num_constraint)
cvxX = cvxpy.Parameter(numX)
cvxY = cvxpy.Variable(numX)
tmp0 = cvxpy.Minimize(cvxpy.sum_squares(cvxX-cvxY)/2)
prob = cvxpy.Problem(tmp0, [cvxG@cvxY<=cvxh])
layer = cvxpylayers.torch.CvxpyLayer(prob, parameters=[cvxG, cvxh, cvxX], variables=[cvxY])

G = torch.empty(num_constraint, numX, dtype=torch.float32).uniform_(-4, 4)
# G = torch.FloatTensor(num_constraint, numX).uniform_(-4, 4)
z0 = torch.full([numX], 0.5)
s0 = torch.full([num_constraint], 0.5)
h = G @ z0 + s0
G_hat = torch.nn.Parameter(torch.empty(num_constraint, numX, dtype=torch.float32).uniform_(-4, 4).requires_grad_())
h_hat = G_hat.mv(z0)+s0

tmp0 = naive_get_all_vertex_of_2d_polygon(hf_to_np(G), hf_to_np(h), [0.5,0.5])
tmp1 = naive_get_all_vertex_of_2d_polygon(hf_to_np(G_hat), hf_to_np(h_hat), [0.5,0.5])
plt_plot_polygon(ground_truth=tmp0, init_guess=tmp1)

opt = torch.optim.Adam([G_hat], lr=0.01)
losses = []
for i in tqdm(range(2500)):
    x = torch.randn(numX)
    y = layer(G, h, x)[0]

    h_hat = G_hat @ z0 + s0
    yhat = layer(G_hat, h_hat, x)[0]
    loss = (yhat-y).norm()
    losses.append(loss.item())

    opt.zero_grad()
    loss.backward()
    opt.step()

fig, ax = plt.subplots(1, 1, figsize=(6,4))
ax.plot(np.array(losses), '.', alpha=0.3)
ax.set_xlabel('Iteration')
ax.set_ylabel('Rolling Loss')
ax.set_ylim(0, None)
