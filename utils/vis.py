# visualization functions
import numpy as np
import networkx as nx
import os, sys
from itertools import product, combinations

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import utils.hyp_functions as hf
import torch

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
from matplotlib import patches
from mpl_toolkits.mplot3d import Axes3D

matplotlib.verbose.set_level("helpful")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# collinearity check. if collinear, draw a line and don't attempt curve
def collinear(a,b,c):
    if np.abs(c[0] - b[0]) < .1**4 and np.abs(c[0]-a[0]) < .1**4:
        return True
    elif np.abs(c[0] - b[0]) < .1**4 or np.abs(c[0]-a[0]) < .1**4:
        return False

    slope1 = np.abs((c[1]-b[1])/(c[0]-b[0]))
    slope2 = np.abs((c[1]-a[1])/(c[0]-a[0]))

    if np.abs(slope1 - slope2) < .1**4:
        return True

    return False

# todo: speed this code up
def get_circle_center(a,b,c):
    m = np.zeros([2,2])
    m[0,0] = 2*(c[0]-a[0])
    m[0,1] = 2*(c[1]-a[1])
    m[1,0] = 2*(c[0]-b[0])
    m[1,1] = 2*(c[1]-b[1])

    v = np.zeros([2,1])
    v[0] = c[0]**2 + c[1]**2 - a[0]**2 - a[1]**2
    v[1] = c[0]**2 + c[1]**2 - b[0]**2 - b[1]**2

    return (np.linalg.inv(m)@v).flatten()

# distance for Euclidean coordinates
def euclid_dist(a,b):
    return np.linalg.norm(a-b)

# angles for arc
def get_angles(center, a):
    if abs(a[0] - center[0]) < 0.1**3:
        if a[1] > center[1] : theta = 90
        else: theta = 270
    else:
        theta = np.rad2deg(np.arctan((a[1]-center[1])/(a[0]-center[0])))    
        # quadrant 3:
        if (a[0]-center[0]) < 0 and (a[1]-center[1]) < 0:
            theta += 180
        # quadrant 2
        if (a[0]-center[0]) < 0 and (a[1]-center[1]) >= 0:
            theta -= 180

    # always use non-negative angles
    if theta < 0: theta += 360    
    return theta

# draw hyperbolic line:
def draw_geodesic(a, b, c, ax, node1=None, node2=None, verbose=False):
    if verbose:
        print("Geodesic points are ", a, "\n", b, "\n", c, "\n")    

    is_collinear = False
    if collinear(a,b,c):
        is_collinear = True
    else:
        cent = get_circle_center(a,b,c)
        radius = euclid_dist(a, cent)
        t1 = get_angles(cent, b)
        t2 = get_angles(cent, a)

        if verbose:
            print("\ncenter at ", cent)
            print("radius is ", radius)
            print("angles are ", t1, " ", t2)
            print("dist(a,center) = ", euclid_dist(cent,a))
            print("dist(b,center) = ", euclid_dist(cent,b))
            print("dist(c,center) = ", euclid_dist(cent,c))

    # if the angle is really tiny, a line is a good approximation
    if is_collinear or (np.abs(t1-t2) < 2):
        coordsA = "data"
        coordsB = "data"
        e = patches.ConnectionPatch(a, b, coordsA, coordsB, linewidth=2)
    else:
        if verbose:
            print("angles are theta_1 = ", t1, " theta_2 = ", t2)
        if (t2>t1 and t2-t1<180) or (t1>t2 and t1-t2>=180):
            e = patches.Arc((cent[0], cent[1]), 2*radius, 2*radius,
                     theta1=t1, theta2=t2, linewidth=2, fill=False, zorder=2)
        else:
            e = patches.Arc((cent[0], cent[1]), 2*radius, 2*radius,
                     theta1=t2, theta2=t1, linewidth=2, fill=False, zorder=2)
    ax.add_patch(e)
    #ax.plot(a[0], a[1], "o")
    #ax.plot(b[0], b[1], "o")

    #if node1 is not None: ax.text(a[0] * (1 + 0.05), a[1] * (1 + 0.05) , node1, fontsize=12)
    #if node2 is not None: ax.text(b[0] * (1 + 0.05), b[1] * (1 + 0.05) , node2, fontsize=12)

# to draw geodesic between a,b, we need
# a third point. easy with inversion
def get_third_point(a,b):
    b0 = hf.reflect_at_zero(a,b)
    c0 = b0/2.0
    c = hf.reflect_at_zero(a,c0)

    return c

def draw_geodesic_on_circle(a, b, ax):
    lp = 5 # number of points for the mesh
    d = np.array(b) - np.array(a)

    vals = np.zeros([3, lp])
    for i in range(lp):
        for j in range(3):
            vals[j,i] = a[j] + d[j]*(i/(lp-1)) 
        
        # let's project back to sphere:
        nrm = vals[0,i]**2 + vals[1,i]**2 + vals[2,i]**2
        for j in range(3):
            vals[j,i] /= np.sqrt(nrm)
        
    # draw the geodesic:
    for i in range(lp-1):
        ax.plot([vals[0,i], vals[0,i+1]], [vals[1,i], vals[1,i+1]], zs=[vals[2,i], vals[2,i+1]], color='r')


# for circle stuff let's just draw the points
def draw_points_on_circle(a, node, ax):
    ax.plot(a[0], a[1], "o", markersize=16)
    ax.text(a[0] * (1 + 0.05), a[1] * (1 + 0.05) , node, fontsize=12)

def draw_points_on_sphere(a, node, ax):
    ax.scatter(a[0], a[1], a[2], c='b', marker='o', s=32)
    ax.text(a[0] * (1 + 0.05), a[1] * (1 + 0.05) , a[2] * (1 + 0.05),  node, fontsize=12)

def draw_points_hyperbolic(a, node, ax):
    ax.plot(a[0], a[1], "o")
    ax.text(a[0] * (1 + 0.05), a[1] * (1 + 0.05) , node, fontsize=12)


# draw the embedding for a graph 
# G is the graph, m is the PyTorch hyperbolic model
def draw_graph(G, m, fig, ax):
    num_spheres = np.minimum(len(m.S), 5)
    num_hypers  = np.minimum(len(m.H), 5)

    sdim = 0 if len(m.S) == 0 else len((m.S[0]).w[0])
    for emb in range(num_spheres):
        if sdim == 3:
            spherical_setup_3d(fig, ax[1, emb])
        else:
            spherical_setup(fig, ax[1, emb])

    for emb in range(num_hypers):
        hyperbolic_setup(fig, ax[0, emb])

    # todo: directly read edge list from csr format
    Gr = nx.from_scipy_sparse_matrix(G)
    for edge in Gr.edges():
        idx = torch.LongTensor([edge[0], edge[1]]).to(device)

        for emb in range(num_hypers):
            a = ((torch.index_select(m.H[emb].w, 0, idx[0])).clone()).detach().cpu().numpy()[0]
            b = ((torch.index_select(m.H[emb].w, 0, idx[1])).clone()).detach().cpu().numpy()[0]
            c = get_third_point(a,b)
            draw_geodesic(a,b,c,ax[0, emb], edge[0], edge[1])

        # let's draw the edges on the sphere; these are geodesics    
        if sdim == 3:
            for emb in range(num_spheres):
                a = ((torch.index_select(m.S[emb].w, 0, idx[0])).clone()).detach().cpu().numpy()[0]
                b = ((torch.index_select(m.S[emb].w, 0, idx[1])).clone()).detach().cpu().numpy()[0]
                draw_geodesic_on_circle(a, b, ax[1, emb])

    for node in Gr.nodes():
        idx = torch.LongTensor([int(node)]).to(device)
        for emb in range(num_spheres):
            v = ((torch.index_select(m.S[emb].w, 0, idx)).clone()).detach().cpu().numpy()[0]
            
            if sdim == 3:
                draw_points_on_sphere(v, node, ax[1, emb])                        
            else:
                draw_points_on_circle(v, node, ax[1, emb])

        for emb in range(num_hypers):
            a = ((torch.index_select(m.H[emb].w, 0, idx)).clone()).detach().cpu().numpy()[0]
            draw_points_hyperbolic(a, node, ax[0, emb])
        

def setup_plot(m, name=None, draw_circle=False):
    # create plot
    num_spheres = np.minimum(len(m.S), 5)
    num_hypers  = np.minimum(len(m.H), 5)
    wid = np.maximum(np.maximum(num_spheres, num_hypers), 2)

    fig, axes = plt.subplots(2, wid, sharey=True, figsize=(wid*10, 20))

    ax = axes
    matplotlib.rcParams['animation.ffmpeg_args'] = '-report'
    writer = animation.FFMpegFileWriter(fps=10, metadata=dict(artist='HazyResearch'))#, bitrate=1800)
    if name is None:
        name = 'HypDistances.mp4'
    else:
        name += '.mp4'

    writer.setup(fig, name, dpi=108)

    sdim = 0 if len(m.S) == 0 else len((m.S[0]).w[0])
    # need these to all be 3D
    if sdim == 3:
        for emb in range(num_spheres):
            ax[1, emb].remove()
            #ax[1, emb] = fig.add_subplot(111, projection='3d')
            ax[1, emb] = fig.add_subplot(2, wid, wid+emb+1, projection='3d')

    if draw_circle:
        for emb in range(num_spheres):
            if sdim == 3:
                spherical_setup_3d(fig, ax[1, emb])
            else:
                spherical_setup(fig, ax[1, emb])
        for emb in range(num_hypers):
            hyperbolic_setup(fig, ax[0, emb])

    return fig, ax, writer


def hyperbolic_setup(fig, ax):
    #fig.set_size_inches(20.0, 10.0, forward=True)
    
    # set axes
    ax.set_ylim([-1.2, 1.2])
    ax.set_xlim([-1.2, 1.2])

    # draw Poincare disk boundary
    e = patches.Arc((0,0), 2.0, 2.0,
                     linewidth=2, fill=False, zorder=2)
    ax.add_patch(e)

def spherical_setup(fig, ax):
#    fig.set_size_inches(20.0, 10.0, forward=True)
    
    # set axes
    ax.set_ylim([-1.2, 1.2])
    ax.set_xlim([-1.2, 1.2])

    # draw circle boundary
    e = patches.Arc((0,0), 2.0, 2.0,
                     linewidth=1, fill=False, zorder=2)
    ax.add_patch(e)

def spherical_setup_3d(fig, ax):
    ax.set_ylim([-1.2, 1.2])
    ax.set_xlim([-1.2, 1.2])

    # draw sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    ax.plot_wireframe(x, y, z, color="y")

def draw_plot():
    plt.show()

def clear_plot():
    plt.cla()
