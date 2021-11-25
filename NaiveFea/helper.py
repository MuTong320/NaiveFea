import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri


def plot_mesh(mesh):
    x=mesh.points.T[0]
    y=mesh.points.T[1]
    elements=mesh.cells_dict['triangle']
    mesh_fig=tri.Triangulation(x,y,elements)
    plt.figure()
    plt.gca().set_aspect('equal')
    plt.triplot(mesh_fig,'k.-',lw=1)
    plt.title('')

def cal_modulus(E='unkown',nv='unkown',K='unkown',G='unkown',lamda='unkown'):
    """You should and can only input two of elastic properties."""
    pass