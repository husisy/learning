import numpy as np
import trimesh


def radius_meshgrid_to_mesh(rlist:np.ndarray, convex:bool=False):
    assert isinstance(rlist, np.ndarray) and (rlist.ndim==2)
    theta_list = np.linspace(0, np.pi, rlist.shape[0])
    phi_list = np.linspace(0, 2*np.pi, rlist.shape[1])
    xdata = rlist * np.sin(theta_list[:, None]) * np.cos(phi_list)
    ydata = rlist * np.sin(theta_list[:, None]) * np.sin(phi_list)
    zdata = rlist * np.cos(theta_list[:, None])
    vertex_list = np.stack([xdata.reshape(-1), ydata.reshape(-1), zdata.reshape(-1)], axis=1)
    a = len(phi_list)
    tmp0 = np.array([(x*a+y, (x+1)*a+y, x*a+y+1, (x+1)*a+y+1) for x in range(len(theta_list)-1) for y in range(len(phi_list)-1)])
    # p0 p1
    # p2 p3
    face_list = np.concatenate([tmp0[:, [0,1,2]], tmp0[:,[2,1,3]]], axis=0) #(p0 p1 p2) (p2 p1 p3)
    mesh = trimesh.Trimesh(vertices=vertex_list, faces=face_list)
    if convex:
        mesh = mesh.convex_hull
    return mesh


def demo_radius_function_to_mesh():
    theta_list = np.linspace(0, np.pi, 49)
    phi_list = np.linspace(0, 2*np.pi, 51)
    hf_radius = lambda theta, phi: np.cos(theta)**2 + 1
    # tmp0,tmp1 = np.meshgrid(theta_list, phi_list, indexing='ij')
    rlist = hf_radius(*np.meshgrid(theta_list, phi_list, indexing='ij'))
    mesh = radius_meshgrid_to_mesh(rlist, convex=True)
    _ = mesh.export('tbd00.stl')
