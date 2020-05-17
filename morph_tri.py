'''
  File name: morph_tri.py
  Author:
  Date created:
'''

'''
  File clarification:
    Image morphing via Triangulation
    - Input im1: target image
    - Input im2: source image
    - Input im1_pts: correspondences coordiantes in the target image
    - Input im2_pts: correspondences coordiantes in the source image
    - Input warp_frac: a vector contains warping parameters
    - Input dissolve_frac: a vector contains cross dissolve parameters

    - Output morphed_im: a set of morphed images obtained from different warp and dissolve parameters.
                         The size should be [number of images, image height, image Width, color channel number]
'''

from helpers import interp2
import numpy as np
from scipy.spatial import Delaunay
import imageio as io
from PIL import Image

def morph_tri(im1, im2, im1_pts, im2_pts, warp_frac, dissolve_frac):
  # TODO: Your code here
  # Tips: use Delaunay() function to get Delaunay triangulation;
  # Tips: use tri.find_simplex(pts) to find the triangulation index that pts locates in.
  dissolve_frac = 1-dissolve_frac
  warp_frac = 1-warp_frac
  images = []
  if im2.shape != im1.shape:
    im2 = Image.fromarray(im2)
    im2 = im2.resize((im1.shape[1],im1.shape[0]))
    im2 = np.asarray(im2)
  imFF = np.zeros(([len(warp_frac),im1.shape[0],im1.shape[1],3]))
  imgc = 0
  for t in warp_frac:
    im_mpts = (t*im1_pts + (1-t)*im2_pts)
    D = Delaunay(im_mpts)
    x = range(im1.shape[0])
    y = range(im1.shape[1])
    xv, yv = np.meshgrid(x, y)
    r = im1.shape[0]
    c = im1.shape[1]
    co_ords = np.zeros([im1.shape[1],im1.shape[0],3])
    co_ords[:,:,0] = yv
    co_ords[:,:,1] = xv
    co_ords[:,:,2] = np.ones(yv.shape)
    points = co_ords.reshape([1,r*c,3])
    triangles = D.find_simplex(points[:,:,:2])
    act_tri = np.transpose(triangles.reshape(co_ords[:,:,0].shape))
    tri_pt = D.simplices
    A_pt = tri_pt[:,0]
    B_pt = tri_pt[:,1]
    C_pt = tri_pt[:,2]
    A_ptc = im_mpts[A_pt]
    B_ptc = im_mpts[B_pt]
    C_ptc = im_mpts[C_pt]
    A_ptc = np.pad(A_ptc,[(0,0),(0,1)],mode = 'constant')
    A_ptc[:,2] = A_ptc[:,2]+1
    B_ptc = np.pad(B_ptc,[(0,0),(0,1)],mode = 'constant')
    B_ptc[:,2] = B_ptc[:,2]+1
    C_ptc = np.pad(C_ptc,[(0,0),(0,1)],mode = 'constant')
    C_ptc[:,2] = C_ptc[:,2]+1
    A_ptc = np.transpose(A_ptc)
    B_ptc = np.transpose(B_ptc)
    C_ptc = np.transpose(C_ptc)
    D1 = np.transpose(A_ptc.copy())
    E1 = np.transpose(B_ptc.copy())
    F1 = np.transpose(C_ptc.copy())
    D12 = D1.reshape([A_ptc.shape[1],3,1])
    E12 = E1.reshape([B_ptc.shape[1],3,1])
    F12 = F1.reshape([C_ptc.shape[1],3,1])
    tri_mat = np.concatenate((D12,E12,F12),axis=2)
    tri_inv = np.linalg.inv(tri_mat)
    tri_inv_ordered = tri_inv[triangles,:,:]
    tri_inv_ordered = tri_inv_ordered[0,:,:,:]
    points1 = points[0,:,:]
    points1 = points.reshape([points.shape[1],3,1])
    BC = np.matmul(tri_inv_ordered,points1)

    As_ptc = im1_pts[A_pt]
    Bs_ptc = im1_pts[B_pt]
    Cs_ptc = im1_pts[C_pt]
    As_ptc = np.pad(As_ptc,[(0,0),(0,1)],mode = 'constant')
    As_ptc[:,2] = As_ptc[:,2]+1
    Bs_ptc = np.pad(Bs_ptc,[(0,0),(0,1)],mode = 'constant')
    Bs_ptc[:,2] = Bs_ptc[:,2]+1
    Cs_ptc = np.pad(Cs_ptc,[(0,0),(0,1)],mode = 'constant')
    Cs_ptc[:,2] = Cs_ptc[:,2]+1
    As_ptc = np.transpose(As_ptc)
    Bs_ptc = np.transpose(Bs_ptc)
    Cs_ptc = np.transpose(Cs_ptc)
    D1s = np.transpose(As_ptc.copy())
    E1s = np.transpose(Bs_ptc.copy())
    F1s = np.transpose(Cs_ptc.copy())
    D12s = D1s.reshape([As_ptc.shape[1],3,1])
    E12s = E1s.reshape([Bs_ptc.shape[1],3,1])
    F12s = F1s.reshape([Cs_ptc.shape[1],3,1])
    tris_mat = np.concatenate((D12s,E12s,F12s),axis=2)
    tris_mat_ordered = tris_mat[triangles,:,:]
    tris_mat_ordered = tris_mat_ordered[0,:,:,:]
    sxyz = np.matmul(tris_mat_ordered,BC)
    sxy = sxyz[:,:2,0]
    sxy = sxy.reshape([im1.shape[1],im1.shape[0],2])
    x_1 = sxy[:,:,0]
    y_1 = sxy[:,:,1]
    im1R = interp2(im1[:,:,0],x_1,y_1)
    im1G = interp2(im1[:,:,1],x_1,y_1)
    im1B = interp2(im1[:,:,2],x_1,y_1)

    At_ptc = im2_pts[A_pt]
    Bt_ptc = im2_pts[B_pt]
    Ct_ptc = im2_pts[C_pt]
    At_ptc = np.pad(At_ptc,[(0,0),(0,1)],mode = 'constant')
    At_ptc[:,2] = At_ptc[:,2]+1
    Bt_ptc = np.pad(Bt_ptc,[(0,0),(0,1)],mode = 'constant')
    Bt_ptc[:,2] = Bt_ptc[:,2]+1
    Ct_ptc = np.pad(Ct_ptc,[(0,0),(0,1)],mode = 'constant')
    Ct_ptc[:,2] = Ct_ptc[:,2]+1
    At_ptc = np.transpose(At_ptc)
    Bt_ptc = np.transpose(Bt_ptc)
    Ct_ptc = np.transpose(Ct_ptc)
    D1t = np.transpose(At_ptc.copy())
    E1t = np.transpose(Bt_ptc.copy())
    F1t = np.transpose(Ct_ptc.copy())
    D12t = D1t.reshape([At_ptc.shape[1],3,1])
    E12t = E1t.reshape([Bt_ptc.shape[1],3,1])
    F12t = F1t.reshape([Ct_ptc.shape[1],3,1])
    trit_mat = np.concatenate((D12t,E12t,F12t),axis=2)
    trit_mat_ordered = trit_mat[triangles,:,:]
    trit_mat_ordered = trit_mat_ordered[0,:,:,:]
    txyz = np.matmul(trit_mat_ordered,BC)
    txy = txyz[:,:2,0]
    txy = txy.reshape([im1.shape[1],im1.shape[0],2])
    x_2 = txy[:,:,0]
    y_2 = txy[:,:,1]
    im2R = interp2(im2[:,:,0],x_2,y_2)
    im2G = interp2(im2[:,:,1],x_2,y_2)
    im2B = interp2(im2[:,:,2],x_2,y_2)

    imFR = (dissolve_frac[imgc]*im1R + (1-dissolve_frac[imgc])*im2R).T
    imFG = (dissolve_frac[imgc]*im1G + (1-dissolve_frac[imgc])*im2G).T
    imFB = (dissolve_frac[imgc]*im1B + (1-dissolve_frac[imgc])*im2B).T
    imF = im1.copy()
    imF[:,:,0] = imFR
    imF[:,:,1] = imFG
    imF[:,:,2] = imFB
    # plt.imshow(imF)
    # plt.show()
    imFF[imgc,:,:,0] = imFR
    imFF[imgc,:,:,1] = imFG
    imFF[imgc,:,:,2] = imFB
    images.append(imF)
    imgc = imgc +1
  morphed_im = imFF
  io.mimsave('Morph.gif', images, duration = 0.05)
  # print(morphed_im.shape)
  return morphed_im
