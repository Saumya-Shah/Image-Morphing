import matplotlib.pyplot as plt
# import os
# from scipy import signal
# from scipy import sparse
# from PIL import Image
import numpy as np
from numpy import genfromtxt
# from scipy.spatial import Delaunay
from morph_tri import morph_tri
from click_correspondences import click_correspondences

filename1 = '/images/Me.jpg'
filename2 = '/images/Old.jpg'
im1 = plt.imread(filename1)
im2 = plt.imread(filename2)
click_correspondences(im1,im2)
r = im1.shape[0]
c = im1.shape[1]
im1_points = genfromtxt('im11.csv',delimiter = ',')
im1_points = np.append(im1_points,(0,0,0,r,c,0,c,r,0,0.5*r,0.5*c,0,0.5*c,r,c,0.5*r),axis = 0)
im1_pts = im1_points.reshape([int(len(im1_points)/2),2])
im2_points = genfromtxt('im22.csv',delimiter = ',')
im2_points = np.append(im2_points,(0,0,0,r,c,0,c,r,0,0.5*r,0.5*c,0,0.5*c,r,c,0.5*r),axis = 0)
im2_pts = im2_points.reshape([int(len(im2_points)/2),2])

warp_frac = np.linspace(0,1,60)
# dissolve_frac = np.ones(len(warp_frac))
dissolve_frac = warp_frac

morphed_im = morph_tri(im1, im2, im1_pts, im2_pts, warp_frac, dissolve_frac)
