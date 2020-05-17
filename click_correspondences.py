'''
  File name: click_correspondences.py
  Author: 
  Date created: 
'''

'''
  File clarification:
    Click correspondences between two images
    - Input im1: target image
    - Input im2: source image
    - Output im1_pts: correspondences coordiantes in the target image
    - Output im2_pts: correspondences coordiantes in the source image
'''

import matplotlib.pyplot as plt
import os
from scipy import signal
from scipy import sparse
from PIL import Image
import numpy as np


def click_correspondences(im1, im2):
  '''
    Tips:
      - use 'matplotlib.pyplot.subplot' to create a figure that shows the source and target image together
      - add arguments in the 'imshow' function for better image view
      - use function 'ginput' and click correspondences in two images in turn
      - please check the 'ginput' function documentation carefully
        + determine the number of correspondences by yourself which is the argument of 'ginput' function
        + when using ginput, left click represents selection, right click represents removing the last click
        + click points in two images in turn and once you finish it, the function is supposed to 
          return a NumPy array contains correspondences position in two images
  '''

  # TODO: Your code here
#
#
# im1 = plt.imread('im1.jpg')
# im2 = plt.imread('im2.jpg')
  points = 40
  if im2.shape != im1.shape:
      im2 = Image.fromarray(im2)
      im2 = im2.resize((im1.shape[1],im1.shape[0]))
      im2 = np.asarray(im2)
  # print(type(im2))
  fig, (Ax0, Ax1) = plt.subplots(1, 2,figsize = (10,10))
  Ax0.imshow(im1)
  Ax1.imshow(im2)
  A = plt.ginput(points*2, show_clicks = True,timeout = 0)

  im1_pts = np.asarray([])
  im2_pts = np.asarray([])
  for i in range(len(A)):
      if (i%2==0):
          im1_pts = np.append(im1_pts,A[i])
      else:
          im2_pts = np.append(im2_pts,A[i])
  np.savetxt("im11.csv", im1_pts, delimiter=",")
  np.savetxt("im22.csv", im2_pts, delimiter=",")
  im1_pts = im1_pts.reshape([int(len(im1_pts)/2),2])
  im2_pts = im2_pts.reshape([int(len(im2_pts)/2),2])

  return im1_pts, im2_pts

