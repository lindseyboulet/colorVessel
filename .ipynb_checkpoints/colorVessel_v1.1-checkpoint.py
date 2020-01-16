# import the necessary packages
from __future__ import print_function
from __future__ import division
import argparse
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
from sklearn.cluster import MeanShift, estimate_bandwidth
from itertools import compress
import math
import glob
import pandas as pd
import re
from tkinter import Tk
from tkinter.filedialog import askopenfilename

def pullScale(img, y = 50, h = 300, x = 0, w =300):
    # crop image scale
    # y,h,x,w = 50, 300, 0, 300
    imageSc = img[y:y+h, x:x+w]
    # plt.imshow(imageSc)
    # extract ultrasound scale
    rows,cols, x = imageSc.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),37,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    y,h,x,w = 0, 300, 150, 50
    dstTest = dst[y:y+h, x:x+w]
    y,h,x,w = 0, 300, 179, 8
    dst = dst[y:y+h, x:x+w]
    #plt.imshow(dst)
    # canny edge detection on scale 
    dst_g=cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(dst_g,50,150,apertureSize = 3)
    #plt.imshow(edges)
    #plt.savefig('cannyScale.png', dpi = 300) 
    # pull scale x-y values
    value = edges.flatten()
    y,x = np.indices(edges.shape).reshape(-1,len(value))
    x = x.astype(float); y = y.astype(float)
    for i in range(0,len(value)):
        if value[i] == 0:
            y[i] = np.nan
            x[i] = np.nan
    x = x[np.logical_not(np.isnan(x))]
    y = y[np.logical_not(np.isnan(y))]
    # cluster y values to find tick values
    Y = list(zip(y,np.zeros(len(y))))
    bandwidth = estimate_bandwidth(Y, quantile=0.1)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(Y)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    # get and sort mean tick pixel values
    ticks = []
    for k in range(n_clusters_):
        my_members = labels == k
        ticks.append(np.mean(np.array(list(compress(Y, my_members)))[:,0]))
    ticks = np.array(ticks)
    ticks = np.sort(ticks)
    # show predicted scale
    # plt.imshow(dstTest)
    # plt.hlines(ticks, xmin = 0, xmax = 20, color = 'white')
    # calculate mean difference between scales
    diffs = []
    for i in range(len(ticks)-1):
            diffs.append(ticks[i+1] - ticks[i])
    pixelPerCm = np.mean(diffs)
    return pixelPerCm

# plot HSV spread on 3D plot
def hsvPlot(image, filename = 'hsv_image'):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixel_colors = image.reshape((np.shape(image)[0]*np.shape(image)[1], 3))
    norm = colors.Normalize(vmin=-1.,vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Hue")
    axis.set_ylabel("Saturation")
    axis.set_zlabel("Value")
    plt.savefig(filename + '.png', dpi = 300)
    
def filterIm(image, filter = 'gaus', x = 5, y = 5):
    if filter == 'gaus':
        blur = cv2.GaussianBlur(image,(x,y),0)
    else:
        blur = cv2.medianBlur(image,5)
    return blur

def cannyMain(result, t1 = 0, t2 = 200):
    # canny edge detection on color signal
    result_g=cv2.cvtColor(result,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(result,t1,t2,apertureSize = 3)
    return(edges)

def angleRot(edges):    
    ret,thresh = cv2.threshold(edges,127,255,0)
    contours,hierarchy = cv2.findContours(thresh, 1, 2)
    cnt = contours[0]
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt,True)
    rows,cols = edges.shape[:2]
    [vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
    lefty = int((-x*vy/vx) + y)
    righty = int(((cols-x)*vy/vx)+y)
    pt2 = (cols-1,righty)
    pt1 = (0,lefty)
    angle = math.degrees(math.atan((pt1[1] - pt2[1])/ (pt2[0] - pt1[0])))
    return angle

def rotMain(result, angle, filename):
    cv2.namedWindow('rot')
    cv2.moveWindow("rot", 700,0)
    # create trackbars for color change
    angle = angle*-1
    cv2.createTrackbar('angle','rot',int(angle),180,callback)

    while(True):
        # get trackbar positions
        a1 = cv2.getTrackbarPos('angle', 'rot')
        rows,cols = result.shape[:2]
        M = cv2.getRotationMatrix2D((cols/2,rows/2),a1,1)
        dst = cv2.warpAffine(result,M,(cols,rows))
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 1000,1000)
        cv2.imshow('image', dst)
        k = cv2.waitKey(1000) & 0xFF # large wait time to remove freezing
        if k == 113 or k == 27 or k == ord("c"):
            break
        
    cv2.imwrite(filename[:-4]+'-rotated.png', cv2.resize(dst,(1000,1000)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return dst, a1

def flatMain(edges, filename, cut1 = None, cut2 = None):
    # convert pixels into x-y datad
    cv2.imwrite(filename[:-4]+'-canny.png', cv2.resize( edges,(1000,1000)))
    value = edges.flatten()
    y1,x1 = np.indices(edges.shape).reshape(-1,len(value))
    x1 = x1.astype(float); y1 = y1.astype(float)
    for i in range(0,len(value)):
        if value[i] == 0:
            y1[i] = np.nan
            x1[i] = np.nan
    x1 = x1[np.logical_not(np.isnan(x1))]
    y1 = y1[np.logical_not(np.isnan(y1))]
    if cut1 == None:
        cut1 = np.amin(x1)
    if cut2 == None:
        cut2 = np.amax(x1)
    x2 = x1[(x1 >= cut1) & (x1 <= cut2)]
    y2 = y1[(x1 >= cut1) & (x1 <= cut2)]    
    plt.subplot(2, 1, 1)
    plt.scatter(x1,y1)
    plt.axvline(x = cut1, color = 'red')
    plt.axvline(x = cut2, color = 'red')    
    plt.subplot(2, 1, 2)    
    plt.scatter(x2,y2)
    plt.savefig(filename[:-4]+'-flat.png', dpi = 300)
    plt.clf()
    return x2, y2

def diamCalc(x1, y1, pxPerCm):
    min = int(np.amin(x1)); max = int(np.amax(x1))
    ranX = range(min, max)
    diams = []
    for i in ranX:
        dist = y1[np.where(x1 == i)]
        diams.append((np.amax(dist) - np.amin(dist))/pxPerCm)
    diams = np.asarray(diams)
    #return np.mean(diams)
    return diams

def createRows(rowD):
    rowUn = pd.DataFrame(columns = ['id', 'cond', 'stage', 'frame','pixel', 'diameter'])
    for i in range(len(rowD[4])):
            rowUn = rowUn.append({'id':rowD[0], 'cond':rowD[1], 'stage':rowD[2], 'frame':rowD[3], 'pixel':i+1,
                                  'diameter':rowD[4][i]}, ignore_index=True)
    return rowUn
               
    
def saveData(imgFileN, x1, y1, pxPerCm):
    words = imgFileN.split('/')
    frameJ = words[len(words)-2]
    id = frameJ.split('_')
    id.append(words[len(words)-1].split('.')[0])
    rowD = [id[0], id[1], id[2], id[3], diamCalc(x1, y1, pxPerCm)]
    saveName =  frameJ + '_diamData.csv'
    if os.path.exists(saveName):
        diamDf = pd.read_csv(saveName)
        inde = diamDf.index[(diamDf.id == id[0]) & (diamDf.cond == id[1]) & (diamDf.stage == id[2]) & (diamDf.frame == id[3])].tolist()
        if len(inde) > 0:
            diamDf = diamDf.drop(inde)
        diamDf = diamDf.append(createRows(rowD))
            
    else:
        diamDf = createRows(rowD)
    diamDf.to_csv(saveName, index = None, header=True)


def shape_selection(event, x, y, flags, param):
    #print(event)
    global ix,iy,drawing,mode,x_,y_, r

    if event == cv2.EVENT_LBUTTONDOWN:
        #print('inside mouse lbutton event....')
        drawing = True
        ix,iy = x,y
        x_,y_ = x,y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        copy = image.copy()
        x_,y_ = x,y
        cv2.rectangle(copy,(ix,iy),(x_,y_),(0,255,0),1)
        cv2.imshow("image", copy)
    elif event == cv2.EVENT_LBUTTONUP:
        #print('inside mouse button up event')
        drawing = False
        cv2.rectangle(image,(ix,iy),(x,y),(0,255,0),1)

def cropper(image):
    global cropping, mode, ix,iy,drawing,mode,x_,y_, r
    # keep looping until the 'q' key is pressed
    # initialize the list of referennce points and boolean indicating
    # whether cropping is being performed or not
    temp_img = np.copy(image)
    #img = np.zeros((512,512,3), np.uint8)
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 1200,1200)
    cv2.setMouseCallback("image", shape_selection)
    while (1):
        # display the image and wait for a keypress
        cv2.imshow("image", image)
        if not cv2.EVENT_MOUSEMOVE:
            copy = image.copy()
            # print('x_: , y_ : '.format(x_,y_))
            # print(x_)
            if mode == True:
                cv2.rectangle(copy,(ix,iy),(x_,y_),(0,255,0),1)
                cv2.imshow('image',copy)
            else:
                cv2.circle(copy,(x_,y_),r,(0,0,255),1)
                cv2.imshow('image',copy)
        
        key = cv2.waitKey(1) & 0xFF

        # if the 'r' key is pressed, reset the cropping region
        if key == ord("r"):
            image = clone.copy()
        
        if key == ord('m'):
            mode = not mode

        # if the 'c' key is pressed, break from the loop
        elif key == ord("c"):
            break

    # if there are two reference points, then crop the region of interest
    # from teh image and display it


    roi = clone[iy:y_, ix:x_]
    cv2.namedWindow("ROI", cv2.WINDOW_NORMAL)
    cv2.resizeWindow('ROI', 300,300) 
    cv2.imshow("ROI", roi)
    cv2.waitKey(0)
    return roi, [iy,y_,ix,x_]

drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1
x_, y_ = 0,0
# now let's initialize the list of reference point

# load the image, clone it, and setup the mouse callback function
Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
filename = askopenfilename(filetypes=[("PNG", "*.png"),("JPG", "*.jpg"),  ("All files", "*")]) # show an "Open" dialog box and return the path to the selected file
image = cv2.imread(filename, -1)
clone = image.copy()
pxPerCm = pullScale(image)

drawing = False; ix,iy = -1,-1; x_, y_ = 0,0; cropping = False
roi2, coords = cropper(image)
roi = filterIm(roi2)
hsvPlot(roi)

cv2.namedWindow("HSV", cv2.WINDOW_NORMAL)
cv2.moveWindow("HSV", 0,0)
cv2.resizeWindow('HSV', 600,600)
hsv = cv2.imread('hsv_image.png')
cv2.imshow("HSV", hsv)
#cv2.waitKey(0)

def callback(x):
    pass

cv2.namedWindow('Mask')
cv2.moveWindow("Mask", 100,650)
# create trackbars for color change
cv2.createTrackbar('lowHue1','Mask',0,255,callback)
cv2.createTrackbar('highHue1','Mask',75,255,callback)

cv2.createTrackbar('lowSat1','Mask',150,255,callback)
cv2.createTrackbar('highSat1','Mask',255,255,callback)

cv2.createTrackbar('lowVal1','Mask',0,255,callback)
cv2.createTrackbar('highVal1','Mask',255,255,callback)

cv2.createTrackbar('lowHue2','Mask',150,255,callback)
cv2.createTrackbar('highHue2','Mask',255,255,callback)

cv2.createTrackbar('lowSat2','Mask',175,255,callback)
cv2.createTrackbar('highSat2','Mask',255,255,callback)

cv2.createTrackbar('lowVal2','Mask',0,255,callback)
cv2.createTrackbar('highVal2','Mask',255,255,callback)
while(True):

    # get trackbar positions
    ilowH = cv2.getTrackbarPos('lowHue1', 'Mask')
    ihighH = cv2.getTrackbarPos('highHue1', 'Mask')
    ilowS = cv2.getTrackbarPos('lowSat1', 'Mask')
    ihighS = cv2.getTrackbarPos('highSat1', 'Mask')
    ilowV = cv2.getTrackbarPos('lowVal1', 'Mask')
    ihighV = cv2.getTrackbarPos('highVal1', 'Mask')
    
    ilowH2 = cv2.getTrackbarPos('lowHue2', 'Mask')
    ihighH2 = cv2.getTrackbarPos('highHue2', 'Mask')
    ilowS2 = cv2.getTrackbarPos('lowSat2', 'Mask')
    ihighS2 = cv2.getTrackbarPos('highSat2', 'Mask')
    ilowV2 = cv2.getTrackbarPos('lowVal2', 'Mask')
    ihighV2 = cv2.getTrackbarPos('highVal2', 'Mask')

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([ilowH, ilowS, ilowV])
    higher_hsv = np.array([ihighH, ihighS, ihighV])
    mask = cv2.inRange(hsv, lower_hsv, higher_hsv)
    
    lower_hsv2 = np.array([ilowH2, ilowS2, ilowV2])
    higher_hsv2 = np.array([ihighH2, ihighS2, ihighV2])
    mask2 = cv2.inRange(hsv, lower_hsv2, higher_hsv2)
    mask = cv2.bitwise_or(mask, mask2)
    frame = cv2.bitwise_and(roi, roi, mask=mask)

    # show thresholded image
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.moveWindow("image", 700,0)
    cv2.resizeWindow('image', 1000,1000)
    cv2.imshow('image', frame)
    
    k = cv2.waitKey(1000) & 0xFF # large wait time to remove freezing
    if k == 113 or k == 27 or k == ord("c"):
        break
cv2.destroyAllWindows()


# drawing = False; ix,iy = -1,-1; x_, y_ = 0,0; cropping = False
# roi2 = cropper(frame)

def cannyFun(roi):
    cv2.namedWindow('Canny')
    cv2.moveWindow("Canny", 300,200)
    # create trackbars for color change
    cv2.createTrackbar('thresh1','Canny',0,500,callback)
    cv2.createTrackbar('thresh2','Canny',200,500,callback)

    while(True):

        # get trackbar positions
        t1 = cv2.getTrackbarPos('thresh1', 'Canny')
        t2 = cv2.getTrackbarPos('thresh2', 'Canny')
        canny1 = cv2.Canny(roi,t1,t2,apertureSize = 3)

        # show thresholded image
        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        cv2.moveWindow("image", 700,0)
        cv2.resizeWindow('image', 1000,1000)
        cv2.imshow('image', canny1)
        k = cv2.waitKey(1000) & 0xFF # large wait time to remove freezing
        if k == 113 or k == 27 or k == ord("c"):
            break
    return canny1



cv2.namedWindow("image", cv2.WINDOW_NORMAL)
cv2.moveWindow("image", 0,0)
cv2.resizeWindow('image', 400,400)
cv2.imshow('image', frame)

roi2Can = cannyFun(frame)
angle = angleRot(roi2Can)
    
    
rotRoi, angle = rotMain(frame, angle, filename)

os.remove('hsv_image.png')
rotCan = cannyFun(rotRoi)

x,y = flatMain(rotCan, filename)

# create trackbars for color change
    
cv2.namedWindow("select_vessel_length", cv2.WINDOW_NORMAL)
cv2.moveWindow("select_vessel_length", 0,700)
cv2.createTrackbar('cut1','select_vessel_length',0, 35,callback)
cv2.createTrackbar('cut2','select_vessel_length',10 ,60,callback)

while(True):

    # get trackbar positions
    cut1 = cv2.getTrackbarPos('cut1', 'select_vessel_length')
    cut2 = cv2.getTrackbarPos('cut2', 'select_vessel_length')
    x,y = flatMain(rotCan, filename, cut1, cut2)

    # show thresholded image
    
    cv2.namedWindow("Flat", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Flat", 0,0)
    cv2.resizeWindow('Flat', 600,600)
    hsv = cv2.imread(filename[:-4]+'-flat.png')
    cv2.imshow("Flat", hsv)
    k = cv2.waitKey(1000) & 0xFF # large wait time to remove freezing
    if k == 113 or k == 27 or k == ord("c"):
        break

saveData(filename, x, y, pxPerCm) #FN 10. Calculate and save diameter values

# close all open windows
cv2.destroyAllWindows()

y1,y2,x1,x2 = coords

s_img = frame.copy()
l_img = clone.copy()
bk_img = clone.copy()

s_img[np.any(s_img != [0, 0, 0], axis=-1)] = [57, 255, 20]

l_img[y1:y2,x1:x2,:] = s_img
l_img[np.any(l_img != [57, 255, 20], axis=-1)] =  [0, 0, 0]

out_img = cv2.add(l_img,bk_img)

cv2.imwrite(filename[:-4]+'-overlay.png', out_img)
cv2.imwrite(filename[:-4]+'-mask.png', l_img)


