#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 00:18:01 2019

@author: lindsey
"""

import numpy as np  
import cv2
import os
import glob

# Funtion to load video file and save frames as image files
def videoFrameExt(filename):
    new_path = filename[0:-4]
    if not os.path.exists(new_path):
        os.makedirs(new_path)

    print(cv2.__version__)
    vidcap = cv2.VideoCapture(filename)
    success,image = vidcap.read()
    count = 0
    success = True
    tag = new_path.split('/')
    tag = tag[len(tag)-1]
    while success:
        cv2.imwrite(new_path +"/" + tag + "_frame%d.jpg" % count, image)   # save frame as JPEG file
        success,image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1

vidList = []        
for file in os.listdir():
    if file.endswith(".avi"):
        vidList.append(file)
        
for i in vidList:
    videoFrameExt(i)
