########################################################################
#
# File:   iSpot.py
# Author: William Lee (and Won Chung) (w/ code taken from Matt Zucker's capture.py)
# Date:   February, 2017 
#
# Written for ENGR 27 - Computer Vision
#
########################################################################
#
# This program demonstrates how to use the VideoCapture and
# VideoWriter objects from OpenCV.
#
# Usage: the program can be run with a filename or a single integer as
# a command line argument.  Integers are camera device ID's (usually
# starting at 0).  If no argument is given, tries to capture from
# the default input 'bunny.mp4'

# Do Python 3-style printing
from __future__ import print_function

import cv2
import numpy as np
import sys
import struct

# Figure out what input we should load:
input_device = None

if len(sys.argv) > 1:
    input_filename = sys.argv[1]
    try:
        input_device = int(input_filename) 
    except:
        pass
else:
    print('Using default input. Specify a device number to try using your camera, e.g.:')
    print()
    print('  python', sys.argv[0], '0')
    print()
    input_filename = 'bunny.mp4'

# Choose camera or file, depending upon whether device was set:
if input_device is not None:
    capture = cv2.VideoCapture(input_device)
    if capture:
        print('Opened camera device number', input_device, '- press Esc to stop capturing.')
else:
    capture = cv2.VideoCapture(input_filename)
    if capture:
        print('Opened file', input_filename)

# Bail if error.
if not capture or not capture.isOpened():
    print('Error opening video capture!')
    sys.exit(1)

# Fetch the first frame and bail if none.
ok, frame = capture.read()

if not ok or frame is None:
    print('No frames in video')
    sys.exit(1)


################################################################################################
#      First frame selected here. ve vill do ze temporal thresholding computations ere!        #
################################################################################################
#TODO: insert temporal averaging code here




# Now set up a VideoWriter to output video.
w = frame.shape[1]
h = frame.shape[0]

fps = 30 #do we want to downgrade? 

# One of these combinations should hopefully work on your platform:
#fourcc, ext = (cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), 'avi')

#huh, so .mov should work
fourcc, ext = (cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), 'mov')

filename = 'captured.'+ext

writer = cv2.VideoWriter(filename, fourcc, fps, (w, h))
if not writer:
    print('Error opening writer')
else:
    print('Opened', filename, 'for output.')
    writer.write(frame)

# Loop until movie is ended or user hits ESC:
frameNumber=0;
while 1:

    # Get the frame.
    frameNumber = frameNumber + 1 #increment frameNumber
    ok, frame = capture.read(frame) 

    # Bail if none.
    if not ok or frame is None:
        print('Bad frame in video! Aborting!')
        break

    ################################################################################################
    #                     ve vill do ze computations ere! (modify the current frame)               #
    ################################################################################################
    print('ve vill do ze computations ere! ', frameNumber)



    #TODO: First, RGB threshold the hands out in this frame
        #we may need to do this with a mask? to preseve color in the other image
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # lower_blue = np.array([210,212,210])
    # upper_blue = np.array([130,255,255])
    # mask = cv2.inRange(frame, lower_blue, upper_blue)
    # res = cv2.bitwise_and(frame,frame, mask= mask)
    # cv2.imshow('frame',frame)
    # cv2.imshow('mask',mask)
    # cv2.imshow('res',res)

        

    #TODO: Here, do temporal threholding to remove all except the bar (non-glove part of arms will likely stay too)

    #do temporal averaging on the first few frames of 125_5x_lf_lowres.mov
        #create comparison frame by averaging the first few of the movie
        #find the difference (absdiff) of every frame and that comparison frame
        #create a mask where only large differences are kept
        #profit??

    #TODO: Morphological Operators to remove noise

    #TODO: Connected Components Analysis
        #Find the centroid of the bar. Track it's (x,y) over time.
        #Find the velocity of the centroid of the bar. Track it over time.



    ################################################################################################
    #                          Write the newly modified frame to the writer                        #
    ################################################################################################
    # Write if we have a writer.
    if writer:
        writer.write(frame)
    # Throw it up on the screen.
    cv2.imshow('Video', frame)



    # Delay for 5ms and get a key
    k = cv2.waitKey(5)
    # Check for ESC hit:
    if k % 0x100 == 27:
        break
