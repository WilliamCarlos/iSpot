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
import pdb
import cvk2

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


w = frame.shape[1]
h = frame.shape[0]

################################################################################################
#                      Create the average frame for temporal thresholding                      #
################################################################################################

##Create the averaged frame
#frame dimensions  (360, 640, 3)
average = np.zeros_like(frame, dtype=np.float32);

#for now, create temporal frame with 20 frames
temporalCapture = cv2.VideoCapture(input_filename)
ok, currentFrame = temporalCapture.read()
for i in range(1, 20):
    frameToAdd = currentFrame.astype(np.float32)
    average = np.add(average, currentFrame)
    okT, currentFrame = temporalCapture.read(currentFrame)

     #print('calculating temporal frame ', i, 'th iteration', average)
average = average/20
#print('final temporal frame ', average)

################################################################################################

# Now set up a VideoWriter to output video. (dependent on w, h above)

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



################################################################################################
#                                    Place computations here                                   #
################################################################################################

while 1:
    frameNumber = frameNumber + 1 #increment frameNumber
    print(frameNumber)
    # Get the frame.
    ok, frame = capture.read(frame)
    # Bail if none.
    if frame is None:
        print('Video Finished!')
        break
    if not ok:
        print('Bad frame in video! Aborting!')
        break

    #TODO: First, RGB threshold the hands out in this frame
        #we may need to do this with a mask? to preseve color in the other image
    video = frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # lower_blue = np.array([210,212,210])
    # upper_blue = np.array([130,255,255])

    # To Get Only Gloves
    # lower_blue = np.array([85,70,20])
    # upper_blue = np.array([130,255,255])

    # To Get Only Gloves without Weights
    lower_blue = np.array([85,70,20])
    upper_blue = np.array([100,255,255])

    # To Get Dumbbells and Gloves
    # lower_blue = np.array([40,80,20])
    # upper_blue = np.array([130,255,255])

    # Get Arms and Gloves
    # lower_blue = np.array([0,90,140])
    # upper_blue = np.array([130,255,255])

    # Get Arm Only Or Skin Only
    # lower_blue = np.array([0,90,110])
    # upper_blue = np.array([50,255,255])

    mask = cv2.inRange(frame, lower_blue, upper_blue)
    # mask = 255-mask
    res = cv2.bitwise_and(frame,frame, mask= mask)
    #
    cv2.imshow('mask',mask)
    # cv2.imshow('res',res)
    # cv2.imshow('frame',frame)
    # cv2.imshow("h", frame[:,:,0])
    # cv2.imshow("s", frame[:,:,1])
    # cv2.imshow("v", frame[:,:,2])



    #TODO: Here, do temporal threholding to remove all except the bar (non-glove part of arms will likely stay too)
    # grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # grayAverage = cv2.cvtColor(average, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # diffMatrix = cv2.absdiff(grayAverage, grayFrame).astype(np.float32)


    diffMatrixRGB = cv2.absdiff(frame.astype(np.float32), average)#.max(axis=2)

    diffMatrix = cv2.cvtColor(diffMatrixRGB, cv2.COLOR_BGR2GRAY)
    print(diffMatrix)



#    temporalThreshold = np.zeros_like(average)
#    temporalThreshold= cv2.threshold(diffMatrix, 50, cv2.THRESH_BINARY)

    ##trippy af bruh
    #temporalThreshold= cv2.threshold(diffMatrix, 0, 255, cv2.THRESH_BINARY)

    something, temporalThreshold = cv2.threshold(diffMatrix, 15, 255, cv2.THRESH_BINARY)
    print(diffMatrix)

    #do temporal averaging on the first few frames of 125_5x_lf_lowres.mov
        #create comparison frame by averaging the first few of the movie
        #find the difference (absdiff) of every frame and that comparison frame
        #create a mask where only large differences are kept
        #profit??

    #TODO: Morphological Operators to remove noise
    kernel = np.ones((10,10), np.uint8)
    erosion = cv2.erode(mask,kernel,iterations = 1)
    dilation = cv2.dilate(mask,kernel,iterations = 1)
    # cv2.imshow('Erosion',erosion)
    # cv2.imshow('Dilation',dilation)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow('Opening',opening)
    # cv2.imshow('Closing',closing)


    #TODO: Connected Components Analysis
    bimodalImg = opening.copy()
        #Find the centroid of the bar. Track it's (x,y) over time.

        #Find the velocity of the centroid of the bar. Track it over time
        # Create an RGB display image which to show the different regions.
    display = np.zeros((bimodalImg.shape[0], bimodalImg.shape[1], 3),
                      dtype='uint8')

    # Get the list of contours in the image. See OpenCV docs for
    # information about the arguments.
    bimodalImg, contours, hierarchy = cv2.findContours(bimodalImg, cv2.RETR_CCOMP,
                                              cv2.CHAIN_APPROX_SIMPLE)

    print('found', len(contours), 'contours')

    # The getccolors function from cvk2 supplies a useful list
    # of different colors to color things in with.
    ccolors = cvk2.getccolors()

    # Define the color white (used below).
    white = (255,255,255)

    # For each contour in the image
    for j in range(len(contours)):

        # Draw the contour as a colored region on the display image.
        cv2.drawContours(display, contours, j, ccolors[j % len(ccolors)], -1 )

        # Compute some statistics about this contour.
        info = cvk2.getcontourinfo(contours[j])

        # Mean location and basis vectors can be useful.
        mu = info['mean']
        b1 = info['b1']
        b2 = info['b2']

        # Annotate the display image with mean and basis vectors.
        cv2.circle( display, cvk2.array2cv_int(mu), 3, white, 1, cv2.LINE_AA )

        cv2.line( display, cvk2.array2cv_int(mu), cvk2.array2cv_int(mu+2*b1),
                  white, 1, cv2.LINE_AA )

        cv2.line( display, cvk2.array2cv_int(mu), cvk2.array2cv_int(mu+2*b2),
                  white, 1, cv2.LINE_AA )

    # Display the output image and wait for a keypress.
    cv2.imshow('Regions', display)

    # cv2.imshow('Contours', contours)

    ################################################################################################
    #                          Write the newly modified frame to the writer                        #
    ################################################################################################
    # Write if we have a writer.
    if writer:
        writer.write(frame)
    # Throw it up on the screen.
    cv2.imshow('Video', video)
    # cv2.imshow('average', average.astype(np.uint8))
    # cv2.imshow('diff matrix', diffMatrix.astype(np.uint8))
    # cv2.imshow('Gray Average', grayAverage.astype(np.uint8))
    # cv2.imshow('Gray Frame', grayFrame.astype(np.uint8))
    # cv2.imshow('Temporal Threshold', temporalThreshold.astype(np.uint8))
    # cv2.imshow('absdiff', diffMatrix.astype(np.uint8))

#    pdb.set_trace()


    # Delay for 5ms and get a key
    k = cv2.waitKey(5)
    # Check for ESC hit:
    if k % 0x100 == 27:
        break
