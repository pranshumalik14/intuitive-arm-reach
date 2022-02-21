# License: Apache 2.0. See LICENSE file in root directory.
# Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2

# set frame resolution
resolutionWidth = 848
resolutionHeight = 480
frameRate = 30

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
config.enable_stream(rs.stream.depth, resolutionWidth,
                     resolutionHeight, rs.format.z16, frameRate)
config.enable_stream(rs.stream.color, resolutionWidth,
                     resolutionHeight, rs.format.bgr8, frameRate)
config.enable_stream(rs.stream.infrared, 1, resolutionWidth,
                     resolutionHeight, rs.format.y8, frameRate)

# Start streaming
pipeline.start(config)

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        infra_frame = frames.get_infrared_frame()
        if not depth_frame or not color_frame or not infra_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        infra_image = np.asanyarray(infra_frame.get_data())

        # crop out calibration area
        infra_thresh = np.zeros_like(infra_image)
        infra_thresh[150:250, 530:625] = infra_image[150:250, 530:625]

        # blob params
        params = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        params.minThreshold = 60
        params.maxThreshold = 255

        # Filter by Area
        params.filterByArea = True
        params.minArea = 1500

        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.1

        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.87

        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.01

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(
            depth_image, alpha=0.03), cv2.COLORMAP_TURBO)
        infra_colormap = cv2.applyColorMap(infra_image, cv2.COLORMAP_HOT)
        _, infra_thresh = cv2.threshold(
            infra_thresh, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        depth_colormap_dim = depth_colormap.shape
        infra_colormap_dim = infra_colormap.shape
        color_colormap_dim = color_image.shape

        # Show images
        cv2.namedWindow('Color', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('Infra', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('Depth', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('Thresh', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Color', color_image)
        cv2.imshow('Infra', infra_colormap)
        cv2.imshow('Depth', depth_colormap)
        cv2.imshow('Thresh', infra_thresh)
        cv2.waitKey(1)

except:
    print("Error")

finally:

    # Stop streaming
    pipeline.stop()
