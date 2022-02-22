import time
import cv2
import numpy as np
import pyrealsense2 as rs
from spatialmath import *


def keypoints_average(keypoints):
    num, sum_x, sum_y = 0, 0, 0

    for keyp in keypoints:
        num += 1
        sum_x += keyp.pt[0]
        sum_y += keyp.pt[1]

    if num == 0:
        return 0, 0

    ave_x = sum_x/num
    ave_y = sum_y/num
    return round(ave_x), round(ave_y)


def blob_param(minArea=10, maxArea=10000):
    # blob params
    params = cv2.SimpleBlobDetector_Params()

    params.minThreshold = 127
    params.maxThreshold = 200

    # thresholds
    params.filterByColor = True
    params.blobColor = 255

    # filter by area
    params.filterByArea = True
    params.minArea = minArea
    params.maxArea = maxArea

    # filter by circularity
    params.filterByCircularity = False

    # filter by convexity
    params.filterByConvexity = False

    # filter by inertia
    params.filterByInertia = False

    return params


def set_config():
    # set frame resolution
    resolutionWidth = 848
    resolutionHeight = 480
    frameRate = 30
    config = rs.config()

    # Get device product line for setting a supporting resolution
    config.enable_stream(rs.stream.depth, resolutionWidth,
                         resolutionHeight, rs.format.z16, frameRate)
    config.enable_stream(rs.stream.color, resolutionWidth,
                         resolutionHeight, rs.format.bgr8, frameRate)
    config.enable_stream(rs.stream.infrared, 1, resolutionWidth,
                         resolutionHeight, rs.format.y8, frameRate)
    return config


def view_stream(pipeline, show_origin='True', show_goal='True'):
    while True:
        # wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        infra_frame = frames.get_infrared_frame()
        if not depth_frame or not color_frame or not infra_frame:
            continue

        # convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        infra_image = np.asanyarray(infra_frame.get_data())

        # apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(
            depth_image, alpha=0.03), cv2.COLORMAP_TURBO)
        infra_colormap = cv2.applyColorMap(infra_image, cv2.COLORMAP_HOT)
        _, infra_thresh = cv2.threshold(
            infra_image, 50, 255, cv2.THRESH_BINARY)
        hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        goal_thresh = cv2.inRange(
            hsv_image, (96, 76, 86), (131, 255, 255))

        # show keypoints and robot base
        if (show_origin == 'True'):
            ave_x, ave_y, keypoints = current_origin(pipeline)
            radius = 10
            drawcolor = (0, 255, 0)  # green
            circlecenter = (ave_x, ave_y)
            thickness = 2
            image = cv2.circle(color_image, circlecenter,
                               radius, drawcolor, thickness)
            image = cv2.line(image, (ave_x-25, ave_y),
                             (ave_x+25, ave_y), drawcolor, thickness)
            image = cv2.line(image, (ave_x, ave_y-25),
                             (ave_x, ave_y+25), drawcolor, thickness)
            infra_thresh = cv2.drawKeypoints(infra_thresh, keypoints, None, color=(
                0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # show goal
        if (show_goal == 'True'):
            ave_x, ave_y, keypoints = current_goal(pipeline)
            radius = 10
            drawcolor = (0, 0, 255)  # red
            circlecenter = (ave_x, ave_y)
            thickness = 2
            image = cv2.circle(color_image, circlecenter,
                               radius, drawcolor, thickness)
            image = cv2.line(image, (ave_x-25, ave_y),
                             (ave_x+25, ave_y), drawcolor, thickness)
            image = cv2.line(image, (ave_x, ave_y-25),
                             (ave_x, ave_y+25), drawcolor, thickness)
            goal_thresh = cv2.drawKeypoints(goal_thresh, keypoints, None, color=(
                0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        cv2.namedWindow('Color', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('Infra', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('Depth', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('Thresh', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('Goal', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Color', color_image)
        cv2.imshow('Infra', infra_colormap)
        cv2.imshow('Depth', depth_colormap)
        cv2.imshow('Thresh', infra_thresh)
        cv2.imshow('Goal', goal_thresh)
        cv2.waitKey(1)
    print("View stream completed")


def current_origin(pipeline):
    # wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()
    infra_frame = frames.get_infrared_frame()
    while not infra_frame:
        infra_frame = frames.get_infrared_frame()

    # convert images to numpy arrays
    infra_image = np.asanyarray(infra_frame.get_data())

    # crop out calibration area
    infra_thresh = np.zeros_like(infra_image)
    infra_thresh[150:250, 530:625] = infra_image[150:250, 530:625]

    # blob params
    params = blob_param(minArea=5, maxArea=1000)

    # apply threshold
    _, infra_thresh = cv2.threshold(
        infra_thresh, 50, 255, cv2.THRESH_BINARY)

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(infra_thresh)

    # show keypoints and images
    ave_x, ave_y = keypoints_average(keypoints)
    return ave_x, ave_y, keypoints


def calibrated_origin(pipeline):
    sum_x, sum_y = 0, 0
    for i in range(5):
        ave_x, ave_y, _ = current_origin(pipeline)
        sum_x += ave_x
        sum_y += ave_y
        time.sleep(1)
    ave_x = sum_x/5
    ave_y = sum_y/5
    return round(ave_x), round(ave_y)  # ox, oy


# y_pix_offset=30 for robot base
def vision2pixpoint_pos(pipeline, px, py, y_pix_offset=0):
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    depth_intrnsc = depth_frame.profile.as_video_stream_profile().intrinsics
    depth_value = depth_frame.get_distance(px, py+y_pix_offset)
    depth_point = rs.rs2_deproject_pixel_to_point(
        depth_intrnsc, [px, py], depth_value)
    return depth_point


def current_goal(pipeline):
    # wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())

    # blob params
    params = blob_param(minArea=50, maxArea=4000)

    # apply threshold
    hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    goal_thresh = cv2.inRange(
        hsv_image, (96, 76, 86), (131, 255, 255))

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(goal_thresh)

    ave_x, ave_y = keypoints_average(keypoints)
    return ave_x, ave_y, keypoints


def origin2vision_frame(vision2origin_pos):
    return (SE3(vision2origin_pos) * SE3.Rz(-np.pi/2) * SE3.Rx(np.pi)).inv()


def origin2point_pos(vision2point_pos, vision2origin_pos):
    return (origin2vision_frame(vision2origin_pos) * SE3(vision2point_pos)).t


def origin2goal_pos(pipeline, vision2origin_pos):
    gx, gy, _ = current_goal(pipeline)
    vision2goal_pos = vision2pixpoint_pos(pipeline, gx, gy)
    return origin2point_pos(vision2goal_pos, vision2origin_pos)


# main: run
pipeline = rs.pipeline()
config = set_config()
pipeline.start(config)
# view_stream(pipeline)
ox, oy = calibrated_origin(pipeline)
print([ox, oy])
vision2origin_pos = vision2pixpoint_pos(pipeline, ox, oy, y_pix_offset=30)
print(vision2origin_pos)
time.sleep(5)

while True:
    origin2goal = origin2goal_pos(pipeline, vision2origin_pos)
    print(origin2goal)
    time.sleep(0.5)

pipeline.stop()
cv2.destroyAllWindows()
