import cv2
import time
import json
import numpy as np
import pyrealsense2 as rs
from multiprocessing import Pipe
from spatialmath import *


def vision_run(vision_subproc_pipe):
    print("[VISION] Thread Started")
    pipeline = init_stream()

    msg = None
    engaged = False
    calib_iter = 0
    calib_tmstmp = None
    calib_ox, calib_oy = 0, 0
    calib_vision2origin_pos = None

    while msg != "vision_stop":
        frames = pipeline.wait_for_frames()

        # follow message and if output ready, send to main; if not, set engaged
        if msg == "vision_calib_orig2vis_frame":
            engaged = True
            if calib_iter == 0:
                calib_iter += 1
                ox, oy, _ = current_origin(frames)
                calib_ox += ox
                calib_oy += oy
                calib_tmstmp = time.time()
            elif calib_iter > 5:
                msg = None
                calib_vision2origin_pos = vision2pixpoint_pos(frames, round(
                    calib_ox/calib_iter), round(calib_oy/calib_iter), y_pix_offset=30)
                calib_ox = 0
                calib_oy = 0
                calib_iter = 0
                engaged = False
                calib_tmstmp = None
                calib_orig2vis_frame = origin2vision_frame(
                    calib_vision2origin_pos)
                vision_subproc_pipe.send(calib_orig2vis_frame)
            elif time.time() - calib_tmstmp > 1.0:
                calib_iter += 1
                ox, oy, _ = current_origin(frames)
                calib_ox += ox
                calib_oy += oy
                calib_tmstmp = time.time()
        elif msg == "vision_curr_orig2vis_frame":
            msg = None
            ox, oy, _ = current_origin(frames)
            vision2origin_pos = vision2pixpoint_pos(
                frames, ox, oy, y_pix_offset=30)
            orig2vis_frame = origin2vision_frame(vision2origin_pos)
            vision_subproc_pipe.send(orig2vis_frame)
        elif msg == "vision_curr_orig2goal_pos":
            msg = None
            if calib_vision2origin_pos == None:
                ox, oy, _ = current_origin(frames)
                vision2origin_pos = vision2pixpoint_pos(
                    frames, ox, oy, y_pix_offset=30)
                orig2goal_pos = origin2goal_pos(frames, vision2origin_pos)
            else:
                orig2goal_pos = origin2goal_pos(
                    frames, calib_vision2origin_pos)
            vision_subproc_pipe.send(orig2goal_pos)

        # view current stream
        view_stream(frames)

        if engaged:
            continue
        elif vision_subproc_pipe.poll():
            msg = vision_subproc_pipe.recv()

    # close stream while exiting and inform main driver thread
    close_stream(pipeline)
    vision_subproc_pipe.send("vision_exiting")
    print("[VISION] Thread Exited")
    return


def init_stream():
    pipeline = rs.pipeline()
    config = set_config()
    config = pipeline.start(config)
    load_presets(config)
    return pipeline


def close_stream(pipeline):
    pipeline.stop()
    cv2.destroyAllWindows()


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

    params.minThreshold = 200
    params.maxThreshold = 255

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


def load_presets(config, filename="src/vision/realsense_presets.json"):
    vis_presets = json.load(open(filename))
    pset_string = str(vis_presets).replace("'", '\"')
    device = config.get_device()
    advmod = rs.rs400_advanced_mode(device)
    advmod.load_json(pset_string)


def set_config():
    # set frame resolution
    resolution_width = 848
    resolution_height = 480
    frameRate = 30
    config = rs.config()

    # Get device product line for setting a supporting resolution
    config.enable_stream(rs.stream.depth, resolution_width,
                         resolution_height, rs.format.z16, frameRate)
    config.enable_stream(rs.stream.color, resolution_width,
                         resolution_height, rs.format.bgr8, frameRate)
    config.enable_stream(rs.stream.infrared, 1, resolution_width,
                         resolution_height, rs.format.y8, frameRate)
    return config


def view_stream(frames, show_origin='True', show_goal='True'):
    colorizer = rs.colorizer()

    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    infra_frame = frames.get_infrared_frame()

    # convert images to numpy arrays
    depth_image = np.asanyarray(colorizer.colorize(depth_frame).get_data())
    color_image = np.asanyarray(color_frame.get_data())
    infra_image = np.asanyarray(infra_frame.get_data())

    # apply colormap on infra image and apply HSV threshold to goal image
    infra_colormap = cv2.applyColorMap(infra_image, cv2.COLORMAP_HOT)
    _, infra_thresh = cv2.threshold(
        infra_image, 50, 255, cv2.THRESH_BINARY)
    hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    goal_thresh = cv2.inRange(
        hsv_image, (96, 120, 86), (131, 255, 255))

    # show keypoints and robot base
    if (show_origin == 'True'):
        ave_x, ave_y, keypoints = current_origin(frames)
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
        ave_x, ave_y, keypoints = current_goal(frames)
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

    # hardcoded windows
    cv2.namedWindow('Color', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Infra', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Depth', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Thresh', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Goal', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Color', 530, 310)
    cv2.resizeWindow('Infra', 530, 310)
    cv2.resizeWindow('Depth', 530, 310)
    cv2.resizeWindow('Thresh', 530, 310)
    cv2.resizeWindow('Goal', 530, 310)
    cv2.moveWindow('Color', 60+530, 140+155+15)
    cv2.moveWindow('Infra', 60+0, 140+310+30)
    cv2.moveWindow('Depth', 60+1060, 140+310+30)
    cv2.moveWindow('Thresh', 60+0, 140+0)
    cv2.moveWindow('Goal', 60+1060, 140+0)
    cv2.imshow('Color', color_image)
    cv2.imshow('Infra', infra_colormap)
    cv2.imshow('Depth', depth_image)
    cv2.imshow('Thresh', infra_thresh)
    cv2.imshow('Goal', goal_thresh)
    cv2.waitKey(1)


def current_origin(frames):
    infra_frame = frames.get_infrared_frame()
    while not infra_frame:
        infra_frame = frames.get_infrared_frame()

    # convert images to numpy arrays
    infra_image = np.asanyarray(infra_frame.get_data())

    # crop out calibration area
    infra_thresh = np.zeros_like(infra_image)
    infra_thresh[150:250, 530:625] = infra_image[150:250, 530:625]

    # blob params
    params = blob_param(minArea=10, maxArea=1000)

    # apply threshold
    _, infra_thresh = cv2.threshold(
        infra_thresh, 50, 255, cv2.THRESH_BINARY)

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(infra_thresh)

    # show keypoints and images
    ave_x, ave_y = keypoints_average(keypoints)
    return ave_x, ave_y, keypoints


# y_pix_offset=30 for robot base
def vision2pixpoint_pos(frames, px, py, y_pix_offset=0):
    # invalid input
    if (px == 0 and py == 0) or (px < 0 or py < 0):
        return [-1, -1, -1]
    depth_frame = frames.get_depth_frame()
    depth_intrnsc = depth_frame.profile.as_video_stream_profile().intrinsics
    depth_value = depth_frame.get_distance(px, py+y_pix_offset)
    depth_point = rs.rs2_deproject_pixel_to_point(
        depth_intrnsc, [px, py], depth_value)
    return depth_point


def current_goal(frames):
    color_frame = frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())

    # blob params
    params = blob_param(minArea=40, maxArea=4000)

    # apply threshold
    hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    goal_thresh = cv2.inRange(
        hsv_image, (96, 120, 86), (131, 255, 255))

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(goal_thresh)

    ave_x, ave_y = keypoints_average(keypoints)
    return ave_x, ave_y, keypoints


def origin2vision_frame(vision2origin_pos):
    return (SE3(vision2origin_pos) * SE3.Rz(-np.pi/2) * SE3.Rx(np.pi)).inv()


def origin2point_pos(vision2point_pos, vision2origin_pos):
    # invalid input
    if vision2point_pos == [-1, -1, -1]:
        return np.array([-1, -1, -1])
    return (origin2vision_frame(vision2origin_pos) * SE3(vision2point_pos)).t


def origin2goal_pos(frames, vision2origin_pos):
    gx, gy, _ = current_goal(frames)
    vision2goal_pos = vision2pixpoint_pos(frames, gx, gy)
    return origin2point_pos(vision2goal_pos, vision2origin_pos)
