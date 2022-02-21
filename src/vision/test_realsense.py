import time
import cv2
import numpy as np
import pyrealsense2 as rs
# import roboticstoolbox as rtb


def keypoints_average(keypoints):
    num, sum_x, sum_y = 0, 0, 0

    for keyp in keypoints:
        num += 1
        sum_x += keyp.pt[0]
        sum_y += keyp.pt[1]
    ave_x = sum_x/num
    ave_y = sum_y/num
    return round(ave_x), round(ave_y)


def blob_param(minArea=10, maxArea=10000):
    # blob params
    params = cv2.SimpleBlobDetector_Params()

    # thresholds
    params.filterByColor = False

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
            ave_x, ave_y = current_goal(pipeline)
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

        cv2.namedWindow('Color', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('Infra', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('Depth', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('Thresh', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Color', color_image)
        cv2.imshow('Infra', infra_colormap)
        cv2.imshow('Depth', depth_colormap)
        cv2.imshow('Thresh', infra_thresh)
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
    params = blob_param(minArea=10, maxArea=10000)

    # apply colormap on depth image (image must be converted to 8-bit per pixel first)
    infra_colormap = cv2.applyColorMap(infra_image, cv2.COLORMAP_HOT)
    _, infra_thresh = cv2.threshold(
        infra_thresh, 50, 255, cv2.THRESH_BINARY)

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(infra_thresh)

    # show keypoints and images
    ave_x, ave_y = keypoints_average(keypoints)
    return ave_x, ave_y, keypoints


def calibrated_origin(pipeline):
    sum_x, sum_y = 0, 0
    for i in range(0, 5):
        ave_x, ave_y, _ = current_origin(pipeline)
        sum_x += ave_x
        sum_y += ave_y
        time.sleep(1)
    ave_x = sum_x/5
    ave_y = sum_y/5
    return round(ave_x), round(ave_y)  # ox, oy


def origin_transform(pipeline, ox, oy, offset_x=0):  # offset_x=20 for robot base
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    depth_intrnsc = depth_frame.profile.as_video_stream_profile().intrinsics
    depth_value = depth_frame.get_distance(ox-offset_x, oy)
    depth_point = rs.rs2_deproject_pixel_to_point(
        depth_intrnsc, [ox, oy], depth_value)
    return depth_point


def current_goal(pipeline):
    # wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()

    # blob params
    params = blob_param(minArea=500, maxArea=20000)

    # apply threshold
    lower_color_bounds = np.array([100, 0, 0])
    upper_color_bounds = np.array([225, 80, 80])
    gray = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(color_frame, lower_color_bounds, upper_color_bounds)
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    color_frame = color_frame & mask_rgb

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(color_frame)

    ave_x, ave_y = keypoints_average(keypoints)
    return ave_x, ave_y


# def base_to_camera():
#     Tx, Ty, Tz = 0.35, -0.33, 1.46  # translation values
#     z_angle = -90
#     x_angle = 180
#     T = (transl(Tx, Ty, Tz) @ trotz(z_angle, 'deg') @ trotx(x_angle, 'deg')).inv()
#     return T

# main: run
pipeline = rs.pipeline()
config = set_config()
pipeline.start(config)
ox, oy = calibrated_origin(pipeline)
print(origin_transform(pipeline, ox, oy, offset_x=20))
# print(calibrated_origin(pipeline))
# view_stream(pipeline, show_goal='False')
# current_goal(pipeline)
