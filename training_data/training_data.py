# Sys imports
from datetime import date, datetime
import time
import imp
from logging.handlers import WatchedFileHandler
from mimetypes import init

from os import sep
import numpy as np

from pathlib import Path
import uuid
import multiprocessing
from joblib import Parallel, delayed

import sys
sys.path.append('../scripts')

# Installed lib imports
import matplotlib.pyplot as plt
import pandas as pd

# Code-base imports
from task_info import numpy_linspace
import robot_arm as rb
import pdff_kinematic_sim_funcs as pdff_sim

time_str = datetime.now().strftime("%Y%m%d_%H%M")
columns = ["init_joint_angles", "x_target", "y_target", "Theta", "iter_count", "cost"]

def cart2pol(x, y):
    """
    Convert cartesian coordinates into polar
    """
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (rho, phi)

def pol2cart(rho, phi):
    """
    Convert polar coordinates into cartesian
    """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)

def generate_2D_target_pos_on_circle(rho, ds):
    Xs, Ys = [], []
    for s in numpy_linspace(0, 2*np.pi*rho, ds):
        phi = s/rho

        # print("Rho is : {} and Phi is: {}".format(rho, phi))
        x, y = pol2cart(rho, phi)
        # print("X coord is : {} and Y coord is: {}".format(x, y))
        Xs.append(x)
        Ys.append(y)

    assert(len(Xs) == len(Ys))

    return np.array(Xs), np.array(Ys)

def generate_2D_target_pos(robot_arm, d_rho=0.01, d_phi=(np.pi/10), closest_reach_radius_factor=1/3):
    """
    Generate a continuously sampled space for target positions, for a 2D Robot arm.
    Target poses belong to circles with a minimum radius of the first link of the robot arm and are further incremented by d_rho
    Because larger circles would need more points, d_phi (angle increments) are transformed into d_s (s = R*phi)

    Params:
    - @robot_arm: The robot arm to generate data for
    - @d_rho: increments of the circle
    - @d_phi: If radius of the circle is 1, then 1/d_phi number of points will be sampled from that space  

    Returns:
    - numpy array of all x coordinates of the sample space
    - numpy array of all y coordinates of the sample space
    """
    n_dims, arm_length, link_lengths = robot_arm.get_arm_params()
    R = arm_length

    # first_link_length = R
    n_circles = 0
    n_target_pts = 0
    closest_reach_radius = link_lengths[0] * closest_reach_radius_factor

    # s = R*theta
    # hence ds = R*d_theta
    ds = d_phi * R

    Xs, Ys = [], []
    # want concentric circles

    for rho in numpy_linspace(closest_reach_radius, R, d_rho):
        X_circle, Y_circle = generate_2D_target_pos_on_circle(rho, ds)

        Xs.append(X_circle)
        Ys.append(Y_circle)
        # Indication circle is over, now new circle
        n_target_pts = n_target_pts + len(X_circle)
        n_circles = n_circles + 1

    assert(len(Xs) == len(Ys))

    return Xs, Ys, n_target_pts

def iterate_circular(buf, start_index, iter_by, cw=True):
    buf_size = len(buf)
    ret = []
    for i in range(0, iter_by):
        index = None
        if cw:
            index = (start_index - i) % buf_size
        else:
            index = (start_index + i) % buf_size
        ret.append(buf[index])
    return np.array(ret)

def find_closest_target_pt(X_target, Y_target, x_ee, y_ee):
    n_target_pts = len(X_target)

    closest_idx = 0
    min_dist = np.Inf

    print("End effector pose is: ({}, {})".format(x_ee, y_ee))

    def euc_dist(x1, y1, x2, y2):
        return np.sqrt((y2-y1)**2 + (x2-x1)**2)
    for i in range(n_target_pts):
        dist = euc_dist(X_target[i], Y_target[i], x_ee, y_ee)

        if dist < min_dist:
            min_dist = dist
            closest_idx = i

    print("Closest point is ({}, {})".format(
        X_target[closest_idx], Y_target[closest_idx]))

    return closest_idx

def visualize_target_poses(Xs, Ys):
    # Xs, Ys = generate_2D_target_pos(robot_arm)
    # X(s) -> len -> n_circles
    for i in range(len(Xs)):
        plt.scatter(Xs[i], Ys[i])

    plt.axis('square')
    plt.savefig('training_data_scatter.png')

def generate_random_init_joint_angle(robot_arm):
    """
    - For a robot arm, generate a random joint configuration
    - For now, velocities are always initialized to 0
    TODO: What about joint constraints?
    """
    n_dims, arm_length, link_lengths = robot_arm.get_arm_params()

    joint_angles = np.zeros(n_dims)
    joint_velocities = np.zeros(n_dims)

    for n in range(n_dims):
        # -π + 2π * rand[0, 1] results in a range of -π to π
        joint_angles[n] = -1*np.pi + (2*np.pi * np.random.random_sample())

    return [joint_angles, joint_velocities]

def save_file(df : pd.DataFrame):
    output_file = '{}.csv'.format(uuid.uuid4().hex)
    output_dir = Path(time_str + "/")
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / output_file, index = False)  # can join path elements with / operator

def explore_random_joint_angles(circles_x, circles_y, robot_arm : rb.RobotArm2D):
    n_circles = len(circles_x)

    init_joint_angles = generate_random_init_joint_angle(robot_arm)
    # init_joint_angles_str = np.array2string(init_joint_angles[0], separator="|") # want only joint angles, not velocity since it is 0

    print("Initial joint angles are :")
    print(init_joint_angles[0])

    # call forward kinematics here, to determine the closest point
    x_ee, y_ee = robot_arm.forward_kinematics(init_joint_angles[0])

    B = 5
    N, _, _ = robot_arm.get_arm_params()


    for circle_idx in range(n_circles):
        print("############### CIRCLE NO: {}/{} ###############".format(circle_idx+1, n_circles+1))
        X_target = circles_x[circle_idx]
        Y_target = circles_y[circle_idx]

        n_pts_on_circle = len(X_target)
        
        index = 0
        final_training_data = np.zeros(n_pts_on_circle * len(columns), dtype=object).reshape((n_pts_on_circle, len(columns)))

        closest_idx = find_closest_target_pt(
            X_target, Y_target, x_ee, y_ee)

        # Get x,y cw
        X_buf_cw = iterate_circular(
            X_target, closest_idx, n_pts_on_circle//2, cw=True)
        Y_buf_cw = iterate_circular(
            Y_target, closest_idx, n_pts_on_circle//2, cw=True)

        # Get x,y ccw
        X_buf_ccw = iterate_circular(
            X_target, closest_idx, n_pts_on_circle//2, cw=False)
        Y_buf_ccw = iterate_circular(
            Y_target, closest_idx, n_pts_on_circle//2, cw=False)

        Theta_matrix = np.zeros(B*N).reshape((B, N))

        # go through CW points
        for j in range(n_pts_on_circle//2):
            final_training_data[index, 0] = init_joint_angles[0]
            final_training_data[index, 1] = X_buf_cw[j]
            final_training_data[index, 2] = Y_buf_cw[j]

            Theta_matrix, iter_count, J_hist, task_info = pdff_sim.gen_theta(
                x_target=np.array([X_buf_cw[j], Y_buf_cw[j]]),
                init_condit=init_joint_angles,
                robot_arm=robot_arm,
                B=B,
                N=N,
                Theta_matrix=Theta_matrix
            )

            final_training_data[index, 3] = Theta_matrix
            final_training_data[index, 4] = iter_count
            final_training_data[index, 5] = J_hist[-1]

            index = index + 1

        # Go through CCW points
        Theta_matrix = np.zeros(B*N).reshape((B, N))
        for j in range(n_pts_on_circle//2):
            final_training_data[index, 0] = init_joint_angles[0]
            final_training_data[index, 1] = X_buf_ccw[j]
            final_training_data[index, 2] = Y_buf_ccw[j]

            Theta_matrix, iter_count, J_hist, task_info = pdff_sim.gen_theta(
                x_target=np.array([X_buf_ccw[j], Y_buf_ccw[j]]),
                init_condit=init_joint_angles,
                robot_arm=robot_arm,
                B=B,
                N=N,
                Theta_matrix=Theta_matrix
            )

            final_training_data[index, 3] = Theta_matrix
            final_training_data[index, 4] = iter_count
            final_training_data[index, 5] = J_hist[-1]

            index = index + 1
    
        df = pd.DataFrame(
            data=final_training_data,
            columns=columns
        )
        save_file(df)

def gen_training_data(robot_arm, n_joint_config=10):
    """
    - take in a robot arm, generate a CSV file with training data. This includes:
        - columns = ["init_joint_angles", "x_target", "y_target", "Theta", "iter_count", "cost"] 
        - Theta (learnt Gaussian weights) for:
            - a pair of an initial joint angle configuration and (x_target, y_target)
        NOTE: For the same joint angle configuration, theta doesn't start from 0 matrix. 
              Each time a new (x_target, y_target) is provided, 
              the Theta matrix from the previous iteration (BUT the same init joint angle)

    Params:
        - robot_arm
        - n_joint_config: number of initial joint configurations

    """
    circles_x, circles_y, n_target_pts = generate_2D_target_pos(
        robot_arm,
        d_rho=0.05,
        d_phi=np.pi/64
    )

    visualize_target_poses(circles_x, circles_y)

    task_info = None

    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores)(delayed(explore_random_joint_angles)(circles_x, circles_y, robot_arm) for i in range(n_joint_config))

    # task_info_df = task_info.to_pd()
    # task_info_df.to_csv(
    #     '{}_task_info.csv'.format(time_str),
    #     index=False
    # )


if __name__ == '__main__':
    # TODO: ask pranshu what Params are ideal for 2D Robot training
    robot_arm = rb.RobotArm2D(
        n_dims=3,
        link_lengths=np.array([0.6, 0.3, 0.1])
    )
    start = time.process_time()
    gen_training_data(robot_arm, n_joint_config=2)
    print("Time taken : {}".format(time.process_time() - start))
