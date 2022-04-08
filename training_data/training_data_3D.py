import numpy as np

import sys
import random
import matplotlib.pyplot as plt
import pandas as pd

from datetime import date, datetime
import multiprocessing
from joblib import Parallel, delayed

sys.path.append('../scripts')
from training_data import save_file, iterate_circular
from task_info import numpy_linspace
import robot_arm as rb
import pdff_kinematic_sim_funcs as pdff_sim

time_str = datetime.now().strftime("%Y%m%d_%H%M")
columns = ["init_joint_angles", "x_target", "y_target", "z_target",  "Theta", "iter_count", "cost"]

def cart2sphe(x, y, z):
    """
    Convert cartesian coordinates into spherical
    """
    # https://keisan.casio.com/exec/system/1359533867
    rho = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x)
    phi = np.arctan2(np.sqrt(x**2+y**2), z)
    return (rho, phi, theta)

def sphe2cart(rho, phi, theta):
    """
    Convert spheical coordinates into cartesian
    https://keisan.casio.com/exec/system/1359534351
    """
    x = rho * np.cos(theta) * np.sin(phi)
    y = rho * np.sin(theta) * np.sin(phi)
    z = rho * np.cos(phi)
    return (x, y, z)

def generate_target_pts_on_arc(rho, d_theta, d_phi):
    X_dum, Y_dum, Z_dum = [], [], []
    ds_phi = rho*d_phi
    for s_phi in numpy_linspace(rho*np.deg2rad(0), rho*np.deg2rad(70), ds_phi):
        phi = s_phi/rho
        R = rho*np.cos(phi)

        ds_theta = R*d_theta

        Xs, Ys, Zs = [], [], []
        for s_theta in numpy_linspace(0, R*np.deg2rad(180), ds_theta):
            theta = s_theta/R
            # print("Rho is : {} and Phi is: {}".format(rho, phi))
            x, y, z = sphe2cart(rho, phi, theta)
            # print("X coord is : {} and Y coord is: {}".format(x, y))
            Xs.append(x)
            Ys.append(y)
            Zs.append(z)
        X_dum.append(Xs)
        Y_dum.append(Ys)
        Z_dum.append(Zs)

    assert(len(X_dum) == len(Y_dum) == len(Z_dum))

    return np.array(X_dum), np.array(Y_dum), np.array(Z_dum)

def generate_target_pts_3D(min_rho, max_rho, d_rho, d_theta, d_phi):
    """
    - Theta strictly starts from 0 to 180
    - Phi goes from 20 to 90
    - Rho goes from min_rho to total robot length
    """
    rhos = numpy_linspace(min_rho, max_rho, d_rho)

    X_circles, Y_circles, Z_circles = [], [], []

    for rho in rhos:
        x, y, z = generate_target_pts_on_arc(rho, d_theta, d_phi)
        X_circles.append(x)
        Y_circles.append(y)
        Z_circles.append(z)
        
    return X_circles, Y_circles, Z_circles

def visualize_circles(Xs, Ys, Zs):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    
    for i in range(len(Xs)):
        for j in range(len(Xs[i])):
            ax.scatter(Xs[i][j], Ys[i][j], Zs[i][j])
    plt.savefig('training_data_scatter3D.png')
    plt.show()

def find_closest_target_pt(x_ee, y_ee, z_ee, X_target, Y_target, Z_target):
    def euc_dist(x1, y1, z1, x2, y2, z2):
        return np.sqrt((y2-y1)**2 + (x2-x1)**2 + (z2-z1)**2)
    
    n_target_pts = len(X_target)

    closest_idx = 0
    min_dist = np.Inf

    for i in range(n_target_pts):
        dist = euc_dist(X_target[i], Y_target[i], x_ee, y_ee)

        if dist < min_dist:
            min_dist = dist
            closest_idx = i
    
    return closest_idx

def explore_random_joint_angles(X_cir, Y_cir, Z_cir, init_joint_angle, robot_arm):
    n_arcs = len(X_cir)

    print("Initial joint angles are :")
    print(init_joint_angle)

    # call forward kinematics here, to determine the closest point
    x_ee, y_ee, z_ee = robot_arm.forward_kinematics(init_joint_angle)

    B = 5
    N, _, _ = robot_arm.get_arm_params()


    for arc_idx in range(n_arcs):
        n_circles = len(X_cir[arc_idx])
        for circle_idx in range(n_circles):

            print("############### CIRCLE NO: {}/{} ###############".format(circle_idx+1, n_circles+1))
            X_target = X_cir[arc_idx][circle_idx]
            Y_target = Y_cir[arc_idx][circle_idx]
            Z_target = Z_cir[arc_idx][circle_idx]

            n_pts_on_circle = len(X_target)
            
            index = 0
            final_training_data = np.zeros(n_pts_on_circle * len(columns), dtype=object).reshape((n_pts_on_circle, len(columns)))

            closest_idx = find_closest_target_pt(
                X_target, Y_target, Z_target, 
                x_ee, y_ee, z_ee
            )

            # TODO: What changes here is that it is not necessary to n_pts_on_circle//2 need to be taken 
            # If 10 pts on circle and closest idx is 3, then 0-3 has one matrix (CW)
            # 3-9 is another (CCW)

            # Get x,y cw
            # X_buf_cw = iterate_circular(
            #     X_target, closest_idx, n_pts_on_circle//2, cw=True)
            # Y_buf_cw = iterate_circular(
            #     Y_target, closest_idx, n_pts_on_circle//2, cw=True)
            X_buf_cw = X_target[0:closest_idx+1]
            Y_buf_cw = Y_target[0:closest_idx+1]
            Z_buf_cw = Z_target[0:closest_idx+1]
            cw_pts_count = len(X_buf_cw)

            # Get x,y ccw
            # X_buf_ccw = iterate_circular(
            #     X_target, closest_idx, n_pts_on_circle//2, cw=False)
            # Y_buf_ccw = iterate_circular(
            #     Y_target, closest_idx, n_pts_on_circle//2, cw=False)
            X_buf_ccw = X_target[closest_idx: -1]
            Y_buf_ccw = Y_target[closest_idx: -1]
            Z_buf_ccw = Z_target[closest_idx: -1]
            ccw_pts_count = len(X_buf_ccw)

            Theta_matrix = np.zeros(B*N).reshape((B, N))

            # go through CW points
            for j in range(cw_pts_count):
                final_training_data[index, 0] = init_joint_angle
                final_training_data[index, 1] = X_buf_cw[j]
                final_training_data[index, 2] = Y_buf_cw[j]
                final_training_data[index, 3] = Z_buf_cw[j]

                Theta_matrix, iter_count, J_hist, task_info = pdff_sim.gen_theta(
                    x_target=np.array([X_buf_cw[j], Y_buf_cw[j], Z_buf_cw[j]]),
                    init_condit = [init_joint_angle, np.zeros(4)],
                    robot_arm=robot_arm,
                    B=B,
                    N=N,
                    Theta_matrix=Theta_matrix
                )

                final_training_data[index, 4] = Theta_matrix
                final_training_data[index, 5] = iter_count
                final_training_data[index, 6] = J_hist[-1]

                index = index + 1

            # Go through CCW points
            Theta_matrix = np.zeros(B*N).reshape((B, N))
            for j in range(ccw_pts_count):
                final_training_data[index, 0] = init_joint_angle
                final_training_data[index, 1] = X_buf_ccw[j]
                final_training_data[index, 2] = Y_buf_ccw[j]
                final_training_data[index, 3] = Z_buf_ccw[j]

                Theta_matrix, iter_count, J_hist, task_info = pdff_sim.gen_theta(
                    x_target=np.array([X_buf_ccw[j], Y_buf_ccw[j], Z_buf_ccw[j]]),
                    init_condit = [init_joint_angle, np.zeros(4)],
                    robot_arm=robot_arm,
                    B=B,
                    N=N,
                    Theta_matrix=Theta_matrix
                )

                final_training_data[index, 4] = Theta_matrix
                final_training_data[index, 5] = iter_count
                final_training_data[index, 6] = J_hist[-1]

                index = index + 1
        
            df = pd.DataFrame(
                data=final_training_data,
                columns=columns
            )
            save_file(df)

def generate_training_data3D(robot_arm, N = 10):
    # TODO: Get robot min length and max length
    X_cir, Y_cir, Z_cir = generate_target_pts_3D(2, 5, 0.05, np.pi/64, np.pi/64)
    assert(len(X_cir) == len(Y_cir) == len(Z_cir))
    visualize_circles(X_cir, Y_cir, Z_cir)

    # Need to pick N number of random target points, on which inverse kine would be applied and that would be chosen as a pseudo random config
    # Need to generate N random indices between 0 and len(X_cir)
    rand_idxs = random.sample(range(0, len(X_cir)), N)
    init_starting_pts = [(X_cir[idx], Y_cir[idx], Z_cir[idx]) for idx in rand_idxs]
    print(init_starting_pts)
    # TODO: how to get curq here?
    cur_q = [0, 90, 90, 90, 0, 0]
    init_joint_angles = [robot_arm.inverse_kinematics(point, cur_q) for point in init_starting_pts]

    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores)(delayed(explore_random_joint_angles)(X_cir, Y_cir, Z_cir, init_joint_angles[i], robot_arm) for i in range(N))

if __name__ == "__main__":
    robot_arm = rb.Braccio3D()
    generate_training_data3D(robot_arm)


