from os import sep
import numpy as np
import sys 
sys.path.append('../scripts')
import pdff_kinematic_sim_funcs as pdff_sim
import robot_arm as rb
from task_info import numpy_linspace
import pandas as pd

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)

def generate_2D_target_pose(robot_arm, d_rho = 0.01, d_phi = (np.pi/10)):

    n_dims, arm_length, link_lengths = robot_arm.get_arm_params()
    R = arm_length

    first_link_length = link_lengths[0]

    ds = d_phi * R

    Xs, Ys = [], []
    # want concentric circles that

    for rho in numpy_linspace(first_link_length, R, d_rho):
        # s = R*theta
        # hence ds = R*d_theta
        for s in numpy_linspace(0, 2*np.pi*rho, ds):
            phi = s/rho

            # print("Rho is : {} and Phi is: {}".format(rho, phi))
            x, y = pol2cart(rho, phi)
            # print("X coord is : {} and Y coord is: {}".format(x, y))
            Xs.append(x)
            Ys.append(y)    
    
    return np.array(Xs), np.array(Ys)

def dummy():
    import matplotlib.pyplot as plt
    robot_arm = rb.RobotArm2D(
        n_dims = 3,
        link_lengths=np.array([0.4, 0.5, 0.1])
    )

    Xs, Ys = generate_2D_target_pose(robot_arm)

    plt.scatter(Xs, Ys)
    plt.savefig('training_data_sc_plot.png')


def generate_random_init_joint_angle(robot_arm):
    n_dims, arm_length, link_lengths = robot_arm.get_arm_params()

    joint_angles = np.zeros(n_dims)
    joint_velocities = np.zeros(n_dims)

    for n in range(n_dims):
        joint_angles[n] = -1*np.pi + (2*np.pi * np.random.random_sample())

    return [joint_angles, joint_velocities]

def gen_training_data(robot_arm, N = 10):
    X_target, Y_target = generate_2D_target_pose(
        robot_arm,
        d_rho = 0.1,
        d_phi = np.pi/5
        )
    N2_rename = len(X_target)

    final_training_data = np.zeros(N * N2_rename * 6, dtype = object).reshape((N * N2_rename, 6))

    for i in range(N):
        # TODO: add multiprocessing or using numpy efficient iterators
        init_joint_angles = generate_random_init_joint_angle(robot_arm)
        # init_joint_angles_str = np.array2string(init_joint_angles[0], separator="|") # want only joint angles, not velocity since it is 0

        for j in range(N2_rename):
            index = (i * N2_rename) + j
            final_training_data[index, 0] = init_joint_angles[0]
            final_training_data[index, 1] = X_target[j]
            final_training_data[index, 2] = Y_target[j]
            Theta, iter_count, J_hist = pdff_sim.gen_theta(
                x_target = np.array([X_target[j], Y_target[j]]),
                init_condit = init_joint_angles,
                robot_arm = robot_arm
            )
            final_training_data[index, 3] = Theta
            final_training_data[index, 4] = iter_count
            final_training_data[index, 5] = J_hist[-1]

    df = pd.DataFrame(
        data = final_training_data, 
        columns = ["init_joint_angles", "x_target", "y_target", "Theta", "iter_count", "cost"]
        )

    df.to_csv(
        'sample_data_new.csv',
        index = False
    )

robot_arm = rb.RobotArm2D(
    n_dims = 3,
    link_lengths=np.array([0.4, 0.5, 0.1])
)

gen_training_data(robot_arm)



    
    

