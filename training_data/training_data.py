from os import sep
import numpy as np
import sys 
sys.path.append('../scripts')
import pdff_kinematic_sim_funcs as pdff_sim
import robot_arm as rb
from task_info import numpy_linspace
import pandas as pd
import matplotlib.pyplot as plt

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

def generate_2D_target_pose(robot_arm, d_rho = 0.01, d_phi = (np.pi/10)):
    """
    Generate a continuously sampled space for target poses, for a 2D Robot arm.
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

def visualize_target_poses(Xs, Ys):
    # robot_arm = rb.RobotArm2D(
    #     n_dims = 3,
    #     link_lengths=np.array([0.4, 0.5, 0.1])
    # )

    # Xs, Ys = generate_2D_target_pose(robot_arm)

    plt.scatter(Xs, Ys)
    plt.savefig('training_data_sc_plot.png')


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

def gen_training_data(robot_arm, n_joint_config = 10):
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
    X_target, Y_target = generate_2D_target_pose(
        robot_arm,
        d_rho = 0.1,
        d_phi = np.pi/8
        )
    assert(len(X_target) == len(Y_target))
    n_target_pts = len(X_target)

    B = 5
    N, _, _ = robot_arm.get_arm_params()

    columns = ["init_joint_angles", "x_target", "y_target", "Theta", "iter_count", "cost"]
    final_training_data = np.zeros(n_joint_config * n_target_pts * len(columns), dtype = object).reshape((n_joint_config* n_target_pts, len(columns)))

    for i in range(n_joint_config):
        # TODO: add multiprocessing or using numpy efficient iterators
        init_joint_angles = generate_random_init_joint_angle(robot_arm)
        # init_joint_angles_str = np.array2string(init_joint_angles[0], separator="|") # want only joint angles, not velocity since it is 0

        # Theta matrix is set to 0 everytime starting from new joint. TODO: ask Pranshu
        Theta_matrix = np.zeros(B*N).reshape((B, N))
        for j in range(n_target_pts):

            index = (i * n_target_pts) + j
            final_training_data[index, 0] = init_joint_angles[0]
            final_training_data[index, 1] = X_target[j]
            final_training_data[index, 2] = Y_target[j]
            
            Theta_matrix, iter_count, J_hist = pdff_sim.gen_theta(
                x_target = np.array([X_target[j], Y_target[j]]),
                init_condit = init_joint_angles,
                robot_arm = robot_arm,
                B = B,
                N = N,
                Theta_matrix = Theta_matrix 
            )
            final_training_data[index, 3] = Theta_matrix
            final_training_data[index, 4] = iter_count
            final_training_data[index, 5] = J_hist[-1]

    df = pd.DataFrame(
        data    = final_training_data,
        columns = columns
        )

    df.to_csv(
        'sample_data_new.csv',
        index = False
    )

if __name__ == '__main__':
    # TODO: ask pranshu what Params are ideal for 2D Robot training
    robot_arm = rb.RobotArm2D(
        n_dims = 2,
        link_lengths=np.array([0.6, 0.4])
    )

    gen_training_data(robot_arm)



    
    

