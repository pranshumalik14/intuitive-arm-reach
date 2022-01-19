import imp


import numpy as np
from robot_arm import RobotArm2D, RobotArm3D


def acceleration_cost(qdotdot, t):
    N = qdotdot.shape[1]
    numerator = 0
    for n in range(qdotdot.shape[1]):
        numerator += (N+1-n) * qdotdot[t, n]**2
    return numerator


def cost_function(target_pos, q, qdotdot, robot_arm, kt=1e-5, kT=1e2):
    x_TN = robot_arm.forward_kinematics(q)

    # reach cost
    # The squared-distance between the end-effector and the goal positions at the end of the movement
    norm_reach_cost = np.linalg.norm(x_TN - target_pos)**2
    # : A cost that corresponds to the largest angle over all the joints at the end of the movement
    comfort_cost = np.max(q[-1, :])
    C_T = kT*(norm_reach_cost) + comfort_cost

    # acce cost
    N = qdotdot.shape[1]
    denom = 0
    for n in range(qdotdot.shape[1]):
        denom += (N+1 - n)

    acce_cost = 0
    for t in range(qdotdot.shape[0]):
        acce_cost += acceleration_cost(qdotdot, t)
    acce_cost = (kt*acce_cost)/denom

    # objective function
    J_Qdd = C_T + acce_cost
    return J_Qdd
