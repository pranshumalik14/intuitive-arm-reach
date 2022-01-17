import numpy as np


class RobotArm2D:
    def __init__(self, n_dims=2, arm_length=0.8, rel_link_lengths=np.array([2, 0.6, 0.4, 0.15, 0.15, 0.1])):
        self.n_dims = n_dims
        self.arm_length = arm_length
        self.link_lengths = arm_length / \
            sum(rel_link_lengths) * rel_link_lengths[0:n_dims+1]
        # np.array([1.5, 1.25, 0.4, 0.15, 0.15, 0.1])

    def get_arm_params(self):
        return self.n_dims, self.arm_length, self.link_lengths


def angles_to_link_positions_2D(q, robot_arm):
    # Forward kinematics
    n_time_steps = q.shape[0]
    n_dims = q.shape[1]

    n_dims_robot, arm_length, link_lengths = robot_arm.get_arm_params()
    assert(n_dims == n_dims_robot)

    links_x = np.zeros((n_time_steps, n_dims+1))
    links_y = np.zeros((n_time_steps, n_dims+1))

    for t in range(n_time_steps):
        sum_angles = 0

        for i_dim in range(n_dims):
            sum_angles += q[t, i_dim]
            links_x[t, i_dim + 1] = links_x[t, i_dim] + \
                np.cos(sum_angles) * link_lengths[i_dim]
            links_y[t, i_dim + 1] = links_y[t, i_dim] + \
                np.sin(sum_angles) * link_lengths[i_dim]

    link_positions = np.zeros((n_time_steps, 2*(n_dims+1)))
    # desired structure of link positions is x y x y x y
    # (first x y are for the base joint)
    for n in range(n_dims + 1):
        link_positions[:, 2*n] = links_x[:, n]
        link_positions[:, 2*n+1] = links_y[:, n]

    return link_positions
