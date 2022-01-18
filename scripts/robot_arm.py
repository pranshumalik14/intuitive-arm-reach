import numpy as np
import roboticstoolbox as rtb


class RobotArm2D:
    def __init__(self, n_dims=2, arm_length=0.8, rel_link_lengths=np.array([2, 0.6, 0.4, 0.15, 0.15, 0.1])):
        self.n_dims = n_dims
        self.arm_length = arm_length
        self.link_lengths = arm_length / \
            sum(rel_link_lengths) * rel_link_lengths[0:n_dims+1]
        # np.array([1.5, 1.25, 0.4, 0.15, 0.15, 0.1])

    def get_arm_params(self):  # TODO: remove relative link lengths
        return self.n_dims, self.arm_length, self.link_lengths

    def angles_to_link_positions(self, q):
        # Forward kinematics
        n_time_steps = q.shape[0]
        q_dims = q.shape[1]

        assert(self.n_dims == q_dims)

        links_x = np.zeros((n_time_steps, self.n_dims+1))
        links_y = np.zeros((n_time_steps, self.n_dims+1))

        for t in range(n_time_steps):
            sum_angles = 0

            for i_dim in range(self.n_dims):
                sum_angles += q[t, i_dim]
                links_x[t, i_dim + 1] = links_x[t, i_dim] + \
                    np.cos(sum_angles) * self.link_lengths[i_dim]
                links_y[t, i_dim + 1] = links_y[t, i_dim] + \
                    np.sin(sum_angles) * self.link_lengths[i_dim]

        link_positions = np.zeros((n_time_steps, 2*(self.n_dims+1)))
        # desired structure of link positions is x y x y x y
        # (first x y are for the base joint)
        for n in range(self.n_dims + 1):
            link_positions[:, 2*n] = links_x[:, n]
            link_positions[:, 2*n+1] = links_y[:, n]

        return link_positions


class RobotArm3D(rtb.DHRobot):
    def __init__(self, name, DHDescription, qz, qr, meshdir):
        super().__init__(DHDescription, name=name, meshdir=meshdir)

        self.addconfiguration("qz", qz)
        self.addconfiguration("qr", qr)

        # links, name, urdf_string, urdf_filepath = self.URDF_read(
        #     rtb.rtb_path_to_datafile("../urdf/braccio.urdf", local=True)
        # )
        # super().__init__(
        #     links,
        #     name=name,
        #     manufacturer="Arduino",
        #     gripper_links=links[12],
        #     urdf_string=urdf_string,
        #     urdf_filepath=urdf_filepath,
        # )

    def get_arm_params(self):
        return 5, 5, np.ones(5)  # TODO: latter 2 returns are meaningless
