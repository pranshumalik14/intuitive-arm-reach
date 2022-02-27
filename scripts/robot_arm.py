from abc import abstractmethod
from traceback import print_tb
import numpy as np
import roboticstoolbox as rtb
from distutils.dir_util import copy_tree
from os import path
import os
import pandas as pd

class RobotArm:

    @abstractmethod
    def forward_kinematics(self, q):
        assert(isinstance(q, np.ndarray))
        # returns coordinates (xyz or just xy) of endeffector at time T
        # q can be one dimension or two dimensional
        to_return = q
        if q.ndim == 1:
            pass
        elif q.ndim == 2:
            to_return = np.array(q[-1, :])

        return to_return

class RobotArm2D(RobotArm):
    def __init__(self, n_dims, link_lengths):
        super().__init__()
        self.n_dims = n_dims
        self.arm_length = sum(link_lengths)
        self.link_lengths = link_lengths
        # Sample input: np.array([2, 0.6, 0.4, 0.15, 0.15, 0.1])

    def get_arm_params(self):  
        return self.n_dims, self.arm_length, self.link_lengths

    def forward_kinematics(self, q):
        q = super().forward_kinematics(q)
        sum_angles = 0
        links_x = np.zeros(self.n_dims+1)
        links_y = np.zeros(self.n_dims+1)

        for i_dim in range(self.n_dims):
            sum_angles += q[i_dim]
            links_x[i_dim + 1] = links_x[i_dim] + \
                np.cos(sum_angles) * self.link_lengths[i_dim]
            links_y[i_dim + 1] = links_y[i_dim] + \
                np.sin(sum_angles) * self.link_lengths[i_dim]

        return np.array(links_x[-1], links_y[-1])

    def angles_to_link_positions(self, q):
        # TODO: have to test this, or remove if it wont be used
        # Could rename it to forward_kinematics_all
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

        # link_positions = np.zeros((n_time_steps, 2*(self.n_dims+1)))
        # # desired structure of link positions is x y x y x y
        # # (first x y are for the base joint)
        # for n in range(self.n_dims + 1):
        #     link_positions[:, 2*n] = links_x[:, n]
        #     link_positions[:, 2*n+1] = links_y[:, n]

        return [links_x, links_y]

    def to_df(self):
        df = {}
        df["n_dims"]        = self.n_dims
        df["arm_length"]    = self.arm_length
        df["link_lenghts"]  = self.link_lengths

        return pd.DataFrame.from_dict(data = df, orient = "index").transpose()

def robot2D_from_df(df):
    return RobotArm2D(df["n_dims"], df["link_lenghts"])



class RobotArm3D(rtb.DHRobot, RobotArm):
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

    def forward_kinematics(self, q):
        q = super().forward_kinematics(q)
        return self.fkine(q).t


class Braccio(rtb.ERobot):

    def __init__(self, xacro_dir):

        links, name, urdf_string, urdf_filepath = self.URDF_read(
            "braccio_description/urdf/braccio.urdf",
            tld=xacro_dir
        )

        # super().__init__(
        #     links,
        #     name=name,
        #     manufacturer="Arduino",
        #     gripper_links=links[12],
        #     urdf_string=urdf_string,
        #     urdf_filepath=urdf_filepath,
        # )

# temp_dir = tf.mkdtemp()

# xacro_dir = path.join(temp_dir, "custom_xacro_folder")

# # Make xacro folder
# mkdir(xacro_dir)

# xacro_dir = os.getcwd()
# print(xacro_dir)

# braccio_dir = path.join(xacro_dir, "braccio_description")
# print(braccio_dir)

# xacro_path = rtb.rtb_path_to_datafile("xacro", local = False)
# print(xacro_path)

# braccio_xacro = xacro_path / "braccio_description"
# if not os.path.isdir(braccio_dir):
#     os.mkdir(braccio_xacro)
# print(braccio_xacro)

# # Copy into temp franka directory
# copy_tree(braccio_xacro, braccio_dir)

# # Make our custom robot
# robot = Braccio(xacro_dir)
# # "braccio_description/urdf/braccio.urdf"


