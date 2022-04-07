import numpy as np
from roboticstoolbox.robot.ERobot import ERobot
from spatialmath import SE3
import os


class Braccio(ERobot):
    """
    Class that imports the Braccio Arm
    """

    def __init__(self):

        args = super().URDF_read(
            "braccio_description/urdf/braccio.urdf", tld=Braccio.load_my_path())
        super().__init__(
            args[0],
            name=args[1])

        self.manufacturer = "Arduino"
        self.ee_links = self.ee_links[0]
        self.tool = SE3([0.08, 0, 0])

        # zero angles, L shaped pose
        self.addconfiguration("qz", np.deg2rad(
            [90, 145, 0, 0]))

        # ready pose, arm up
        self.addconfiguration("qr", np.deg2rad(
            [0, 90, 90, 90]))

    @staticmethod
    def load_my_path():
        # print(__file__)
        os.chdir(os.path.dirname(__file__))
