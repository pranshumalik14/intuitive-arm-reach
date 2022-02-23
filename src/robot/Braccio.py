import numpy as np
from roboticstoolbox.robot.ERobot import ERobot
from math import pi
import os

class Braccio(ERobot):
    """
    Class that imports a planar3DOF Robot
    """

    def __init__(self):

        args = super().URDF_read(
            "braccio_description/urdf/braccio.urdf",tld=Braccio.load_my_path())
        super().__init__(
            args[0],
            name=args[1])

        self.manufacturer = "Utadeo"
        # self.ee_link = self.ets[9]

        # zero angles, L shaped pose
        self.addconfiguration("qz", np.array([np.pi/2, np.pi/4, np.pi, np.pi, np.pi/2, np.pi/2, np.pi/2]))

        # ready pose, arm up
        self.addconfiguration("qr", np.array([np.pi/2, np.pi/4, np.pi, np.pi, np.pi/2, 0, 0]))

        # # straight and horizontal
        # self.addconfiguration("qs", np.array([0, 0, -pi/2]))

        # # nominal table top picking pose
        # self.addconfiguration("qn", np.array([0, pi/4, pi]))
    @staticmethod
    def load_my_path():
            #print(__file__)
            os.chdir(os.path.dirname(__file__))
if __name__ == '__main__':   # pragma nocover

    robot = Braccio()
    print(robot)