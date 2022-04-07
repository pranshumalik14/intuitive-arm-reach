import numpy as np
from roboticstoolbox.robot.ERobot import ERobot
from spatialmath import SE3
import scipy.optimize as opt
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

    def __closed_form_solution__(self, xyz, curr_q, L):
        # Arctan2(y, x)
        a1 = L[0]
        a2 = L[1]
        a3 = L[2]

        # wrist is xyz
        x_wrist = xyz[0]
        y_wrist = xyz[1]
        z_wrist = xyz[2]
        l = np.sqrt(x_wrist**2 + y_wrist**2)

        # Theta 2
        stuff = (a1**2 + a2**2 -l**2)/(2*a1*a2)
        theta2 = np.arctan2( np.sqrt(1-stuff**2) , stuff)

        # Theta 1
        in_angle  = np.arctan2(a2*np.sin(theta2), a1 + a2*np.cos(theta2))
        out_angle = np.arctan2(y_wrist, x_wrist)
        theta1 = in_angle + out_angle

        # TODO: This is wrong must fix
        theta3 = np.pi - (theta1 + theta2)
        return theta1, theta2, theta3
        
    def inverse_kinematics(self, xyz, curr_q):  # curr_q is 6D
        """
        Inverse kinematics for the Braccio Arm
        """
        L = np.array([0.125, 0.125, 0.17])
        q0 = self.qr[1:]
        zb = 0.072
        qb = [np.arctan2(xyz[1], xyz[0])]

        curr_q = curr_q[1:-2]

        min_q = self.qlim[0, 1:]
        max_q = self.qlim[1, 1:]

        R = np.sqrt(xyz[0]**2 + xyz[1]**2)
        rz = (R, xyz[2]-zb)

        def distance_to_q0(q, *args):
            w = [1, 1, 1.3]
            return np.sqrt(
                np.sum([(qi - q0i)**2 * wi
                        for qi, q0i, wi in zip(q, q0, w)])
            )

        def r_constraint(q, rz):
            qprime = np.deg2rad(np.rad2deg(q).astype(int))  # match reality
            xyzprime = self.fkine(np.append(qb, qprime)).t
            r = np.sqrt(xyzprime[0]**2 + xyzprime[1]**2)
            return r - rz[0]

        def z_constraint(q, rz):
            qprime = np.deg2rad(np.rad2deg(q).astype(int))  # match reality
            z = self.fkine(np.append(qb, qprime)).t[2] - zb
            return z - rz[1]

        def qlim_upper_constraint(q, *args):
            return max_q - q

        def qlim_lower_constraint(q, *args):
            return q - min_q

        # the optimal [shoulder, elbow, wrist] angles
        qsew = opt.fmin_slsqp(
            func=distance_to_q0,
            x0=curr_q,
            eqcons=[r_constraint,
                    z_constraint],
            ieqcons=[qlim_upper_constraint,
                     qlim_lower_constraint],
            args=(rz,),
            iprint=0
        )

        return np.append(qb, qsew)
