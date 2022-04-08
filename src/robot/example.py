from roboticstoolbox.backends.swift import *
from Braccio import Braccio
from spatialmath import SE3
import roboticstoolbox as rtb
import numpy as np
import time

robot = Braccio()
env = Swift()

q = [0, 60, 27, 90]
print(
    "Current error {}".format(
        np.linalg.norm(
            robot.fkine(np.deg2rad(q)).t -
            np.array([0.18207023, 0.03188366, 0.33000004])
        )
    )
)

ik_start_time = time.time()
qstart = np.deg2rad(q+[90, 0])
qik = robot.inverse_kinematics(
    [0.18207023, 0.03188366, 0.33000004], curr_q=qstart)
print(
    "IK took {}s with error {}".format(
        time.time() - ik_start_time,
        np.linalg.norm(
            robot.fkine(qik).t -
            np.array([0.26217005, 0.18896779, 0.051])
        )
    )
)

# Tend = SE3([-0.017, 0,  0.48])
# q = robot.ikine_LM(Tend, mask=[1, 1, 1, 0, 0, 0]).q
# q = robot.ikine_min(Tend, qlim=True).q
# robot.q = np.deg2rad(q)
robot.plot(qik)

# qt = rtb.tools.trajectory.jtraj(np.array(
#     [0, 0.3, pi/2, pi/2, pi/2]), np.array([pi/2, pi/2, pi, pi/2, pi/2]), 1000)

# for q in qt.q:
#     robot.q = q
#     env.step(3)
env.hold()
