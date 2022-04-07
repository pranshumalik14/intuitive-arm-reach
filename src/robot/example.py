from roboticstoolbox.backends.swift import *
from Braccio import Braccio
from spatialmath import SE3
import roboticstoolbox as rtb
import numpy as np
import time

robot = Braccio()
env = Swift()

print(robot)

q = [65, 45, 38, 87]
# q = [90, 145, 0, 90]
print(robot.fkine(np.deg2rad(q)).t)

ik_start_time = time.time()
qik = robot.inverse_kinematics(
    # fails to reach if start from 90,90,90,90,10...
    [0.26217005, 0.18896779, 0.051], np.deg2rad([0, 60, 27, 90, 90, 10])
)
print("IK took {}s".format(time.time() - ik_start_time))

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
