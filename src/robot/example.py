from roboticstoolbox.backends.swift import *
from Braccio import Braccio
from spatialmath import SE3
import roboticstoolbox as rtb
import numpy as np

robot = Braccio()
env = Swift()

q = [90, 145, 0, 90]
print(robot.fkine(np.deg2rad(q)).t)

q = robot.inverse_kinematics(
    [0.26217005, 0.18896779, 0.051], np.deg2rad([37, 66, 27, 48, 90, 10])
)

# Tend = SE3([-0.017, 0,  0.48])
# q = robot.ikine_LM(Tend, mask=[1, 1, 1, 0, 0, 0]).q
# q = robot.ikine_min(Tend, qlim=True).q
# robot.q = np.deg2rad(q)
robot.plot(q)

# qt = rtb.tools.trajectory.jtraj(np.array(
#     [0, 0.3, pi/2, pi/2, pi/2]), np.array([pi/2, pi/2, pi, pi/2, pi/2]), 1000)

# for q in qt.q:
#     robot.q = q
#     env.step(3)
env.hold()
