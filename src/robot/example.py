from roboticstoolbox.backends.swift import *
from Braccio import Braccio
from spatialmath import SE3
import roboticstoolbox as rtb
import numpy as np
from numpy import pi

robot = Braccio()
env = Swift()

print(robot)

q = [65, 45, 38, 87]
print(robot.fkine(np.deg2rad(q)).t)

# Tend = SE3([-0.017, 0,  0.48])
# q = robot.ikine_LM(Tend, mask=[1, 1, 1, 0, 0, 0]).q
# q = robot.ikine_min(Tend, qlim=True).q
robot.q = np.deg2rad(q)
robot.plot(robot.q)

# qt = rtb.tools.trajectory.jtraj(np.array(
#     [0, 0.3, pi/2, pi/2, pi/2]), np.array([pi/2, pi/2, pi, pi/2, pi/2]), 1000)

# for q in qt.q:
#     robot.q = q
#     env.step(3)
env.hold()
